import numpy as np
import tifffile as tiff
import os
from glob import glob
import matplotlib.pyplot as plt
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.measure import regionprops, label, perimeter
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
from skimage.segmentation import clear_border
from skimage.transform import resize
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import tensorflow as tf
from datetime import datetime

# TensorFlowがGPUを使用しないように設定
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.experimental.set_visible_devices([], 'GPU')

class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# データセットの準備
normal_folder_path = "/Users/matsuokoujirou/Documents/Data/Screening/Noemal_cells"
anomalous_folder_paths = [
    '/Users/matsuokoujirou/Documents/Data/imaging_data/240404/RV5_LC',
    '/Users/matsuokoujirou/Documents/Data/imaging_data/240520_RG2_LC',
    '/Users/matsuokoujirou/Documents/Data/imaging_data/RG2_hyg001/1',
    '/Users/matsuokoujirou/Documents/Data/imaging_data/RG2_hyg001/2',
    '/Users/matsuokoujirou/Documents/Data/imaging_data/RG2_hyg001/3',
    '/Users/matsuokoujirou/Documents/Data/imaging_data/RG2_hyg001/4',
    '/Users/matsuokoujirou/Documents/Data/imaging_data/RG2_hyg002/1',
    '/Users/matsuokoujirou/Documents/Data/imaging_data/RG2_hyg002/2',
    '/Users/matsuokoujirou/Documents/Data/imaging_data/RG2_hyg002/3',
    '/Users/matsuokoujirou/Documents/Data/imaging_data/RG2_hyg002/4',
    "/Users/matsuokoujirou/Documents/Data/imaging_data/240206_RV5ccm1_LC_pyrearea", 
    '/Users/matsuokoujirou/Documents/Data/imaging_data/240501_KO-60',
    '/Users/matsuokoujirou/Documents/Data/imaging_data/240501_KO-62'
]
anomalous_labels = ['RV5', 'RG2', '1-1', "1-2", '1-3', '1-4', '2-1', "2-2", '2-3', '2-4', 'ccm1', 'KO-60', 'KO-62']

# 今日の日付を取得
today = datetime.now().strftime("%Y%m%d")

# 結果を保存するパスを設定
result_path = f"/Users/matsuokoujirou/Documents/Data/Screening/Result/{today}/"
os.makedirs(result_path, exist_ok=True)

normal_file_paths = glob(os.path.join(normal_folder_path, '*.tif'))
anomalous_file_paths_list = [glob(os.path.join(path, '*.tif')) for path in anomalous_folder_paths]

# Stardistモデルのロード
stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')

def extract_cells(image, labels):
    cell_images = []
    cell_props = []
    labeled_cells = label(labels)
    labeled_cells = clear_border(labeled_cells)
    props = regionprops(labeled_cells, intensity_image=image)
    
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        cell_image = image[minr:maxr, minc:maxc]
        cell_images.append(cell_image)
        cell_props.append(prop)
    
    return cell_images, cell_props

def circularity(prop):
    perim = perimeter(prop.image)
    area = prop.area
    if perim == 0:
        return 0
    return 4 * np.pi * area / (perim ** 2)

# すべての.tifファイルから細胞画像を抽出
def extract_features_and_images(file_paths):
    all_cell_images_green = []
    all_props_red = []
    for file_path in file_paths:
        image = tiff.imread(file_path)
        magenta_channel = image[..., 2]
        normalized_image = normalize(magenta_channel)
        
        labels, details = stardist_model.predict_instances(normalized_image)
        
        height, width = labels.shape
        filtered_labels = np.copy(labels)
        props = regionprops(labels)
        for prop in props:
            minr, minc, maxr, maxc = prop.bbox
            if minr < 10 or minc < 10 or maxr > (height - 10) or maxc > (width - 10):
                filtered_labels[labels == prop.label] = 0
        
        green_channel = image[..., 1]
        red_channel = image[..., 0]
        
        cell_images_green, _ = extract_cells(green_channel, filtered_labels)
        _, cell_props_red = extract_cells(red_channel, filtered_labels)
        all_cell_images_green.extend(cell_images_green)
        all_props_red.extend(cell_props_red)

    return all_cell_images_green, all_props_red

# 外れ値除去のための真円度計算とフィルタリング
def filter_by_circularity(cell_images_green, cell_props_red, threshold=0.6):
    circularities = [circularity(prop) for prop in cell_props_red]
    valid_indices = [i for i, circ in enumerate(circularities) if circ > threshold]
    filtered_cell_images_green = [cell_images_green[i] for i in valid_indices]
    return filtered_cell_images_green, circularities

normal_images_green, normal_props_red = extract_features_and_images(normal_file_paths)
normal_images_green, normal_circularities = filter_by_circularity(normal_images_green, normal_props_red)

# Debug: Print the number of normal images
print(f"Number of normal images: {len(normal_images_green)}")

anomalous_images_green_list = []
anomalous_props_red_list = []
anomalous_circularities_list = []

for anomalous_file_paths in anomalous_file_paths_list:
    anomalous_images_green, anomalous_props_red = extract_features_and_images(anomalous_file_paths)
    anomalous_images_green, anomalous_circularities = filter_by_circularity(anomalous_images_green, anomalous_props_red)
    anomalous_images_green_list.append(anomalous_images_green)
    anomalous_circularities_list.append(anomalous_circularities)

# すべての画像をリサイズして統一
def resize_images(cell_images):
    cell_size = (64, 64)
    resized_cell_images = [resize(cell, cell_size, anti_aliasing=True) for cell in cell_images]
    resized_cell_images = [cell for cell in resized_cell_images if cell.shape == (64, 64)]
    return np.array(resized_cell_images)

normal_images_green = resize_images(normal_images_green)
anomalous_images_green_list = [resize_images(anomalous_images_green) for anomalous_images_green in anomalous_images_green_list]

# Debug: Print the shape of normal_images_green after resizing
print(f"Shape of normal_images_green after resizing: {normal_images_green.shape}")

# 学習済みエンコーダのロード
encoder_path = '/Users/matsuokoujirou/Documents/Data/Screening/Noemal_cells/RG2_encoder.h5'
encoder = load_model(encoder_path, custom_objects={'Sampling': Sampling})

# 入力画像を2チャンネルに変換
def convert_to_2channel(images):
    images = np.expand_dims(images, axis=-1)  # 1チャンネルに次元を追加
    images_2channel = np.concatenate([images, images], axis=-1)  # 2チャンネルに拡張
    return images_2channel

normal_images_green_2channel = convert_to_2channel(normal_images_green)
anomalous_images_green_list_2channel = [convert_to_2channel(anomalous_images_green) for anomalous_images_green in anomalous_images_green_list]

# Debug: Print the shape of normal_images_green_2channel
print(f"Shape of normal_images_green_2channel: {normal_images_green_2channel.shape}")

# 特徴抽出
normal_features = encoder.predict(normal_images_green_2channel)
anomalous_features_list = [encoder.predict(anomalous_images_green_2channel) for anomalous_images_green_2channel in anomalous_images_green_list_2channel]

# z_meanとz_log_varを個別に分離
z_mean_normal = normal_features[0]
z_log_var_normal = normal_features[1]
anomalous_features_list_mean = [features[0] for features in anomalous_features_list]

# 訓練データとテストデータに分割
normal_features_train, normal_features_test = train_test_split(z_mean_normal, test_size=0.2, random_state=42)

# フラット化とスケーリング
scaler = StandardScaler()
normal_features_train_flatten = normal_features_train.reshape(len(normal_features_train), -1)
normal_features_test_flatten = normal_features_test.reshape(len(normal_features_test), -1)
normal_features_train_scaled = scaler.fit_transform(normal_features_train_flatten)
normal_features_test_scaled = scaler.transform(normal_features_test_flatten)

# PCAとOC-SVMのパラメータを調整
dimension = min(100, normal_features_train_flatten.shape[0], normal_features_train_flatten.shape[1])
pca = PCA(n_components=dimension)
normal_features_train_reduced = pca.fit_transform(normal_features_train_scaled)

# OC-SVMの学習
oc_svm = OneClassSVM(gamma='scale', nu=0.01).fit(normal_features_train_reduced)

# 異常率を計算
def calculate_anomaly_rate(features):
    features_reshaped = features.reshape(features.shape[0], -1)
    features_scaled = scaler.transform(features_reshaped)
    features_reduced = pca.transform(features_scaled)
    predictions = oc_svm.predict(features_reduced)
    anomaly_rate = np.mean(predictions == -1) * 100
    return anomaly_rate

# 各異常細胞データセットに対して異常率を計算
anomaly_rates = [calculate_anomaly_rate(features) for features in anomalous_features_list_mean]

# 各異常株ごとに異常率を表示
for label, rate in zip(anomalous_labels, anomaly_rates):
    print(f"Abnormal Cell Type: {label}, Anomaly Rate: {rate:.2f}%")

# 異常率の棒グラフを作成
plt.figure(figsize=(12, 8))
plt.bar(anomalous_labels, anomaly_rates, color='skyblue')
plt.xlabel('Abnormal Cell Types', fontsize=14)
plt.ylabel('Anomaly Rate (%)', fontsize=14)
plt.title('Anomaly Rates for Different Abnormal Cell Types', fontsize=16)
plt.xticks(rotation=45)
plt.savefig(os.path.join(result_path, 'anomaly_rates.png'))
plt.show()
