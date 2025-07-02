import numpy as np
import tifffile as tiff
import os
from glob import glob
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.measure import regionprops, label
from skimage.segmentation import clear_border
from skimage.transform import resize
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from datetime import datetime

# TensorFlowがGPUを使用しないように設定
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.experimental.set_visible_devices([], 'GPU')

def extract_cells(image, labels):
    cell_images = []
    labeled_cells = label(labels)
    labeled_cells = clear_border(labeled_cells)
    props = regionprops(labeled_cells, intensity_image=image)
    
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        cell_image = image[minr:maxr, minc:maxc]
        cell_images.append(cell_image)
    
    return cell_images

# データセットの準備
normal_folder_path = "/Users/matsuokoujirou/Documents/Data/Screening/Noemal_cells"
model_path = '/Users/matsuokoujirou/Documents/Data/Screening/Noemal_cells/RG2_CAE_green.h5'
anomalous_folder_paths = [
    "/Users/matsuokoujirou/Documents/Data/Screening/Noemal_cells",
    '/Users/matsuokoujirou/Documents/Data/imaging_data/240404/12',
    '/Users/matsuokoujirou/Documents/Data/imaging_data/RG2_hyg001/1',
    '/Users/matsuokoujirou/Documents/Data/imaging_data/RG2_hyg001/2',
    '/Users/matsuokoujirou/Documents/Data/imaging_data/RG2_hyg001/3',
    '/Users/matsuokoujirou/Documents/Data/imaging_data/RG2_hyg001/4',
    '/Users/matsuokoujirou/Documents/Data/imaging_data/RG2_hyg002/1',
    '/Users/matsuokoujirou/Documents/Data/imaging_data/RG2_hyg002/2',
    '/Users/matsuokoujirou/Documents/Data/imaging_data/RG2_hyg002/3',
    '/Users/matsuokoujirou/Documents/Data/imaging_data/RG2_hyg002/4'
]
anomalous_labels = ['RG2_normal', 'RG2', '1-1', "1-2", '1-3', '1-4', '2-1', "2-2", '2-3', '2-4']

# 今日の日付を取得
today = datetime.now().strftime("%Y%m%d")

# 結果を保存するパスを設定
result_path = f"/Users/matsuokoujirou/Documents/Data/Screening/Result/{today}"
os.makedirs(result_path, exist_ok=True)

normal_file_paths = glob(os.path.join(normal_folder_path, '*.tif'))
anomalous_file_paths_list = [glob(os.path.join(path, '*.tif')) for path in anomalous_folder_paths]

# Stardistモデルのロード
stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')

# 学習済みオートエンコーダのロード
autoencoder = load_model(model_path)

# エンコーダ部分の抽出
encoder_input = autoencoder.input
encoder_output = autoencoder.get_layer(index=6).output  # エンコーダの最終層の出力（ここでは6層目のMaxPooling2D層）
encoder = Model(inputs=encoder_input, outputs=encoder_output)

# セグメンテーションと特徴抽出の計算
def extract_features_and_images(file_paths, encoder):
    all_cell_images_green = []
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

        cell_images_green = extract_cells(green_channel, filtered_labels)
        all_cell_images_green.extend(cell_images_green)

    cell_size = (64, 64)
    resized_cell_images_green = [resize(cell, cell_size, anti_aliasing=True) for cell in all_cell_images_green]
    cell_images_array_green = np.array(resized_cell_images_green)
    cell_images_array = np.expand_dims(cell_images_array_green, axis=-1)
    cell_images_array = cell_images_array.astype('float32') / 255.

    features = encoder.predict(cell_images_array)
    return features, resized_cell_images_green

# 正常細胞と異常細胞の全データセットから特徴量と画像を抽出
normal_features, normal_images = extract_features_and_images(normal_file_paths, encoder)
anomalous_features_list = []
anomalous_images_list = []
for anomalous_file_paths in anomalous_file_paths_list:
    features, images = extract_features_and_images(anomalous_file_paths, encoder)
    anomalous_features_list.append(features)
    anomalous_images_list.append(images)

# 訓練データとテストデータに分割
normal_features_train, normal_features_test = train_test_split(normal_features, test_size=0.2, random_state=42)

# フラット化とスケーリング
scaler = StandardScaler()
normal_features_train_flatten = normal_features_train.reshape(len(normal_features_train), -1)
normal_features_train_scaled = scaler.fit_transform(normal_features_train_flatten)

# 次元数200に次元削減
dimension = 200
pca = PCA(n_components=dimension)
normal_features_train_reduced = pca.fit_transform(normal_features_train_scaled)

# OC-SVMの学習
oc_svm = OneClassSVM(gamma='auto', nu=0.05).fit(normal_features_train_reduced)

# 異常細胞の画像を表示するためのリストを準備
anomalous_images_to_display = []
anomaly_rates = []

# 各異常株に対して異常細胞の画像を抽出
for anomalous_features, anomalous_images in zip(anomalous_features_list, anomalous_images_list):
    anomalous_features_flatten = anomalous_features.reshape(len(anomalous_features), -1)
    anomalous_features_scaled = scaler.transform(anomalous_features_flatten)
    anomalous_features_reduced = pca.transform(anomalous_features_scaled)
    predictions = oc_svm.predict(anomalous_features_reduced)
    anomaly_indices = np.where(predictions == -1)[0]
    
    # 異常率を計算
    anomaly_rate = len(anomaly_indices) / len(predictions)
    anomaly_rates.append(anomaly_rate)
    
    # 異常細胞を5つランダムに選択
    if len(anomaly_indices) > 5:
        anomaly_indices = np.random.choice(anomaly_indices, 5, replace=False)
    selected_images = [anomalous_images[i] for i in anomaly_indices]
    anomalous_images_to_display.append(selected_images)

# 異常細胞を表示
fig, axes = plt.subplots(len(anomalous_labels), 6, figsize=(15, 2 * len(anomalous_labels)))

for i, (label, images, rate) in enumerate(zip(anomalous_labels, anomalous_images_to_display, anomaly_rates)):
    for j, image in enumerate(images):
        ax = axes[i, j]
        ax.imshow(image, cmap='gray')
        if j == 0:
            ax.set_ylabel(label, rotation=0, size='large', labelpad=80)
        ax.axis('off')
    
    # 異常率の棒グラフ
    ax_bar = axes[i, 5]
    ax_bar.barh([0], [rate * 100], color='red')
    ax_bar.set_xlim(0, 100)
    ax_bar.set_yticks([])
    ax_bar.set_xticks([0, 100])
    ax_bar.set_xticklabels(['0%', '100%'])
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['bottom'].set_visible(False)
    ax_bar.spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig(result_path + "/anomalous_cells_v2.png")
plt.show()

# 異常率のパーセンテージとそれに対応する名前をプリント
for label, rate in zip(anomalous_labels, anomaly_rates):
    print(f'{label}: {rate * 100:.2f}%')
