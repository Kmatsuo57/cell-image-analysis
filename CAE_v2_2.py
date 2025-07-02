import numpy as np
import tifffile as tiff
import os
from glob import glob
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.measure import regionprops, label as skimage_label
from skimage.segmentation import clear_border
from skimage.transform import resize
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
from datetime import datetime

# TensorFlowがGPUを使用しないように設定
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.experimental.set_visible_devices([], 'GPU')

def calculate_fluorescence(roi):
    mean_value = np.mean(roi)
    std_dev = np.std(roi)
    return mean_value, std_dev

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def extract_cells(image, labels):
    cell_images = []
    circularities = []
    labeled_cells = skimage_label(labels)
    labeled_cells = clear_border(labeled_cells)
    props = regionprops(labeled_cells, intensity_image=image)
    
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        cell_image = image[minr:maxr, minc:maxc]
        cell_images.append(cell_image)
        
        # Calculate circularity
        perimeter = prop.perimeter
        area = prop.area
        if perimeter != 0:
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            circularities.append(circularity)
    
    return cell_images, circularities

# データセットの準備
normal_folder_path = "/Users/matsuokoujirou/Documents/Data/Screening/Noemal_cells"
model_path = '/Users/matsuokoujirou/Documents/Data/Screening/Noemal_cells/RG2_CAE_green.h5'
anomalous_base_paths = [
        '/Volumes/NO NAME/240622',
        '/Volumes/NO NAME/240630/plate7',
        '/Volumes/NO NAME/240630/plate8',
        '/Volumes/NO NAME/240630/plate9',
        '/Volumes/NO NAME/240630/plate10',
        ]
# 今日の日付を取得
today = datetime.now().strftime("%Y%m%d")

# 結果を保存するパスを設定
result_path = f"/Users/matsuokoujirou/Documents/Data/Screening/Result/{today}"
os.makedirs(result_path, exist_ok=True)

normal_file_paths = glob(os.path.join(normal_folder_path, '*.tif'))

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
    all_circularities = []
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

        cell_images_green, circularities = extract_cells(green_channel, filtered_labels)
        all_cell_images_green.extend(cell_images_green)
        all_circularities.extend(circularities)

    # Remove outliers based on circularity
    circularity_df = pd.DataFrame({'Circularity': all_circularities})
    cleaned_df = remove_outliers(circularity_df, 'Circularity')
    valid_indices = cleaned_df.index

    cell_size = (64, 64)
    resized_cell_images_green = [resize(all_cell_images_green[i], cell_size, anti_aliasing=True) for i in valid_indices]
    cell_images_array_green = np.array(resized_cell_images_green)
    cell_images_array = np.expand_dims(cell_images_array_green, axis=-1)
    cell_images_array = cell_images_array.astype('float32') / 255.

    features = encoder.predict(cell_images_array)
    return features, resized_cell_images_green

# 正常細胞と異常細胞の全データセットから特徴量と画像を抽出
normal_features, normal_images = extract_features_and_images(normal_file_paths, encoder)

# 訓練データとテストデータに分割
normal_features_train, normal_features_test = train_test_split(normal_features, test_size=0.2, random_state=42)

# フラット化とスケーリング
scaler = StandardScaler()
normal_features_train_flatten = normal_features_train.reshape(len(normal_features_train), -1)
normal_features_train_scaled = scaler.fit_transform(normal_features_train_flatten)

# 次元数200に次元削減
dimension = 150
pca = PCA(n_components=dimension)
normal_features_train_reduced = pca.fit_transform(normal_features_train_scaled)

# OC-SVMの学習
oc_svm = OneClassSVM(gamma='auto', nu=0.08).fit(normal_features_train_reduced)

# 異常率を計算
anomaly_rates = []

# 正常細胞の異常率
normal_features_test_flatten = normal_features_test.reshape(len(normal_features_test), -1)
normal_features_test_scaled = scaler.transform(normal_features_test_flatten)
normal_features_test_reduced = pca.transform(normal_features_test_scaled)
normal_predictions = oc_svm.predict(normal_features_test_reduced)
normal_anomaly_rate = np.sum(normal_predictions == -1) / len(normal_predictions)
anomaly_rates.append(normal_anomaly_rate)

all_anomalous_labels = []
all_anomalous_anomaly_rates = []

for base_path in anomalous_base_paths:
    anomalous_folders = [os.path.join(base_path, folder) for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]
    anomalous_labels = [os.path.basename(folder) for folder in anomalous_folders]
    
    # 各異常フォルダ内の.tifファイルを取得
    anomalous_file_paths_list = [glob(os.path.join(folder, '*.tif')) for folder in anomalous_folders]
    
    anomalous_features_list = []
    anomalous_images_list = []
    for anomalous_file_paths in anomalous_file_paths_list:
        features, images = extract_features_and_images(anomalous_file_paths, encoder)
        anomalous_features_list.append(features)
        anomalous_images_list.append(images)

    # 各異常株に対して異常率を計算
    for label, anomalous_features in zip(anomalous_labels, anomalous_features_list):
        anomalous_features_flatten = anomalous_features.reshape(len(anomalous_features), -1)
        anomalous_features_scaled = scaler.transform(anomalous_features_flatten)
        anomalous_features_reduced = pca.transform(anomalous_features_scaled)
        predictions = oc_svm.predict(anomalous_features_reduced)
        anomaly_rate = np.sum(predictions == -1) / len(predictions)
        all_anomalous_labels.append(f"{label}_{base_path.split('/')[-1]}")
        all_anomalous_anomaly_rates.append(anomaly_rate)

# ラベルに"Normal"を追加
all_anomalous_labels.insert(0, "Normal")
all_anomalous_anomaly_rates.insert(0, normal_anomaly_rate)

# ラベルと異常率を組み合わせてソート（"Normal"は最初に）
label_rate_pairs = sorted(zip(all_anomalous_labels[1:], all_anomalous_anomaly_rates[1:]), key=lambda x: x[0])
sorted_labels, sorted_anomaly_rates = zip(*label_rate_pairs)
sorted_labels = ["Normal"] + list(sorted_labels)
sorted_anomaly_rates = [normal_anomaly_rate] + list(sorted_anomaly_rates)

# 異常率の棒グラフを作成
plt.figure(figsize=(12, 8))
plt.bar(sorted_labels, [rate * 100 for rate in sorted_anomaly_rates], color='skyblue')
plt.xlabel('Samples')
plt.ylabel('Anomaly Rate (%)')
plt.title('Anomaly Rate by Sample')
plt.xticks(rotation=90)

# Normal 細胞の異常率プラマイ3%の帯を追加
normal_rate = normal_anomaly_rate * 100
plt.axhline(y=normal_rate, color='r', linestyle='--', label='Normal Anomaly Rate')
plt.fill_between([-0.5, len(sorted_labels) - 0.5], normal_rate, normal_rate + 3, color='red', alpha=0.2, label='Normal Anomaly Rate + 3%')

plt.legend()
plt.tight_layout()
plt.savefig(result_path + "/anomaly_rates_candidates.png")
plt.show()

# 異常率のパーセンテージとそれに対応する名前をプリント
for label, rate in zip(sorted_labels, sorted_anomaly_rates):
    print(f'{label}: {rate * 100:.2f}%')

# OC-SVMの境界を可視化
xx, yy = np.meshgrid(np.linspace(normal_features_train_reduced[:, 0].min(), normal_features_train_reduced[:, 0].max(), 500),
                     np.linspace(normal_features_train_reduced[:, 1].min(), normal_features_train_reduced[:, 1].max(), 500))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid)
grid_reduced = pca.transform(grid_scaled)
Z = oc_svm.decision_function(grid_reduced)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(12, 8))
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
plt.scatter(normal_features_train_reduced[:, 0], normal_features_train_reduced[:, 1], c='white', s=20, edgecolors='k')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('OC-SVM Decision Boundary')
plt.savefig(result_path + "/oc_svm_boundary_150.png")
plt.show()
