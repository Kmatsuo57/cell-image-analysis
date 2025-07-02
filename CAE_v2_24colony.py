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
import tensorflow as tf
from datetime import datetime
import re

# TensorFlowがGPUを使用しないように設定
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.experimental.set_visible_devices([], 'GPU')

def extract_cells(image, labels):
    cell_images = []
    labeled_cells = skimage_label(labels)
    labeled_cells = clear_border(labeled_cells)
    props = regionprops(labeled_cells, intensity_image=image)
    
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        cell_image = image[minr:maxr, minc:maxc]
        cell_images.append(cell_image)
    print(len(cell_images))
    return cell_images

# データセットの準備
normal_folder_path = "/Users/matsuokoujirou/Documents/Data/Screening/Noemal_cells"
model_path = '/Users/matsuokoujirou/Documents/Data/Screening/Noemal_cells/RG2_CAE_green.h5'
anomalous_base_paths = [
   '/Volumes/KOJIRO MATS/240712/results'
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

# 訓練データとテストデータに分割
normal_features_train, normal_features_test = train_test_split(normal_features, test_size=0.2, random_state=42)

# フラット化とスケーリング
scaler = StandardScaler()
normal_features_train_flatten = normal_features_train.reshape(len(normal_features_train), -1)
normal_features_train_scaled = scaler.fit_transform(normal_features_train_flatten)

# 次元数200に次元削減
dimension = 250
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

# 各異常フォルダ内の個別の.tifファイルに対して異常率を計算
for base_path in anomalous_base_paths:
    anomalous_file_paths = glob(os.path.join(base_path, '*.tif'))
    for file_path in anomalous_file_paths:
        features, images = extract_features_and_images([file_path], encoder)
        features_flatten = features.reshape(len(features), -1)
        features_scaled = scaler.transform(features_flatten)
        features_reduced = pca.transform(features_scaled)
        predictions = oc_svm.predict(features_reduced)
        anomaly_rate = np.sum(predictions == -1) / len(predictions)
        label = os.path.basename(file_path)
        all_anomalous_labels.append(label)
        all_anomalous_anomaly_rates.append(anomaly_rate)

# 自然順序でソートする関数
def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]

# ラベルと異常率を組み合わせて自然順でソート
label_rate_pairs = sorted(zip(all_anomalous_labels, all_anomalous_anomaly_rates), key=lambda x: natural_sort_key(x[0]))
sorted_labels, sorted_anomaly_rates = zip(*label_rate_pairs)

# ラベルに"Normal"を追加（最後に）
sorted_labels = list(sorted_labels) + ["Normal"]
sorted_anomaly_rates = list(sorted_anomaly_rates) + [normal_anomaly_rate]

# 異常率の棒グラフを作成
plt.figure(figsize=(12, 8))
plt.bar(sorted_labels, [rate * 100 for rate in sorted_anomaly_rates], color='skyblue')
plt.xlabel('Samples')
plt.ylabel('Anomaly Rate (%)')
plt.title('Anomaly Rate by Image')
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig(result_path + "/anomaly_rates_by_image.png")
plt.show()

# 異常率のパーセンテージとそれに対応する名前をプリント
for label, rate in zip(sorted_labels, sorted_anomaly_rates):
    print(f'{label}: {rate * 100:.2f}%')
