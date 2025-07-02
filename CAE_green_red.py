import numpy as np
import tifffile as tiff
import os
from glob import glob
import matplotlib.pyplot as plt
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.measure import regionprops, label, perimeter
from skimage.segmentation import clear_border
from skimage.transform import resize
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from datetime import datetime

# TensorFlowがGPUを使用しないように設定
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.experimental.set_visible_devices([], 'GPU')

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
result_path = f"/Users/matsuokoujirou/Documents/Data/Screening/Result/{today}"

# フォルダが存在しない場合は作成
os.makedirs(result_path, exist_ok=True)

normal_file_paths = glob(os.path.join(normal_folder_path, '*.tif'))
anomalous_file_paths_list = [glob(os.path.join(path, '*.tif')) for path in anomalous_folder_paths]

# Stardistモデルのロード
stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')

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

# Greenチャンネルのオートエンコーダー学習
# エポック数を増やし、データ拡張の範囲を広げる
def train_autoencoder(cell_images_array, color, folder_path):
    cell_images_array = cell_images_array.astype('float32') / 255.0
    cell_images_array = np.expand_dims(cell_images_array, axis=-1)
    
    num_cells = cell_images_array.shape[0]
    print(f"抽出された{color} ch細胞画像の数: {num_cells}")
    
    if num_cells == 0:
        raise ValueError("抽出された細胞画像がありません。データセットを確認してください。")

    input_img = Input(shape=(64, 64, 1))

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    autoencoder.summary()

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    autoencoder.fit(datagen.flow(cell_images_array, cell_images_array, batch_size=16),
                    steps_per_epoch=int(len(cell_images_array) / 16), epochs=100)

    encoder = Model(input_img, encoded)

    encoded_images = encoder.predict(cell_images_array)
    encoded_images_flat = encoded_images.reshape((encoded_images.shape[0], -1))

    scaler = StandardScaler()
    encoded_images_flat = scaler.fit_transform(encoded_images_flat)

    pca = PCA(n_components=100)
    encoded_images_reduced = pca.fit_transform(encoded_images_flat)

    oc_svm = OneClassSVM(kernel='rbf', gamma=0.01, nu=0.1).fit(encoded_images_reduced)

    autoencoder.save(os.path.join(folder_path, f'RG2_CAE_{color}.keras'))

    return encoder, oc_svm, scaler, pca, cell_images_array



# Greenチャンネル用オートエンコーダーとOC-SVM
encoder_green, oc_svm_green, scaler_green, pca_green, cell_images_array_green = train_autoencoder(normal_images_green, 'green', normal_folder_path)



# 異常細胞を表示するためのリストを準備
anomalous_images_to_display = []

# 各異常株に対して異常細胞の画像を抽出
for anomalous_images_green in anomalous_images_green_list:
    if len(anomalous_images_green.shape) == 3:
        anomalous_images_green = np.expand_dims(anomalous_images_green, axis=-1)
        
    anomalous_features_flatten_green = encoder_green.predict(anomalous_images_green).reshape(len(anomalous_images_green), -1)
    
    anomalous_features_scaled_green = scaler_green.transform(anomalous_features_flatten_green)
    
    anomalous_features_reduced_green = pca_green.transform(anomalous_features_scaled_green)
    
    predictions_green = oc_svm_green.predict(anomalous_features_reduced_green)
    
    anomaly_indices = np.where(predictions_green == -1)[0]
    if len(anomaly_indices) > 10:
        anomaly_indices = np.random.choice(anomaly_indices, 10, replace=False)
    selected_images_green = [anomalous_images_green[i] for i in anomaly_indices]
    anomalous_images_to_display.append(selected_images_green)

# 再度異常率の表示と異常細胞のプロット
fig, axes = plt.subplots(len(anomalous_labels), 11, figsize=(22, 2 * len(anomalous_labels)))

for i, (label, images_green, rate) in enumerate(zip(anomalous_labels, anomalous_images_to_display, anomaly_rates)):
    for j, image_green in enumerate(images_green):
        ax_green = axes[i, j]
        ax_green.imshow(image_green, cmap='gray')
        ax_green.axis('off')
    
    ax_bar = axes[i, 10]
    ax_bar.barh([0], [rate * 100], color='red')
    ax_bar.set_xlim(0, 10)
    ax_bar.set_yticks([])
    ax_bar.set_xticks([0, 10])
    ax_bar.set_xticklabels(['0%', '10%'])
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['bottom'].set_visible(False)
    ax_bar.spines['left'].set_visible(False)
    if j == 0:
        ax_bar.set_ylabel(label, rotation=0, size='large', labelpad=80)

plt.tight_layout()
plt.show()

# 真円度の分布表示
all_circularities = [circ for circularities in anomalous_circularities_list for circ in circularities]
plt.hist(all_circularities, bins=50)
plt.xlabel('Circularity')
plt.ylabel('Frequency')
plt.title('Distribution of Circularity for Anomalous Cells')
plt.show()

import subprocess

subprocess.Popen(["python", "Documents/Data/imaging_data/scripts/notify.py"])
