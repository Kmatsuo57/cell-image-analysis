import numpy as np
import tifffile as tiff
import os
from glob import glob
import matplotlib.pyplot as plt
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.measure import regionprops, label
from skimage.segmentation import clear_border
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import tensorflow as tf
from datetime import datetime

# 外れ値除去のための真円度計算とフィルタリング
def circularity(prop):
    return 4 * np.pi * (prop.area / (prop.perimeter ** 2))

def filter_by_circularity(cell_images_green, cell_props_red, threshold=0.6):
    circularities = [circularity(prop) for prop in cell_props_red]
    valid_indices = [i for i, circ in enumerate(circularities) if circ > threshold]
    filtered_cell_images_green = [cell_images_green[i] for i in valid_indices if i < len(cell_images_green)]
    return filtered_cell_images_green, circularities

def build_autoencoder(optimizer='adam', filters_1=64, filters_2=32, filters_3=16):
    input_img = Input(shape=(64, 64, 1))
    x = Conv2D(filters_1, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(filters_2, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(filters_3, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(filters_3, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(filters_2, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(filters_1, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')
    return autoencoder

def load_data(folder_path):
    file_paths = glob(os.path.join(folder_path, '*.tif'))
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    all_cell_images_green = []
    for file_path in file_paths:
        image = tiff.imread(file_path)
        magenta_channel = image[..., 2]
        normalized_image = normalize(magenta_channel)
        labels, _ = model.predict_instances(normalized_image)
        height, width = labels.shape
        filtered_labels = np.copy(labels)
        props = regionprops(labels)
        for prop in props:
            minr, minc, maxr, maxc = prop.bbox
            if minr < 10 or minc < 10 or maxr > (height - 10) or maxc > (width - 10):
                filtered_labels[labels == prop.label] = 0
        green_channel = image[..., 1]
        cell_props_red = regionprops(labels, intensity_image=magenta_channel)
        cell_images_green = extract_cells(green_channel, filtered_labels)
        if len(cell_images_green) == len(cell_props_red):
            cell_images_green, _ = filter_by_circularity(cell_images_green, cell_props_red)
        all_cell_images_green.extend(cell_images_green)
    cell_size = (64, 64)
    resized_cell_images_green = [resize(cell, cell_size, anti_aliasing=True) for cell in all_cell_images_green]
    cell_images_array_green = np.array(resized_cell_images_green)
    cell_images_array = np.expand_dims(cell_images_array_green, axis=-1)
    cell_images_array = cell_images_array.astype('float32') / 255.
    return cell_images_array

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

def manual_grid_search_ocsvm(normal_data, ccm1_data, ko60_data, dataset24_data, autoencoder):
    encoder = Model(autoencoder.input, autoencoder.layers[-7].output)
    normal_features = encoder.predict(normal_data)
    ccm1_features = encoder.predict(ccm1_data)
    ko60_features = encoder.predict(ko60_data)
    dataset24_features = encoder.predict(dataset24_data)

    scaler = StandardScaler()
    normal_features_flat = normal_features.reshape(len(normal_features), -1)
    normal_features_scaled = scaler.fit_transform(normal_features_flat)

    param_grid = {
        'nu': [0.01, 0.05, 0.1, 0.2, 0.3],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],
        'pca_components': [100, 150, 200, 250]
    }

    best_score = -np.inf
    best_params = None

    for nu in param_grid['nu']:
        for gamma in param_grid['gamma']:
            for pca_components in param_grid['pca_components']:
                pca = PCA(n_components=pca_components)
                normal_features_reduced = pca.fit_transform(normal_features_scaled)
                oc_svm = OneClassSVM(nu=nu, gamma=gamma)
                oc_svm.fit(normal_features_reduced)

                normal_anomaly_rate = calculate_anomaly_rate(normal_features, scaler, pca, oc_svm)
                ccm1_anomaly_rate = calculate_anomaly_rate(ccm1_features, scaler, pca, oc_svm)
                ko60_anomaly_rate = calculate_anomaly_rate(ko60_features, scaler, pca, oc_svm)
                dataset24_anomaly_rate = calculate_anomaly_rate(dataset24_features, scaler, pca, oc_svm)

                score = ((ccm1_anomaly_rate - normal_anomaly_rate) +
                         (ko60_anomaly_rate - normal_anomaly_rate) +
                         (normal_anomaly_rate - dataset24_anomaly_rate))

                if score > best_score:
                    best_score = score
                    best_params = {'nu': nu, 'gamma': gamma, 'pca_components': pca_components}

    return best_params

def calculate_anomaly_rate(features, scaler, pca, oc_svm):
    features_flat = features.reshape(len(features), -1)
    features_scaled = scaler.transform(features_flat)
    features_reduced = pca.transform(features_scaled)
    predictions = oc_svm.predict(features_reduced)
    anomaly_indices = np.where(predictions == -1)[0]
    anomaly_rate = len(anomaly_indices) / len(predictions)
    return anomaly_rate

def optimize_autoencoder_and_ocsvm(normal_data, ccm1_data, ko60_data, dataset24_data):
    autoencoder = build_autoencoder()

    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    autoencoder.fit(datagen.flow(normal_data, normal_data, batch_size=16),
                    steps_per_epoch=int(len(normal_data) / 16), epochs=100, verbose=1)

    best_params = manual_grid_search_ocsvm(normal_data, ccm1_data, ko60_data, dataset24_data, autoencoder)

    return autoencoder, best_params

# データのロード
normal_data = load_data("/Users/matsuokoujirou/Documents/Data/Screening/Noemal_cells")
ccm1_data = load_data("/Users/matsuokoujirou/Documents/Data/imaging_data/240206_RV5ccm1_LC_pyrearea")
ko60_data = load_data("/Users/matsuokoujirou/Documents/Data/imaging_data/240501_KO-60")
dataset24_data = load_data("/Users/matsuokoujirou/Documents/Data/imaging_data/240404/24")

# モデルの最適化
best_autoencoder, best_params = optimize_autoencoder_and_ocsvm(normal_data, ccm1_data, ko60_data, dataset24_data)

# モデルの保存
best_autoencoder.save('/Users/matsuokoujirou/Documents/Data/Screening/Noemal_cells/best_autoencoder_v3.h5')

print(f"最適なオートエンコーダのパラメータ: {best_autoencoder.get_config()}")
print(f"最適なnuとgammaの組み合わせ: {best_params}")

# 異常率の計算
encoder = Model(best_autoencoder.input, best_autoencoder.layers[-7].output)

normal_features = encoder.predict(normal_data)
ccm1_features = encoder.predict(ccm1_data)
ko60_features = encoder.predict(ko60_data)
dataset24_features = encoder.predict(dataset24_data)

scaler = StandardScaler()
normal_features_flat = normal_features.reshape(len(normal_features), -1)
normal_features_scaled = scaler.fit_transform(normal_features_flat)
pca = PCA(n_components=best_params['pca_components'])
normal_features_reduced = pca.fit_transform(normal_features_scaled)

oc_svm = OneClassSVM(nu=best_params['nu'], gamma=best_params['gamma'])
oc_svm.fit(normal_features_reduced)

normal_anomaly_rate = calculate_anomaly_rate(normal_features, scaler, pca, oc_svm)
ccm1_anomaly_rate = calculate_anomaly_rate(ccm1_features, scaler, pca, oc_svm)
ko60_anomaly_rate = calculate_anomaly_rate(ko60_features, scaler, pca, oc_svm)
dataset24_anomaly_rate = calculate_anomaly_rate(dataset24_features, scaler, pca, oc_svm)

print(f'RG2_normal: {normal_anomaly_rate * 100:.2f}%')
print(f'ccm1: {ccm1_anomaly_rate * 100:.2f}%')
print(f'KO-60: {ko60_anomaly_rate * 100:.2f}%')
print(f'dataset24: {dataset24_anomaly_rate * 100:.2f}%')
