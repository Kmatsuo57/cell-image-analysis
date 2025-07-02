import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from stardist.models import StarDist2D
from skimage.exposure import rescale_intensity
from glob import glob
import tifffile as tiff
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM

# セル画像抽出関数
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

# セル画像のサイズ調整関数
def pad_and_resize(cell, target_size=(64, 64)):
    h, w = cell.shape[:2]
    if h > target_size[0] or w > target_size[1]:
        cell = cv2.resize(cell, target_size, interpolation=cv2.INTER_AREA)
    h, w = cell.shape[:2]
    pad_h = (target_size[0] - h) // 2
    pad_w = (target_size[1] - w) // 2
    padded = np.pad(cell, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=0)
    return padded[:target_size[0], :target_size[1]]

# データの増強関数
def augment_images(images, datagen, augment_size):
    augmented_images = []
    for img in images:
        img = img.reshape((1, ) + img.shape + (1,))
        i = 0
        for batch in datagen.flow(img, batch_size=1):
            augmented_images.append(batch[0].reshape(img.shape[1:3]))
            i += 1
            if i >= augment_size:
                break
    return augmented_images

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# フォルダ内のすべての.tifファイルを取得
folder_path = '/Users/matsuokoujirou/Documents/Data/Screening/Noemal_cells'
file_paths = glob(os.path.join(folder_path, '*.tif'))

# Stardistモデルのロード
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# すべての.tifファイルから細胞画像を抽出
all_cell_images_green = []
for file_path in file_paths:
    image = tiff.imread(file_path)
    magenta_channel = image[..., 2]
    normalized_image = rescale_intensity(magenta_channel)

    labels, details = model.predict_instances(normalized_image)

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

# サイズ調整
padded_cells = [pad_and_resize(cell) for cell in all_cell_images_green]

# データセットの分割と増強
train_size = int(0.8 * len(padded_cells))
train_cells = padded_cells[:train_size]
test_cells = padded_cells[train_size:]

augment_size = 5
augmented_train_cells = augment_images(train_cells, datagen, augment_size)
train_cells_extended = train_cells + augmented_train_cells

print(f'増強後の学習データセットのサイズ: {len(train_cells_extended)}')

# すべてのセル画像が64x64の形状を持つことを確認し、必要なら修正
train_cells_extended = [pad_and_resize(cell) for cell in train_cells_extended if cell.shape == (64, 64)]

# CAEの構築と学習
input_img = Input(shape=(64, 64, 1))

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

train_cells_extended_array = np.array(train_cells_extended).reshape(-1, 64, 64, 1)
autoencoder.fit(train_cells_extended_array, train_cells_extended_array, epochs=50, batch_size=128, shuffle=True, validation_split=0.2)

# 特徴抽出と次元削減、OC-SVMによる学習
encoder = Model(input_img, encoded)
encoded_cells_extended = encoder.predict(train_cells_extended_array)

pca = PCA(n_components=32)
pca_features_extended = pca.fit_transform(encoded_cells_extended.reshape(len(encoded_cells_extended), -1))

oc_svm = OneClassSVM(kernel='rbf', gamma='auto').fit(pca_features_extended)

test_cells_array = np.array(test_cells).reshape(-1, 64, 64, 1)
encoded_test_cells = encoder.predict(test_cells_array)
pca_test_features = pca.transform(encoded_test_cells.reshape(len(encoded_test_cells), -1))
anomaly_scores = oc_svm.decision_function(pca_test_features)
anomalies = anomaly_scores < 0

# 結果の表示
print(f'学習に使用した細胞数: {len(train_cells_extended)}')
reconstructed_imgs = autoencoder.predict(test_cells_array)

# ランダムに5つの再構成画像と入力画像を表示
num_images = 5
random_indices = np.random.choice(len(test_cells_array), num_images, replace=False)
for idx in random_indices:
    plt.figure(figsize=(4, 2))
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(test_cells_array[idx].reshape(64, 64), cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title('Reconstructed')
    plt.imshow(reconstructed_imgs[idx].reshape(64, 64), cmap='gray')
    plt.axis('off')
    plt.show()

anomaly_indices = np.where(anomalies)[0]
random_anomalies = np.random.choice(anomaly_indices, 10, replace=False)

for idx in random_anomalies:
    plt.imshow(test_cells_array[idx].reshape(64, 64), cmap='gray')
    plt.title('Anomalous Cell')
    plt.show()
