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

# フォルダ内のすべての.tifファイルを取得
folder_path = "/Users/matsuokoujirou/Documents/Data/Screening/Noemal_cells"
file_paths = glob(os.path.join(folder_path, '*.tif'))

# Stardistモデルのロード
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# すべての.tifファイルから細胞画像を抽出
all_cell_images_green = []
for file_path in file_paths:
    image = tiff.imread(file_path)
    magenta_channel = image[..., 2]
    normalized_image = normalize(magenta_channel)
    
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

cell_size = (64, 64)
resized_cell_images_green = [resize(cell, cell_size, anti_aliasing=True) for cell in all_cell_images_green]

cell_images_array_green = np.array(resized_cell_images_green)

num_cells_green = cell_images_array_green.shape[0]
print(f"抽出されたgreen ch細胞画像の数: {num_cells_green}")

if num_cells_green == 0:
    raise ValueError("抽出された細胞画像がありません。データセットを確認してください。")

cell_images_array = np.expand_dims(cell_images_array_green, axis=-1)

# 細胞画像のサンプルを可視化
num_images_to_display = min(10, len(resized_cell_images_green))
plt.figure(figsize=(12, 12))
for i in range(num_images_to_display):
    plt.subplot(1, num_images_to_display, i+1)
    plt.imshow(resized_cell_images_green[i], cmap='gray')
    plt.title("Green Channel")
    plt.axis('off')
plt.show()

# データ拡張
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# オートエンコーダーの構築
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
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()

# 損失を記録しながらトレーニング
cell_images_array = cell_images_array.astype('float32') / 255.

history = autoencoder.fit(
    datagen.flow(cell_images_array, cell_images_array, batch_size=16),
    steps_per_epoch=int(len(cell_images_array) / 16),
    epochs=50
)

# 損失のプロット
losses = history.history['loss']
epochs = range(1, len(losses) + 1)
num_cells = len(cell_images_array)

plt.figure(figsize=(12, 6))

# 損失のグラフ
plt.subplot(1, 2, 1)
plt.plot(epochs, losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()

# 使用した細胞数のグラフ
plt.subplot(1, 2, 2)
plt.bar(epochs, [num_cells] * len(epochs))
plt.xlabel('Epoch')
plt.ylabel('Number of Cells')
plt.title('Number of Cells Used in Training Over Epochs')

plt.tight_layout()
plt.show()

# エンコーダー部分の抽出
encoder = Model(input_img, encoded)

# エンコードされた画像を取得
encoded_images = encoder.predict(cell_images_array)

# エンコードされた特徴を1次元に変換
encoded_images_flat = encoded_images.reshape((encoded_images.shape[0], -1))

# One-Class SVMのトレーニング
oc_svm = OneClassSVM(kernel='rbf', gamma='auto').fit(encoded_images_flat)

# 再構成画像と再構成誤差を計算
reconstructed_images = autoencoder.predict(cell_images_array)
reconstruction_error = np.mean(np.power(cell_images_array - reconstructed_images, 2), axis=(1, 2, 3))

# 再構成誤差のヒストグラムを表示
plt.hist(reconstruction_error, bins=50)
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Reconstruction Error for Normal Cells')
plt.show()

# 異常検出
threshold = np.percentile(reconstruction_error, 95)
anomalies = reconstruction_error > threshold
anomaly_rate = np.mean(anomalies)

print(f"異常細胞の割合: {anomaly_rate * 100:.2f}%")

# ランダムに5つの再構成画像と入力画像を表示
num_images_to_display = 5
random_indices = np.random.choice(len(cell_images_array), num_images_to_display, replace=False)
plt.figure(figsize=(12, 6))
for i, idx in enumerate(random_indices):
    plt.subplot(2, num_images_to_display, i+1)
    plt.imshow(cell_images_array[idx].reshape(64, 64), cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(2, num_images_to_display, i+1+num_images_to_display)
    plt.imshow(reconstructed_images[idx].reshape(64, 64), cmap='gray')
    plt.title('Reconstructed')
    plt.axis('off')
plt.show()

# モデルの保存
autoencoder.save(folder_path + '/RG2_CAE_green_test.h5')
