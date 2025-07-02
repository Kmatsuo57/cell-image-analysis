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
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape, Conv2DTranspose, Lambda, BatchNormalization, Layer, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
import random
import time
import tensorflow as tf

# TensorFlowがGPUを使用しないように設定
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.experimental.set_visible_devices([], 'GPU')

# 開始時間の計測
start_time = time.time()

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

# データ前処理部分
def preprocess_data(folder_path):
    file_paths = glob(os.path.join(folder_path, '*.tif'))
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    all_cell_images_green = []
    all_cell_images_red = []
    segmentation_results = []
    for file_path in file_paths:
        image = tiff.imread(file_path)
        magenta_channel = image[..., 2]
        normalized_image = normalize(magenta_channel)
        
        labels, details = model.predict_instances(normalized_image)
        segmentation_results.append((image, labels))
        
        height, width = labels.shape
        filtered_labels = np.copy(labels)
        props = regionprops(labels)
        for prop in props:
            minr, minc, maxr, maxc = prop.bbox
            if minr < 10 or minc < 10 or maxr > (height - 10) or maxc > (width - 10):
                filtered_labels[labels == prop.label] = 0
        
        green_channel = image[..., 1]
        red_channel = image[..., 0]
        
        cell_images_green = extract_cells(green_channel, filtered_labels)
        cell_images_red = extract_cells(red_channel, filtered_labels)
        all_cell_images_green.extend(cell_images_green)
        all_cell_images_red.extend(cell_images_red)

    cell_size = (64, 64)
    resized_cell_images_green = [resize(cell, cell_size, anti_aliasing=True) for cell in all_cell_images_green]
    resized_cell_images_red = [resize(cell, cell_size, anti_aliasing=True) for cell in all_cell_images_red]

    cell_images_array_green = np.array(resized_cell_images_green)
    cell_images_array_red = np.array(resized_cell_images_red)

    num_cells_green = cell_images_array_green.shape[0]
    num_cells_red = cell_images_array_red.shape[0]
    print(f"抽出されたgreen ch細胞画像の数: {num_cells_green}")
    print(f"抽出されたred ch細胞画像の数: {num_cells_red}")

    if num_cells_green == 0 or num_cells_red == 0:
        raise ValueError("抽出された細胞画像がありません。データセットを確認してください。")

    cell_images_array = np.stack((cell_images_array_green, cell_images_array_red), axis=-1)
    return cell_images_array, segmentation_results

folder_path = '/Users/matsuokoujirou/Documents/Data/Screening/Noemal_cells/'
cell_images_array, segmentation_results = preprocess_data(folder_path)

# データのスケーリング
cell_images_array = cell_images_array.astype('float32') / 255.

# 入力データの確認
print(f"Input data range: min={np.min(cell_images_array)}, max={np.max(cell_images_array)}")

num_images_to_display = min(10, cell_images_array.shape[0])
plt.figure(figsize=(12, 12))
for i in range(num_images_to_display):
    plt.subplot(1, num_images_to_display, i+1)
    plt.imshow(cell_images_array[i, :, :, 0], cmap='gray')
    plt.title("Green Channel")
    plt.axis('off')
plt.savefig('green_channel_images.png')
plt.close()

plt.figure(figsize=(12, 12))
for i in range(num_images_to_display):
    plt.subplot(1, num_images_to_display, i+1)
    plt.imshow(cell_images_array[i, :, :, 1], cmap='gray')
    plt.title("Chlorophyll")
    plt.axis('off')
plt.savefig('red_channel_images.png')
plt.close()

input_shape = (64, 64, 2)

class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = Sampling()([z_mean, z_log_var])
        reconstructed = self.decoder(z)
        reconstruction_loss = MeanSquaredError()(inputs, reconstructed)
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.add_loss(vae_loss)
        return reconstructed

def create_model():
    latent_dim = 128  # 潜在次元をさらに増やす
    inputs = Input(shape=input_shape)
    x = Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(inputs)  # フィルター数を増やし、正則化を追加
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, padding='same')(x)
    x = Dropout(0.5)(x)  # ドロップアウトを追加
    x = Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)  # フィルター数を増やし、正則化を追加
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, padding='same')(x)
    x = Dropout(0.5)(x)  # ドロップアウトを追加
    x = Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)  # フィルター数を増やし、正則化を追加
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, padding='same')(x)
    x = Dropout(0.5)(x)  # ドロップアウトを追加
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)  # ユニット数を増やし、正則化を追加
    x = BatchNormalization()(x)

    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    encoder = Model(inputs, [z_mean, z_log_var], name='encoder')

    decoder_input = Input(shape=(latent_dim,))
    x = Dense(8*8*256, activation='relu', kernel_regularizer=l2(0.001))(decoder_input)  # ユニット数を増やし、正則化を追加
    x = Reshape((8, 8, 256))(x)
    x = Conv2DTranspose(256, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(2)(x)
    x = Dropout(0.5)(x)  # ドロップアウトを追加
    x = Conv2DTranspose(128, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(2)(x)
    x = Dropout(0.5)(x)  # ドロップアウトを追加
    x = Conv2DTranspose(64, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(2)(x)
    outputs = Conv2DTranspose(2, 3, activation='sigmoid', padding='same', kernel_regularizer=l2(0.001))(x)  # 最後の層の正則化を追加

    decoder = Model(decoder_input, outputs, name='decoder')

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=Adam(learning_rate=0.0001))  # 学習率をさらに低く設定
    return vae, encoder, decoder

best_vae, encoder, decoder = create_model()

# エンコーダー出力の確認
z_mean, z_log_var = best_vae.encoder.predict(cell_images_array)
print(f"z_mean: {z_mean}")
print(f"z_log_var: {z_log_var}")

# サンプル潜在変数を生成してデコーダーに通す
sample_z = z_mean[0:1]  # 最初のサンプルを使用
reconstructed_sample = best_vae.decoder.predict(sample_z)
print(f"Reconstructed sample shape: {reconstructed_sample.shape}")

# 再構成サンプルの表示
plt.figure(figsize=(6, 6))
plt.subplot(1, 2, 1)
plt.imshow(cell_images_array[0, :, :, 0], cmap='gray')
plt.title("Original Green")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_sample[0, :, :, 0], cmap='gray')
plt.title("Reconstructed Green")
plt.axis('off')
plt.savefig('sample_reconstruction_check.png')
plt.close()

# デコーダーが生成した出力の確認
reconstructed_images = best_vae.predict(cell_images_array)
print(f"Reconstructed images shape: {reconstructed_images.shape}")

# 再構成データの範囲の確認
print(f"Reconstructed data range: min={np.min(reconstructed_images)}, max={np.max(reconstructed_images)}")

# 再構成画像の表示
num_images_to_display = 5
random_indices = random.sample(range(cell_images_array.shape[0]), num_images_to_display)

plt.figure(figsize=(12, 6))
for i, idx in enumerate(random_indices):
    plt.subplot(2, num_images_to_display, i + 1)
    plt.imshow(cell_images_array[idx, :, :, 0], cmap='gray')
    plt.title("Original Green")
    plt.axis('off')

    plt.subplot(2, num_images_to_display, i + 1 + num_images_to_display)
    plt.imshow(reconstructed_images[idx, :, :, 0], cmap='gray', vmin=0, vmax=1)
    plt.title("Reconstructed Green")
    plt.axis('off')
plt.savefig(folder_path + 'original_vs_reconstructed_green.png')
plt.close()

# データ増強
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# モデルのトレーニング
batch_size = 16
epochs = 50 # エポック数を100に設定

early_stopping = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, min_lr=1e-5)

steps_per_epoch = len(cell_images_array) // batch_size

# データジェネレータをtf.data.Datasetに変換して繰り返す
def generator():
    while True:
        for batch in datagen.flow(cell_images_array, batch_size=batch_size):
            yield batch, batch

dataset = tf.data.Dataset.from_generator(generator, output_signature=(tf.TensorSpec(shape=(None, 64, 64, 2), dtype=tf.float32), tf.TensorSpec(shape=(None, 64, 64, 2), dtype=tf.float32)))

history = best_vae.fit(dataset,
                       steps_per_epoch=steps_per_epoch,
                       epochs=epochs,
                       callbacks=[early_stopping, reduce_lr])

plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(folder_path + 'training_loss.png')
plt.close()

# 再構成誤差の計算
reconstruction_error = np.mean(np.power(cell_images_array - reconstructed_images, 2), axis=(1, 2, 3))

plt.hist(reconstruction_error, bins=50)
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Reconstruction Error for Normal Cells')
plt.savefig(folder_path + 'reconstruction_error_distribution.png')
plt.close()

threshold = np.percentile(reconstruction_error, 95)

anomalies = reconstruction_error > threshold

anomaly_rate = np.mean(anomalies)


# エンコーダとデコーダを個別に保存
encoder.save(folder_path + 'RG2_encoder.h5')
decoder.save(folder_path + 'RG2_decoder.h5')




best_vae.save(folder_path + 'RG2_VAE_all_pyrenoid_v2.h5')

print("Models have been saved.")

print(f"異常細胞の割合: {anomaly_rate * 100:.2f}%")

# 終了時間の計測
end_time = time.time()
print("Execution time: ", end_time - start_time, "seconds")


