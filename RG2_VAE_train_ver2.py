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
from sklearn.model_selection import train_test_split
from sklearn import svm

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
segmentation_results = []
for file_path in file_paths:
    image = tiff.imread(file_path)
    magenta_channel = image[..., 2]
    normalized_image = normalize(magenta_channel)
    
    # セグメンテーションの実施
    labels, details = model.predict_instances(normalized_image)
    
    # セグメンテーション結果を保存（ランダム表示用）
    segmentation_results.append((image, labels))
    
    # 画像のサイズを取得
    height, width = labels.shape
    
    # 枠から10ピクセル以内にあるセグメントを除去
    filtered_labels = np.copy(labels)
    props = regionprops(labels)
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        if minr < 10 or minc < 10 or maxr > (height - 10) or maxc > (width - 10):
            filtered_labels[labels == prop.label] = 0
    
    # Venus蛍光チャンネル（green ch）の抽出
    green_channel = image[..., 1]
    
    # セグメンテーション結果から細胞画像を抽出
    cell_images_green = extract_cells(green_channel, filtered_labels)
    all_cell_images_green.extend(cell_images_green)

# すべての細胞画像を同じサイズにリサイズ
cell_size = (64, 64)
resized_cell_images_green = [resize(cell, cell_size, anti_aliasing=True) for cell in all_cell_images_green]

# NumPy配列に変換
cell_images_array_green = np.array(resized_cell_images_green)

# 抽出した細胞画像の数を確認
num_cells_green = cell_images_array_green.shape[0]
print(f"抽出されたgreen ch細胞画像の数: {num_cells_green}")

# エラー処理：細胞画像が存在しない場合
if num_cells_green == 0:
    raise ValueError("抽出された細胞画像がありません。データセットを確認してください。")

# データの形状を (64, 64, 1) に変更
cell_images_array = np.expand_dims(cell_images_array_green, axis=-1)

# データの正規化
cell_images_array = cell_images_array.astype('float32') / 255.

# トレーニングデータと検証データに分割
train_images, val_images = train_test_split(cell_images_array, test_size=0.2, random_state=42)

# VAEを用いて特徴量を抽出する
class VAE:
    def __init__(self, latent_dim=128):
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        from keras.models import Model
        from keras.layers import Input, Conv2D, Flatten, Dense, Lambda
        import tensorflow as tf

        input_img = Input(shape=(64, 64, 1))
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        z_mean = Dense(self.latent_dim)(x)
        z_log_var = Dense(self.latent_dim)(x)

        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])
        
        encoder = Model(input_img, [z_mean, z_log_var, z], name='encoder')
        return encoder
    
    def build_decoder(self):
        from keras.models import Model
        from keras.layers import Input, Conv2DTranspose, Dense, Reshape

        latent_inputs = Input(shape=(self.latent_dim,))
        x = Dense(256, activation='relu')(latent_inputs)
        x = Dense(4096, activation='relu')(x)
        x = Reshape((8, 8, 64))(x)
        x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
        x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
        x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(x)
        x = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)

        # Output shape should be (64, 64, 1)
        decoder = Model(latent_inputs, x, name='decoder')
        return decoder

    def extract_features(self, images):
        features = self.encoder.predict(images)
        return features[2]

    def reconstruct(self, latent_vectors):
        reconstructed_images = self.decoder.predict(latent_vectors)
        return reconstructed_images

vae = VAE()

# トレーニングデータから特徴量を抽出
train_features = vae.extract_features(train_images)
train_features = train_features.reshape((train_features.shape[0], -1))

# One-Class SVMのトレーニング
oc_svm = svm.OneClassSVM(kernel='rbf', gamma='auto').fit(train_features)

# 検証データから特徴量を抽出
val_features = vae.extract_features(val_images)
val_features = val_features.reshape((val_features.shape[0], -1))

# 検証データに対する異常判定
val_predictions = oc_svm.predict(val_features)
val_anomaly_rate = np.mean(val_predictions == -1)

print(f"異常細胞の割合: {val_anomaly_rate * 100:.2f}%")

# 再構成画像の表示
import random
import matplotlib.pyplot as plt

latent_vectors = vae.extract_features(val_images)
reconstructed_images = vae.reconstruct(latent_vectors)

num_images_to_display = 5
random_indices = random.sample(range(len(val_images)), num_images_to_display)

plt.figure(figsize=(12, 6))
for i, idx in enumerate(random_indices):
    plt.subplot(2, num_images_to_display, i + 1)
    plt.imshow(val_images[idx].squeeze(), cmap='gray')
    plt.title("Original Green")
    plt.axis('off')

    plt.subplot(2, num_images_to_display, i + 1 + num_images_to_display)
    plt.imshow(reconstructed_images[idx].squeeze(), cmap='gray')
    plt.title("Reconstructed Green")
    plt.axis('off')
plt.show()

# 再構成誤差の計算
reconstruction_error = np.mean(np.abs(val_images - reconstructed_images), axis=(1, 2, 3))

# 再構成誤差の分布をプロット
plt.hist(reconstruction_error, bins=50)
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Reconstruction Error for Normal Cells')
plt.show()

# 異常判定の閾値
threshold = np.percentile(reconstruction_error, 95)  # 上位5%を異常とする

# 異常細胞の判定
anomalies = reconstruction_error > threshold

# 異常細胞の割合を計算
anomaly_rate = np.mean(anomalies)

print(f"異常細胞の割合: {anomaly_rate * 100:.2f}%")
