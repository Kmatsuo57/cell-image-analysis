import os
import random
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

from glob import glob

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input

from stardist.models import StarDist2D
from csbdeep.utils import normalize

from skimage.measure import regionprops, label
from skimage.segmentation import clear_border
from skimage.transform import resize

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


###############################
# パラメータやパスの設定
###############################
normal_folder_path = '/Users/matsuokoujirou/Documents/Data/imaging_data/240404/RV5_teacher'
model_path         = '/Users/matsuokoujirou/Documents/Data/imaging_data/240404/RV5_teacher/RV5_CAE.h5'

anomalous_folder_paths = [
    "/Users/matsuokoujirou/Documents/Data/imaging_data/240206_RV5ccm1_LC_pyrearea", 
    '/Users/matsuokoujirou/Documents/Data/imaging_data/240501_KO-60',
    '/Users/matsuokoujirou/Documents/Data/imaging_data/240501_KO-62'
]
anomalous_labels = ['ccm1', 'KO-60', 'KO-62']


###############################
# 基本的な関数定義
###############################
def extract_cells(image, labels):
    """
    ラベル画像(labels)から、個々のセル領域の画素値を
    image から切り出して返す。
    """
    cell_images = []
    labeled_cells = label(labels)
    labeled_cells = clear_border(labeled_cells)
    props = regionprops(labeled_cells, intensity_image=image)
    
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        cell_image = image[minr:maxr, minc:maxc]
        cell_images.append(cell_image)
    
    return cell_images


def extract_features_with_images(file_paths, encoder, stardist_model):
    """
    画像ファイル群を読み込み、Stardistでセグメンテーションしてセル領域を抽出。
    各セル画像(緑チャネル + 赤チャネル)を(64,64)にリサイズし、
    オートエンコーダのEncoder部分で特徴量を返すと同時に、
    可視化用に緑チャネル(64x64)のリサイズ後セル画像をリストとして返す。
    """
    all_cell_images_green = []
    all_cell_images_red   = []
    
    # 可視化用(ここでは緑チャネルのみを表示する想定)
    vis_images_green = []
    
    for file_path in file_paths:
        image = tiff.imread(file_path)

        # Stardist用のマゼンタチャネル(=image[...,2])を正規化
        magenta_channel = image[..., 2]
        normalized_image = normalize(magenta_channel)

        # Stardistで細胞セグメンテーション
        labels, details = stardist_model.predict_instances(normalized_image)

        # 端のセルを除外 (例: 10ピクセルのマージン)
        height, width = labels.shape
        filtered_labels = np.copy(labels)
        props = regionprops(labels)
        for prop in props:
            minr, minc, maxr, maxc = prop.bbox
            if (minr < 10) or (minc < 10) or (maxr > (height - 10)) or (maxc > (width - 10)):
                filtered_labels[labels == prop.label] = 0

        # 緑チャネル, 赤チャネルを切り出し
        green_channel = image[..., 1]
        red_channel   = image[..., 0]

        # セル画像をextract
        cell_images_green = extract_cells(green_channel, filtered_labels)
        cell_images_red   = extract_cells(red_channel,   filtered_labels)
        
        # 可視化用にも格納 (緑チャネルのみ)
        vis_images_green.extend(cell_images_green)
        
        all_cell_images_green.extend(cell_images_green)
        all_cell_images_red.extend(cell_images_red)

    # (64,64) にリサイズ (CAEと同じサイズ)
    cell_size = (64, 64)
    resized_cell_images_green = [resize(cell, cell_size, anti_aliasing=True) for cell in all_cell_images_green]
    resized_cell_images_red   = [resize(cell, cell_size, anti_aliasing=True) for cell in all_cell_images_red]

    # 形状を (N, 64, 64, 2) にまとめ、0~1にスケーリング
    cell_images_array_green = np.array(resized_cell_images_green)
    cell_images_array_red   = np.array(resized_cell_images_red)
    cell_images_array = np.stack((cell_images_array_green, cell_images_array_red), axis=-1)
    cell_images_array = cell_images_array.astype('float32') / 255.

    # エンコーダの出力(特徴量)を取得
    features = encoder.predict(cell_images_array)

    # 可視化用の緑チャネル (64x64) の配列も 0~1スケールに
    vis_images_green_resized = [
        resize(img, cell_size, anti_aliasing=True) for img in vis_images_green
    ]
    vis_images_green_resized = np.array(vis_images_green_resized).astype('float32') / 255.

    return features, vis_images_green_resized


###############################
# Stardistモデル・CAEのロード
###############################
stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
autoencoder    = load_model(model_path)

# Encoder部分の切り出し
input_img      = Input(shape=(64, 64, 2))  # CAEが学習するときの入力形状に合わせる
encoded_output = autoencoder.layers[5].output  # エンコーダの最後の層を取り出す(例:第5層)
encoder        = Model(inputs=autoencoder.input, outputs=encoded_output)


#################################
# 1) 正常データ: 全セルから抽出
#################################
normal_file_paths = glob(os.path.join(normal_folder_path, '*.tif'))
normal_features_all, normal_images_all = extract_features_with_images(normal_file_paths, encoder, stardist_model)

# この "normal_features_all" を学習・推定に使うが、学習はそのうちの一部でOK (train_test_split)
normal_features_train, normal_features_test = train_test_split(
    normal_features_all, test_size=0.2, random_state=42
)

# スケーリング
scaler = StandardScaler()
normal_features_train_flat = normal_features_train.reshape(len(normal_features_train), -1)
normal_features_train_scaled = scaler.fit_transform(normal_features_train_flat)

# One-Class SVM 学習
oc_svm = OneClassSVM(gamma='auto', nu=0.05).fit(normal_features_train_scaled)

# 学習後、正常データ全体を推定 (可視化用に +1 判定セルを取る)
normal_features_all_flat   = normal_features_all.reshape(len(normal_features_all), -1)
normal_features_all_scaled = scaler.transform(normal_features_all_flat)
normal_preds = oc_svm.predict(normal_features_all_scaled)  # +1 => 正常判定, -1 => 異常判定

# 正常データのうち、正常判定されたインデックスを取得
normal_correct_indices = np.where(normal_preds == 1)[0]
if len(normal_correct_indices) < 5:
    print(f"警告: 正常データで正常判定されたセルが {len(normal_correct_indices)} 個しかありません。")
# ランダムに5つピックアップ (無ければあるだけ)
selected_normal_indices = random.sample(list(normal_correct_indices), min(5, len(normal_correct_indices)))


#################################
# 2) 異常データ: 全セルから抽出
#################################
anomalous_file_paths_list = [glob(os.path.join(path, '*.tif')) for path in anomalous_folder_paths]

anomalous_features_list_vis = []
anomalous_images_list_vis   = []
for paths_ in anomalous_file_paths_list:
    feats, imgs = extract_features_with_images(paths_, encoder, stardist_model)
    anomalous_features_list_vis.append(feats)
    anomalous_images_list_vis.append(imgs)


###############################
# 3) 異常データを SVM で判定
#    → -1(異常)セルを可視化
###############################
anomaly_indices_list = []
for i, feats in enumerate(anomalous_features_list_vis):
    feats_flat   = feats.reshape(len(feats), -1)
    feats_scaled = scaler.transform(feats_flat)
    preds        = oc_svm.predict(feats_scaled)  # -1 が異常判定

    anomaly_indices = np.where(preds == -1)[0]
    if len(anomaly_indices) < 5:
        print(f"警告: {anomalous_labels[i]} は異常判定されたセルが {len(anomaly_indices)} 個しかありません。")
    selected_anomaly = random.sample(list(anomaly_indices), min(5, len(anomaly_indices)))
    anomaly_indices_list.append(selected_anomaly)


###############################
# 4) 正常 & 異常セルを 5×4 枚で並べて表示
###############################
fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(10, 12))

# -- 1列目(0番目のcol)に「正常(+1判定)」のセルを表示 --
for row in range(5):
    ax = axes[row, 0]
    ax.axis('off')
    if row < len(selected_normal_indices):
        idx = selected_normal_indices[row]
        ax.imshow(normal_images_all[idx], cmap='gray')
    else:
        ax.text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=12)
    if row == 0:
        ax.set_title("Normal")

# -- 2～4列目に各異常株の異常セルを表示 --
for col in range(1, 4):
    i = col - 1  # anomalous_labels / images_list のインデックス
    label_name    = anomalous_labels[i]
    anomaly_imgs  = anomalous_images_list_vis[i]
    selected_inds = anomaly_indices_list[i]

    for row in range(5):
        ax = axes[row, col]
        ax.axis('off')
        if row < len(selected_inds):
            idx = selected_inds[row]
            ax.imshow(anomaly_imgs[idx], cmap='gray')
        else:
            ax.text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=12)
        if row == 0:
            ax.set_title(label_name)

plt.tight_layout()
plt.show()
