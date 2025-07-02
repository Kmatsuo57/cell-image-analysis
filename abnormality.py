import numpy as np
import pandas as pd
import os
from glob import glob
import tifffile as tiff

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.measure import regionprops, label as skimage_label
from skimage.segmentation import clear_border
from skimage.transform import resize
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import math
from datetime import datetime

# TensorFlowがGPUを使用しないように設定
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.experimental.set_visible_devices([], 'GPU')


################################################################################
# 1. 前処理・設定
################################################################################

def remove_outliers(df, column):
    """
    特定カラム(column)の外れ値をIQRに基づき除外する。
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def extract_cells(image, labels):
    """
    セグメンテーションラベル(labels)に基づき、個々の細胞領域を切り出す。
    circularity（円形度）も算出して返す。
    """
    cell_images = []
    circularities = []
    
    labeled_cells = skimage_label(labels)
    labeled_cells = clear_border(labeled_cells)
    props = regionprops(labeled_cells, intensity_image=image)
    
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        cell_image = image[minr:maxr, minc:maxc]
        cell_images.append(cell_image)
        
        # Calculate circularity = (4π * area) / (perimeter^2)
        perimeter = prop.perimeter
        area = prop.area
        if perimeter != 0:
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            circularities.append(circularity)
        else:
            circularities.append(0)
    
    return cell_images, circularities


################################################################################
# 2. Stardist + オートエンコーダEncoderで特徴抽出
################################################################################

def extract_features_and_images(file_paths, stardist_model, encoder):
    """
    - file_paths: 対象となる tif ファイルのパス一覧
    - stardist_model: StarDist2D のモデル
    - encoder: 事前学習済みオートエンコーダのエンコーダ部分

    返り値:
      features: (N, latent_dim, ...) の特徴ベクトル（flattenする前でもOK）
      resized_cell_images: それに対応する細胞画像のリスト(要素数N)
    """
    all_cell_images = []
    all_circularities = []
    
    for file_path in file_paths:
        image_stack = tiff.imread(file_path)  # (H, W, C)
        
        # 例: マゼンタチャネルでセグメンテーション
        #     [R, G, B] = [0, 1, 2] という順番の例
        magenta_channel = image_stack[..., 2]
        
        # Stardistで予測するために0~1正規化
        normalized_image = normalize(magenta_channel)
        labels, details = stardist_model.predict_instances(normalized_image)

        # 境界の外側10pxを切るなどの例
        height, width = labels.shape
        filtered_labels = np.copy(labels)
        props = regionprops(labels)
        for prop in props:
            minr, minc, maxr, maxc = prop.bbox
            if (minr < 10 or minc < 10 or 
                maxr > (height - 10) or maxc > (width - 10)):
                filtered_labels[labels == prop.label] = 0
        
        # 今度はGチャネルの画素を細胞画像として切り出す
        green_channel = image_stack[..., 1]
        
        cell_imgs, circularities = extract_cells(green_channel, filtered_labels)
        all_cell_images.extend(cell_imgs)
        all_circularities.extend(circularities)

    # 円形度に基づいて外れ値を除去
    circularity_df = pd.DataFrame({'Circularity': all_circularities})
    cleaned_df = remove_outliers(circularity_df, 'Circularity')
    valid_indices = cleaned_df.index

    # すべて64x64にリサイズ
    cell_size = (64, 64)
    resized_cell_images = [
        resize(all_cell_images[i], cell_size, anti_aliasing=True)
        for i in valid_indices
    ]

    # エンコーダへの入力形状に合わせる (batch, H, W, 1)
    cell_images_array = np.expand_dims(resized_cell_images, axis=-1)
    cell_images_array = cell_images_array.astype('float32') / 255.

    # 特徴抽出 (encoder)
    features = encoder.predict(cell_images_array)

    return features, resized_cell_images


################################################################################
# 3. メイン実行部
################################################################################
if __name__ == "__main__":
    
    # === (A) ファイルパスやモデルの設定 ===
    normal_folder_path = "/Users/xxx/.../Noemal_cells"  # 正常細胞画像が入ったフォルダ
    model_path = "/Users/xxx/.../RG2_CAE_green.h5"      # 学習済みオートエンコーダ
    anomalous_base_paths = [
       "/Volumes/xxx/240726/"  # 異常細胞フォルダ
    ]
    
    # 今日の日付
    today = datetime.now().strftime("%Y%m%d")
    # 結果を保存するパス (存在しなければ作成)
    result_path = f"/Users/xxx/.../Result/{today}"
    os.makedirs(result_path, exist_ok=True)
    
    # === (B) データ読み込み & ファイルリスト ===
    normal_file_paths = glob(os.path.join(normal_folder_path, '*.tif'))
    
    # === (C) Stardistモデルとオートエンコーダのロード ===
    stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
    autoencoder = load_model(model_path)

    # オートエンコーダのEncoder部分だけを取り出す
    # 例: index=6 の MaxPooling2D の出力を最終とするモデル、といったイメージ
    encoder_input = autoencoder.input
    encoder_output = autoencoder.get_layer(index=6).output
    encoder = Model(inputs=encoder_input, outputs=encoder_output)

    # === (D) 正常細胞から特徴抽出 ===
    normal_features, normal_images = extract_features_and_images(
        normal_file_paths,
        stardist_model,
        encoder
    )

    # === (E) One-Class SVM の学習準備 ===
    #     - train_test_splitで分ける（正常データ内で学習とテストを分割）
    normal_features_train, normal_features_test = train_test_split(
        normal_features, test_size=0.2, random_state=42
    )
    
    # スケーリング & PCA
    scaler = StandardScaler()
    normal_features_train_flat = normal_features_train.reshape(len(normal_features_train), -1)
    normal_features_train_scaled = scaler.fit_transform(normal_features_train_flat)

    dimension = 150  # PCA次元
    pca = PCA(n_components=dimension)
    normal_features_train_reduced = pca.fit_transform(normal_features_train_scaled)

    # === (F) OC-SVM の学習 ===
    oc_svm = OneClassSVM(gamma='auto', nu=0.08).fit(normal_features_train_reduced)

    # === (G) 正常細胞テストでの異常率を算出 ===
    normal_features_test_flat = normal_features_test.reshape(len(normal_features_test), -1)
    normal_features_test_scaled = scaler.transform(normal_features_test_flat)
    normal_features_test_reduced = pca.transform(normal_features_test_scaled)
    normal_predictions = oc_svm.predict(normal_features_test_reduced)
    normal_anomaly_rate = np.mean(normal_predictions == -1)
    print(f"Normal test anomaly rate: {normal_anomaly_rate*100:.2f}%")

    # === (H) 異常データを同様に処理 & 異常度を算出 ===
    for base_path in anomalous_base_paths:
        # base_path配下にあるフォルダ全てを異常サンプルとして取り扱う例
        anomalous_folders = [
            os.path.join(base_path, folder)
            for folder in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, folder))
        ]
        
        for folder in anomalous_folders:
            label = os.path.basename(folder)
            anomalous_files = glob(os.path.join(folder, '*.tif'))
            if len(anomalous_files) == 0:
                continue

            # 特徴抽出
            anomalous_features, anomalous_images = extract_features_and_images(
                anomalous_files, stardist_model, encoder
            )

            # SVM予測
            anomalous_features_flat = anomalous_features.reshape(len(anomalous_features), -1)
            anomalous_features_scaled = scaler.transform(anomalous_features_flat)
            anomalous_features_reduced = pca.transform(anomalous_features_scaled)

            predictions = oc_svm.predict(anomalous_features_reduced)
            scores = oc_svm.decision_function(anomalous_features_reduced)
            # scores が「大きいほど正常寄り / 小さいほど異常寄り」となります(RBFカーネルOC-SVMの場合)
            
            anomaly_rate = np.mean(predictions == -1)
            print(f"[{label}] total cells={len(predictions)}  anomaly_rate={anomaly_rate*100:.2f}%")

            #------------------------------------------------------------------
            # ここで「異常度が高い順(= scoresが小さい順)」に細胞画像を並べる
            #------------------------------------------------------------------
            sort_idx = np.argsort(scores)  # scoresが小さい(=もっとも異常)順に並びます
            sorted_images = [anomalous_images[i] for i in sort_idx]
            sorted_scores = [scores[i] for i in sort_idx]
            
            # 上位N枚を可視化したい場合
            N = min(30, len(sorted_images))  # 例: 30枚だけ表示
            most_anomalous_imgs = sorted_images[:N]
            most_anomalous_scores = sorted_scores[:N]
            
            # サブプロットで表示する
            num_cols = 5
            num_rows = math.ceil(N / num_cols)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(3*num_cols, 3*num_rows))
            axes = axes.flatten() if num_rows > 1 else [axes]

            for ax_i, ax in enumerate(axes):
                if ax_i < N:
                    ax.imshow(most_anomalous_imgs[ax_i], cmap='gray')
                    ax.set_title(f"score={most_anomalous_scores[ax_i]:.3f}")
                    ax.axis('off')
                else:
                    ax.axis('off')
            
            plt.suptitle(f"Most anomalous cells: {label}")
            plt.tight_layout()
            
            # 画像保存
            save_path = os.path.join(result_path, f"{label}_top{N}_anomalous_cells.png")
            plt.savefig(save_path)
            plt.close()  # 画面には表示せずファイル保存のみの場合

    print("Done.")
