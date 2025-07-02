import numpy as np
import tifffile as tiff
import os
from glob import glob
from tensorflow.keras.models import load_model, Model
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.exposure import equalize_hist
import matplotlib.pyplot as plt
from datetime import datetime

# フォルダパスとモデルパスの設定
normal_folder_path = "/Users/matsuokoujirou/Documents/Data/Screening/Noemal_cells"
anomalous_base_paths = ['/Users/matsuokoujirou/Documents/Data/Screening/Pools/241129_plate11']
result_path = f"/Users/matsuokoujirou/Documents/Data/Screening/Result/{datetime.now().strftime('%Y%m%d')}"
os.makedirs(result_path, exist_ok=True)

# Stardistモデルのロード
stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')

# ヒストグラム均質化を各チャンネルに適用
def apply_histogram_equalization(image):
    equalized_channels = [equalize_hist(image[..., i]) for i in range(image.shape[-1])]
    return np.stack(equalized_channels, axis=-1)

# グループ画像を一枚の画像にまとめて保存
def process_and_save_group_images(group_names, file_paths_list, save_path, stardist_model):
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 20))
    axes = axes.ravel()

    for i, (group_name, file_paths) in enumerate(zip(group_names, file_paths_list)):
        if not file_paths:
            print(f"No files found for {group_name}. Skipping.")
            continue

        # 最初の画像だけ処理
        file_path = file_paths[0]
        image = tiff.imread(file_path)

        # オリジナル画像（ヒストグラム均質化適用後のRGB画像）
        original_image = apply_histogram_equalization(image[..., :3])

        # マスク画像
        magenta_channel = image[..., 2]  # Magenta channel
        normalized_image = normalize(magenta_channel)
        labels, _ = stardist_model.predict_instances(normalized_image)

        # オリジナル画像を表示
        axes[i * 2].imshow(original_image)
        axes[i * 2].set_title(f'{group_name}: Original Image (Equalized)')
        axes[i * 2].axis('off')

        # マスク画像を表示
        axes[i * 2 + 1].imshow(labels, cmap='viridis')
        axes[i * 2 + 1].set_title(f'{group_name}: Mask Image')
        axes[i * 2 + 1].axis('off')

    # 余ったプロットを消す
    for j in range(len(group_names) * 2, len(axes)):
        axes[j].axis('off')

    # 保存
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "group_images_and_masks_equalized.png"))
    plt.close()
    print(f"Saved group images and masks (equalized) at {save_path}.")

# グループの名前とファイルパスのリストを作成
group_names = ["Normal"]
file_paths_list = [glob(os.path.join(normal_folder_path, '*.tif'))]

for base_path in anomalous_base_paths:
    anomalous_folders = [os.path.join(base_path, folder) for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]
    for i, folder in enumerate(anomalous_folders):
        group_names.append(f"Group_{i+1}")
        file_paths_list.append(glob(os.path.join(folder, '*.tif')))

# グループ画像を処理して保存
process_and_save_group_images(group_names, file_paths_list, result_path, stardist_model)
