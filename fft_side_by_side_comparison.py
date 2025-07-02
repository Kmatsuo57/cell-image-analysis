import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import glob
from skimage.io import imread


def plot_side_by_side_fft_comparison(wt_cells, mutant_cells, wt_name, mutant_name, output_path, n_samples=50):
    """
    WTと変異株の平均FFT像（振幅）を上段に、
    各50個のオリジナル画像とそのFFT像を下段にグリッドで並べて比較する
    """
    # サンプル数を揃える
    n_samples = min(n_samples, len(wt_cells), len(mutant_cells))
    wt_cells = wt_cells[:n_samples]
    mutant_cells = mutant_cells[:n_samples]

    # FFT振幅画像を計算
    def calc_fft_magnitude(img):
        img = img.astype(np.float64)
        img = (img - np.mean(img)) / np.std(img)
        fft = np.fft.fftshift(np.fft.fft2(img))
        mag = np.abs(fft)
        return np.log1p(mag)

    wt_fft_imgs = [calc_fft_magnitude(img) for img in wt_cells]
    mutant_fft_imgs = [calc_fft_magnitude(img) for img in mutant_cells]

    # 平均FFT像
    wt_fft_mean = np.mean(wt_fft_imgs, axis=0)
    mutant_fft_mean = np.mean(mutant_fft_imgs, axis=0)

    # 図のレイアウト
    grid_cols = 10
    grid_rows = (n_samples + grid_cols - 1) // grid_cols
    fig_height = 2 + 2 * grid_rows  # 上段+下段
    fig, axes = plt.subplots(2 + 2*grid_rows, 2, figsize=(2*grid_cols, fig_height))

    # 上段: 平均FFT像
    axes[0, 0].imshow(wt_fft_mean, cmap='viridis')
    axes[0, 0].set_title(f'{wt_name} Mean FFT', fontsize=12)
    axes[0, 0].axis('off')
    axes[0, 1].imshow(mutant_fft_mean, cmap='viridis')
    axes[0, 1].set_title(f'{mutant_name} Mean FFT', fontsize=12)
    axes[0, 1].axis('off')

    # 下段: オリジナル画像とFFT像
    for i in range(n_samples):
        row = i // grid_cols
        col = i % grid_cols
        # WT
        axes[1 + row, 0].imshow(wt_cells[i], cmap='gray')
        axes[1 + row, 0].set_title(f'{wt_name} Cell {i+1}', fontsize=8)
        axes[1 + row, 0].axis('off')
        axes[1 + grid_rows + row, 0].imshow(wt_fft_imgs[i], cmap='viridis')
        axes[1 + grid_rows + row, 0].set_title(f'{wt_name} FFT {i+1}', fontsize=8)
        axes[1 + grid_rows + row, 0].axis('off')
        # Mutant
        axes[1 + row, 1].imshow(mutant_cells[i], cmap='gray')
        axes[1 + row, 1].set_title(f'{mutant_name} Cell {i+1}', fontsize=8)
        axes[1 + row, 1].axis('off')
        axes[1 + grid_rows + row, 1].imshow(mutant_fft_imgs[i], cmap='viridis')
        axes[1 + grid_rows + row, 1].set_title(f'{mutant_name} FFT {i+1}', fontsize=8)
        axes[1 + grid_rows + row, 1].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def load_cell_images_from_folder(folder, max_cells=1000):
    """指定フォルダ内の画像ファイルをすべて読み込む（tif, png, jpg対応）"""
    image_files = sorted(glob.glob(os.path.join(folder, '*.tif')) +
                         glob.glob(os.path.join(folder, '*.png')) +
                         glob.glob(os.path.join(folder, '*.jpg')))
    images = [imread(f) for f in image_files[:max_cells]]
    return images


def plot_all_strains_fft_grid(strain_cells_dict, strain_names, output_path, n_samples=50, orientation='row'):
    """
    全ての株の個別細胞FFT像をグリッドで並べて比較表示
    orientation='row'なら株ごとに行、'col'なら株ごとに列
    """
    # 各株ごとにn_samples個の細胞画像を取得しFFT像を計算
    fft_imgs_per_strain = []
    for strain in strain_names:
        cells = strain_cells_dict[strain][:n_samples]
        fft_imgs = []
        for img in cells:
            img = img.astype(np.float64)
            img = (img - np.mean(img)) / np.std(img)
            fft = np.fft.fftshift(np.fft.fft2(img))
            mag = np.abs(fft)
            fft_img = np.log1p(mag)
            fft_imgs.append(fft_img)
        fft_imgs_per_strain.append(fft_imgs)

    # グリッドサイズ
    n_strains = len(strain_names)
    n_cells = n_samples
    if orientation == 'row':
        nrows, ncols = n_strains, n_cells
    else:
        nrows, ncols = n_cells, n_strains

    # 画像サイズ取得
    img_shape = fft_imgs_per_strain[0][0].shape
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*1.2, nrows*1.2))

    for i, strain in enumerate(strain_names):
        for j in range(n_cells):
            if orientation == 'row':
                ax = axes[i, j]
            else:
                ax = axes[j, i]
            if j < len(fft_imgs_per_strain[i]):
                ax.imshow(fft_imgs_per_strain[i][j], cmap='viridis')
            ax.axis('off')
            if j == 0:
                # 行ラベル
                ax.set_ylabel(strain, fontsize=10, rotation=0, labelpad=30, va='center')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
def main():
    # 出力ディレクトリをcell_fft_analyzer.pyと同じ形式で作成
    output_dir = f"/Users/matsuokoujirou/Documents/Data/Screening/FFT_Analysis_RG2_WT/{datetime.now().strftime('%Y%m%d_%H%M')}"
    os.makedirs(output_dir, exist_ok=True)

    # WTと変異株のパス
    base_dir = "/Users/matsuokoujirou/Documents/Data/Screening/Candidates/LConly"
    wt_name = 'RG2_LC (WT)'
    wt_folder = os.path.join(base_dir, 'RG2_LC')
    mutant_list = [
        ('10A_LC', '10A_LC'),
        ('10B_LC', '10B_LC'),
        ('10D_LC', '10D_LC'),
        ('6C1_LC', '6C1_LC'),
    ]

    # WT細胞画像をロード
    wt_cells = load_cell_images_from_folder(wt_folder, max_cells=1000)

    for mutant_label, mutant_folder in mutant_list:
        mutant_cells = load_cell_images_from_folder(os.path.join(base_dir, mutant_folder), max_cells=1000)
        if len(wt_cells) == 0 or len(mutant_cells) == 0:
            print(f"Skipping {mutant_label}: No images found.")
            continue
        output_path = os.path.join(output_dir, f'fft_comparison_RG2_LC_vs_{mutant_label}.png')
        plot_side_by_side_fft_comparison(wt_cells, mutant_cells, wt_name, mutant_label, output_path, n_samples=50)

if __name__ == "__main__":
    main() 