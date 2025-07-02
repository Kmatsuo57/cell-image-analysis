import numpy as np
import tifffile as tiff
import os
from glob import glob
import matplotlib.pyplot as plt
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.measure import regionprops
from skimage.transform import resize
from skimage import exposure
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import pandas as pd
from datetime import datetime
import shutil

class CellFFTAnalyzer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.setup_environment()
        
    def setup_environment(self):
        """環境設定"""
        RANDOM_SEED = 42
        np.random.seed(RANDOM_SEED)
        
    def extract_quality_cells(self, image_path, stardist_model):
        """品質管理付き細胞抽出"""
        try:
            image = tiff.imread(image_path)
            
            # チャンネル分離
            if image.ndim == 3 and image.shape[-1] >= 3:
                seg_channel = image[..., 2]  # セグメンテーション用
                green_channel = image[..., 1]  # 解析用
            else:
                seg_channel = image
                green_channel = image
            
            # StarDist セグメンテーション
            normalized_seg = normalize(seg_channel)
            labels, details = stardist_model.predict_instances(normalized_seg)
            
            # 品質フィルタリング
            height, width = labels.shape
            props = regionprops(labels)
            
            quality_cells = []
            cell_stats = []
            
            for prop in props:
                minr, minc, maxr, maxc = prop.bbox
                
                # 境界チェック
                if (minr < 10 or minc < 10 or maxr > (height - 10) or maxc > (width - 10)):
                    continue
                
                # サイズチェック
                if prop.area < 200 or prop.area > 8000:
                    continue
                
                # 形状チェック（細長すぎる細胞を除外）
                if prop.eccentricity > 0.95:
                    continue
                
                # 細胞画像抽出
                cell_image = green_channel[minr:maxr, minc:maxc]
                
                # 強度チェック
                cell_mean = np.mean(cell_image)
                cell_std = np.std(cell_image)
                
                # 極端に暗い細胞や均一すぎる細胞を除外
                if cell_mean < 0.5 or cell_std < 0.1:
                    continue
                
                # 画像の正規化とリサイズ
                cell_image_eq = exposure.equalize_adapthist(cell_image, clip_limit=0.02)
                cell_image_resized = resize(cell_image_eq, (64, 64), anti_aliasing=True)
                
                quality_cells.append(cell_image_resized)
                
                # 統計情報記録
                cell_stats.append({
                    'area': prop.area,
                    'eccentricity': prop.eccentricity,
                    'solidity': prop.solidity,
                    'mean_intensity': cell_mean,
                    'std_intensity': cell_std,
                    'file': os.path.basename(image_path)
                })
            
            return quality_cells, cell_stats
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return [], []
    
    def calculate_average_fft(self, cell_images):
        """各細胞の2次元フーリエ変換を計算し、平均FFT像を出力"""
        print("=== Calculating Average 2D Fourier Transform ===")
        
        if len(cell_images) == 0:
            print("No cell images available for FFT analysis")
            return None
        
        # 各細胞のFFTを計算
        fft_magnitudes = []
        
        for i, cell_img in enumerate(cell_images):
            if i % 100 == 0:
                print(f"Processing FFT for cell {i+1}/{len(cell_images)}")
            
            # 画像をfloat32に変換し、正規化
            cell_float = cell_img.astype(np.float32)
            
            # 2次元フーリエ変換
            fft_result = fft2(cell_float)
            
            # フーリエ変換結果をシフト（低周波成分を中心に）
            fft_shifted = fftshift(fft_result)
            
            # マグニチュード（振幅）を計算
            magnitude = np.abs(fft_shifted)
            
            # 対数スケールで表示（低周波成分の詳細を見やすくする）
            magnitude_log = np.log1p(magnitude)
            
            fft_magnitudes.append(magnitude_log)
        
        # 平均FFT像を計算
        average_fft = np.mean(fft_magnitudes, axis=0)
        
        print(f"Average FFT calculated from {len(cell_images)} cells")
        
        # 結果を可視化
        self.visualize_fft_results(average_fft, fft_magnitudes)
        
        # 平均FFT像を保存
        np.save(os.path.join(self.output_dir, 'average_fft.npy'), average_fft)
        
        return average_fft, fft_magnitudes
    
    def visualize_fft_results(self, average_fft, fft_magnitudes):
        """FFT結果の可視化"""
        print("=== Visualizing FFT Results ===")
        
        # 平均FFT像の可視化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 平均FFT像
        im1 = axes[0, 0].imshow(average_fft, cmap='viridis')
        axes[0, 0].set_title('Average FFT Magnitude (Log Scale)')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 平均FFT像の中心部分（低周波成分）
        center_size = min(32, average_fft.shape[0]//2)
        center_fft = average_fft[
            average_fft.shape[0]//2 - center_size:average_fft.shape[0]//2 + center_size,
            average_fft.shape[1]//2 - center_size:average_fft.shape[1]//2 + center_size
        ]
        im2 = axes[0, 1].imshow(center_fft, cmap='viridis')
        axes[0, 1].set_title('Center Region (Low Frequency)')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 平均FFT像のプロファイル（中心から放射状）
        center_y, center_x = average_fft.shape[0]//2, average_fft.shape[1]//2
        max_radius = min(center_x, center_y)
        radial_profile = []
        radii = []
        
        for r in range(1, max_radius):
            circle_mask = np.zeros_like(average_fft, dtype=bool)
            y, x = np.ogrid[:average_fft.shape[0], :average_fft.shape[1]]
            mask = (x - center_x)**2 + (y - center_y)**2 <= r**2
            circle_mask[mask] = True
            
            # 前の円との差分を取る
            if r > 1:
                prev_mask = np.zeros_like(average_fft, dtype=bool)
                prev_mask[(x - center_x)**2 + (y - center_y)**2 <= (r-1)**2] = True
                ring_mask = circle_mask & ~prev_mask
            else:
                ring_mask = circle_mask
            
            if np.any(ring_mask):
                mean_value = np.mean(average_fft[ring_mask])
                radial_profile.append(mean_value)
                radii.append(r)
        
        axes[0, 2].plot(radii, radial_profile)
        axes[0, 2].set_xlabel('Radius (pixels)')
        axes[0, 2].set_ylabel('Average Magnitude')
        axes[0, 2].set_title('Radial Profile')
        axes[0, 2].grid(True)
        
        # 個別細胞のFFT例（最初の3個）
        n_examples = min(3, len(fft_magnitudes))
        for i in range(n_examples):
            row = 1
            col = i
            
            im = axes[row, col].imshow(fft_magnitudes[i], cmap='viridis')
            axes[row, col].set_title(f'Cell {i+1} FFT')
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fft_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(0.1)  # 短い一時停止でグラフを表示
        plt.close()  # 図を閉じてメモリを解放
        
        # FFT統計情報
        fft_stats = {
            'total_cells': len(fft_magnitudes),
            'fft_shape': average_fft.shape,
            'mean_magnitude': np.mean(average_fft),
            'std_magnitude': np.std(average_fft),
            'max_magnitude': np.max(average_fft),
            'min_magnitude': np.min(average_fft)
        }
        
        # 統計情報を保存
        with open(os.path.join(self.output_dir, 'fft_statistics.txt'), 'w') as f:
            f.write("=== FFT ANALYSIS STATISTICS ===\n\n")
            f.write(f"Total cells analyzed: {fft_stats['total_cells']}\n")
            f.write(f"FFT image shape: {fft_stats['fft_shape']}\n")
            f.write(f"Mean magnitude: {fft_stats['mean_magnitude']:.6f}\n")
            f.write(f"Std magnitude: {fft_stats['std_magnitude']:.6f}\n")
            f.write(f"Max magnitude: {fft_stats['max_magnitude']:.6f}\n")
            f.write(f"Min magnitude: {fft_stats['min_magnitude']:.6f}\n")
        
        print(f"FFT analysis completed. Results saved to {self.output_dir}")
        return fft_stats
    
    def analyze_directory(self, folder_path):
        """ディレクトリ内の画像を解析"""
        print("=== Cell FFT Analysis Pipeline ===")
        
        # StarDistモデル読み込み
        stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
        
        # ファイル取得
        file_paths = sorted(glob(os.path.join(folder_path, '*.tif')))
        print(f"Found {len(file_paths)} image files")
        
        all_cells = []
        all_stats = []
        file_summary = []
        
        for i, file_path in enumerate(file_paths):
            filename = os.path.basename(file_path)
            print(f"Processing {i+1}/{len(file_paths)}: {filename}")
            
            cells, stats = self.extract_quality_cells(file_path, stardist_model)
            
            all_cells.extend(cells)
            all_stats.extend(stats)
            
            file_summary.append({
                'filename': filename,
                'cells_extracted': len(cells),
                'mean_cell_intensity': np.mean([s['mean_intensity'] for s in stats]) if stats else 0
            })
            
            print(f"  Extracted {len(cells)} quality cells")
        
        print(f"\nTotal quality cells extracted: {len(all_cells)}")
        
        # 統計サマリー保存
        stats_df = pd.DataFrame(all_stats)
        file_summary_df = pd.DataFrame(file_summary)
        
        stats_df.to_csv(os.path.join(self.output_dir, 'cell_statistics.csv'), index=False)
        file_summary_df.to_csv(os.path.join(self.output_dir, 'file_summary.csv'), index=False)
        
        # FFT解析実行
        if len(all_cells) > 0:
            print(f"\nStarting FFT analysis for {len(all_cells)} cells...")
            average_fft, fft_magnitudes = self.calculate_average_fft(all_cells)
            fft_stats = self.visualize_fft_results(average_fft, fft_magnitudes)
        else:
            print("No cells available for FFT analysis")
            fft_stats = None
        
        # 最終レポート生成
        self.generate_final_report(stats_df, file_summary_df, fft_stats)
        
        return all_cells, stats_df, fft_stats
    
    def generate_final_report(self, stats_df, file_summary_df, fft_stats):
        """最終レポート生成"""
        with open(os.path.join(self.output_dir, 'analysis_report.txt'), 'w') as f:
            f.write("=== CELL FFT ANALYSIS REPORT ===\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("CELL EXTRACTION SUMMARY:\n")
            f.write(f"Total files processed: {len(file_summary_df)}\n")
            f.write(f"Total cells extracted: {len(stats_df)}\n")
            f.write(f"Average cells per file: {len(stats_df)/len(file_summary_df):.1f}\n\n")
            
            f.write("CELL MORPHOLOGY STATISTICS:\n")
            f.write(f"Area: {stats_df['area'].mean():.1f} ± {stats_df['area'].std():.1f}\n")
            f.write(f"Eccentricity: {stats_df['eccentricity'].mean():.3f} ± {stats_df['eccentricity'].std():.3f}\n")
            f.write(f"Solidity: {stats_df['solidity'].mean():.3f} ± {stats_df['solidity'].std():.3f}\n\n")
            
            f.write("INTENSITY STATISTICS:\n")
            f.write(f"Mean intensity: {stats_df['mean_intensity'].mean():.3f} ± {stats_df['mean_intensity'].std():.3f}\n")
            f.write(f"Std intensity: {stats_df['std_intensity'].mean():.3f} ± {stats_df['std_intensity'].std():.3f}\n\n")
            
            if fft_stats:
                f.write("FFT ANALYSIS RESULTS:\n")
                f.write(f"Total cells analyzed: {fft_stats['total_cells']}\n")
                f.write(f"FFT image shape: {fft_stats['fft_shape']}\n")
                f.write(f"Mean magnitude: {fft_stats['mean_magnitude']:.6f}\n")
                f.write(f"Std magnitude: {fft_stats['std_magnitude']:.6f}\n")
                f.write(f"Max magnitude: {fft_stats['max_magnitude']:.6f}\n")
                f.write(f"Min magnitude: {fft_stats['min_magnitude']:.6f}\n\n")
            
            f.write("FILE-WISE SUMMARY:\n")
            for _, row in file_summary_df.iterrows():
                f.write(f"{row['filename']}: {row['cells_extracted']} cells, "
                       f"avg intensity: {row['mean_cell_intensity']:.3f}\n")
            
            f.write("\nOUTPUT FILES:\n")
            f.write("- average_fft.npy: Average 2D FFT image\n")
            f.write("- fft_analysis.png: FFT visualization\n")
            f.write("- fft_statistics.txt: FFT analysis statistics\n")
            f.write("- cell_statistics.csv: Individual cell statistics\n")
            f.write("- file_summary.csv: File-wise summary\n")
            f.write("- analysis_report.txt: This report\n")

    def analyze_wt_and_mutants(self, wt_directory, candidate_base_directory, candidate_folders):
        """
        Normal_cells vs LC Candidates比較解析
        """
        print(f"=== Normal_cells vs LC Candidates FFT Analysis ===")
        print(f"WT Directory: {wt_directory}")
        print(f"Candidate Base Directory: {candidate_base_directory}")
        print(f"Candidate Folders: {candidate_folders}")
        
        # 結果保存用辞書
        all_results = {}
        
        # WT (Normal_cells) 解析
        print(f"\n--- Analyzing WT (Normal_cells) ---")
        wt_result = self.analyze_single_sample(wt_directory, "Normal_cells")
        all_results["Normal_cells"] = wt_result
        
        # 各LC候補の解析
        for candidate_folder in candidate_folders:
            candidate_path = os.path.join(candidate_base_directory, candidate_folder)
            print(f"\n--- Analyzing {candidate_folder} ---")
            candidate_result = self.analyze_single_sample(candidate_path, candidate_folder)
            all_results[candidate_folder] = candidate_result
        
        # 比較解析とレポート生成
        print(f"\n--- Generating Comparison Reports ---")
        self.generate_comparison_report(all_results, "Normal_cells")
        
        return all_results

    def analyze_rg2_vs_candidates(self, wt_directory, candidate_base_directory, candidate_folders):
        """
        RG2_LC vs Other LC Candidates比較解析
        """
        print(f"=== RG2_LC vs Other LC Candidates FFT Analysis ===")
        print(f"WT Directory (RG2_LC): {wt_directory}")
        print(f"Candidate Base Directory: {candidate_base_directory}")
        print(f"Candidate Folders: {candidate_folders}")
        
        # 結果保存用辞書
        all_results = {}
        
        # WT (RG2_LC) 解析
        print(f"\n--- Analyzing WT (RG2_LC) ---")
        wt_result = self.analyze_single_sample(wt_directory, "RG2_LC")
        all_results["RG2_LC"] = wt_result
        
        # 各LC候補の解析
        for candidate_folder in candidate_folders:
            candidate_path = os.path.join(candidate_base_directory, candidate_folder)
            print(f"\n--- Analyzing {candidate_folder} ---")
            candidate_result = self.analyze_single_sample(candidate_path, candidate_folder)
            all_results[candidate_folder] = candidate_result
        
        # 比較解析とレポート生成
        print(f"\n--- Generating Comparison Reports ---")
        self.generate_comparison_report(all_results, "RG2_LC")
        
        return all_results
    
    def analyze_single_sample(self, directory, sample_name):
        """
        単一サンプルの解析
        """
        print(f"\n--- Analyzing {sample_name} ---")
        print(f"Directory: {directory}")
        
        # 解析実行
        cells, stats_df, fft_stats = self.analyze_directory(directory)
        
        # 結果保存
        result = {
            'cells': cells,
            'stats': stats_df,
            'fft_stats': fft_stats,
            'output_dir': self.output_dir
        }
        
        print(f"✓ {sample_name}: {len(cells)} cells analyzed")
        
        return result
    
    def generate_comparison_report(self, all_results, reference_sample):
        """
        比較レポート生成
        """
        print(f"\n--- Generating Comparison Report for {reference_sample} ---")
        
        comparison_data = []
        
        for sample_name, result in all_results.items():
            if result['fft_stats'] is not None:
                # RG2_LCを基準とした比較の場合のType設定
                if reference_sample == "RG2_LC":
                    sample_type = 'WT' if sample_name == 'RG2_LC' else 'LC_Candidate'
                else:
                    sample_type = 'Normal_cells' if sample_name == 'Normal_cells' else 'LC_Candidate'
                
                comparison_data.append({
                    'Sample': sample_name,
                    'Type': sample_type,
                    'Total_Cells': result['fft_stats']['total_cells'],
                    'Mean_Magnitude': result['fft_stats']['mean_magnitude'],
                    'Std_Magnitude': result['fft_stats']['std_magnitude'],
                    'Max_Magnitude': result['fft_stats']['max_magnitude'],
                    'Min_Magnitude': result['fft_stats']['min_magnitude']
                })
        
        if comparison_data:
            # 比較データフレーム作成
            comparison_df = pd.DataFrame(comparison_data)
            
            # 適切なファイル名を生成
            if reference_sample == "RG2_LC":
                csv_filename = f'RG2_LC_vs_LC_Candidates_comparison.csv'
                report_filename = f'RG2_LC_vs_LC_Candidates_comparison_report.txt'
            else:
                csv_filename = f'{reference_sample}_vs_LC_Candidates_comparison.csv'
                report_filename = f'{reference_sample}_vs_LC_Candidates_comparison_report.txt'
            
            comparison_df.to_csv(os.path.join(self.output_dir, csv_filename), index=False)
            
            # 比較可視化
            self.visualize_mutant_comparison(comparison_df)
            
            # 比較レポート
            with open(os.path.join(self.output_dir, report_filename), 'w') as f:
                if reference_sample == "RG2_LC":
                    f.write(f"=== RG2_LC (WT) vs LC CANDIDATES COMPARISON REPORT ===\n\n")
                else:
                    f.write(f"=== {reference_sample} vs LC CANDIDATES COMPARISON REPORT ===\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("SUMMARY BY SAMPLE:\n")
                for _, row in comparison_df.iterrows():
                    f.write(f"{row['Sample']} ({row['Type']}):\n")
                    f.write(f"  Total cells: {row['Total_Cells']}\n")
                    f.write(f"  Mean magnitude: {row['Mean_Magnitude']:.6f}\n")
                    f.write(f"  Std magnitude: {row['Std_Magnitude']:.6f}\n")
                    f.write(f"  Max magnitude: {row['Max_Magnitude']:.6f}\n")
                    f.write(f"  Min magnitude: {row['Min_Magnitude']:.6f}\n\n")
                
                f.write("OVERALL STATISTICS:\n")
                f.write(f"Total samples analyzed: {len(comparison_df)}\n")
                f.write(f"Total cells across all samples: {comparison_df['Total_Cells'].sum()}\n")
                f.write(f"Average cells per sample: {comparison_df['Total_Cells'].mean():.1f}\n")
                f.write(f"Mean magnitude range: {comparison_df['Mean_Magnitude'].min():.6f} - {comparison_df['Mean_Magnitude'].max():.6f}\n")
        
        # エラーサマリー
        error_samples = [name for name, result in all_results.items() if result.get('fft_stats') is None]
        if error_samples:
            with open(os.path.join(self.output_dir, 'error_summary.txt'), 'w') as f:
                f.write("=== ERROR SUMMARY ===\n\n")
                for sample_name in error_samples:
                    error_msg = all_results[sample_name].get('error', 'Unknown error')
                    f.write(f"{sample_name}: {error_msg}\n")
    
    def visualize_mutant_comparison(self, comparison_df):
        """
        サンプル間比較の可視化
        """
        print("Creating comparison visualizations...")
        
        # 色の設定
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # 1. 細胞数の比較
        plt.figure(figsize=(10, 6))
        bars = plt.bar(comparison_df['Sample'], comparison_df['Total_Cells'], 
                      color=colors[:len(comparison_df)], alpha=0.7)
        plt.title('Number of Cells Analyzed by Sample', fontsize=14, fontweight='bold')
        plt.xlabel('Sample', fontsize=12)
        plt.ylabel('Number of Cells', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # バーの上に値を表示
        for bar, value in zip(bars, comparison_df['Total_Cells']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    str(value), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cell_count_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. FFT振幅統計の比較
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('FFT Magnitude Statistics Comparison', fontsize=16, fontweight='bold')
        
        # Mean Magnitude
        axes[0,0].bar(comparison_df['Sample'], comparison_df['Mean_Magnitude'], 
                     color=colors[:len(comparison_df)], alpha=0.7)
        axes[0,0].set_title('Mean FFT Magnitude')
        axes[0,0].set_ylabel('Mean Magnitude')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # Std Magnitude
        axes[0,1].bar(comparison_df['Sample'], comparison_df['Std_Magnitude'], 
                     color=colors[:len(comparison_df)], alpha=0.7)
        axes[0,1].set_title('Standard Deviation of FFT Magnitude')
        axes[0,1].set_ylabel('Std Magnitude')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # Max Magnitude
        axes[1,0].bar(comparison_df['Sample'], comparison_df['Max_Magnitude'], 
                     color=colors[:len(comparison_df)], alpha=0.7)
        axes[1,0].set_title('Maximum FFT Magnitude')
        axes[1,0].set_ylabel('Max Magnitude')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        
        # Min Magnitude
        axes[1,1].bar(comparison_df['Sample'], comparison_df['Min_Magnitude'], 
                     color=colors[:len(comparison_df)], alpha=0.7)
        axes[1,1].set_title('Minimum FFT Magnitude')
        axes[1,1].set_ylabel('Min Magnitude')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fft_statistics_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 箱ひげ図風の比較（Mean ± Std）
        plt.figure(figsize=(12, 8))
        
        x_pos = np.arange(len(comparison_df))
        means = comparison_df['Mean_Magnitude']
        stds = comparison_df['Std_Magnitude']
        
        bars = plt.bar(x_pos, means, yerr=stds, capsize=5, 
                      color=colors[:len(comparison_df)], alpha=0.7, 
                      error_kw=dict(elinewidth=2, capthick=2))
        
        plt.title('FFT Magnitude: Mean ± Standard Deviation', fontsize=14, fontweight='bold')
        plt.xlabel('Sample', fontsize=12)
        plt.ylabel('FFT Magnitude', fontsize=12)
        plt.xticks(x_pos, comparison_df['Sample'], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # 値の表示
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.001, 
                    f'{mean:.4f}±{std:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fft_magnitude_with_error_bars.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Comparison visualizations saved successfully!")
    
    def calculate_average_fft_with_phase(self, cell_images):
        """位相情報を保持したFFT解析（改良版）"""
        print("=== Calculating Average 2D Fourier Transform with Phase ===")
        
        if len(cell_images) == 0:
            print("No cell images available for FFT analysis")
            return None, None
        
        # 各細胞のFFTを計算（複素数で保持）
        fft_results = []
        fft_magnitudes = []
        fft_phases = []
        
        for i, cell_img in enumerate(cell_images):
            if i % 100 == 0:
                print(f"Processing FFT for cell {i+1}/{len(cell_images)}")
            
            # 画像をfloat64に変換し、正規化
            cell_float = cell_img.astype(np.float64)
            cell_normalized = (cell_float - np.mean(cell_float)) / np.std(cell_float)
            
            # 2次元フーリエ変換（複素数結果）
            fft_result = fft2(cell_normalized)
            fft_shifted = fftshift(fft_result)
            
            # 振幅と位相を分離
            magnitude = np.abs(fft_shifted)
            phase = np.angle(fft_shifted)
            
            fft_results.append(fft_shifted)
            fft_magnitudes.append(magnitude)
            fft_phases.append(phase)
        
        # 平均振幅を計算（対数変換なし）
        average_magnitude = np.mean(fft_magnitudes, axis=0)
        
        # 改良された位相平均化
        print("Calculating improved phase average...")
        
        # 方法1: 複素数での平均化
        complex_average_phase = self.calculate_complex_average_phase(fft_results)
        
        # 方法2: 円周統計での平均化
        circular_average_phase = self.calculate_circular_mean_phase(fft_phases)
        
        # 方法3: 中央値
        median_phase = np.median(fft_phases, axis=0)
        
        # 方法4: 最も代表的な細胞の位相を使用
        magnitude_similarities = []
        for mag in fft_magnitudes:
            similarity = np.corrcoef(average_magnitude.flatten(), mag.flatten())[0,1]
            magnitude_similarities.append(similarity)
        
        best_cell_idx = np.argmax(magnitude_similarities)
        representative_phase = fft_phases[best_cell_idx]
        
        print(f"Using phase from cell {best_cell_idx} (similarity: {magnitude_similarities[best_cell_idx]:.3f})")
        
        # 各方法での逆FFT品質を比較
        phase_methods = {
            'Complex_Average': complex_average_phase,
            'Circular_Average': circular_average_phase,
            'Median': median_phase,
            'Representative': representative_phase
        }
        
        # 品質評価（最初の細胞でテスト）
        test_cell = cell_images[0]
        test_cell_float = test_cell.astype(np.float64)
        test_cell_normalized = (test_cell_float - np.mean(test_cell_float)) / np.std(test_cell_float)
        
        print("\nPhase averaging quality comparison:")
        for method_name, phase_method in phase_methods.items():
            # 逆FFTで品質を評価
            inverse_result, _ = self.perform_proper_inverse_fft(average_magnitude, phase_method)
            inverse_adjusted = (inverse_result - np.mean(inverse_result)) / np.std(inverse_result)
            reconstruction_error = np.mean(np.abs(test_cell_normalized - inverse_adjusted))
            print(f"  {method_name}: reconstruction error = {reconstruction_error:.6f}")
        
        return {
            'average_magnitude': average_magnitude,
            'complex_average_phase': complex_average_phase,
            'circular_average_phase': circular_average_phase,
            'median_phase': median_phase,
            'representative_phase': representative_phase,
            'all_magnitudes': fft_magnitudes,
            'all_phases': fft_phases,
            'all_fft_results': fft_results
        }
    
    def calculate_complex_average_phase(self, fft_results):
        """複素数での位相平均化"""
        # 複素数FFTの平均
        complex_average = np.mean(fft_results, axis=0)
        
        # 平均位相を抽出
        average_phase = np.angle(complex_average)
        
        return average_phase
    
    def calculate_circular_mean_phase(self, phases):
        """円周統計での位相平均化"""
        # 複素数に変換
        complex_phases = np.exp(1j * np.array(phases))
        
        # 複素数の平均
        mean_complex = np.mean(complex_phases, axis=0)
        
        # 位相を抽出
        mean_phase = np.angle(mean_complex)
        
        return mean_phase
    
    def perform_proper_inverse_fft(self, magnitude, phase):
        """適切な逆フーリエ変換"""
        # 複素数FFTデータを再構成
        complex_fft = magnitude * np.exp(1j * phase)
        
        # シフトを戻す（fftshiftの逆操作）
        complex_fft_unshifted = ifftshift(complex_fft)
        
        # 逆フーリエ変換
        inverse_result = ifft2(complex_fft_unshifted)
        
        # 実部を取得（虚部は理論的には0に近いはず）
        inverse_real = np.real(inverse_result)
        
        # 正規化（0-1の範囲に）
        inverse_normalized = (inverse_real - np.min(inverse_real)) / (np.max(inverse_real) - np.min(inverse_real))
        
        return inverse_normalized, np.imag(inverse_result)
    
    def analyze_individual_cell_fft(self, cell_image):
        """個別細胞のFFT解析とinverse FFT"""
        # 元画像
        cell_float = cell_image.astype(np.float64)
        cell_normalized = (cell_float - np.mean(cell_float)) / np.std(cell_float)
        
        # FFT
        fft_result = fft2(cell_normalized)
        fft_shifted = fftshift(fft_result)
        
        # 振幅と位相
        magnitude = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)
        
        # 逆FFT（完全復元）
        inverse_image, imaginary_part = self.perform_proper_inverse_fft(magnitude, phase)
        
        # 元の正規化された画像と同じスケールに調整
        # 元画像の統計量に合わせて正規化
        inverse_adjusted = (inverse_image - np.mean(inverse_image)) / np.std(inverse_image)
        
        # 復元誤差を計算（正規化された画像との比較）
        reconstruction_error = np.mean(np.abs(cell_normalized - inverse_adjusted))
        
        # 元のスケールに戻す（表示用）
        inverse_original_scale = inverse_adjusted * np.std(cell_float) + np.mean(cell_float)
        
        return {
            'original': cell_normalized,  # 正規化された元画像
            'original_scale': cell_float,  # 元のスケールの画像
            'magnitude': magnitude,
            'phase': phase,
            'inverse': inverse_adjusted,  # 正規化された逆FFT結果
            'inverse_original_scale': inverse_original_scale,  # 元のスケールの逆FFT結果
            'imaginary_part': imaginary_part,
            'reconstruction_error': reconstruction_error
        }
    
    def create_magnitude_only_inverse(self, magnitude):
        """振幅のみからの逆変換（位相=0と仮定）"""
        # 位相を0と仮定
        zero_phase = np.zeros_like(magnitude)
        inverse_image, _ = self.perform_proper_inverse_fft(magnitude, zero_phase)
        return inverse_image
    
    def create_random_phase_inverse(self, magnitude):
        """振幅 + ランダム位相からの逆変換"""
        # ランダム位相を生成
        random_phase = np.random.uniform(-np.pi, np.pi, magnitude.shape)
        inverse_image, _ = self.perform_proper_inverse_fft(magnitude, random_phase)
        return inverse_image
    
    def visualize_fft_comparison(self, cell_images, max_cells=5):
        """FFTの比較可視化"""
        n_cells = min(len(cell_images), max_cells)
        
        fig, axes = plt.subplots(n_cells, 7, figsize=(28, 4*n_cells))
        if n_cells == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_cells):
            # 個別細胞解析
            result = self.analyze_individual_cell_fft(cell_images[i])
            
            # 振幅のみの逆変換
            magnitude_only_inverse = self.create_magnitude_only_inverse(result['magnitude'])
            
            # 元画像（正規化）
            axes[i, 0].imshow(result['original'], cmap='gray')
            axes[i, 0].set_title(f'Cell {i+1}\nNormalized Original')
            axes[i, 0].axis('off')
            
            # 元画像（元のスケール）
            axes[i, 1].imshow(result['original_scale'], cmap='gray')
            axes[i, 1].set_title('Original Scale')
            axes[i, 1].axis('off')
            
            # FFT振幅
            axes[i, 2].imshow(np.log1p(result['magnitude']), cmap='viridis')
            axes[i, 2].set_title('FFT Magnitude\n(log scale)')
            axes[i, 2].axis('off')
            
            # FFT位相
            axes[i, 3].imshow(result['phase'], cmap='hsv')
            axes[i, 3].set_title('FFT Phase')
            axes[i, 3].axis('off')
            
            # 完全復元（正規化）
            axes[i, 4].imshow(result['inverse'], cmap='gray')
            axes[i, 4].set_title(f'Perfect Reconstruction\n(Normalized)\nError: {result["reconstruction_error"]:.6f}')
            axes[i, 4].axis('off')
            
            # 完全復元（元のスケール）
            axes[i, 5].imshow(result['inverse_original_scale'], cmap='gray')
            axes[i, 5].set_title('Perfect Reconstruction\n(Original Scale)')
            axes[i, 5].axis('off')
            
            # 振幅のみ復元
            axes[i, 6].imshow(magnitude_only_inverse, cmap='gray')
            axes[i, 6].set_title('Magnitude Only\n(Phase = 0)')
            axes[i, 6].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fft_comparison_detailed.png'), dpi=300, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()  # 図を閉じてメモリを解放
    
    def demonstrate_phase_importance(self, cell_image):
        """位相の重要性を示すデモ"""
        # 個別解析
        result = self.analyze_individual_cell_fft(cell_image)
        
        # 異なるケースを試す（正規化されたスケール）
        cases_normalized = {
            'Normalized Original': result['original'],
            'Perfect Reconstruction': result['inverse'],
            'Magnitude Only (Phase=0)': self.create_magnitude_only_inverse(result['magnitude']),
        }
        
        # 異なるケースを試す（元のスケール）
        cases_original_scale = {
            'Original Scale': result['original_scale'],
            'Perfect Reconstruction': result['inverse_original_scale'],
        }
        
        # 可視化（正規化されたスケール）
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 正規化されたスケール
        for i, (title, image) in enumerate(cases_normalized.items()):
            axes[0, i].imshow(image, cmap='gray')
            axes[0, i].set_title(f'{title}\n(Normalized)', fontsize=12)
            axes[0, i].axis('off')
        
        # 元のスケール
        for i, (title, image) in enumerate(cases_original_scale.items()):
            axes[1, i].imshow(image, cmap='gray')
            axes[1, i].set_title(f'{title}\n(Original Scale)', fontsize=12)
            axes[1, i].axis('off')
        
        # ランダム位相の例
        random_phase_inverse = self.create_random_phase_inverse(result['magnitude'])
        axes[0, 2].imshow(random_phase_inverse, cmap='gray')
        axes[0, 2].set_title('Random Phase\n(Normalized)', fontsize=12)
        axes[0, 2].axis('off')
        
        # 統計情報
        axes[1, 2].text(0.1, 0.5, f"Reconstruction Error: {result['reconstruction_error']:.6f}\n\n" +
                       f"Original mean: {np.mean(result['original']):.3f}\n" +
                       f"Original std: {np.std(result['original']):.3f}\n\n" +
                       f"Reconstructed mean: {np.mean(result['inverse']):.3f}\n" +
                       f"Reconstructed std: {np.std(result['inverse']):.3f}", 
                       transform=axes[1, 2].transAxes, fontsize=10, 
                       verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 2].set_title('Statistics', fontsize=12)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'phase_importance_demo.png'), dpi=300, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()  # 図を閉じてメモリを解放
        
        # 統計情報を出力
        print("=== Phase Importance Analysis ===")
        print(f"Original image stats: mean={np.mean(result['original']):.3f}, std={np.std(result['original']):.3f}")
        print(f"Perfect reconstruction error: {result['reconstruction_error']:.6f}")
        
        mag_only_error = np.mean(np.abs(result['original'] - cases_normalized['Magnitude Only (Phase=0)']))
        print(f"Magnitude-only reconstruction error: {mag_only_error:.6f}")
        print(f"Error increase without phase: {mag_only_error/result['reconstruction_error']:.1f}x")
        
        # 元のスケールでの比較
        print(f"\nOriginal scale comparison:")
        print(f"Original mean: {np.mean(result['original_scale']):.3f}, std: {np.std(result['original_scale']):.3f}")
        print(f"Reconstructed mean: {np.mean(result['inverse_original_scale']):.3f}, std: {np.std(result['inverse_original_scale']):.3f}")
        original_scale_error = np.mean(np.abs(result['original_scale'] - result['inverse_original_scale']))
        print(f"Original scale reconstruction error: {original_scale_error:.6f}")
    
    def create_improved_mutant_comparison(self, all_results):
        """改良されたmutant比較（位相情報付き）"""
        print("=== Creating Improved Mutant Comparison with Phase Information ===")
        
        # 成功したmutantのデータを収集
        successful_mutants = []
        all_cells = []
        
        for mutant_name, result in all_results.items():
            if result['fft_stats'] is not None and len(result['cells']) > 0:
                successful_mutants.append(mutant_name)
                all_cells.extend(result['cells'])
        
        if len(all_cells) == 0:
            print("No cell data available for improved analysis")
            return
        
        # 位相の重要性デモ（最初の細胞で）
        print("Demonstrating phase importance...")
        self.demonstrate_phase_importance(all_cells[0])
        
        # 詳細FFT比較（最初の5個の細胞で）
        print("Creating detailed FFT comparison...")
        self.visualize_fft_comparison(all_cells[:5], max_cells=5)
        
        # 各mutantの位相情報付きFFT解析
        mutant_fft_data = {}
        for mutant_name, result in all_results.items():
            if result['fft_stats'] is not None and len(result['cells']) > 0:
                print(f"Analyzing {mutant_name} with phase information...")
                fft_data = self.calculate_average_fft_with_phase(result['cells'])
                mutant_fft_data[mutant_name] = fft_data
        
        # 位相情報付きの比較可視化
        self.visualize_phase_aware_comparison(mutant_fft_data)
        
        print("Improved mutant comparison completed")
    
    def visualize_phase_aware_comparison(self, mutant_fft_data):
        """位相情報付きの比較可視化"""
        print("=== Visualizing Phase-Aware Comparison ===")
        
        n_mutants = len(mutant_fft_data)
        cols = min(4, n_mutants)
        rows = (n_mutants + cols - 1) // cols
        
        # 振幅比較
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (mutant_name, fft_data) in enumerate(mutant_fft_data.items()):
            row = i // cols
            col = i % cols
            
            im = axes[row, col].imshow(np.log1p(fft_data['average_magnitude']), cmap='viridis')
            axes[row, col].set_title(f'{mutant_name}\nAvg Magnitude (log)', fontsize=12)
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], shrink=0.8)
        
        # 空のサブプロットを非表示
        for i in range(n_mutants, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'phase_aware_magnitude_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()  # 図を閉じてメモリを解放
        
        # 位相比較
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (mutant_name, fft_data) in enumerate(mutant_fft_data.items()):
            row = i // cols
            col = i % cols
            
            im = axes[row, col].imshow(fft_data['representative_phase'], cmap='hsv')
            axes[row, col].set_title(f'{mutant_name}\nRepresentative Phase', fontsize=12)
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], shrink=0.8)
        
        # 空のサブプロットを非表示
        for i in range(n_mutants, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'phase_aware_phase_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()  # 図を閉じてメモリを解放
        
        # 適切な逆FFT比較
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (mutant_name, fft_data) in enumerate(mutant_fft_data.items()):
            row = i // cols
            col = i % cols
            
            # 適切な逆FFT（振幅+位相）
            proper_inverse, _ = self.perform_proper_inverse_fft(
                fft_data['average_magnitude'], 
                fft_data['representative_phase']
            )
            
            im = axes[row, col].imshow(proper_inverse, cmap='gray')
            axes[row, col].set_title(f'{mutant_name}\nProper Inverse FFT', fontsize=12)
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], shrink=0.8)
        
        # 空のサブプロットを非表示
        for i in range(n_mutants, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'phase_aware_inverse_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()  # 図を閉じてメモリを解放

    def calculate_radial_profile(self, fft_magnitude):
        """放射状プロファイル解析: 中心からの距離に対する強度変化"""
        height, width = fft_magnitude.shape
        center_y, center_x = height // 2, width // 2
        
        # 最大半径を計算
        max_radius = min(center_x, center_y)
        
        # 各半径での平均強度を計算
        radial_profile = []
        radii = []
        
        for r in range(1, max_radius):
            # 円周上の点をサンプリング
            angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
            intensities = []
            
            for angle in angles:
                x = int(center_x + r * np.cos(angle))
                y = int(center_y + r * np.sin(angle))
                
                if 0 <= x < width and 0 <= y < height:
                    intensities.append(fft_magnitude[y, x])
            
            if intensities:
                radial_profile.append(np.mean(intensities))
                radii.append(r)
        
        return np.array(radii), np.array(radial_profile)
    
    def calculate_angular_profile(self, fft_magnitude, num_sectors=8):
        """角度方向解析: 対称性の定量化"""
        height, width = fft_magnitude.shape
        center_y, center_x = height // 2, width // 2
        
        # 最大半径を計算
        max_radius = min(center_x, center_y)
        
        # 各セクターでの平均強度を計算
        angular_profiles = []
        sector_angles = np.linspace(0, 2*np.pi, num_sectors+1)[:-1]
        
        for sector_angle in sector_angles:
            sector_intensities = []
            
            for r in range(1, max_radius):
                # セクター内の点をサンプリング
                angles = np.linspace(sector_angle, sector_angle + 2*np.pi/num_sectors, 45)
                
                for angle in angles:
                    x = int(center_x + r * np.cos(angle))
                    y = int(center_y + r * np.sin(angle))
                    
                    if 0 <= x < width and 0 <= y < height:
                        sector_intensities.append(fft_magnitude[y, x])
            
            if sector_intensities:
                angular_profiles.append(np.mean(sector_intensities))
            else:
                angular_profiles.append(0)
        
        return sector_angles, np.array(angular_profiles)
    
    def analyze_frequency_domains(self, fft_magnitude):
        """周波数領域別解析: 低周波・高周波成分の比較"""
        height, width = fft_magnitude.shape
        center_y, center_x = height // 2, width // 2
        
        # 低周波領域（中心付近）
        low_freq_mask = np.zeros_like(fft_magnitude, dtype=bool)
        low_freq_radius = min(center_x, center_y) // 4
        
        y_coords, x_coords = np.ogrid[:height, :width]
        distance_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        low_freq_mask = distance_from_center <= low_freq_radius
        
        # 高周波領域（外側）
        high_freq_mask = np.zeros_like(fft_magnitude, dtype=bool)
        high_freq_radius = min(center_x, center_y) // 2
        
        high_freq_mask = distance_from_center > high_freq_radius
        
        # 中周波領域
        mid_freq_mask = ~(low_freq_mask | high_freq_mask)
        
        # 各領域の統計を計算
        low_freq_stats = {
            'mean': np.mean(fft_magnitude[low_freq_mask]),
            'std': np.std(fft_magnitude[low_freq_mask]),
            'total_energy': np.sum(fft_magnitude[low_freq_mask]**2),
            'area': np.sum(low_freq_mask)
        }
        
        mid_freq_stats = {
            'mean': np.mean(fft_magnitude[mid_freq_mask]),
            'std': np.std(fft_magnitude[mid_freq_mask]),
            'total_energy': np.sum(fft_magnitude[mid_freq_mask]**2),
            'area': np.sum(mid_freq_mask)
        }
        
        high_freq_stats = {
            'mean': np.mean(fft_magnitude[high_freq_mask]),
            'std': np.std(fft_magnitude[high_freq_mask]),
            'total_energy': np.sum(fft_magnitude[high_freq_mask]**2),
            'area': np.sum(high_freq_mask)
        }
        
        return {
            'low_freq': low_freq_stats,
            'mid_freq': mid_freq_stats,
            'high_freq': high_freq_stats,
            'masks': {
                'low_freq': low_freq_mask,
                'mid_freq': mid_freq_mask,
                'high_freq': high_freq_mask
            }
        }
    
    def perform_statistical_testing(self, all_results):
        """統計的検定: 有意差の確認"""
        print("\n=== Statistical Testing ===")
        
        # 成功したサンプルのデータを収集
        successful_samples = {}
        for sample_name, result in all_results.items():
            if result['fft_stats'] is not None and len(result['cells']) > 0:
                successful_samples[sample_name] = result
        
        if len(successful_samples) < 2:
            print("Insufficient samples for statistical testing")
            return
        
        # 各サンプルのFFTデータを計算
        sample_fft_data = {}
        for sample_name, result in successful_samples.items():
            print(f"Calculating FFT data for {sample_name}...")
            fft_data = self.calculate_average_fft_with_phase(result['cells'])
            sample_fft_data[sample_name] = fft_data
        
        # 1. 放射状プロファイルの比較
        self.compare_radial_profiles(sample_fft_data)
        
        # 2. 角度方向解析の比較
        self.compare_angular_profiles(sample_fft_data)
        
        # 3. 周波数領域別解析の比較
        self.compare_frequency_domains(sample_fft_data)
        
        # 4. 統計的検定結果の保存
        self.save_statistical_results(sample_fft_data)
    
    def compare_radial_profiles(self, sample_fft_data):
        """放射状プロファイルの比較"""
        print("\n--- Radial Profile Comparison ---")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 色の定義
        candidate_colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
        
        # 各サンプルの放射状プロファイルを計算
        profiles = {}
        for i, (sample_name, fft_data) in enumerate(sample_fft_data.items()):
            radii, profile = self.calculate_radial_profile(fft_data['average_magnitude'])
            profiles[sample_name] = {'radii': radii, 'profile': profile}
            
            # プロット
            if sample_name == 'Normal_cells':
                color = 'blue'
                linewidth = 3
            else:
                color = candidate_colors[(i-1) % len(candidate_colors)]
                linewidth = 2
            
            ax1.plot(radii, profile, label=sample_name, color=color, linewidth=linewidth)
        
        ax1.set_xlabel('Distance from Center (pixels)')
        ax1.set_ylabel('Average FFT Magnitude')
        ax1.set_title('Radial Profile Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 正規化された比較
        normal_profile = profiles.get('Normal_cells', None)
        if normal_profile is not None:
            for i, (sample_name, profile_data) in enumerate(profiles.items()):
                if sample_name != 'Normal_cells':
                    # 共通の半径範囲で比較
                    common_radii = np.intersect1d(normal_profile['radii'], profile_data['radii'])
                    if len(common_radii) > 0:
                        normal_interp = np.interp(common_radii, normal_profile['radii'], normal_profile['profile'])
                        sample_interp = np.interp(common_radii, profile_data['radii'], profile_data['profile'])
                        
                        # 正規化（Normal_cellsで割る）
                        normalized_profile = sample_interp / normal_interp
                        
                        color = candidate_colors[(i-1) % len(candidate_colors)]
                        ax2.plot(common_radii, normalized_profile, label=f'{sample_name}/Normal_cells', 
                               color=color, linewidth=2)
            
            ax2.axhline(y=1, color='blue', linestyle='--', alpha=0.7, label='Normal_cells (reference)')
            ax2.set_xlabel('Distance from Center (pixels)')
            ax2.set_ylabel('Normalized FFT Magnitude')
            ax2.set_title('Normalized Radial Profile Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'radial_profile_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()
        
        # 統計情報を出力
        print("Radial Profile Statistics:")
        for sample_name, profile_data in profiles.items():
            profile = profile_data['profile']
            print(f"  {sample_name}: mean={np.mean(profile):.6f}, std={np.std(profile):.6f}")
    
    def compare_angular_profiles(self, sample_fft_data):
        """角度方向解析の比較"""
        print("\n--- Angular Profile Comparison ---")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 色の定義
        candidate_colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
        
        # 各サンプルの角度方向プロファイルを計算
        profiles = {}
        for i, (sample_name, fft_data) in enumerate(sample_fft_data.items()):
            angles, profile = self.calculate_angular_profile(fft_data['average_magnitude'])
            profiles[sample_name] = {'angles': angles, 'profile': profile}
            
            # プロット
            if sample_name == 'Normal_cells':
                color = 'blue'
                linewidth = 3
                markersize = 8
            else:
                color = candidate_colors[(i-1) % len(candidate_colors)]
                linewidth = 2
                markersize = 6
            
            ax1.plot(np.degrees(angles), profile, label=sample_name, color=color, 
                    marker='o', linewidth=linewidth, markersize=markersize)
        
        ax1.set_xlabel('Angle (degrees)')
        ax1.set_ylabel('Average FFT Magnitude')
        ax1.set_title('Angular Profile Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(np.arange(0, 360, 45))
        
        # 対称性の定量化
        symmetry_scores = {}
        for sample_name, profile_data in profiles.items():
            profile = profile_data['profile']
            # 対称性スコア（隣接セクター間の相関）
            symmetry_score = np.corrcoef(profile, np.roll(profile, len(profile)//2))[0,1]
            symmetry_scores[sample_name] = symmetry_score
            
            print(f"  {sample_name} symmetry score: {symmetry_score:.3f}")
        
        # 対称性スコアの比較
        sample_names = list(symmetry_scores.keys())
        scores = list(symmetry_scores.values())
        
        # 色の割り当て
        colors = []
        for i, name in enumerate(sample_names):
            if name == 'Normal_cells':
                colors.append('blue')
            else:
                colors.append(candidate_colors[(i-1) % len(candidate_colors)])
        
        bars = ax2.bar(sample_names, scores, color=colors)
        ax2.set_ylabel('Symmetry Score')
        ax2.set_title('Angular Symmetry Comparison')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Normal_cellsのバーを強調
        for i, name in enumerate(sample_names):
            if name == 'Normal_cells':
                bars[i].set_edgecolor('black')
                bars[i].set_linewidth(2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'angular_profile_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()
    
    def compare_frequency_domains(self, sample_fft_data):
        """周波数領域別解析の比較"""
        print("\n--- Frequency Domain Comparison ---")
        
        # 色の定義
        candidate_colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
        
        # 各サンプルの周波数領域解析を実行
        domain_data = {}
        for sample_name, fft_data in sample_fft_data.items():
            domain_data[sample_name] = self.analyze_frequency_domains(fft_data['average_magnitude'])
        
        # 比較用のデータフレームを作成
        comparison_data = []
        for sample_name, data in domain_data.items():
            for domain in ['low_freq', 'mid_freq', 'high_freq']:
                comparison_data.append({
                    'Sample': sample_name,
                    'Domain': domain,
                    'Mean_Magnitude': data[domain]['mean'],
                    'Std_Magnitude': data[domain]['std'],
                    'Total_Energy': data[domain]['total_energy'],
                    'Energy_Density': data[domain]['total_energy'] / data[domain]['area']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 可視化
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 平均マグニチュードの比較
        domains = ['low_freq', 'mid_freq', 'high_freq']
        domain_labels = ['Low Frequency', 'Mid Frequency', 'High Frequency']
        
        for i, (domain, label) in enumerate(zip(domains, domain_labels)):
            domain_data_subset = comparison_df[comparison_df['Domain'] == domain]
            
            # 色の割り当て
            colors = []
            for sample in domain_data_subset['Sample']:
                if sample == 'Normal_cells':
                    colors.append('blue')
                else:
                    # 候補のインデックスを取得
                    sample_names = list(sample_fft_data.keys())
                    candidate_idx = sample_names.index(sample) - 1  # Normal_cellsを除く
                    colors.append(candidate_colors[candidate_idx % len(candidate_colors)])
            
            bars = axes[0, i].bar(domain_data_subset['Sample'], 
                                domain_data_subset['Mean_Magnitude'], color=colors)
            axes[0, i].set_title(f'{label} - Mean Magnitude')
            axes[0, i].set_ylabel('Mean Magnitude')
            axes[0, i].tick_params(axis='x', rotation=45)
            
            # Normal_cellsのバーを強調
            for j, sample in enumerate(domain_data_subset['Sample']):
                if sample == 'Normal_cells':
                    bars[j].set_edgecolor('black')
                    bars[j].set_linewidth(2)
        
        # 2. エネルギー密度の比較
        for i, (domain, label) in enumerate(zip(domains, domain_labels)):
            domain_data_subset = comparison_df[comparison_df['Domain'] == domain]
            
            # 色の割り当て
            colors = []
            for sample in domain_data_subset['Sample']:
                if sample == 'Normal_cells':
                    colors.append('blue')
                else:
                    # 候補のインデックスを取得
                    sample_names = list(sample_fft_data.keys())
                    candidate_idx = sample_names.index(sample) - 1  # Normal_cellsを除く
                    colors.append(candidate_colors[candidate_idx % len(candidate_colors)])
            
            bars = axes[1, i].bar(domain_data_subset['Sample'], 
                                domain_data_subset['Energy_Density'], color=colors)
            axes[1, i].set_title(f'{label} - Energy Density')
            axes[1, i].set_ylabel('Energy Density')
            axes[1, i].tick_params(axis='x', rotation=45)
            
            # Normal_cellsのバーを強調
            for j, sample in enumerate(domain_data_subset['Sample']):
                if sample == 'Normal_cells':
                    bars[j].set_edgecolor('black')
                    bars[j].set_linewidth(2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'frequency_domain_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()
        
        # 統計情報を出力
        print("Frequency Domain Statistics:")
        for sample_name, data in domain_data.items():
            print(f"  {sample_name}:")
            for domain in ['low_freq', 'mid_freq', 'high_freq']:
                print(f"    {domain}: mean={data[domain]['mean']:.6f}, "
                      f"energy={data[domain]['total_energy']:.6f}")
        
        # データフレームを保存
        comparison_df.to_csv(os.path.join(self.output_dir, 'frequency_domain_analysis.csv'), index=False)
    
    def save_statistical_results(self, sample_fft_data):
        """統計的検定結果の保存"""
        print("\n--- Saving Statistical Results ---")
        
        with open(os.path.join(self.output_dir, 'advanced_statistical_analysis.txt'), 'w') as f:
            f.write("=== ADVANCED STATISTICAL ANALYSIS REPORT ===\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 放射状プロファイル解析
            f.write("1. RADIAL PROFILE ANALYSIS:\n")
            f.write("   Center-to-edge intensity changes:\n")
            for sample_name, fft_data in sample_fft_data.items():
                radii, profile = self.calculate_radial_profile(fft_data['average_magnitude'])
                f.write(f"   {sample_name}: mean={np.mean(profile):.6f}, std={np.std(profile):.6f}\n")
            f.write("\n")
            
            # 角度方向解析
            f.write("2. ANGULAR PROFILE ANALYSIS:\n")
            f.write("   Symmetry quantification:\n")
            for sample_name, fft_data in sample_fft_data.items():
                angles, profile = self.calculate_angular_profile(fft_data['average_magnitude'])
                symmetry_score = np.corrcoef(profile, np.roll(profile, len(profile)//2))[0,1]
                f.write(f"   {sample_name}: symmetry score={symmetry_score:.3f}\n")
            f.write("\n")
            
            # 周波数領域別解析
            f.write("3. FREQUENCY DOMAIN ANALYSIS:\n")
            for sample_name, fft_data in sample_fft_data.items():
                domain_data = self.analyze_frequency_domains(fft_data['average_magnitude'])
                f.write(f"   {sample_name}:\n")
                for domain in ['low_freq', 'mid_freq', 'high_freq']:
                    f.write(f"     {domain}: mean={domain_data[domain]['mean']:.6f}, "
                           f"energy={domain_data[domain]['total_energy']:.6f}\n")
                f.write("\n")
            
            # 比較分析
            f.write("4. COMPARATIVE ANALYSIS:\n")
            normal_data = sample_fft_data.get('Normal_cells', None)
            if normal_data is not None:
                normal_radii, normal_profile = self.calculate_radial_profile(normal_data['average_magnitude'])
                normal_angles, normal_angular = self.calculate_angular_profile(normal_data['average_magnitude'])
                normal_domains = self.analyze_frequency_domains(normal_data['average_magnitude'])
                
                for sample_name, fft_data in sample_fft_data.items():
                    if sample_name != 'Normal_cells':
                        f.write(f"   {sample_name} vs Normal_cells:\n")
                        
                        # 放射状プロファイルの比較
                        radii, profile = self.calculate_radial_profile(fft_data['average_magnitude'])
                        profile_diff = np.mean(profile) - np.mean(normal_profile)
                        f.write(f"     Radial profile difference: {profile_diff:+.6f}\n")
                        
                        # 角度方向の比較
                        angles, angular = self.calculate_angular_profile(fft_data['average_magnitude'])
                        angular_diff = np.mean(angular) - np.mean(normal_angular)
                        f.write(f"     Angular profile difference: {angular_diff:+.6f}\n")
                        
                        # 周波数領域の比較
                        domains = self.analyze_frequency_domains(fft_data['average_magnitude'])
                        for domain in ['low_freq', 'mid_freq', 'high_freq']:
                            energy_diff = domains[domain]['total_energy'] - normal_domains[domain]['total_energy']
                            f.write(f"     {domain} energy difference: {energy_diff:+.6f}\n")
                        f.write("\n")
        
        print("Advanced statistical analysis completed and saved!")

    def visualize_sample_cells_with_fft(self, all_results, n_cells=30):
        """
        各株ごとに、抽出した細胞画像からn_cells個をランダムに選び、
        左: FFT像（log振幅）、右: グレースケール画像（green channel）
        を2列で並べたグリッド画像をoutput_dirに保存
        """
        print(f"\n=== Visualizing {n_cells} Sample Cells with FFT for Each Strain ===")
        for sample_name, result in all_results.items():
            cells = result.get('cells', [])
            if not cells or len(cells) < 1:
                print(f"  {sample_name}: No cells to visualize.")
                continue
            n_show = min(n_cells, len(cells))
            # ランダムにn_show個選ぶ
            idxs = np.random.choice(len(cells), n_show, replace=False) if len(cells) > n_show else np.arange(len(cells))
            selected_cells = [cells[i] for i in idxs]
            # FFT像を計算
            fft_imgs = [np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(cell)))) for cell in selected_cells]
            # グリッド描画
            fig, axes = plt.subplots(n_show, 2, figsize=(6, n_show*2.2))
            if n_show == 1:
                axes = axes.reshape(1, 2)
            for i, (cell, fft_img) in enumerate(zip(selected_cells, fft_imgs)):
                # FFT像（左）
                axes[i, 1].imshow(fft_img, cmap='viridis')
                axes[i, 1].set_title('FFT (log)', fontsize=10)
                axes[i, 1].axis('off')
                # グレースケール画像（右）
                axes[i, 0].imshow(cell, cmap='gray')
                axes[i, 0].set_title('Cell (Green channel)', fontsize=10)
                axes[i, 0].axis('off')
            plt.tight_layout()
            out_path = os.path.join(self.output_dir, f'{sample_name}_sample_cells_fft_grid.png')
            plt.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"  {sample_name}: Saved {out_path}")

    def visualize_all_samples_cells_with_fft(self, all_results, n_cells=30, cells_per_page=10):
        """
        全ての株のsample_cells_fft_gridを1つの画像にまとめて出力。
        各株ごとにn_cells個をランダム抽出し、
        左: グレースケール, 右: FFT (log振幅) の2列、
        （株数×n_cells）行のグリッド画像をoutput_dirに保存。
        画像が大きくなりすぎないよう、cells_per_pageごとに分割保存。
        """
        print(f"\n=== Visualizing {n_cells} Sample Cells with FFT for All Strains in Multiple Pages ===")
        # 株ごとに細胞を抽出
        sample_names = []
        all_cells = []
        all_fft_imgs = []
        for sample_name, result in all_results.items():
            cells = result.get('cells', [])
            if not cells or len(cells) < 1:
                print(f"  {sample_name}: No cells to visualize.")
                continue
            n_show = min(n_cells, len(cells))
            idxs = np.random.choice(len(cells), n_show, replace=False) if len(cells) > n_show else np.arange(len(cells))
            selected_cells = [cells[i] for i in idxs]
            fft_imgs = [np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(cell)))) for cell in selected_cells]
            all_cells.extend(selected_cells)
            all_fft_imgs.extend(fft_imgs)
            sample_names.extend([sample_name]*n_show)
        total = len(all_cells)
        if total == 0:
            print("No cells to visualize in total.")
            return
        # ページ分割
        num_pages = (total + cells_per_page - 1) // cells_per_page
        for page in range(num_pages):
            start = page * cells_per_page
            end = min((page + 1) * cells_per_page, total)
            n_this_page = end - start
            fig, axes = plt.subplots(n_this_page, 2, figsize=(6, n_this_page*2))
            if n_this_page == 1:
                axes = axes.reshape(1, 2)
            for i in range(n_this_page):
                idx = start + i
                cell = all_cells[idx]
                fft_img = all_fft_imgs[idx]
                sample_name = sample_names[idx]
                # グレースケール画像（左）
                axes[i, 0].imshow(cell, cmap='gray')
                axes[i, 0].set_title(f'{sample_name}', fontsize=10)
                axes[i, 0].axis('off')
                # FFT像（右）
                axes[i, 1].imshow(fft_img, cmap='viridis')
                axes[i, 1].set_title('FFT (log)', fontsize=10)
                axes[i, 1].axis('off')
            plt.tight_layout()
            out_path = os.path.join(self.output_dir, f'all_samples_sample_cells_fft_grid_page{page+1}.png')
            plt.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved page {page+1}: {out_path}")

def main():
    # 設定
    wt_directory = "/Users/matsuokoujirou/Documents/Data/Screening/Candidates/LConly/RG2_LC"  # RG2_LCをWTとして使用
    candidate_base_directory = "/Users/matsuokoujirou/Documents/Data/Screening/Candidates/LConly"
    output_dir = f"/Users/matsuokoujirou/Documents/Data/Screening/FFT_Analysis_RG2_WT/{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    # LC候補フォルダのリスト（RG2_LCを除く）
    candidate_folders = ['10A_LC', '10B_LC', '6C1_LC', '10D_LC']
    
    # FFT解析パイプライン実行
    analyzer = CellFFTAnalyzer(output_dir)
    
    # === ここで実行スクリプトを保存 ===
    try:
        script_path = os.path.abspath(__file__)
        shutil.copy(script_path, os.path.join(output_dir, os.path.basename(script_path)))
        print(f"実行スクリプト {os.path.basename(script_path)} を {output_dir} に保存しました。")
    except Exception as e:
        print(f"スクリプト保存時のエラー: {e}")
    
    # RG2_LC vs Other LC Candidates比較解析実行
    all_results = analyzer.analyze_rg2_vs_candidates(wt_directory, candidate_base_directory, candidate_folders)
    
    print(f"\n=== RG2_LC vs OTHER LC CANDIDATES FFT ANALYSIS COMPLETED ===")
    print(f"Results saved to: {output_dir}")
    
    # 成功したサンプルの数をカウント
    successful_samples = sum(1 for result in all_results.values() if result['fft_stats'] is not None)
    total_cells = sum(result['fft_stats']['total_cells'] for result in all_results.values() if result['fft_stats'] is not None)
    
    print(f"Successfully analyzed {successful_samples}/{len(all_results)} samples")
    print(f"Total cells analyzed: {total_cells}")
    
    # 各サンプルの結果サマリー
    print("\nResults Summary:")
    for sample_name, result in all_results.items():
        if result['fft_stats'] is not None:
            print(f"  {sample_name}: {result['fft_stats']['total_cells']} cells")
        else:
            print(f"  {sample_name}: Failed - {result.get('error', 'Unknown error')}")

    # 追加: 各株ごとに30細胞のグリッド画像を出力
    analyzer.visualize_sample_cells_with_fft(all_results, n_cells=30)
    # 追加: 全株まとめて30細胞ずつのグリッド画像を分割出力
    analyzer.visualize_all_samples_cells_with_fft(all_results, n_cells=30, cells_per_page=10)

def test_improved_fft():
    """改良されたFFT解析のテスト"""
    import os
    
    # テスト用の合成画像を作成
    def create_test_cell():
        # 細胞様の構造を持つテスト画像
        x, y = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64))
        r = np.sqrt(x**2 + y**2)
        
        # 中心が明るい円形構造
        cell = np.exp(-r**2 / 0.3)
        
        # 内部構造を追加
        cell += 0.3 * np.sin(4 * np.arctan2(y, x)) * np.exp(-r**2 / 0.2)
        
        # ノイズを追加
        cell += 0.1 * np.random.randn(64, 64)
        
        return cell
    
    # テスト用細胞画像を複数作成
    test_cells = [create_test_cell() for _ in range(10)]
    
    # 解析実行
    output_dir = './improved_fft_test'
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = CellFFTAnalyzer(output_dir)
    
    # 位相の重要性デモ
    analyzer.demonstrate_phase_importance(test_cells[0])
    
    # 詳細比較
    analyzer.visualize_fft_comparison(test_cells, max_cells=3)
    
    # 逆FFTの品質チェック
    print("\n=== Inverse FFT Quality Check ===")
    result = analyzer.analyze_individual_cell_fft(test_cells[0])
    
    print(f"Original image range: [{np.min(result['original']):.3f}, {np.max(result['original']):.3f}]")
    print(f"Inverse FFT range: [{np.min(result['inverse']):.3f}, {np.max(result['inverse']):.3f}]")
    print(f"Reconstruction error: {result['reconstruction_error']:.6f}")
    
    # 四隅の値をチェック
    h, w = result['inverse'].shape
    corners = [
        result['inverse'][0, 0],      # 左上
        result['inverse'][0, w-1],    # 右上
        result['inverse'][h-1, 0],    # 左下
        result['inverse'][h-1, w-1]   # 右下
    ]
    print(f"Corner values: {[f'{c:.3f}' for c in corners]}")
    
    # 中心の値と比較
    center = result['inverse'][h//2, w//2]
    print(f"Center value: {center:.3f}")
    print(f"Corner vs Center ratio: {np.mean(corners)/center:.3f}")
    
    print("Improved FFT analysis completed!")

def test_inverse_fft_fix():
    """逆FFT修正のテスト"""
    import os
    import numpy as np
    from scipy.fft import fft2, ifft2, fftshift, ifftshift
    
    # テスト用の単純な画像を作成
    test_image = np.zeros((32, 32))
    test_image[12:20, 12:20] = 1.0  # 中心に正方形
    
    # 正規化
    test_normalized = (test_image - np.mean(test_image)) / np.std(test_image)
    
    # FFT
    fft_result = fft2(test_normalized)
    fft_shifted = fftshift(fft_result)
    
    # 振幅と位相
    magnitude = np.abs(fft_shifted)
    phase = np.angle(fft_shifted)
    
    # 修正された逆FFT
    complex_fft = magnitude * np.exp(1j * phase)
    complex_fft_unshifted = ifftshift(complex_fft)
    inverse_result = ifft2(complex_fft_unshifted)
    inverse_real = np.real(inverse_result)
    inverse_normalized = (inverse_real - np.min(inverse_real)) / (np.max(inverse_real) - np.min(inverse_real))
    inverse_adjusted = (inverse_normalized - np.mean(inverse_normalized)) / np.std(inverse_normalized)
    
    # 結果をチェック
    reconstruction_error = np.mean(np.abs(test_normalized - inverse_adjusted))
    
    print("=== Inverse FFT Fix Test ===")
    print(f"Original image range: [{np.min(test_normalized):.3f}, {np.max(test_normalized):.3f}]")
    print(f"Inverse FFT range: [{np.min(inverse_adjusted):.3f}, {np.max(inverse_adjusted):.3f}]")
    print(f"Reconstruction error: {reconstruction_error:.6f}")
    
    # 四隅の値をチェック
    h, w = inverse_adjusted.shape
    corners = [
        inverse_adjusted[0, 0],      # 左上
        inverse_adjusted[0, w-1],    # 右上
        inverse_adjusted[h-1, 0],    # 左下
        inverse_adjusted[h-1, w-1]   # 右下
    ]
    center = inverse_adjusted[h//2, w//2]
    
    print(f"Corner values: {[f'{c:.3f}' for c in corners]}")
    print(f"Center value: {center:.3f}")
    print(f"Corner vs Center ratio: {np.mean(corners)/center:.3f}")
    
    if reconstruction_error < 1e-10:
        print("✓ Inverse FFT working correctly!")
    else:
        print("✗ Inverse FFT has issues!")
    
    if np.mean(corners)/center < 0.1:
        print("✓ No corner artifacts!")
    else:
        print("✗ Corner artifacts detected!")

if __name__ == "__main__":
    main()  # 通常の解析
    # test_improved_fft()  # 改良されたFFT解析のテスト
    # test_inverse_fft_fix()  # 逆FFT修正のテスト