import numpy as np
import tifffile as tiff
import os
from glob import glob
from tensorflow.keras.models import load_model
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.measure import regionprops, label as skimage_label
from skimage.segmentation import clear_border
from skimage.transform import resize
from skimage import exposure
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import pickle

class ProductionMutantScreening:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.load_trained_models()
        
    def load_trained_models(self):
        """訓練済みモデルの読み込み"""
        print("Loading trained models...")
        
        # AutoencoderとEncoder
        self.autoencoder = load_model(os.path.join(self.model_dir, 'best_autoencoder.keras'))
        self.encoder = load_model(os.path.join(self.model_dir, 'encoder.keras'))
        
        # 前処理器
        with open(os.path.join(self.model_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(self.model_dir, 'pca.pkl'), 'rb') as f:
            self.pca = pickle.load(f)
        
        # 異常検知器
        with open(os.path.join(self.model_dir, 'detector_conservative.pkl'), 'rb') as f:
            self.detector_conservative = pickle.load(f)
        with open(os.path.join(self.model_dir, 'detector_moderate.pkl'), 'rb') as f:
            self.detector_moderate = pickle.load(f)
        
        # StarDist
        self.stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
        
        print("All models loaded successfully!")
    
    def extract_quality_cells(self, image_path):
        """品質管理付き細胞抽出（訓練時と同じ条件）"""
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
            labels, details = self.stardist_model.predict_instances(normalized_seg)
            
            # 品質フィルタリング（訓練時と同じ条件）
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
                
                # 形状チェック
                if prop.eccentricity > 0.95:
                    continue
                
                # 細胞画像抽出
                cell_image = green_channel[minr:maxr, minc:maxc]
                
                # 強度チェック
                cell_mean = np.mean(cell_image)
                cell_std = np.std(cell_image)
                
                if cell_mean < 0.5 or cell_std < 0.1:
                    continue
                
                # 訓練時と同じ前処理
                cell_image_eq = exposure.equalize_adapthist(cell_image, clip_limit=0.02)
                cell_image_resized = resize(cell_image_eq, (64, 64), anti_aliasing=True)
                
                quality_cells.append(cell_image_resized)
                
                cell_stats.append({
                    'area': prop.area,
                    'eccentricity': prop.eccentricity,
                    'solidity': prop.solidity,
                    'mean_intensity': cell_mean,
                    'std_intensity': cell_std
                })
            
            return quality_cells, cell_stats
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return [], []
    
    def compute_anomaly_scores(self, cell_images):
        """包括的異常スコア計算"""
        if len(cell_images) == 0:
            return {}
        
        X = np.expand_dims(np.array(cell_images), axis=-1).astype('float32')
        
        # 1. 再構成誤差
        reconstructed = self.autoencoder.predict(X, verbose=0)
        mse_errors = np.mean(np.square(X - reconstructed), axis=(1, 2, 3))
        mae_errors = np.mean(np.abs(X - reconstructed), axis=(1, 2, 3))
        
        # 2. エンコーダ特徴量ベース
        encoded_features = self.encoder.predict(X, verbose=0)
        encoded_flat = encoded_features.reshape(len(encoded_features), -1)
        
        # 前処理（訓練時と同じ）
        encoded_scaled = self.scaler.transform(encoded_flat)
        encoded_pca = self.pca.transform(encoded_scaled)
        
        # 異常検知
        conservative_predictions = self.detector_conservative.predict(encoded_pca)
        moderate_predictions = self.detector_moderate.predict(encoded_pca)
        
        conservative_scores = self.detector_conservative.decision_function(encoded_pca)
        moderate_scores = self.detector_moderate.decision_function(encoded_pca)
        
        return {
            'reconstruction_mse': mse_errors,
            'reconstruction_mae': mae_errors,
            'conservative_predictions': conservative_predictions,
            'moderate_predictions': moderate_predictions,
            'conservative_scores': -conservative_scores,  # 高いほど異常
            'moderate_scores': -moderate_scores,
            'conservative_anomaly_rate': np.sum(conservative_predictions == -1) / len(conservative_predictions),
            'moderate_anomaly_rate': np.sum(moderate_predictions == -1) / len(moderate_predictions)
        }
    
    def screen_mutant_samples(self, test_folders_dict, output_dir):
        """変異株スクリーニング実行"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("=== Starting Mutant Screening with Improved Model ===")
        
        results = {}
        detailed_results = []
        
        for sample_name, folder_path in test_folders_dict.items():
            print(f"\nProcessing {sample_name}...")
            
            tif_files = sorted(glob(os.path.join(folder_path, '*.tif')))
            if not tif_files:
                print(f"  No .tif files found in {folder_path}")
                continue
            
            sample_cells = []
            sample_stats = []
            file_summary = []
            
            # 各ファイルの処理
            for file_path in tif_files:
                filename = os.path.basename(file_path)
                cells, stats = self.extract_quality_cells(file_path)
                
                sample_cells.extend(cells)
                sample_stats.extend(stats)
                
                file_summary.append({
                    'filename': filename,
                    'cells_extracted': len(cells),
                    'mean_intensity': np.mean([s['mean_intensity'] for s in stats]) if stats else 0
                })
                
                print(f"  {filename}: {len(cells)} cells")
            
            print(f"  Total {sample_name} cells: {len(sample_cells)}")
            
            if len(sample_cells) == 0:
                print(f"  No quality cells extracted from {sample_name}")
                continue
            
            # 異常スコア計算
            anomaly_scores = self.compute_anomaly_scores(sample_cells)
            
            # 結果サマリー
            sample_result = {
                'sample_name': sample_name,
                'total_cells': len(sample_cells),
                'files_processed': len(tif_files),
                'conservative_anomaly_rate': anomaly_scores['conservative_anomaly_rate'],
                'moderate_anomaly_rate': anomaly_scores['moderate_anomaly_rate'],
                'mean_mse': np.mean(anomaly_scores['reconstruction_mse']),
                'std_mse': np.std(anomaly_scores['reconstruction_mse']),
                'mean_mae': np.mean(anomaly_scores['reconstruction_mae']),
                'std_mae': np.std(anomaly_scores['reconstruction_mae'])
            }
            
            results[sample_name] = sample_result
            
            # 詳細結果（細胞レベル）
            for i, (mse, mae, cons_pred, mod_pred, cons_score, mod_score) in enumerate(zip(
                anomaly_scores['reconstruction_mse'],
                anomaly_scores['reconstruction_mae'],
                anomaly_scores['conservative_predictions'],
                anomaly_scores['moderate_predictions'],
                anomaly_scores['conservative_scores'],
                anomaly_scores['moderate_scores']
            )):
                detailed_results.append({
                    'sample_name': sample_name,
                    'cell_id': i,
                    'mse': mse,
                    'mae': mae,
                    'conservative_anomaly': cons_pred == -1,
                    'moderate_anomaly': mod_pred == -1,
                    'conservative_score': cons_score,
                    'moderate_score': mod_score
                })
            
            # 進捗表示
            print(f"    Conservative anomaly rate: {sample_result['conservative_anomaly_rate']*100:.2f}%")
            print(f"    Moderate anomaly rate: {sample_result['moderate_anomaly_rate']*100:.2f}%")
            print(f"    Mean MSE: {sample_result['mean_mse']:.6f}")
        
        # 結果の保存と可視化
        self.save_and_visualize_results(results, detailed_results, output_dir)
        
        return results, detailed_results
    
    def save_and_visualize_results(self, results, detailed_results, output_dir):
        """結果の保存と可視化"""
        
        # サマリー結果のDataFrame化
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df.to_csv(os.path.join(output_dir, 'screening_summary.csv'))
        
        # 詳細結果の保存
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv(os.path.join(output_dir, 'detailed_cell_results.csv'), index=False)
        
        # 可視化
        self.create_screening_visualizations(results_df, detailed_df, output_dir)
        
        # レポート生成
        self.generate_screening_report(results_df, output_dir)
    
    def create_screening_visualizations(self, results_df, detailed_df, output_dir):
        """スクリーニング結果の可視化"""
        
        # 1. 異常率比較
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        sample_names = results_df.index.tolist()
        conservative_rates = results_df['conservative_anomaly_rate'] * 100
        moderate_rates = results_df['moderate_anomaly_rate'] * 100
        
        # Conservative
        bars1 = ax1.bar(sample_names, conservative_rates, color='lightcoral', alpha=0.8)
        ax1.axhline(y=5, color='blue', linestyle='--', alpha=0.7, label='Expected Normal (~5%)')
        ax1.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='High Anomaly Threshold')
        ax1.set_title('Conservative Model - Anomaly Rates')
        ax1.set_ylabel('Anomaly Rate (%)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()
        
        # # 値の表示
        # for bar, rate in zip(bars1, conservative_rates):
        #     ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
        #             f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Moderate
        bars2 = ax2.bar(sample_names, moderate_rates, color='lightblue', alpha=0.8)
        ax2.axhline(y=10, color='blue', linestyle='--', alpha=0.7, label='Expected Normal (~10%)')
        ax2.axhline(y=25, color='red', linestyle='--', alpha=0.7, label='High Anomaly Threshold')
        ax2.set_title('Moderate Model - Anomaly Rates')
        ax2.set_ylabel('Anomaly Rate (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        
        for bar, rate in zip(bars2, moderate_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'anomaly_rates_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 再構成誤差分布
        plt.figure(figsize=(12, 8))
        
        sample_names = detailed_df['sample_name'].unique()
        n_samples = len(sample_names)
        
        fig, axes = plt.subplots(2, (n_samples + 1) // 2, figsize=(15, 8))
        axes = axes.flatten() if n_samples > 1 else [axes]
        
        for i, sample_name in enumerate(sample_names):
            if i >= len(axes):
                break
                
            sample_data = detailed_df[detailed_df['sample_name'] == sample_name]
            
            axes[i].hist(sample_data['mse'], bins=30, alpha=0.7, density=True)
            axes[i].set_title(f'{sample_name}\n(n={len(sample_data)})')
            axes[i].set_xlabel('MSE')
            axes[i].set_ylabel('Density')
            
            # 平均線
            mean_mse = sample_data['mse'].mean()
            axes[i].axvline(mean_mse, color='red', linestyle='--', label=f'Mean: {mean_mse:.4f}')
            axes[i].legend()
        
        # 空のサブプロットを隠す
        for i in range(len(sample_names), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mse_distributions.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. 相関行列
        if len(results_df) > 1:
            plt.figure(figsize=(10, 8))
            
            correlation_data = results_df[['conservative_anomaly_rate', 'moderate_anomaly_rate', 'mean_mse', 'mean_mae']]
            correlation_matrix = correlation_data.corr()
            
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, cbar_kws={'label': 'Correlation Coefficient'})
            plt.title('Correlation Matrix of Anomaly Metrics')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_screening_report(self, results_df, output_dir):
        """スクリーニングレポート生成"""
        
        with open(os.path.join(output_dir, 'mutant_screening_report.txt'), 'w') as f:
            f.write("=== MUTANT SCREENING REPORT (IMPROVED MODEL) ===\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("MODEL PERFORMANCE BASELINE:\n")
            f.write("- Conservative model: ~5% anomaly rate for normal cells\n")
            f.write("- Moderate model: ~10% anomaly rate for normal cells\n\n")
            
            f.write("SCREENING RESULTS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Sample':<20} {'Cells':<8} {'Conservative':<12} {'Moderate':<12} {'Mean MSE':<12}\n")
            f.write("-" * 80 + "\n")
            
            for sample_name, row in results_df.iterrows():
                f.write(f"{sample_name:<20} {row['total_cells']:<8} "
                       f"{row['conservative_anomaly_rate']*100:>8.1f}% "
                       f"{row['moderate_anomaly_rate']*100:>10.1f}% "
                       f"{row['mean_mse']:>10.6f}\n")
            
            f.write("\n")
            
            # 異常候補の特定
            f.write("ANOMALY ANALYSIS:\n")
            
            # Conservative modelで15%以上
            high_conservative = results_df[results_df['conservative_anomaly_rate'] > 0.15]
            if not high_conservative.empty:
                f.write("\nHIGH ANOMALY CANDIDATES (Conservative >15%):\n")
                for sample_name, row in high_conservative.iterrows():
                    f.write(f"- {sample_name}: {row['conservative_anomaly_rate']*100:.1f}%\n")
            
            # Moderate modelで25%以上
            high_moderate = results_df[results_df['moderate_anomaly_rate'] > 0.25]
            if not high_moderate.empty:
                f.write("\nHIGH ANOMALY CANDIDATES (Moderate >25%):\n")
                for sample_name, row in high_moderate.iterrows():
                    f.write(f"- {sample_name}: {row['moderate_anomaly_rate']*100:.1f}%\n")
            
            # 正常レベル
            normal_conservative = results_df[results_df['conservative_anomaly_rate'] <= 0.10]
            if not normal_conservative.empty:
                f.write("\nNORMAL-LEVEL SAMPLES (Conservative ≤10%):\n")
                for sample_name, row in normal_conservative.iterrows():
                    f.write(f"- {sample_name}: {row['conservative_anomaly_rate']*100:.1f}%\n")
            
            f.write("\n\nRECOMMENDATIONS:\n")
            f.write("1. Focus on samples with Conservative >15% for detailed analysis\n")
            f.write("2. Samples with Conservative ≤10% are likely normal phenotype\n")
            f.write("3. Consider morphological analysis for high-anomaly candidates\n")
            f.write("4. Validate results with independent experimental methods\n")


def main():
    """メイン実行関数"""
    # 設定
    model_dir = "/Users/matsuokoujirou/Documents/Data/Screening/Models/20250614_1152_improved"
    
    # テストサンプル
    test_folders = {
        'RG2_current': "/Users/matsuokoujirou/Documents/Data/Screening/250603_check/RG2",
        '3-1':'/Volumes/NO NAME/240622/3-1',
        '3-2':'/Volumes/NO NAME/240622/3-2',
        '3-3':'/Volumes/NO NAME/240622/3-3',
        '3-4':'/Volumes/NO NAME/240622/3-4',
        "4-1":'/Volumes/NO NAME/240622/4-1',
        "4-2":'/Volumes/NO NAME/240622/4-2',
        "4-3":'/Volumes/NO NAME/240622/4-3',
        "4-4":'/Volumes/NO NAME/240622/4-4',
        "5-1":'/Volumes/NO NAME/240622/5-1',
        "5-2":'/Volumes/NO NAME/240622/5-2',
        "5-3":'/Volumes/NO NAME/240622/5-3',
        "5-4":'/Volumes/NO NAME/240622/5-4',
        "6-1":'/Volumes/NO NAME/240622/6-1',
        "6-2":'/Volumes/NO NAME/240622/6-2',
        "6-3":'/Volumes/NO NAME/240622/6-3',
        "6-4":'/Volumes/NO NAME/240622/6-4',
        "7-1":'/Volumes/NO NAME/240630/plate7/7-1',
        "7-2":'/Volumes/NO NAME/240630/plate7/7-2',
        "7-3":'/Volumes/NO NAME/240630/plate7/7-3',
        "7-4":'/Volumes/NO NAME/240630/plate7/7-4',
        "8-1":'/Volumes/NO NAME/240630/plate8/8-1',
        "8-2":'/Volumes/NO NAME/240630/plate8/8-2',
        "8-3":'/Volumes/NO NAME/240630/plate8/8-3',
        "8-4":'/Volumes/NO NAME/240630/plate8/8-4',
        "9-1":'/Volumes/NO NAME/240630/plate9/9-1',
        "9-2":'/Volumes/NO NAME/240630/plate9/9-2',
        "9-3":'/Volumes/NO NAME/240630/plate9/9-3',
        "9-4":'/Volumes/NO NAME/240630/plate9/9-4',
        "10-1":'/Volumes/NO NAME/240630/plate10/10-1',
        "10-2":'/Volumes/NO NAME/240630/plate10/10-2',
        "10-3":'/Volumes/NO NAME/240630/plate10/10-3',
        "10-4":'/Volumes/NO NAME/240630/plate10/10-4',
        '10A': "/Users/matsuokoujirou/Documents/Data/Screening/250603_check/10A",
        '10B': "/Users/matsuokoujirou/Documents/Data/Screening/250603_check/10B",
        '10C': "/Users/matsuokoujirou/Documents/Data/Screening/250603_check/10C",
        '10D': "/Users/matsuokoujirou/Documents/Data/Screening/250603_check/10D",
        '6C1': "/Users/matsuokoujirou/Documents/Data/Screening/250603_check/6C1",
        '6D6': "/Users/matsuokoujirou/Documents/Data/Screening/250603_check/6D6",
        "KO-60":"/Users/matsuokoujirou/Documents/Data/imaging_data/240501_KO-60",
        "KO-62":"/Users/matsuokoujirou/Documents/Data/imaging_data/240501_KO-62",
        "ccm1":"/Users/matsuokoujirou/Documents/Data/imaging_data/240206_RV5ccm1_LC_pyrearea",
        "WT":"/Users/matsuokoujirou/Documents/Data/imaging_data/240404/RV5_teacher",
        "EPYC1":"/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/epyc1",
        "MITH1":"/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/mith1",
        "RBMP1":"/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/rbmp1",
        "RBMP2":"/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/rbmp2",
        "RG2_AM":"/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/RG2",
        "SAGA1":"/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/saga1",
        "SAGA2":"/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/saga2",
        # "6-2-1":'/Volumes/NO NAME/240628/6-2/6-2-1',
        # "6-2-2":'/Volumes/NO NAME/240628/6-2/6-2-2',
        # "6-2-3":'/Volumes/NO NAME/240628/6-2/6-2-3',
        # "6-2-4":'/Volumes/NO NAME/240628/6-2/6-2-4',
        # "6-2-5":'/Volumes/NO NAME/240628/6-2/6-2-5',
        # "6-2-6":'/Volumes/NO NAME/240628/6-2/6-2-6',
        # "6-2-7":'/Volumes/NO NAME/240628/6-2/6-2-7',
        # "6-2-8":'/Volumes/NO NAME/240628/6-2/6-2-8', 
        # "6-2-9":'/Volumes/NO NAME/240628/6-2/6-2-9',
        # "6-2-10":'/Volumes/NO NAME/240628/6-2/6-2-10',
        # "6-2-11":'/Volumes/NO NAME/240628/6-2/6-2-11',
        # "6-2-12":'/Volumes/NO NAME/240628/6-2/6-2-12', 
        # "6-2-13":'/Volumes/NO NAME/240628/6-2/6-2-13',
        # "6-2-14":'/Volumes/NO NAME/240628/6-2/6-2-14',
        # "6-2-15":'/Volumes/NO NAME/240628/6-2/6-2-15',
        # "6-2-16":'/Volumes/NO NAME/240628/6-2/6-2-16',
        # "6-2-17":'/Volumes/NO NAME/240628/6-2/6-2-17',
        # "6-2-18":'/Volumes/NO NAME/240628/6-2/6-2-18',
        # "6-2-19":'/Volumes/NO NAME/240628/6-2/6-2-19',
        # "6-2-20":'/Volumes/NO NAME/240628/6-2/6-2-20',
        # "6-2-21":'/Volumes/NO NAME/240628/6-2/6-2-21',
        # "6-2-22":'/Volumes/NO NAME/240628/6-2/6-2-22',
        # "6-2-23":'/Volumes/NO NAME/240628/6-2/6-2-23',
        # "6-2-24":'/Volumes/NO NAME/240628/6-2/6-2-24',
    }
    
    # 出力ディレクトリ
    output_dir = f"/Users/matsuokoujirou/Documents/Data/Screening/Results/{datetime.now().strftime('%Y%m%d_%H%M')}_improved_screening"
    
    # スクリーニング実行
    screener = ProductionMutantScreening(model_dir)
    results, detailed_results = screener.screen_mutant_samples(test_folders, output_dir)
    
    print(f"\n=== SCREENING COMPLETED ===")
    print(f"Results saved to: {output_dir}")
    
    # 簡易サマリー表示
    print(f"\nQUICK SUMMARY:")
    print(f"{'Sample':<15} {'Conservative':<12} {'Moderate':<12}")
    print("-" * 40)
    
    for sample_name, result in results.items():
        print(f"{sample_name:<15} {result['conservative_anomaly_rate']*100:>8.1f}% "
              f"{result['moderate_anomaly_rate']*100:>10.1f}%")


if __name__ == "__main__":
    main()