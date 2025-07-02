import numpy as np
import tifffile as tiff
import os
from glob import glob
from tensorflow.keras.models import load_model, Model
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.measure import regionprops, label as skimage_label
from skimage.segmentation import clear_border
from skimage.transform import resize
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from datetime import datetime
import pandas as pd
import seaborn as sns
from scipy import stats

class GreenChannelAnalysis:
    def __init__(self):
        self.setup_environment()
        
    def setup_environment(self):
        """環境設定"""
        RANDOM_SEED = 42
        np.random.seed(RANDOM_SEED)
        tf.random.set_seed(RANDOM_SEED)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
    def extract_cells_green_only(self, image_path, stardist_model):
        """緑チャンネル専用細胞抽出"""
        try:
            image = tiff.imread(image_path)
            
            # チャンネル分離
            if image.ndim == 3 and image.shape[-1] >= 3:
                # セグメンテーション用（青/マゼンタチャンネル）
                seg_channel = image[..., 2]
                # 解析用（緑チャンネルのみ）
                green_channel = image[..., 1]
            else:
                seg_channel = image
                green_channel = image
            
            # 緑チャンネルの統計情報
            green_stats = {
                'mean': np.mean(green_channel),
                'std': np.std(green_channel),
                'median': np.median(green_channel),
                'nonzero_fraction': np.sum(green_channel > 0) / green_channel.size
            }
            
            # StarDist セグメンテーション
            normalized_seg = normalize(seg_channel)
            labels, details = stardist_model.predict_instances(normalized_seg)
            
            # 境界・サイズフィルタリング
            height, width = labels.shape
            filtered_labels = np.copy(labels)
            props = regionprops(labels)
            
            original_count = len(props)
            filtered_count = 0
            
            for prop in props:
                minr, minc, maxr, maxc = prop.bbox
                # 境界条件とサイズ条件
                if (minr < 10 or minc < 10 or maxr > (height - 10) or 
                    maxc > (width - 10) or prop.area < 100 or prop.area > 10000):
                    filtered_labels[labels == prop.label] = 0
                else:
                    filtered_count += 1
            
            # 緑チャンネルから細胞画像を抽出
            cell_images = []
            labeled_cells = skimage_label(filtered_labels)
            labeled_cells = clear_border(labeled_cells)
            final_props = regionprops(labeled_cells)
            
            for prop in final_props:
                minr, minc, maxr, maxc = prop.bbox
                cell_image = green_channel[minr:maxr, minc:maxc]
                
                # 細胞の緑チャンネル強度チェック
                cell_mean_intensity = np.mean(cell_image)
                
                # 極端に暗い細胞は除外
                if cell_image.size > 0 and cell_mean_intensity > 0.5:
                    resized_cell = resize(cell_image, (64, 64), anti_aliasing=True)
                    cell_images.append(resized_cell)
            
            return cell_images, green_stats, original_count, filtered_count, len(cell_images)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return [], {}, 0, 0, 0
    
    def analyze_samples(self, normal_folder, test_folders_dict, model_path, output_path):
        """サンプル解析"""
        os.makedirs(output_path, exist_ok=True)
        
        # モデル読み込み
        stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
        autoencoder = load_model(model_path)
        
        # エンコーダ部分の抽出
        encoder_input = autoencoder.input
        encoder_output = autoencoder.get_layer(index=6).output
        encoder = Model(inputs=encoder_input, outputs=encoder_output)
        
        # 正常細胞の処理
        print("=== Processing Normal Cells (Training Data) ===")
        normal_files = sorted(glob(os.path.join(normal_folder, '*.tif')))
        
        all_normal_cells = []
        normal_stats_list = []
        normal_processing_log = []
        
        for i, file_path in enumerate(normal_files):
            filename = os.path.basename(file_path)
            print(f"Processing {i+1}/{len(normal_files)}: {filename}")
            
            cells, green_stats, orig_count, filt_count, final_count = self.extract_cells_green_only(
                file_path, stardist_model
            )
            
            all_normal_cells.extend(cells)
            normal_stats_list.append(green_stats)
            
            log_entry = {
                'file': filename,
                'original_detected': orig_count,
                'filtered_detected': filt_count,
                'final_extracted': final_count,
                'green_mean': green_stats.get('mean', 0),
                'green_std': green_stats.get('std', 0)
            }
            normal_processing_log.append(log_entry)
            
            print(f"  Extracted: {final_count} cells (Green mean: {green_stats.get('mean', 0):.2f})")
        
        print(f"\nTotal normal cells extracted: {len(all_normal_cells)}")
        
        # 正常細胞の統計サマリー
        normal_stats_df = pd.DataFrame(normal_stats_list)
        normal_summary = {
            'green_mean_avg': normal_stats_df['mean'].mean(),
            'green_mean_std': normal_stats_df['mean'].std(),
            'green_std_avg': normal_stats_df['std'].mean(),
            'total_cells': len(all_normal_cells)
        }
        
        print(f"Normal cells summary:")
        print(f"  Green channel mean: {normal_summary['green_mean_avg']:.3f} ± {normal_summary['green_mean_std']:.3f}")
        print(f"  Green channel std: {normal_summary['green_std_avg']:.3f}")
        
        # 十分な正常細胞があるかチェック
        if len(all_normal_cells) < 500:
            print(f"Warning: Only {len(all_normal_cells)} normal cells available. Recommend >500 for stable training.")
        
        # 特徴量抽出
        print("\n=== Extracting Features ===")
        normal_cell_array = np.array(all_normal_cells)
        normal_cell_array = np.expand_dims(normal_cell_array, axis=-1)
        normal_cell_array = normal_cell_array.astype('float32') / 255.0
        
        # バッチ処理で特徴量抽出
        batch_size = 32
        normal_features_list = []
        
        for i in range(0, len(normal_cell_array), batch_size):
            batch = normal_cell_array[i:i+batch_size]
            batch_features = encoder.predict(batch, verbose=0)
            normal_features_list.append(batch_features)
        
        normal_features = np.vstack(normal_features_list)
        print(f"Extracted features shape: {normal_features.shape}")
        
        # 訓練・テスト分割
        train_features, test_features = train_test_split(
            normal_features, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"Training features: {train_features.shape}")
        print(f"Test features: {test_features.shape}")
        
        # 前処理
        scaler = StandardScaler()
        train_features_flat = train_features.reshape(len(train_features), -1)
        train_features_scaled = scaler.fit_transform(train_features_flat)
        
        # PCA（次元数を調整）
        n_components = min(200, train_features_scaled.shape[1], train_features_scaled.shape[0] - 1)
        pca = PCA(n_components=n_components)
        train_features_reduced = pca.fit_transform(train_features_scaled)
        
        print(f"PCA reduced to {n_components} components")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_[:5]}")
        
        # 異常検知モデル（保守的なパラメータ）
        anomaly_detectors = {
            'Conservative': OneClassSVM(gamma='scale', nu=0.05),  # 5%の異常率
            'Moderate': OneClassSVM(gamma='scale', nu=0.10),      # 10%の異常率
            'Liberal': OneClassSVM(gamma='scale', nu=0.15)        # 15%の異常率
        }
        
        # モデル学習
        for name, detector in anomaly_detectors.items():
            detector.fit(train_features_reduced)
        
        # 正常細胞テストセットでの基準異常率
        test_features_flat = test_features.reshape(len(test_features), -1)
        test_features_scaled = scaler.transform(test_features_flat)
        test_features_reduced = pca.transform(test_features_scaled)
        
        baseline_anomaly_rates = {}
        print(f"\n=== Baseline Anomaly Rates (Normal Test Set) ===")
        
        for name, detector in anomaly_detectors.items():
            predictions = detector.predict(test_features_reduced)
            anomaly_rate = np.sum(predictions == -1) / len(predictions)
            baseline_anomaly_rates[name] = anomaly_rate
            print(f"{name}: {anomaly_rate*100:.2f}%")
        
        # 結果辞書
        results = {
            'normal_summary': normal_summary,
            'baseline_anomaly_rates': baseline_anomaly_rates,
            'test_results': {},
            'processing_logs': {'normal': normal_processing_log}
        }
        
        # テストサンプルの解析
        print(f"\n=== Processing Test Samples ===")
        
        for sample_name, folder_path in test_folders_dict.items():
            print(f"\nProcessing {sample_name}...")
            
            test_files = sorted(glob(os.path.join(folder_path, '*.tif')))
            if not test_files:
                print(f"  No .tif files found in {folder_path}")
                continue
            
            sample_cells = []
            sample_stats_list = []
            sample_processing_log = []
            
            for file_path in test_files:
                filename = os.path.basename(file_path)
                cells, green_stats, orig_count, filt_count, final_count = self.extract_cells_green_only(
                    file_path, stardist_model
                )
                
                sample_cells.extend(cells)
                sample_stats_list.append(green_stats)
                
                log_entry = {
                    'file': filename,
                    'original_detected': orig_count,
                    'filtered_detected': filt_count,
                    'final_extracted': final_count,
                    'green_mean': green_stats.get('mean', 0),
                    'green_std': green_stats.get('std', 0)
                }
                sample_processing_log.append(log_entry)
                
                print(f"  {filename}: {final_count} cells (Green mean: {green_stats.get('mean', 0):.2f})")
            
            print(f"  Total {sample_name} cells: {len(sample_cells)}")
            
            if len(sample_cells) == 0:
                print(f"  No cells extracted for {sample_name}")
                continue
            
            # サンプルの統計サマリー
            sample_stats_df = pd.DataFrame(sample_stats_list)
            sample_summary = {
                'green_mean_avg': sample_stats_df['mean'].mean(),
                'green_mean_std': sample_stats_df['mean'].std(),
                'green_std_avg': sample_stats_df['std'].mean(),
                'total_cells': len(sample_cells)
            }
            
            # 特徴量抽出
            sample_cell_array = np.array(sample_cells)
            sample_cell_array = np.expand_dims(sample_cell_array, axis=-1)
            sample_cell_array = sample_cell_array.astype('float32') / 255.0
            
            sample_features_list = []
            for i in range(0, len(sample_cell_array), batch_size):
                batch = sample_cell_array[i:i+batch_size]
                batch_features = encoder.predict(batch, verbose=0)
                sample_features_list.append(batch_features)
            
            sample_features = np.vstack(sample_features_list)
            
            # 前処理
            sample_features_flat = sample_features.reshape(len(sample_features), -1)
            sample_features_scaled = scaler.transform(sample_features_flat)
            sample_features_reduced = pca.transform(sample_features_scaled)
            
            # 異常検知
            sample_result = {
                'summary': sample_summary,
                'anomaly_rates': {},
                'relative_scores': {}
            }
            
            for name, detector in anomaly_detectors.items():
                predictions = detector.predict(sample_features_reduced)
                anomaly_rate = np.sum(predictions == -1) / len(predictions)
                
                # 相対スコア（基準からの偏差）
                baseline_rate = baseline_anomaly_rates[name]
                relative_score = (anomaly_rate - baseline_rate) / (baseline_rate + 1e-6)
                
                sample_result['anomaly_rates'][name] = anomaly_rate
                sample_result['relative_scores'][name] = relative_score
                
                print(f"    {name}: {anomaly_rate*100:.2f}% (Relative: {relative_score:.2f})")
            
            results['test_results'][sample_name] = sample_result
            results['processing_logs'][sample_name] = sample_processing_log
        
        # 結果の可視化と保存
        self.visualize_results(results, output_path)
        self.generate_report(results, output_path)
        
        return results
    
    def visualize_results(self, results, output_path):
        """結果の可視化"""
        
        # 1. 異常率の比較（保守的モデル）
        plt.figure(figsize=(12, 8))
        
        conservative_model = 'Conservative'
        sample_names = ['Normal_Baseline']
        anomaly_rates = [results['baseline_anomaly_rates'][conservative_model] * 100]
        
        for sample_name, sample_result in results['test_results'].items():
            sample_names.append(sample_name)
            anomaly_rates.append(sample_result['anomaly_rates'][conservative_model] * 100)
        
        # 色分け（基準の2倍以上を赤、1.5倍以上をオレンジ）
        baseline_rate = results['baseline_anomaly_rates'][conservative_model] * 100
        colors = ['blue']  # 基準
        
        for rate in anomaly_rates[1:]:
            if rate > baseline_rate * 2:
                colors.append('red')
            elif rate > baseline_rate * 1.5:
                colors.append('orange')
            else:
                colors.append('lightblue')
        
        bars = plt.bar(sample_names, anomaly_rates, color=colors)
        
        # 基準線
        plt.axhline(y=baseline_rate, color='blue', linestyle='--', alpha=0.7, label='Baseline')
        plt.axhline(y=baseline_rate * 1.5, color='orange', linestyle='--', alpha=0.7, label='1.5x Baseline')
        plt.axhline(y=baseline_rate * 2, color='red', linestyle='--', alpha=0.7, label='2x Baseline')
        
        # 値の表示
        for bar, rate in zip(bars, anomaly_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.title(f'Anomaly Rates - Green Channel Only ({conservative_model} Model)')
        plt.xlabel('Samples')
        plt.ylabel('Anomaly Rate (%)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'green_channel_anomaly_rates.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 相対スコアのヒートマップ
        plt.figure(figsize=(10, 6))
        
        if results['test_results']:
            sample_names = list(results['test_results'].keys())
            model_names = list(results['baseline_anomaly_rates'].keys())
            
            relative_scores = []
            for sample_name in sample_names:
                scores = [results['test_results'][sample_name]['relative_scores'][model] 
                         for model in model_names]
                relative_scores.append(scores)
            
            heatmap_data = np.array(relative_scores)
            sns.heatmap(heatmap_data, 
                       xticklabels=model_names,
                       yticklabels=sample_names,
                       annot=True, fmt='.2f', cmap='RdYlBu_r', center=0,
                       cbar_kws={'label': 'Relative Anomaly Score'})
            
            plt.title('Relative Anomaly Scores - Green Channel Analysis')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'green_channel_relative_scores.png'), 
                       dpi=300, bbox_inches='tight')
            plt.show()
        
        # 3. 細胞数と緑チャンネル強度の比較
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 細胞数
        sample_names = []
        cell_counts = []
        green_means = []
        
        sample_names.append('Normal')
        cell_counts.append(results['normal_summary']['total_cells'])
        green_means.append(results['normal_summary']['green_mean_avg'])
        
        for sample_name, sample_result in results['test_results'].items():
            sample_names.append(sample_name)
            cell_counts.append(sample_result['summary']['total_cells'])
            green_means.append(sample_result['summary']['green_mean_avg'])
        
        ax1.bar(sample_names, cell_counts, color='lightgreen')
        ax1.set_title('Total Cell Count')
        ax1.set_ylabel('Number of Cells')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.bar(sample_names, green_means, color='green', alpha=0.7)
        ax2.set_title('Average Green Channel Intensity')
        ax2.set_ylabel('Green Channel Mean Intensity')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'green_channel_cell_stats.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, results, output_path):
        """レポート生成"""
        with open(os.path.join(output_path, 'green_channel_analysis_report.txt'), 'w') as f:
            f.write("=== GREEN CHANNEL ONLY ANOMALY ANALYSIS REPORT ===\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 正常細胞のサマリー
            f.write("NORMAL CELLS SUMMARY:\n")
            normal_summary = results['normal_summary']
            f.write(f"  Total cells: {normal_summary['total_cells']}\n")
            f.write(f"  Green channel mean: {normal_summary['green_mean_avg']:.3f} ± {normal_summary['green_mean_std']:.3f}\n")
            f.write(f"  Green channel std: {normal_summary['green_std_avg']:.3f}\n\n")
            
            # 基準異常率
            f.write("BASELINE ANOMALY RATES (Normal test set):\n")
            for model_name, rate in results['baseline_anomaly_rates'].items():
                f.write(f"  {model_name}: {rate*100:.2f}%\n")
            f.write("\n")
            
            # テストサンプル結果
            f.write("TEST SAMPLE RESULTS:\n")
            for sample_name, sample_result in results['test_results'].items():
                f.write(f"\n{sample_name}:\n")
                f.write(f"  Cells: {sample_result['summary']['total_cells']}\n")
                f.write(f"  Green mean: {sample_result['summary']['green_mean_avg']:.3f}\n")
                
                f.write("  Anomaly rates:\n")
                for model_name, rate in sample_result['anomaly_rates'].items():
                    relative_score = sample_result['relative_scores'][model_name]
                    f.write(f"    {model_name}: {rate*100:.2f}% (Relative: {relative_score:.2f})")
                    
                    if relative_score > 1.0:
                        f.write(" **HIGH ANOMALY**")
                    elif relative_score > 0.5:
                        f.write(" *Moderate anomaly*")
                    elif relative_score < -0.3:
                        f.write(" *Lower than normal*")
                    
                    f.write("\n")
            
            # 候補の推定
            f.write("\n\nPROMISING MUTANT CANDIDATES:\n")
            candidates = []
            
            for sample_name, sample_result in results['test_results'].items():
                # 保守的モデルでの相対スコアを使用
                conservative_score = sample_result['relative_scores']['Conservative']
                if conservative_score > 0.3:  # 30%以上の増加
                    candidates.append((sample_name, conservative_score))
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            if candidates:
                for sample_name, score in candidates:
                    f.write(f"  {sample_name}: Relative score = {score:.2f}\n")
            else:
                f.write("  No samples show consistently high anomaly scores\n")
            
            f.write("\n\nINTERPretation Guidelines:\n")
            f.write("- Relative score > 1.0: Very high anomaly (>2x baseline)\n")
            f.write("- Relative score 0.5-1.0: High anomaly\n")
            f.write("- Relative score 0.3-0.5: Moderate anomaly\n")
            f.write("- Relative score < 0.3: Similar to normal\n")
            f.write("- Negative scores: Lower anomaly than baseline\n")

def main():
    # パス設定
    normal_folder = "/Users/matsuokoujirou/Documents/Data/Screening/Noemal_cells"
    model_path = '/Users/matsuokoujirou/Documents/Data/Screening/Noemal_cells/RG2_CAE_green.h5'
    
    # テストフォルダの設定
    test_folders = {
        'RG2_current': "/Users/matsuokoujirou/Documents/Data/Screening/250603_check/RG2",
        '10A': "/Users/matsuokoujirou/Documents/Data/Screening/250603_check/10A",
        '10B': "/Users/matsuokoujirou/Documents/Data/Screening/250603_check/10B",
        '10C': "/Users/matsuokoujirou/Documents/Data/Screening/250603_check/10C",
        '10D': "/Users/matsuokoujirou/Documents/Data/Screening/250603_check/10D",
        '6C1': "/Users/matsuokoujirou/Documents/Data/Screening/250603_check/6C1",
        '6D6': "/Users/matsuokoujirou/Documents/Data/Screening/250603_check/6D6"
    }
    
    # 出力パス
    today = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = f"/Users/matsuokoujirou/Documents/Data/Screening/Result/{today}_green_channel_only"
    
    # 解析実行
    analyzer = GreenChannelAnalysis()
    results = analyzer.analyze_samples(normal_folder, test_folders, model_path, output_path)
    
    if results:
        print(f"\n=== GREEN CHANNEL ANALYSIS COMPLETED ===")
        print(f"Results saved to: {output_path}")
        
        # 結果サマリー
        print(f"\nBASELINE ANOMALY RATES:")
        for model_name, rate in results['baseline_anomaly_rates'].items():
            print(f"  {model_name}: {rate*100:.2f}%")
        
        print(f"\nTEST SAMPLE RESULTS (Conservative Model):")
        for sample_name, sample_result in results['test_results'].items():
            conservative_rate = sample_result['anomaly_rates']['Conservative']
            relative_score = sample_result['relative_scores']['Conservative']
            print(f"  {sample_name}: {conservative_rate*100:.2f}% (Relative: {relative_score:.2f})")
        
        # 推奨候補
        candidates = []
        for sample_name, sample_result in results['test_results'].items():
            score = sample_result['relative_scores']['Conservative']
            if score > 0.3:
                candidates.append((sample_name, score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if candidates:
            print(f"\nPROMISING CANDIDATES:")
            for sample_name, score in candidates:
                print(f"  {sample_name}: {score:.2f}")
        else:
            print(f"\nNo clear candidates identified")
    
    else:
        print("Analysis failed")

if __name__ == "__main__":
    main()