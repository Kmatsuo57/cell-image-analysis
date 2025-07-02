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
        """訓練済みモデルの読み込み（高感度版追加）"""
        print("Loading trained models...")
        
        # AutoencoderとEncoder
        self.autoencoder = load_model(os.path.join(self.model_dir, 'best_autoencoder.keras'))
        self.encoder = load_model(os.path.join(self.model_dir, 'encoder.keras'))
        
        # 前処理器
        with open(os.path.join(self.model_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(self.model_dir, 'pca.pkl'), 'rb') as f:
            self.pca = pickle.load(f)
        
        # 異常検知器（高感度版含む）
        self.detectors = {}
        detector_files = {
            'extreme_conservative': 'detector_extremeconservative.pkl',
            'ultra_conservative': 'detector_ultraconservative.pkl',
            'super_conservative': 'detector_superconservative.pkl', 
            'very_conservative': 'detector_veryconservative.pkl',
            'conservative': 'detector_conservative.pkl',
            'moderate': 'detector_moderate.pkl'
        }
        
        for name, filename in detector_files.items():
            filepath = os.path.join(self.model_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    self.detectors[name] = pickle.load(f)
                print(f"  Loaded {name} detector")
            else:
                print(f"  Warning: {filename} not found")
        
        # StarDist
        self.stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
        
        print(f"All models loaded successfully! ({len(self.detectors)} detectors)")
    
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
        """基本的異常スコア計算"""
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
        
        # 全検知器での異常検知
        results = {
            'reconstruction_mse': mse_errors,
            'reconstruction_mae': mae_errors,
        }
        
        for detector_name, detector in self.detectors.items():
            predictions = detector.predict(encoded_pca)
            scores = detector.decision_function(encoded_pca)
            
            results[f'{detector_name}_predictions'] = predictions
            results[f'{detector_name}_scores'] = -scores  # 高いほど異常
            results[f'{detector_name}_anomaly_rate'] = np.sum(predictions == -1) / len(predictions)
        
        return results
    
    def compute_threshold_based_anomaly_scores(self, cell_images):
        """閾値ベースの動的異常検知"""
        if len(cell_images) == 0:
            return {}
        
        X = np.expand_dims(np.array(cell_images), axis=-1).astype('float32')
        
        # 1. 再構成誤差による検知
        reconstructed = self.autoencoder.predict(X, verbose=0)
        mse_errors = np.mean(np.square(X - reconstructed), axis=(1, 2, 3))
        mae_errors = np.mean(np.abs(X - reconstructed), axis=(1, 2, 3))
        
        # 2. エンコーダ特徴量ベース
        encoded_features = self.encoder.predict(X, verbose=0)
        encoded_flat = encoded_features.reshape(len(encoded_features), -1)
        encoded_scaled = self.scaler.transform(encoded_flat)
        encoded_pca = self.pca.transform(encoded_scaled)
        
        # 3. 動的閾値の計算
        results = {
            'reconstruction_mse': mse_errors,
            'reconstruction_mae': mae_errors,
        }
        
        # 各検知器でのスコア計算
        for detector_name, detector in self.detectors.items():
            scores = detector.decision_function(encoded_pca)
            results[f'{detector_name}_scores'] = -scores  # 高いほど異常
        
        # 4. パーセンタイル閾値による異常判定
        percentile_thresholds = [99.9, 99.5, 99.0, 95.0, 90.0, 80.0, 70.0, 60.0, 50.0]
        
        for detector_name in self.detectors.keys():
            score_key = f'{detector_name}_scores'
            if score_key in results:
                scores = results[score_key]
                
                for percentile in percentile_thresholds:
                    threshold = np.percentile(scores, percentile)
                    anomaly_predictions = scores > threshold
                    anomaly_rate = np.sum(anomaly_predictions) / len(anomaly_predictions)
                    
                    results[f'{detector_name}_p{int(percentile)}_predictions'] = anomaly_predictions
                    results[f'{detector_name}_p{int(percentile)}_anomaly_rate'] = anomaly_rate
                    results[f'{detector_name}_p{int(percentile)}_threshold'] = threshold
        
        # 5. 再構成誤差閾値
        for error_type in ['mse', 'mae']:
            errors = results[f'reconstruction_{error_type}']
            
            for percentile in percentile_thresholds:
                threshold = np.percentile(errors, percentile)
                anomaly_predictions = errors > threshold
                anomaly_rate = np.sum(anomaly_predictions) / len(anomaly_predictions)
                
                results[f'reconstruction_{error_type}_p{int(percentile)}_predictions'] = anomaly_predictions
                results[f'reconstruction_{error_type}_p{int(percentile)}_anomaly_rate'] = anomaly_rate
                results[f'reconstruction_{error_type}_p{int(percentile)}_threshold'] = threshold
        
        return results
    
    def adaptive_anomaly_detection(self, cell_images, target_sensitivity=0.90):
        """適応的異常検知（目標感度を指定）"""
        if len(cell_images) == 0:
            return {}
        
        # 全スコア計算
        all_scores = self.compute_threshold_based_anomaly_scores(cell_images)
        
        # 最適閾値の選択
        best_results = {}
        
        for detector_name in self.detectors.keys():
            score_key = f'{detector_name}_scores'
            if score_key in all_scores:
                scores = all_scores[score_key]
                
                # 目標感度に最も近い閾値を選択
                best_percentile = None
                best_rate = None
                target_rate = target_sensitivity
                
                percentile_thresholds = [99.9, 99.5, 99.0, 95.0, 90.0, 80.0, 70.0, 60.0, 50.0]
                
                for percentile in percentile_thresholds:
                    rate_key = f'{detector_name}_p{int(percentile)}_anomaly_rate'
                    if rate_key in all_scores:
                        rate = all_scores[rate_key]
                        
                        if best_rate is None or abs(rate - target_rate) < abs(best_rate - target_rate):
                            best_rate = rate
                            best_percentile = percentile
                
                if best_percentile is not None:
                    pred_key = f'{detector_name}_p{int(best_percentile)}_predictions'
                    threshold_key = f'{detector_name}_p{int(best_percentile)}_threshold'
                    
                    best_results[f'{detector_name}_adaptive_predictions'] = all_scores[pred_key]
                    best_results[f'{detector_name}_adaptive_anomaly_rate'] = best_rate
                    best_results[f'{detector_name}_adaptive_threshold'] = all_scores[threshold_key]
                    best_results[f'{detector_name}_adaptive_percentile'] = best_percentile
        
        # 基本情報も含める
        best_results['reconstruction_mse'] = all_scores['reconstruction_mse']
        best_results['reconstruction_mae'] = all_scores['reconstruction_mae']
        
        return best_results
    
    def ensemble_anomaly_detection(self, cell_images, consensus_threshold=0.5):
        """複数検知器のアンサンブル投票による異常検知"""
        if len(cell_images) == 0:
            return {}
        
        X = np.expand_dims(np.array(cell_images), axis=-1).astype('float32')
        
        # 1. 再構成誤差
        reconstructed = self.autoencoder.predict(X, verbose=0)
        mse_errors = np.mean(np.square(X - reconstructed), axis=(1, 2, 3))
        mae_errors = np.mean(np.abs(X - reconstructed), axis=(1, 2, 3))
        
        # 2. 特徴量抽出と前処理
        encoded_features = self.encoder.predict(X, verbose=0)
        encoded_flat = encoded_features.reshape(len(encoded_features), -1)
        encoded_scaled = self.scaler.transform(encoded_flat)
        encoded_pca = self.pca.transform(encoded_scaled)
        
        # 3. 各検知器の投票
        votes = []
        detector_results = {}
        
        for detector_name, detector in self.detectors.items():
            predictions = detector.predict(encoded_pca)
            scores = detector.decision_function(encoded_pca)
            
            # 異常(-1)を1に、正常(+1)を0に変換
            anomaly_votes = (predictions == -1).astype(int)
            votes.append(anomaly_votes)
            
            detector_results[f'{detector_name}_predictions'] = predictions
            detector_results[f'{detector_name}_scores'] = -scores
            detector_results[f'{detector_name}_anomaly_rate'] = np.mean(anomaly_votes)
        
        # 4. 再構成誤差による投票
        mse_threshold = np.percentile(mse_errors, 95)
        mae_threshold = np.percentile(mae_errors, 95)
        
        mse_votes = (mse_errors > mse_threshold).astype(int)
        mae_votes = (mae_errors > mae_threshold).astype(int)
        
        votes.extend([mse_votes, mae_votes])
        
        # 5. アンサンブル投票
        if len(votes) > 0:
            vote_matrix = np.array(votes).T  # (n_cells, n_voters)
            vote_counts = np.sum(vote_matrix, axis=1)  # 各細胞の異常票数
            total_voters = vote_matrix.shape[1]
            
            # 投票率による異常判定
            vote_ratios = vote_counts / total_voters
            
            # 複数の閾値での判定
            ensemble_results = {
                'reconstruction_mse': mse_errors,
                'reconstruction_mae': mae_errors,
                'vote_counts': vote_counts,
                'vote_ratios': vote_ratios,
                'total_voters': total_voters
            }
            
            # 各検知器の結果も含める
            ensemble_results.update(detector_results)
            
            # 異なる合意レベルでの判定
            consensus_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            
            for level in consensus_levels:
                consensus_predictions = vote_ratios >= level
                consensus_rate = np.mean(consensus_predictions)
                
                ensemble_results[f'consensus_{int(level*100)}_predictions'] = consensus_predictions
                ensemble_results[f'consensus_{int(level*100)}_anomaly_rate'] = consensus_rate
            
            # 推奨レベル（50%合意）
            recommended_predictions = vote_ratios >= consensus_threshold
            ensemble_results['recommended_predictions'] = recommended_predictions
            ensemble_results['recommended_anomaly_rate'] = np.mean(recommended_predictions)
            ensemble_results['recommended_threshold'] = consensus_threshold
            
            return ensemble_results
        
        return {}
    
    def comprehensive_screening_analysis(self, test_folders_dict, output_dir):
        """包括的スクリーニング解析"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("=== COMPREHENSIVE SCREENING ANALYSIS ===")
        
        all_results = {}
        
        for sample_name, folder_path in test_folders_dict.items():
            print(f"\nProcessing {sample_name}...")
            
            tif_files = sorted(glob(os.path.join(folder_path, '*.tif')))
            sample_cells = []
            
            for file_path in tif_files:
                cells, _ = self.extract_quality_cells(file_path)
                sample_cells.extend(cells)
            
            if len(sample_cells) == 0:
                print(f"  No cells extracted")
                continue
            
            print(f"  Analyzing {len(sample_cells)} cells...")
            
            # 1. 標準検知
            standard_results = self.compute_anomaly_scores(sample_cells)
            
            # 2. 適応的検知
            adaptive_results = self.adaptive_anomaly_detection(sample_cells, target_sensitivity=0.90)
            
            # 3. アンサンブル検知
            ensemble_results = self.ensemble_anomaly_detection(sample_cells, consensus_threshold=0.5)
            
            # 結果統合
            comprehensive_result = {
                'sample_name': sample_name,
                'total_cells': len(sample_cells),
                'standard': standard_results,
                'adaptive': adaptive_results,
                'ensemble': ensemble_results
            }
            
            all_results[sample_name] = comprehensive_result
            
            # 結果表示
            print(f"  Results summary:")
            
            # 標準検知結果
            if 'conservative_anomaly_rate' in standard_results:
                rate = standard_results['conservative_anomaly_rate'] * 100
                print(f"    Standard Conservative: {rate:.1f}%")
            
            # 適応的検知結果
            for detector_name in ['extreme_conservative', 'ultra_conservative', 'conservative']:
                if detector_name in self.detectors:
                    rate_key = f'{detector_name}_adaptive_anomaly_rate'
                    if rate_key in adaptive_results:
                        rate = adaptive_results[rate_key] * 100
                        percentile = adaptive_results.get(f'{detector_name}_adaptive_percentile', 'N/A')
                        print(f"    Adaptive {detector_name}: {rate:.1f}% (P{percentile})")
            
            # アンサンブル結果
            if 'recommended_anomaly_rate' in ensemble_results:
                rate = ensemble_results['recommended_anomaly_rate'] * 100
                print(f"    Ensemble (50% consensus): {rate:.1f}%")
            
            # 高合意レベル
            if 'consensus_80_anomaly_rate' in ensemble_results:
                rate = ensemble_results['consensus_80_anomaly_rate'] * 100
                print(f"    Ensemble (80% consensus): {rate:.1f}%")
        
        # 結果の保存
        self.save_comprehensive_results(all_results, output_dir)
        
        return all_results
    
    def save_comprehensive_results(self, all_results, output_dir):
        """包括的結果の保存"""
        
        # サマリーテーブル作成
        summary_data = []
        
        for sample_name, result in all_results.items():
            row = {
                'Sample': sample_name,
                'Total_Cells': result['total_cells']
            }
            
            # 標準結果
            standard = result.get('standard', {})
            for detector_name in self.detectors.keys():
                rate_key = f'{detector_name}_anomaly_rate'
                if rate_key in standard:
                    row[f'Standard_{detector_name}'] = standard[rate_key] * 100
            
            # 適応的結果
            adaptive = result.get('adaptive', {})
            for detector_name in self.detectors.keys():
                rate_key = f'{detector_name}_adaptive_anomaly_rate'
                if rate_key in adaptive:
                    row[f'Adaptive_{detector_name}'] = adaptive[rate_key] * 100
            
            # アンサンブル結果
            ensemble = result.get('ensemble', {})
            if 'recommended_anomaly_rate' in ensemble:
                row['Ensemble_50pct'] = ensemble['recommended_anomaly_rate'] * 100
            if 'consensus_80_anomaly_rate' in ensemble:
                row['Ensemble_80pct'] = ensemble['consensus_80_anomaly_rate'] * 100
            if 'consensus_90_anomaly_rate' in ensemble:
                row['Ensemble_90pct'] = ensemble['consensus_90_anomaly_rate'] * 100
            
            summary_data.append(row)
        
        # CSV保存
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, 'comprehensive_screening_summary.csv'), index=False)
        
        # 詳細レポート
        with open(os.path.join(output_dir, 'comprehensive_screening_report.txt'), 'w') as f:
            f.write("=== COMPREHENSIVE SCREENING REPORT ===\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("ANALYSIS METHODS:\n")
            f.write("1. Standard Detection: Original SVM-based detection\n")
            f.write("2. Adaptive Detection: Dynamic threshold selection for target sensitivity (90%)\n")
            f.write("3. Ensemble Detection: Multi-detector voting system\n\n")
            
            f.write("SCREENING RESULTS:\n")
            f.write("-" * 130 + "\n")
            
            # ヘッダー
            header = f"{'Sample':<15} {'Cells':<6}"
            header += f" {'Std_Cons':<10} {'Adapt_Ext':<11} {'Adapt_Ultra':<12} {'Ens_50%':<9} {'Ens_80%':<9} {'Ens_90%':<9}"
            f.write(header + "\n")
            f.write("-" * 130 + "\n")
            
            # データ行
            for _, row in summary_df.iterrows():
                line = f"{row['Sample']:<15} {row['Total_Cells']:<6}"
                
                # 各列のデータを安全に取得
                std_cons = row.get('Standard_conservative', 0)
                adapt_ext = row.get('Adaptive_extreme_conservative', 0)
                adapt_ultra = row.get('Adaptive_ultra_conservative', 0) 
                ens_50 = row.get('Ensemble_50pct', 0)
                ens_80 = row.get('Ensemble_80pct', 0)
                ens_90 = row.get('Ensemble_90pct', 0)
                
                line += f" {std_cons:>8.1f}% {adapt_ext:>9.1f}% {adapt_ultra:>10.1f}% {ens_50:>7.1f}% {ens_80:>7.1f}% {ens_90:>7.1f}%"
                f.write(line + "\n")
            
            f.write("\n")
            
            # 高異常率候補の特定
            f.write("HIGH ANOMALY CANDIDATES:\n")
            
            # 90%アンサンブル基準
            high_ensemble_90 = summary_df[summary_df.get('Ensemble_90pct', 0) > 50]
            if not high_ensemble_90.empty:
                f.write("Based on 90% ensemble consensus (>50%):\n")
                for _, row in high_ensemble_90.iterrows():
                    f.write(f"🔥 {row['Sample']}: {row.get('Ensemble_90pct', 0):.1f}% - EXTREMELY HIGH CONFIDENCE\n")
                f.write("\n")
            
            # 80%アンサンブル基準
            high_ensemble_80 = summary_df[summary_df.get('Ensemble_80pct', 0) > 30]
            if not high_ensemble_80.empty:
                f.write("Based on 80% ensemble consensus (>30%):\n")
                for _, row in high_ensemble_80.iterrows():
                    f.write(f"⚠️  {row['Sample']}: {row.get('Ensemble_80pct', 0):.1f}% - HIGH CONFIDENCE\n")
                f.write("\n")
            
            # 適応的検知基準
            adapt_ultra_high = summary_df[summary_df.get('Adaptive_ultra_conservative', 0) > 70]
            if not adapt_ultra_high.empty:
                f.write("Based on Adaptive Ultra-Conservative (>70%):\n")
                for _, row in adapt_ultra_high.iterrows():
                    f.write(f"📋 {row['Sample']}: {row.get('Adaptive_ultra_conservative', 0):.1f}% - ADAPTIVE HIGH\n")
                f.write("\n")
            
            if high_ensemble_90.empty and high_ensemble_80.empty and adapt_ultra_high.empty:
                f.write("No samples exceeded high-confidence thresholds.\n")
                
                # 中程度の候補も表示
                moderate_candidates = summary_df[
                    (summary_df.get('Ensemble_50pct', 0) > 20) |
                    (summary_df.get('Adaptive_ultra_conservative', 0) > 50)
                ]
                
                if not moderate_candidates.empty:
                    f.write("\nMODERATE ANOMALY CANDIDATES:\n")
                    for _, row in moderate_candidates.iterrows():
                        ens_50 = row.get('Ensemble_50pct', 0)
                        adapt_ultra = row.get('Adaptive_ultra_conservative', 0)
                        f.write(f"• {row['Sample']}: Ens50={ens_50:.1f}%, AdaptUltra={adapt_ultra:.1f}%\n")
            
            f.write("\nRECOMMENDATIONS:\n")
            f.write("1. Prioritize samples with 90% ensemble consensus >50%\n")
            f.write("2. Validate samples with 80% ensemble consensus >30%\n")
            f.write("3. Consider adaptive ultra-conservative results >70%\n")
            f.write("4. Use multiple detection methods for confidence assessment\n")
            f.write("5. Perform biological validation for all high-confidence candidates\n")
        
        print(f"Comprehensive results saved to: {output_dir}")
    
    def screen_mutant_samples(self, test_folders_dict, output_dir):
        """標準的なスクリーニング（後方互換性のため）"""
        return self.comprehensive_screening_analysis(test_folders_dict, output_dir)

def main():
    """メイン関数（包括的解析対応）"""
    # 設定
    model_dir = "/Users/matsuokoujirou/Documents/Data/Screening/Models/20250613_1909_ultra_sensitive"
    
    # テストサンプル
    test_folders = {
        'RG2_current': "/Users/matsuokoujirou/Documents/Data/Screening/250603_check/RG2",
        "EPYC1":"/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/epyc1",
        "MITH1":"/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/mith1",
        "RBMP1":"/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/rbmp1",
        "RBMP2":"/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/rbmp2",
        "RG2_AM":"/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/RG2",
        "SAGA1":"/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/saga1",
        "SAGA2":"/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/saga2",
    }
    
    # 出力ディレクトリ
    output_dir = f"/Users/matsuokoujirou/Documents/Data/Screening/Results/{datetime.now().strftime('%Y%m%d_%H%M')}_comprehensive_screening"
    
    # 包括的スクリーニング実行
    screener = ProductionMutantScreening(model_dir)
    all_results = screener.comprehensive_screening_analysis(test_folders, output_dir)
    
    print(f"\n=== COMPREHENSIVE SCREENING COMPLETED ===")
    print(f"Results saved to: {output_dir}")
    
    # 最終サマリー表示
    print(f"\n=== FINAL SUMMARY ===")
    
    high_confidence_found = False
    
    for sample_name, result in all_results.items():
        ensemble = result.get('ensemble', {})
        adaptive = result.get('adaptive', {})
        
        # 高信頼度候補の特定
        ens_90 = ensemble.get('consensus_90_anomaly_rate', 0) * 100
        ens_80 = ensemble.get('consensus_80_anomaly_rate', 0) * 100
        ens_50 = ensemble.get('recommended_anomaly_rate', 0) * 100
        
        adapt_ultra = adaptive.get('ultra_conservative_adaptive_anomaly_rate', 0) * 100
        adapt_extreme = adaptive.get('extreme_conservative_adaptive_anomaly_rate', 0) * 100
        
        # 結果分類
        if ens_90 > 50 or adapt_extreme > 80:
            print(f"🔥 {sample_name}: EXTREMELY HIGH ANOMALY")
            print(f"   90% Consensus: {ens_90:.1f}%, Adaptive Extreme: {adapt_extreme:.1f}%")
            high_confidence_found = True
        elif ens_80 > 30 or adapt_ultra > 70:
            print(f"⚠️  {sample_name}: HIGH ANOMALY")
            print(f"   80% Consensus: {ens_80:.1f}%, Adaptive Ultra: {adapt_ultra:.1f}%")
            high_confidence_found = True
        elif ens_50 > 20 or adapt_ultra > 50:
            print(f"📋 {sample_name}: MODERATE ANOMALY")
            print(f"   50% Consensus: {ens_50:.1f}%, Adaptive Ultra: {adapt_ultra:.1f}%")
        else:
            print(f"✅ {sample_name}: NORMAL")
            print(f"   50% Consensus: {ens_50:.1f}%, Adaptive Ultra: {adapt_ultra:.1f}%")
    
    if high_confidence_found:
        print(f"\n🎯 HIGH-CONFIDENCE ANOMALIES DETECTED!")
        print(f"   Recommend immediate biological validation.")
    else:
        print(f"\n⚡ No high-confidence anomalies detected.")
        print(f"   Consider reviewing moderate candidates or adjusting sensitivity.")

def run_adaptive_screening_example():
    """適応的スクリーニングの実行例"""
    
    model_dir = "/Users/matsuokoujirou/Documents/Data/Screening/Models/20250613_1524_improved"
    
    test_folders = {
        'RG2_current': "/Users/matsuokoujirou/Documents/Data/Screening/250603_check/RG2",
        "SAGA2":"/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/saga2",
        "RBMP1":"/Users/matsuokoujirou/Documents/Data/Screening/Abnormal_data/rbmp1",
    }
    
    screener = ProductionMutantScreening(model_dir)
    
    print("=== ADAPTIVE SCREENING RESULTS ===")
    
    for sample_name, folder_path in test_folders.items():
        print(f"\nProcessing {sample_name}...")
        
        tif_files = sorted(glob(os.path.join(folder_path, '*.tif')))
        sample_cells = []
        
        for file_path in tif_files:
            cells, _ = screener.extract_quality_cells(file_path)
            sample_cells.extend(cells)
        
        if len(sample_cells) == 0:
            continue
        
        # 適応的異常検知
        adaptive_results = screener.adaptive_anomaly_detection(
            sample_cells, 
            target_sensitivity=0.90  # 90%の異常検知を目標
        )
        
        print(f"  Total cells: {len(sample_cells)}")
        print(f"  Adaptive anomaly rates:")
        
        for detector_name in screener.detectors.keys():
            rate_key = f'{detector_name}_adaptive_anomaly_rate'
            percentile_key = f'{detector_name}_adaptive_percentile'
            
            if rate_key in adaptive_results:
                rate = adaptive_results[rate_key] * 100
                percentile = adaptive_results[percentile_key]
                print(f"    {detector_name}: {rate:.1f}% (using {percentile}th percentile)")

class PooledMutantScreening(ProductionMutantScreening):
    """プール解析対応のスクリーニングシステム"""
    
    def create_pools(self, colony_paths, pool_size=24):
        """コロニーをプールに分割"""
        pools = []
        for i in range(0, len(colony_paths), pool_size):
            pool = colony_paths[i:i+pool_size]
            pools.append({
                'pool_id': f'Pool_{i//pool_size + 1:03d}',
                'colonies': pool,
                'colony_names': [os.path.basename(p) for p in pool]
            })
        return pools
    
    def screen_pool(self, pool_colonies, detection_method='adaptive'):
        """プール全体の異常率を測定"""
        all_cells = []
        colony_info = []
        
        print(f"  Processing {len(pool_colonies)} colonies in pool...")
        
        for colony_path in pool_colonies:
            cells, stats = self.extract_quality_cells(colony_path)
            all_cells.extend(cells)
            
            colony_info.append({
                'colony': os.path.basename(colony_path),
                'cell_count': len(cells),
                'path': colony_path
            })
        
        if len(all_cells) == 0:
            return {
                'total_cells': 0,
                'colony_info': colony_info,
                'anomaly_rate': 0.0,
                'status': 'no_cells'
            }
        
        # プール全体の異常率（選択された方法で）
        if detection_method == 'adaptive':
            pool_results = self.adaptive_anomaly_detection(all_cells, target_sensitivity=0.90)
            # 最も厳格な検知器の結果を使用
            primary_detector = 'ultra_conservative'
            if primary_detector in self.detectors:
                rate_key = f'{primary_detector}_adaptive_anomaly_rate'
                anomaly_rate = pool_results.get(rate_key, 0.0)
            else:
                anomaly_rate = 0.0
        elif detection_method == 'ensemble':
            pool_results = self.ensemble_anomaly_detection(all_cells, consensus_threshold=0.8)
            anomaly_rate = pool_results.get('consensus_80_anomaly_rate', 0.0)
        else:  # standard
            pool_results = self.compute_anomaly_scores(all_cells)
            anomaly_rate = pool_results.get('conservative_anomaly_rate', 0.0)
        
        return {
            'total_cells': len(all_cells),
            'colony_info': colony_info,
            'anomaly_rate': anomaly_rate,
            'all_scores': pool_results,
            'status': 'success'
        }
    
    def hierarchical_screening(self, colony_paths, pool_size=24, 
                             detection_method='adaptive',
                             threshold=0.20):  # 20%閾値
        """階層的スクリーニング実行"""
        
        print(f"=== HIERARCHICAL POOLED SCREENING ===")
        print(f"Total colonies: {len(colony_paths)}")
        print(f"Pool size: {pool_size}")
        print(f"Detection method: {detection_method}")
        print(f"Threshold: {threshold*100:.1f}%")
        
        # Step 1: プール作成
        pools = self.create_pools(colony_paths, pool_size)
        print(f"Created {len(pools)} pools")
        
        pool_results = []
        high_anomaly_pools = []
        
        # Step 2: プールスクリーニング
        print(f"\n--- Pool Screening Phase ---")
        for i, pool in enumerate(pools):
            print(f"Screening {pool['pool_id']} ({len(pool['colonies'])} colonies)...")
            
            pool_result = self.screen_pool(pool['colonies'], detection_method)
            
            if pool_result['status'] == 'no_cells':
                print(f"  No cells extracted - skipping")
                continue
                
            anomaly_rate = pool_result['anomaly_rate']
            
            pool_summary = {
                'pool_id': pool['pool_id'],
                'anomaly_rate': anomaly_rate,
                'cell_count': pool_result['total_cells'],
                'colonies': pool['colonies'],
                'colony_names': pool['colony_names'],
                'detailed_result': pool_result
            }
            
            pool_results.append(pool_summary)
            
            # 高異常率プールを特定
            if anomaly_rate > threshold:
                high_anomaly_pools.append(pool_summary)
                print(f"  🚨 HIGH ANOMALY: {anomaly_rate*100:.2f}%")
            else:
                print(f"  ✅ Normal: {anomaly_rate*100:.2f}%")
        
        # Step 3: 高異常率プールの個別解析
        individual_results = []
        
        if high_anomaly_pools:
            print(f"\n--- Individual Analysis Phase ---")
            print(f"Analyzing {len(high_anomaly_pools)} high-anomaly pools individually...")
            
            for pool_summary in high_anomaly_pools:
                print(f"\nDetailed analysis of {pool_summary['pool_id']}...")
                
                for colony_path in pool_summary['colonies']:
                    colony_name = os.path.basename(colony_path)
                    cells, stats = self.extract_quality_cells(colony_path)
                    
                    if len(cells) > 0:
                        # 包括的解析
                        standard_scores = self.compute_anomaly_scores(cells)
                        adaptive_scores = self.adaptive_anomaly_detection(cells, target_sensitivity=0.90)
                        ensemble_scores = self.ensemble_anomaly_detection(cells, consensus_threshold=0.8)
                        
                        # 結果統合
                        individual_result = {
                            'colony': colony_name,
                            'colony_path': colony_path,
                            'pool_id': pool_summary['pool_id'],
                            'cell_count': len(cells)
                        }
                        
                        # 各検知方法の結果を追加
                        # 標準検知
                        for detector_name in self.detectors.keys():
                            rate_key = f'{detector_name}_anomaly_rate'
                            if rate_key in standard_scores:
                                individual_result[f'std_{rate_key}'] = standard_scores[rate_key]
                        
                        # 適応的検知
                        for detector_name in self.detectors.keys():
                            rate_key = f'{detector_name}_adaptive_anomaly_rate'
                            if rate_key in adaptive_scores:
                                individual_result[f'adapt_{rate_key}'] = adaptive_scores[rate_key]
                        
                        # アンサンブル検知
                        if 'consensus_80_anomaly_rate' in ensemble_scores:
                            individual_result['ensemble_80_anomaly_rate'] = ensemble_scores['consensus_80_anomaly_rate']
                        if 'recommended_anomaly_rate' in ensemble_scores:
                            individual_result['ensemble_50_anomaly_rate'] = ensemble_scores['recommended_anomaly_rate']
                        
                        individual_results.append(individual_result)
                        
                        # 主要結果の表示
                        primary_rate = adaptive_scores.get('ultra_conservative_adaptive_anomaly_rate', 0)
                        ensemble_rate = ensemble_scores.get('consensus_80_anomaly_rate', 0)
                        
                        if primary_rate > 0.8 or ensemble_rate > 0.6:  # 80%または60%閾値
                            print(f"  🔥 {colony_name}: Adapt={primary_rate*100:.1f}%, Ens80={ensemble_rate*100:.1f}% (VERY HIGH)")
                        elif primary_rate > 0.5 or ensemble_rate > 0.3:
                            print(f"  ⚠️  {colony_name}: Adapt={primary_rate*100:.1f}%, Ens80={ensemble_rate*100:.1f}% (HIGH)")
                        elif primary_rate > 0.2 or ensemble_rate > 0.1:
                            print(f"  📋 {colony_name}: Adapt={primary_rate*100:.1f}%, Ens80={ensemble_rate*100:.1f}% (Moderate)")
                        else:
                            print(f"  ✅ {colony_name}: Adapt={primary_rate*100:.1f}%, Ens80={ensemble_rate*100:.1f}% (Normal)")
                    else:
                        print(f"  ❌ {colony_name}: No cells extracted")
        else:
            print(f"\n✅ No high-anomaly pools detected!")
        
        # 結果サマリー
        self.save_pooled_results(pool_results, individual_results, detection_method, threshold)
        
        return pool_results, individual_results
    
    def save_pooled_results(self, pool_results, individual_results, detection_method, threshold):
        """プール解析結果の保存"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        output_dir = f"/Users/matsuokoujirou/Documents/Data/Screening/Results/{timestamp}_pooled_screening"
        os.makedirs(output_dir, exist_ok=True)
        
        # プール結果の保存
        pool_df = pd.DataFrame([{
            'pool_id': r['pool_id'],
            'anomaly_rate': r['anomaly_rate'],
            'cell_count': r['cell_count'],
            'colony_count': len(r['colonies']),
            'status': 'HIGH' if r['anomaly_rate'] > threshold else 'NORMAL'
        } for r in pool_results])
        
        pool_df.to_csv(os.path.join(output_dir, 'pool_screening_results.csv'), index=False)
        
        # 個別結果の保存
        if individual_results:
            individual_df = pd.DataFrame(individual_results)
            individual_df.to_csv(os.path.join(output_dir, 'individual_analysis_results.csv'), index=False)
            
            # 超高異常率コロニーの特定
            if 'adapt_ultra_conservative_adaptive_anomaly_rate' in individual_df.columns:
                super_high_colonies = individual_df[
                    individual_df['adapt_ultra_conservative_adaptive_anomaly_rate'] > 0.8
                ].sort_values('adapt_ultra_conservative_adaptive_anomaly_rate', ascending=False)
                
                super_high_colonies.to_csv(
                    os.path.join(output_dir, 'super_high_anomaly_colonies.csv'), index=False
                )
        
        # サマリーレポート
        with open(os.path.join(output_dir, 'pooled_screening_report.txt'), 'w') as f:
            f.write("=== POOLED MUTANT SCREENING REPORT ===\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Detection method: {detection_method}\n")
            f.write(f"Threshold: {threshold*100:.1f}%\n\n")
            
            f.write("POOL SCREENING SUMMARY:\n")
            f.write(f"Total pools: {len(pool_results)}\n")
            high_pools = [r for r in pool_results if r['anomaly_rate'] > threshold]
            f.write(f"High-anomaly pools: {len(high_pools)}\n")
            f.write(f"Normal pools: {len(pool_results) - len(high_pools)}\n\n")
            
            if high_pools:
                f.write("HIGH-ANOMALY POOLS:\n")
                for pool in sorted(high_pools, key=lambda x: x['anomaly_rate'], reverse=True):
                    f.write(f"- {pool['pool_id']}: {pool['anomaly_rate']*100:.2f}% "
                           f"({pool['cell_count']} cells, {len(pool['colonies'])} colonies)\n")
            
            if individual_results:
                f.write(f"\nINDIVIDUAL ANALYSIS:\n")
                f.write(f"Colonies analyzed: {len(individual_results)}\n")
                
                individual_df = pd.DataFrame(individual_results)
                
                # 超高異常率
                if 'adapt_ultra_conservative_adaptive_anomaly_rate' in individual_df.columns:
                    super_high = individual_df[individual_df['adapt_ultra_conservative_adaptive_anomaly_rate'] > 0.8]
                    f.write(f"Super-high anomaly (>80%): {len(super_high)}\n")
                    
                    for _, row in super_high.iterrows():
                        rate = row['adapt_ultra_conservative_adaptive_anomaly_rate']
                        f.write(f"  - {row['colony']}: {rate*100:.1f}%\n")
        
        print(f"\nResults saved to: {output_dir}")

def run_pooled_screening_example():
    """プール解析の実行例"""
    
    # モデルディレクトリ
    model_dir = "/Users/matsuokoujirou/Documents/Data/Screening/Models/20250613_1524_improved"
    
    # コロニーファイルのパス（例：240個のコロニー）
    colony_base_path = "/path/to/colonies"
    colony_paths = sorted(glob(os.path.join(colony_base_path, "*.tif")))
    
    if len(colony_paths) == 0:
        print("No colony files found!")
        print("Please update colony_base_path in the function")
        return
    
    print(f"Found {len(colony_paths)} colony files")
    
    # プールスクリーニング実行
    screener = PooledMutantScreening(model_dir)
    
    pool_results, individual_results = screener.hierarchical_screening(
        colony_paths,
        pool_size=24,                    # 24コロニー/プール
        detection_method='adaptive',     # 適応的検知使用
        threshold=0.20                   # 20%閾値
    )
    
    # 結果サマリー表示
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Pools screened: {len(pool_results)}")
    
    high_pools = [r for r in pool_results if r['anomaly_rate'] > 0.20]
    print(f"High-anomaly pools: {len(high_pools)}")
    
    if individual_results:
        print(f"Individual colonies analyzed: {len(individual_results)}")
        
        # 最高異常率のコロニー
        individual_df = pd.DataFrame(individual_results)
        if 'adapt_ultra_conservative_adaptive_anomaly_rate' in individual_df.columns:
            max_idx = individual_df['adapt_ultra_conservative_adaptive_anomaly_rate'].idxmax()
            top_colony = individual_df.loc[max_idx]
            rate = top_colony['adapt_ultra_conservative_adaptive_anomaly_rate']
            print(f"Highest anomaly colony: {top_colony['colony']} ({rate*100:.1f}%)")

if __name__ == "__main__":
    # 通常の包括的スクリーニング
    main()
    
    # 適応的スクリーニングの例
    # run_adaptive_screening_example()
    
    # プール解析の例
    # run_pooled_screening_example()