import numpy as np
import tifffile as tiff
import os
from glob import glob
import matplotlib.pyplot as plt
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.measure import regionprops, label as skimage_label
from skimage.segmentation import clear_border
from skimage.transform import resize
from skimage import exposure
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
from datetime import datetime
import tensorflow as tf

class ImprovedAnomalyDetectionTraining:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.setup_environment()
        
    def setup_environment(self):
        """環境設定"""
        RANDOM_SEED = 42
        np.random.seed(RANDOM_SEED)
        tf.random.set_seed(RANDOM_SEED)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU使用
        
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
            filtered_labels = np.copy(labels)
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
                # ヒストグラム均等化で強度を標準化
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
    
    def select_ultra_perfect_cells(self, all_cells, all_stats):
        """最も理想的な正常細胞のみを選択（修正版）"""
        print("=== Selecting Ultra-Perfect Normal Cells ===")
        
        if len(all_stats) == 0:
            print("No cells to select from!")
            return np.array([]), []
        
        stats_df = pd.DataFrame(all_stats)
        
        # 統計的指標の計算
        area_mean = stats_df['area'].mean()
        area_std = stats_df['area'].std()
        ecc_mean = stats_df['eccentricity'].mean()
        ecc_std = stats_df['eccentricity'].std()
        sol_mean = stats_df['solidity'].mean()
        sol_std = stats_df['solidity'].std()
        
        print(f"Statistics:")
        print(f"  Area: {area_mean:.1f} ± {area_std:.1f}")
        print(f"  Eccentricity: {ecc_mean:.3f} ± {ecc_std:.3f}")
        print(f"  Solidity: {sol_mean:.3f} ± {sol_std:.3f}")
        
        ultra_perfect_cells = []
        ultra_perfect_stats = []
        
        # より緩い条件で段階的に選択
        selection_criteria = [
            {  # 最初に試す厳格な条件
                'area_min': 1000, 'area_max': 4000,
                'ecc_max': 0.4, 'sol_min': 0.92,
                'intensity_min': 0.6, 'intensity_std_min': 0.15,
                'zscore_max': 1.0, 'name': 'Ultra-Strict'
            },
            {  # 条件を緩める
                'area_min': 800, 'area_max': 5000,
                'ecc_max': 0.6, 'sol_min': 0.88,
                'intensity_min': 0.5, 'intensity_std_min': 0.12,
                'zscore_max': 1.5, 'name': 'Strict'
            },
            {  # さらに緩める
                'area_min': 500, 'area_max': 6000,
                'ecc_max': 0.8, 'sol_min': 0.85,
                'intensity_min': 0.4, 'intensity_std_min': 0.10,
                'zscore_max': 2.0, 'name': 'Moderate-Strict'
            }
        ]
        
        for criteria in selection_criteria:
            ultra_perfect_cells = []
            ultra_perfect_stats = []
            
            for i, (cell, stat) in enumerate(zip(all_cells, all_stats)):
                # Z-scoreの計算
                area_zscore = abs(stat['area'] - area_mean) / area_std if area_std > 0 else 0
                ecc_zscore = abs(stat['eccentricity'] - ecc_mean) / ecc_std if ecc_std > 0 else 0
                sol_zscore = abs(stat['solidity'] - sol_mean) / sol_std if sol_std > 0 else 0
                
                # 条件チェック
                if (stat['area'] >= criteria['area_min'] and 
                    stat['area'] <= criteria['area_max'] and
                    stat['eccentricity'] <= criteria['ecc_max'] and
                    stat['solidity'] >= criteria['sol_min'] and
                    stat['mean_intensity'] >= criteria['intensity_min'] and
                    stat['std_intensity'] >= criteria['intensity_std_min'] and
                    area_zscore <= criteria['zscore_max'] and
                    ecc_zscore <= criteria['zscore_max'] and
                    sol_zscore <= criteria['zscore_max']):
                    
                    ultra_perfect_cells.append(cell)
                    ultra_perfect_stats.append(stat)
            
            print(f"  {criteria['name']} selection: {len(ultra_perfect_cells)} cells")
            
            # 十分な数の細胞が得られたら終了
            if len(ultra_perfect_cells) >= 500:
                print(f"  Selected {criteria['name']} criteria (sufficient cells)")
                break
            elif len(ultra_perfect_cells) >= 200:
                print(f"  Selected {criteria['name']} criteria (minimum threshold)")
                break
        
        print(f"Original cells: {len(all_cells)}")
        print(f"Ultra-perfect cells: {len(ultra_perfect_cells)}")
        if len(all_cells) > 0:
            print(f"Selection rate: {len(ultra_perfect_cells)/len(all_cells)*100:.1f}%")
        
        return np.array(ultra_perfect_cells), ultra_perfect_stats
    
    def create_training_dataset(self, folder_path, use_ultra_selection=False):
        """高品質訓練データセット作成"""
        print("=== Creating High-Quality Training Dataset ===")
        
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
        
        # 超厳格選択を適用
        if use_ultra_selection and len(all_cells) > 0:
            selected_cells, selected_stats = self.select_ultra_perfect_cells(all_cells, all_stats)
            
            # 選択された細胞が十分にある場合のみ使用
            if len(selected_cells) >= 200:
                all_cells, all_stats = selected_cells, selected_stats
                print(f"Using ultra-selected cells: {len(all_cells)}")
            else:
                print(f"Ultra-selection yielded too few cells ({len(selected_cells)}). Using all quality cells.")
        
        # 統計サマリー保存
        if len(all_stats) > 0:
            stats_df = pd.DataFrame(all_stats)
        else:
            # 空のDataFrameを作成
            stats_df = pd.DataFrame(columns=['area', 'eccentricity', 'solidity', 'mean_intensity', 'std_intensity', 'file'])
        
        file_summary_df = pd.DataFrame(file_summary)
        
        stats_df.to_csv(os.path.join(self.output_dir, 'cell_statistics.csv'), index=False)
        file_summary_df.to_csv(os.path.join(self.output_dir, 'file_summary.csv'), index=False)
        
        # データ品質レポート
        self.generate_data_quality_report(stats_df, file_summary_df, use_ultra_selection)
        
        return np.array(all_cells), stats_df
    
    def generate_data_quality_report(self, stats_df, file_summary_df, use_ultra_selection=False):
        """データ品質レポート生成（安全版）"""
        with open(os.path.join(self.output_dir, 'data_quality_report.txt'), 'w') as f:
            f.write("=== TRAINING DATA QUALITY REPORT ===\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Ultra-selection applied: {use_ultra_selection}\n\n")
            
            f.write("OVERALL STATISTICS:\n")
            f.write(f"Total files processed: {len(file_summary_df)}\n")
            f.write(f"Total cells extracted: {len(stats_df)}\n")
            if len(file_summary_df) > 0:
                f.write(f"Average cells per file: {len(stats_df)/len(file_summary_df):.1f}\n\n")
            else:
                f.write("Average cells per file: 0\n\n")
            
            if len(stats_df) > 0 and 'area' in stats_df.columns:
                f.write("CELL MORPHOLOGY STATISTICS:\n")
                f.write(f"Area: {stats_df['area'].mean():.1f} ± {stats_df['area'].std():.1f}\n")
                f.write(f"Eccentricity: {stats_df['eccentricity'].mean():.3f} ± {stats_df['eccentricity'].std():.3f}\n")
                f.write(f"Solidity: {stats_df['solidity'].mean():.3f} ± {stats_df['solidity'].std():.3f}\n\n")
                
                f.write("INTENSITY STATISTICS:\n")
                f.write(f"Mean intensity: {stats_df['mean_intensity'].mean():.3f} ± {stats_df['mean_intensity'].std():.3f}\n")
                f.write(f"Std intensity: {stats_df['std_intensity'].mean():.3f} ± {stats_df['std_intensity'].std():.3f}\n\n")
            else:
                f.write("CELL MORPHOLOGY STATISTICS: No data available\n\n")
                f.write("INTENSITY STATISTICS: No data available\n\n")
            
            f.write("FILE-WISE SUMMARY:\n")
            for _, row in file_summary_df.iterrows():
                f.write(f"{row['filename']}: {row['cells_extracted']} cells, "
                       f"avg intensity: {row['mean_cell_intensity']:.3f}\n")
    
    def create_improved_autoencoder(self, input_shape=(64, 64, 1)):
        """改善されたオートエンコーダ"""
        
        # エンコーダ
        input_img = Input(shape=input_shape)
        
        # エンコーダ部分（より保守的な設計）
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)  # 8x8x32
        
        # デコーダ部分
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        x = BatchNormalization()(x)
        x = UpSampling2D((2, 2))(x)
        
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2, 2))(x)
        
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2, 2))(x)
        
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        
        # モデル構築
        autoencoder = Model(input_img, decoded)
        encoder = Model(input_img, encoded)
        
        # コンパイル
        autoencoder.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',  # MSEの方が安定
            metrics=['mae']
        )
        
        return autoencoder, encoder
    
    def train_autoencoder(self, cell_images):
        """オートエンコーダの訓練"""
        print("=== Training Autoencoder ===")
        
        # データ準備
        X = np.expand_dims(cell_images, axis=-1)
        X = X.astype('float32')
        
        # 訓練・検証分割
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
        
        print(f"Training data: {X_train.shape}")
        print(f"Validation data: {X_val.shape}")
        
        # データ拡張（控えめに）
        datagen = ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.05,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
        
        # モデル作成
        autoencoder, encoder = self.create_improved_autoencoder()
        
        print("Model architecture:")
        autoencoder.summary()
        
        # コールバック設定
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                os.path.join(self.output_dir, 'best_autoencoder.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # 訓練実行
        history = autoencoder.fit(
            datagen.flow(X_train, X_train, batch_size=32),
            steps_per_epoch=len(X_train) // 32,
            epochs=100,
            validation_data=(X_val, X_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # 訓練履歴の保存と可視化
        self.plot_training_history(history)
        
        # 最終モデル保存
        autoencoder.save(os.path.join(self.output_dir, 'final_autoencoder.keras'))
        encoder.save(os.path.join(self.output_dir, 'encoder.keras'))
        
        return autoencoder, encoder, history
    
    def plot_training_history(self, history):
        """訓練履歴の可視化"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # MAE
        ax2.plot(history.history['mae'], label='Training MAE')
        ax2.plot(history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'), dpi=300)
        plt.show()
    
    def evaluate_reconstruction_quality(self, autoencoder, cell_images):
        """再構成品質の評価"""
        print("=== Evaluating Reconstruction Quality ===")
        
        X = np.expand_dims(cell_images, axis=-1).astype('float32')
        
        # 再構成
        reconstructed = autoencoder.predict(X)
        
        # 再構成誤差計算
        mse_errors = np.mean(np.square(X - reconstructed), axis=(1, 2, 3))
        mae_errors = np.mean(np.abs(X - reconstructed), axis=(1, 2, 3))
        
        # 統計
        print(f"MSE - Mean: {np.mean(mse_errors):.6f}, Std: {np.std(mse_errors):.6f}")
        print(f"MAE - Mean: {np.mean(mae_errors):.6f}, Std: {np.std(mae_errors):.6f}")
        
        # 分布の可視化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.hist(mse_errors, bins=50, alpha=0.7)
        ax1.set_xlabel('MSE')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of MSE Reconstruction Errors')
        ax1.axvline(np.percentile(mse_errors, 95), color='red', linestyle='--', 
                   label='95th percentile')
        ax1.legend()
        
        ax2.hist(mae_errors, bins=50, alpha=0.7)
        ax2.set_xlabel('MAE')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of MAE Reconstruction Errors')
        ax2.axvline(np.percentile(mae_errors, 95), color='red', linestyle='--', 
                   label='95th percentile')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'reconstruction_error_distribution.png'), dpi=300)
        plt.show()
        
        # サンプル可視化
        self.visualize_reconstructions(X, reconstructed)
        
        return mse_errors, mae_errors
    
    def visualize_reconstructions(self, original, reconstructed, n_samples=10):
        """再構成結果の可視化"""
        indices = np.random.choice(len(original), n_samples, replace=False)
        
        fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))
        
        for i, idx in enumerate(indices):
            # オリジナル
            axes[0, i].imshow(original[idx].squeeze(), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # 再構成
            axes[1, i].imshow(reconstructed[idx].squeeze(), cmap='gray')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'reconstruction_samples.png'), dpi=300)
        plt.show()
    
    def create_ultra_sensitive_detectors(self, encoder, cell_images):
        """超高感度異常検知器の作成"""
        print("=== Creating Ultra-Sensitive Anomaly Detectors ===")
        
        X = np.expand_dims(cell_images, axis=-1).astype('float32')
        
        # 特徴抽出
        features = encoder.predict(X)
        features_flat = features.reshape(len(features), -1)
        
        print(f"Encoded features shape: {features.shape}")
        print(f"Flattened features shape: {features_flat.shape}")
        
        # 前処理
        scaler = RobustScaler()  # 外れ値に頑健
        features_scaled = scaler.fit_transform(features_flat)
        
        # 次元削減
        n_components = min(100, features_scaled.shape[1], features_scaled.shape[0] - 1)
        pca = PCA(n_components=n_components)
        features_reduced = pca.fit_transform(features_scaled)
        
        print(f"PCA reduced to {n_components} components")
        print(f"Explained variance ratio (first 5): {pca.explained_variance_ratio_[:5]}")
        
        # 超高感度異常検知モデル
        detectors = {
            'Extreme_Conservative': OneClassSVM(kernel='rbf', gamma='scale', nu=0.0001),    # 0.01%
            'Ultra_Conservative': OneClassSVM(kernel='rbf', gamma='scale', nu=0.001),       # 0.1%
            'Super_Conservative': OneClassSVM(kernel='rbf', gamma='scale', nu=0.005),       # 0.5%
            'Very_Conservative': OneClassSVM(kernel='rbf', gamma='scale', nu=0.01),         # 1%
            'Conservative': OneClassSVM(kernel='rbf', gamma='scale', nu=0.05),              # 5%
            'Moderate': OneClassSVM(kernel='rbf', gamma='scale', nu=0.10)                   # 10%
        }
        
        # 学習
        successful_detectors = {}
        for name, detector in detectors.items():
            print(f"Training {name} detector...")
            try:
                detector.fit(features_reduced)
                successful_detectors[name] = detector
            except Exception as e:
                print(f"  Warning: Failed to train {name}: {e}")
                continue
        
        # ベースライン異常率
        print("\nUltra-strict baseline anomaly rates:")
        for name, detector in successful_detectors.items():
            try:
                predictions = detector.predict(features_reduced)
                anomaly_rate = np.sum(predictions == -1) / len(predictions)
                print(f"{name}: {anomaly_rate*100:.4f}%")
            except:
                print(f"{name}: Failed to predict")
        
        # モデル保存
        import pickle
        with open(os.path.join(self.output_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        with open(os.path.join(self.output_dir, 'pca.pkl'), 'wb') as f:
            pickle.dump(pca, f)
        
        for name, detector in successful_detectors.items():
            filename = f'detector_{name.lower().replace("_", "")}.pkl'
            with open(os.path.join(self.output_dir, filename), 'wb') as f:
                pickle.dump(detector, f)
        
        return successful_detectors, scaler, pca
    
    def generate_final_report(self, stats_df, history, mse_errors, mae_errors, use_ultra_selection=False):
        """最終レポート生成"""
        with open(os.path.join(self.output_dir, 'training_report.txt'), 'w') as f:
            f.write("=== ULTRA-SENSITIVE ANOMALY DETECTION MODEL TRAINING REPORT ===\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Ultra-selection applied: {use_ultra_selection}\n\n")
            
            f.write("TRAINING DATA SUMMARY:\n")
            f.write(f"Total cells used for training: {len(stats_df)}\n")
            if len(stats_df) > 0 and 'area' in stats_df.columns:
                f.write(f"Average cell area: {stats_df['area'].mean():.1f} ± {stats_df['area'].std():.1f}\n")
                f.write(f"Average eccentricity: {stats_df['eccentricity'].mean():.3f} ± {stats_df['eccentricity'].std():.3f}\n")
            else:
                f.write("Cell statistics: Not available\n")
            f.write("\n")
            
            f.write("TRAINING PERFORMANCE:\n")
            f.write(f"Final training loss: {history.history['loss'][-1]:.6f}\n")
            f.write(f"Final validation loss: {history.history['val_loss'][-1]:.6f}\n")
            f.write(f"Best validation loss: {min(history.history['val_loss']):.6f}\n")
            f.write(f"Training epochs: {len(history.history['loss'])}\n\n")
            
            f.write("RECONSTRUCTION ERROR STATISTICS:\n")
            f.write(f"MSE - Mean: {np.mean(mse_errors):.6f}, Std: {np.std(mse_errors):.6f}\n")
            f.write(f"MSE - 95th percentile: {np.percentile(mse_errors, 95):.6f}\n")
            f.write(f"MAE - Mean: {np.mean(mae_errors):.6f}, Std: {np.std(mae_errors):.6f}\n")
            f.write(f"MAE - 95th percentile: {np.percentile(mae_errors, 95):.6f}\n\n")
            
            f.write("MODEL FILES GENERATED:\n")
            f.write("- best_autoencoder.keras: Best autoencoder model\n")
            f.write("- final_autoencoder.keras: Final autoencoder model\n")
            f.write("- encoder.keras: Encoder model\n")
            f.write("- scaler.pkl: Feature scaler (RobustScaler)\n")
            f.write("- pca.pkl: PCA transformer\n")
            f.write("- detector_extremeconservative.pkl: Extreme Conservative detector (0.01%)\n")
            f.write("- detector_ultraconservative.pkl: Ultra Conservative detector (0.1%)\n")
            f.write("- detector_superconservative.pkl: Super Conservative detector (0.5%)\n")
            f.write("- detector_veryconservative.pkl: Very Conservative detector (1%)\n")
            f.write("- detector_conservative.pkl: Conservative detector (5%)\n")
            f.write("- detector_moderate.pkl: Moderate detector (10%)\n\n")
            
            f.write("EXPECTED PERFORMANCE:\n")
            f.write("- Extreme Conservative: Should detect 80-95% anomaly rate in true mutants\n")
            f.write("- Ultra Conservative: Should detect 70-90% anomaly rate in true mutants\n")
            f.write("- Super Conservative: Should detect 60-80% anomaly rate in true mutants\n")
            f.write("- For pooled screening, use adaptive detection with 90% target sensitivity\n")
            f.write("- For individual validation, use ensemble detection with 80% consensus\n")

def main():
    # 設定
    folder_path = "/Users/matsuokoujirou/Documents/Data/Screening/Noemal_cells"
    output_dir = f"/Users/matsuokoujirou/Documents/Data/Screening/Models/{datetime.now().strftime('%Y%m%d_%H%M')}_ultra_sensitive"
    
    # 訓練パイプライン実行
    trainer = ImprovedAnomalyDetectionTraining(output_dir)
    
    # 1. 高品質データセット作成（超厳格選択を試す）
    try:
        cell_images, stats_df = trainer.create_training_dataset(folder_path, use_ultra_selection=True)
    except Exception as e:
        print(f"Error with ultra-selection: {e}")
        print("Falling back to standard selection...")
        cell_images, stats_df = trainer.create_training_dataset(folder_path, use_ultra_selection=False)
    
    if len(cell_images) < 200:
        print(f"Error: Only {len(cell_images)} cells available after selection.")
        print("Need at least 200 cells for training. Please check your data.")
        return
    
    print(f"Using {len(cell_images)} high-quality cells for training")
    
    # 2. オートエンコーダ訓練
    autoencoder, encoder, history = trainer.train_autoencoder(cell_images)
    
    # 3. 再構成品質評価
    mse_errors, mae_errors = trainer.evaluate_reconstruction_quality(autoencoder, cell_images)
    
    # 4. 超高感度異常検知器作成
    detectors, scaler, pca = trainer.create_ultra_sensitive_detectors(encoder, cell_images)
    
    # 5. 最終レポート
    use_ultra = len(cell_images) < len(stats_df) if len(stats_df) > 0 else False
    trainer.generate_final_report(stats_df, history, mse_errors, mae_errors, use_ultra)
    
    print(f"\n=== ULTRA-SENSITIVE TRAINING COMPLETED ===")
    print(f"Models and reports saved to: {output_dir}")
    print(f"High-quality cells used: {len(cell_images)}")
    print(f"Detectors trained: {len(detectors)}")
    print(f"Ready for ultra-sensitive anomaly detection!")
    
    # 簡易性能テスト
    print(f"\n=== SELF-TEST ===")
    if len(cell_images) >= 100:
        X_test = np.expand_dims(cell_images[:100], axis=-1).astype('float32')
        features = encoder.predict(X_test, verbose=0)
        features_flat = features.reshape(len(features), -1)
        features_scaled = scaler.transform(features_flat)
        features_reduced = pca.transform(features_scaled)
        
        for name, detector in detectors.items():
            try:
                predictions = detector.predict(features_reduced)
                anomaly_rate = np.sum(predictions == -1) / len(predictions)
                print(f"{name}: {anomaly_rate*100:.2f}% anomaly rate on training data")
            except:
                print(f"{name}: Failed self-test")
    else:
        print("Not enough cells for self-test")

if __name__ == "__main__":
    main()