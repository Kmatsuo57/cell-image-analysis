# Cell Image Analysis and Anomaly Detection Pipeline

This repository contains Python scripts for cell image analysis, including cell extraction, FFT analysis, and autoencoder-based anomaly detection for different strains.

## Main Scripts
- `CAE_improved_modeltrain.py`: Cell extraction, autoencoder training, and anomaly detection model creation
- `improved_detection.py`: Mutant screening using trained models with comprehensive anomaly detection

## Usage
1. Prepare necessary directories and image files
2. Modify path settings in the `main()` function of each script to match your environment
3. Run in terminal:
   ```bash
   python CAE_improved_modeltrain.py
   # or
   python improved_detection.py
   ```
4. Results, models, and reports will be saved to the output directory

## Dependencies
- numpy, pandas, matplotlib, tifffile, scikit-image, scikit-learn
- tensorflow, keras
- stardist, csbdeep

Install if needed:
```bash
pip install numpy pandas matplotlib tifffile scikit-image scikit-learn tensorflow keras stardist csbdeep
```

## Output Examples
- Analysis reports (.txt, .csv)
- Images (.png, .jpg)
- Trained models (.keras, .pkl)

## For Reproducibility
- Execution scripts are automatically saved to the output directory
- Analysis parameters and environment information are documented in reports

---

## CAE and improved_detection Script Usage and Integration

### 1. CAE Scripts (e.g., `CAE_improved_modeltrain.py`)
- Extract high-quality cells from normal cell images and create autoencoder (CAE) models for feature extraction and reconstruction error-based anomaly detection.
- Main workflow:
  1. Cell extraction and quality control using StarDist
  2. Resize to 64x64 and normalize
  3. CAE learning (minimize reconstruction error)
  4. Create anomaly detection models using encoder features with PCA and SVM
  5. Save models, scalers, PCA, etc. to output folder
- **Usage example:**
  ```bash
  python CAE_improved_modeltrain.py
  # Models and reports will be saved to output_dir
  ```

### 2. improved_detection Scripts (e.g., `improved_detection.py`, `improved_detection_v2.py`)
- Use trained CAE and anomaly detectors (SVM, etc.) to perform batch screening of multiple strains/samples.
- Calculate anomaly rates using various indicators such as reconstruction error, encoder features, SVM, and ensemble methods, and output anomaly rates and detailed scores for each sample.
- **Usage example:**
  1. Place CAE, SVM, etc. model files in `model_dir`
  2. Specify paths of samples to screen in `test_folders` in the script
  3. Execute:
     ```bash
     python improved_detection.py
     # or
     python improved_detection_v2.py
     ```
  4. Anomaly rate summaries and detailed scores will be saved as CSV, etc. in output_dir

### 3. Integration Points
- Models, scalers, PCA, SVM, etc. created by CAE scripts can be used directly in improved_detection scripts.
- Consistency between training and screening is maintained by aligning cell extraction and preprocessing conditions using StarDist.
- For reproducibility, execution scripts and parameters are automatically saved to output.

---

---

# 細胞画像解析・異常検知パイプライン

本リポジトリは、細胞画像の抽出・FFT解析・オートエンコーダによる異常検知など、細胞画像解析のためのPythonスクリプト群です。

## 主なスクリプト
- `CAE_improved_modeltrain.py`: 高品質細胞画像の抽出・オートエンコーダ訓練・異常検知モデル作成
- `improved_detection.py`: 訓練済みモデルを使った包括的異常検知による変異株スクリーニング

## 使い方
1. 必要なディレクトリ・画像ファイルを用意
2. スクリプト内の`main()`関数のパス設定を自分の環境に合わせて修正
3. ターミナルで
   ```bash
   python CAE_improved_modeltrain.py
   # または
   python improved_detection.py
   ```
4. 出力ディレクトリに解析結果・モデル・レポートが保存されます

## 依存パッケージ
- numpy, pandas, matplotlib, tifffile, scikit-image, scikit-learn
- tensorflow, keras
- stardist, csbdeep

必要に応じて以下でインストール：
```bash
pip install numpy pandas matplotlib tifffile scikit-image scikit-learn tensorflow keras stardist csbdeep
```

## 出力例
- 解析レポート（.txt, .csv）
- 画像（.png, .jpg）
- 学習済みモデル（.keras, .pkl）

## 再現性のために
- 実行時のスクリプトが出力先に自動保存されます
- 解析パラメータや環境情報もレポートに記載

---

## CAE系・improved_detection系スクリプトの使い分けと連携

### 1. CAE系（例: `CAE_improved_modeltrain.py`）
- 正常細胞画像から高品質な細胞を抽出し、オートエンコーダ（CAE）で特徴抽出・再構成誤差による異常検知モデルを作成します。
- 主な流れ：
  1. StarDistで細胞抽出・品質管理
  2. 64x64にリサイズ・正規化
  3. CAEで学習（再構成誤差最小化）
  4. エンコーダ特徴量をPCA・SVMで異常検知モデル化
  5. モデル・スケーラー・PCAなどをoutputフォルダに保存
- **使い方例：**
  ```bash
  python CAE_improved_modeltrain.py
  # output_dirにモデル・レポートが保存されます
  ```

### 2. improved_detection系（例: `improved_detection.py`, `improved_detection_v2.py`）
- 訓練済みCAEと異常検知器（SVM等）を使い、複数株・サンプルの細胞画像を一括スクリーニングします。
- 再構成誤差・エンコーダ特徴量・SVM・アンサンブルなど多様な指標で異常率を算出し、サンプルごとに異常率・詳細スコアを出力します。
- **使い方例：**
  1. `model_dir`にCAEやSVM等のモデルファイルを配置
  2. スクリプト内の`test_folders`にスクリーニングしたいサンプルのパスを指定
  3. 実行：
     ```bash
     python improved_detection.py
     # または
     python improved_detection_v2.py
     ```
  4. output_dirに異常率サマリーや詳細スコアがCSV等で保存されます

### 3. 連携のポイント
- CAE系で作成したモデル・スケーラー・PCA・SVM等をimproved_detection系でそのまま利用できます。
- StarDistによる細胞抽出・前処理条件を揃えることで、訓練・スクリーニングの一貫性が保たれます。
- 再現性のため、実行時のスクリプトやパラメータもoutputに自動保存されます。

---

Please report questions or issues via Issues or Pull Requests.

ご質問・不具合はIssueまたはPull Requestでご連絡ください。 