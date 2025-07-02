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

Please report questions or issues via Issues or Pull Requests. 