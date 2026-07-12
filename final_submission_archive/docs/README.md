# ANRF AISEHack 2.0 SAR Crop Mapping Challenge Solution

This repository contains the winning production-grade solution for Round 1 of the ANRF AISEHack 2.0 SAR Crop Mapping Challenge.

## Project Overview
The objective of this challenge is to estimate the acreage (in hectares) of five key crops (**Rice, Cotton, Maize, Bajra, and Groundnut**) across 29 villages near Vadodara, Gujarat, India, using multi-temporal Capella SAR imagery. 

A major geospatial constraint is that **12 out of the 29 test villages lie completely outside the satellite swath** (0% image coverage). To solve this, this pipeline utilizes a hybrid spatial-temporal architecture that blends direct pixel-level temporal signatures (for covered villages) with stacked crop-specific machine learning estimators trained on spatial-geometric features (for zero-coverage villages).

---

## Repository Structure
```
final_submission_archive/
├── REPRODUCIBILITY.md     # Step-by-step reproduction instructions and environment specs
├── RELEASE_NOTES.md       # Version information and architecture summary
├── code/                  # Source code directories
│   ├── preprocessing/     # Raster alignment and reprojection
│   ├── features/          # Geometry and SAR feature extraction
│   ├── models/            # Stacking/blending ensemble definitions
│   ├── training/          # Training pipeline and serialization
│   └── inference/         # Prediction and checkpoint loader
├── models/                # Serialized models, imputers, and feature lists (.pkl)
├── data/                  # Dataset manifest file (raw images are not duplicated)
├── outputs/               # Final submission and validation results
├── configs/               # Hyperparameter and settings YAML
├── environment/           # requirements.txt and pip freeze environment specs
├── logs/                  # Console training and execution logs
├── checksums/             # SHA256SUMS.txt and file_manifest.csv
└── docs/                  # In-depth technical documentation
    ├── README.md
    ├── PIPELINE.md
    ├── FEATURE_ENGINEERING.md
    ├── MODEL_SELECTION.md
    ├── VALIDATION.md
    ├── IMPUTATION.md
    └── FINAL_REPORT.md
```

---

## Installation & Environment Setup
1. **Python Version**: Python 3.12.x is recommended.
2. **Install Dependencies**:
   ```bash
   pip install -r environment/requirements.txt
   ```

### Required Packages:
- `numpy>=2.0.0`
- `pandas>=2.0.0`
- `geopandas>=1.0.0`
- `rasterio>=1.3.0`
- `shapely>=2.0.0`
- `pyproj>=3.0.0`
- `opencv-python>=4.8.0`
- `scipy>=1.11.0`
- `scikit-learn>=1.3.0`
- `xgboost>=2.0.0`
- `lightgbm>=4.0.0`
- `catboost>=1.2.0`

---

## Training & Inference Workflows

### 1. Training and Serialization
To execute the preprocessing, extract features, fit the ensembles, and serialize the checkpoints:
```bash
python code/training/train.py
```
This script performs:
- Alignment of raw TIFFs to a 10m grid.
- Spatial-temporal feature extraction.
- Training of crop-specific ensembles (Random Forest, Extra Trees, CatBoost).
- Checkpoint serialization to `models/`.
- Blending and generation of `outputs/submission.csv`.

### 2. Inference from Checkpoints
To run inference from the pre-trained checkpoints (without retraining):
```bash
python code/inference/predict.py
```
This loads the serialized imputers and models, processes the imagery, and outputs `outputs/submission_regenerated.csv`.

---

## Troubleshooting
- **Permission Denied (`cv2` error)**: Ensure `opencv-python-headless` is installed if running in headless server environments.
- **GDAL/Rasterio Issues on Windows**: Install pre-compiled wheel files for `rasterio` and `fiona` from Christoph Gohlke's archive if standard `pip install` fails.
