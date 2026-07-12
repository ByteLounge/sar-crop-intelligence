# Imputation Strategy Documentation

This document analyzes feature imputation for the 12 zero-coverage villages.

## Imputation Audit & Comparisons
We evaluated three imputation strategies under 40% missingness (simulating the hidden test set) across 20 iterations:
1. **KNN Imputer (k=6)**: Imputes from 6 nearest neighbors in feature space.
2. **Median Imputer**: Replaces NaNs with the overall training median.
3. **Spatial Nearest Neighbor (1-NN)**: Copies the SAR features of the closest covered village using centroid coordinates.

### Imputer Performance (LOVO MSE Under 40% Masking)

| Target Crop | KNN-6 Imputer | Median Imputer | Spatial Nearest Neighbor | Optimal Strategy |
| :--- | :---: | :---: | :---: | :---: |
| **Rice_frac** | **0.017996** | 0.023002 | 0.023600 | **KNN-6** |
| **Cotton_frac** | **0.003684** | 0.004008 | 0.004629 | **KNN-6** |
| **Maize_frac** | **0.002101** | 0.002367 | 0.002187 | **KNN-6** |
| **Bajra_frac** | 0.010636 | 0.009512 | **0.009472** | **Spatial-NN** |
| **Groundnut_frac**| 0.002451 | 0.002598 | **0.002277** | **Spatial-NN** |

---

## Adopted Hybrid Imputation Strategy
- **KNN-6** performs best for Rice, Cotton, and Maize by smoothing out local noise and capturing regional crop trends.
- **Spatial 1-NN** performs best for Bajra and Groundnut. Copying the exact profile of the nearest geographic neighbor preserves the sharp local crop contrast that KNN neighbor averaging smooths out.
- **Implementation**: `train.py` fits both imputers and maps the correct feature matrices per crop target.
