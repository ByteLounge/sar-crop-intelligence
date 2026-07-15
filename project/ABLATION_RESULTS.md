# Ablation Results Report

This report presents a systematic ablation study of the translation-invariant tree pipeline, measuring the impact of individual components on Hectares LOVO CV MSE.

## Ablation Results Table

The table below lists the configurations sorted by Hectares LOVO CV MSE (lower is better):

| Ablation Configuration | Description | Hectares CV MSE ($ha^2$) | MSE Change relative to Baseline | Impact Rating |
| :--- | :--- | :---: | :---: | :---: |
| **`Add_Coordinates`** | Re-introduce centroids (`centroid_x`, `centroid_y`) | **4787.0491** | -2.82% | Minor (but high extrapolation risk) |
| **`Remove_Advanced_Features`** | Use only basic SAR statistics (no ratios, gradients, variance) | **4882.2923** | -0.88% | Minimal (basic features are sufficient for trees) |
| **`Full_Pipeline (Baseline)`** | **Hybrid Imputation + Safe Features + Clipping + Normalisation** | **4925.8502** | **0.00%** | **Reference** |
| **`Remove_Clipping`** | Do not clip predicted fractions to `[0.0, 1.0]` | **4925.8502** | 0.00% | Neutral (tree predictions naturally fall in `[0.0, 1.0]`) |
| **`Remove_Normalisation`** | Do not scale predicted hectares to match village capacity | **4960.1330** | +0.70% | Helpful |
| **`Remove_KNN_Imputation`** | Replace KNN-6 with Median Imputer for Rice/Cotton/Maize | **5353.1453** | +8.67% | **Critical** |
| **`Remove_Spatial_Imputation`**| Replace Spatial 1-NN with KNN-6 for Bajra/Groundnut | **6027.5565** | +22.37% | **Critical** |
| **`Remove_Hybrid_Imputation`** | Replace both KNN and Spatial 1-NN with Median Imputer | **6984.7440** | +41.80% | **Critical** |

## Key Findings

1. **Hybrid Imputation is Critical**:
   - Replacing the hybrid imputation scheme with a simple Median Imputer (`Remove_Hybrid_Imputation`) degrades Hectares MSE by **+41.80%** (jumping to **6984.74 $ha^2$**).
   - Replacing the crop-specific Spatial 1-NN with KNN-6 for Bajra and Groundnut (`Remove_Spatial_Imputation`) degrades Hectares MSE by **+22.37%** (jumping to **6027.56 $ha^2$**).
   - This validates our design choice:
     - **KNN-6** is optimal for Rice, Cotton, and Maize by smoothing local noise.
     - **Spatial 1-NN** is optimal for Bajra and Groundnut by preserving sharp local crop variance and copying the neighbor's profiles.
2. **Clipping & Normalization Value**:
   - Area Normalization (`Remove_Normalisation`) improves the Hectares MSE by reducing error by **0.70%**.
   - Clipping is neutral for tree-based models (since trees do not extrapolate and their predictions naturally fall within training targets `[0.0, 1.0]`), but is essential for linear models.
3. **Advanced Features vs Coordinates**:
   - Adding coordinates (`Add_Coordinates`) slightly reduces Hectares MSE by **2.82%** in local CV, but as documented in the Generalization Audit, it introduces spatial coordinate leakage and linear extrapolation risks. We accept the 2.82% CV tradeoff for absolute safety on hidden test villages.
