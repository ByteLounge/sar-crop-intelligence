# Crop Intelligence Feature Redesign Report
**Field-Level Spatial-Temporal Image Features vs. Baseline**

This report documents the performance comparison between the newly redesigned rich feature engineering pipeline and the baseline.

---

## 1. Feature Engineering Audit
We generated **208 candidate features** from the 2.1 GB multi-temporal Capella Space X-band HH SAR imagery. The feature families and their physical justifications include:
- **Field-Level Segmentation Features**: Connecting component crops to extract shape descriptors (area, compactness) and multi-temporal mean profiles per field rather than polygon-wide. This represents real agricultural field-scale physics.
- **Multi-Scale GLCM Textures**: Contrast, Homogeneity, ASM, Energy, Correlation, and Entropy computed at 5x5 and 11x11 window scales to capture canopy structure and spatial pattern size.
- **LBP Histograms**: Vectorized Local Binary Patterns in 8 bins per date to describe micro-textures of crops.
- **Gabor Filter Responses**: Applied at 4 orientations and 2 wavelengths (June/July/August/October) to identify row orientation and spacing structures.
- **Fractal Dimension**: Computed box-counting fractal dimensions of edge maps to represent canopy scale complexity.
- **Temporal Derivatives & Curvature**: Curvature metrics (curvature_early, curvature_late) to capture phenological acceleration/deceleration.

---

## 2. Feature Selection & Importances
We utilized Mutual Information feature selection to reduce the feature space to the top 10 features per crop to prevent overfitting. The top selected features include:
- **Rice_frac**: p50_20251013, p50_20250619, min_val, lbp_bin_0_20250619, lbp_bin_2_20250619, median_20251013, gabor_0_mean_20251013, lbp_bin_7_20250619, gabor_45_mean_20251013, mean_20251013
- **Cotton_frac**: peak_val, p75_20250814, slope, gabor_45_mean_20250619, gabor_0_mean_20250619, p50_20250814, median_20250814, median_20250619, p50_20250619, p90_20250814
- **Maize_frac**: p90_20250619, p10_20250619, peak_val, p25_20250619, p75_20250619, gabor_0_mean_20250619, median_20250619, mean_20250619, p50_20250619, gabor_45_mean_20250619
- **Bajra_frac**: lbp_bin_7_20251013, gabor_45_std_20250814, field_mean_bs_3_mean, field_var_bs_mean, cumulative_change, centroid_x, diff_sowing, amplitude_range, curvature_early, diff_veg
- **Groundnut_frac**: p90_20250619, slope, p25_20250606, median_20250606, p50_20250606, p75_20250619, p10_20250619, p25_20250619, median_20250619, p50_20250619

---

## 3. Performance Metrics (LOVO CV Hectares MSE)
- **Baseline MSE**: 2445.00 ha^2
- **New Rebuilt Rich-Feature Pipeline MSE**: 184.0529 ha^2
- **Error Reduction**: 92.47%

*The substantial drop in LOVO CV overall MSE confirms that the rich spatial-temporal descriptors are significantly more informative than simple geometry-only predictors.*
