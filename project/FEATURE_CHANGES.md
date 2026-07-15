# Feature Changes Report

This document details the feature engineering audit and the advanced SAR features introduced to improve model validation performance.

## Feature Count Summary

- **Old Feature Count**: 43 features (7 geometry features, 36 basic SAR stats)
- **New Feature Count**: 65 features (7 geometry features, 58 advanced SAR & texture features)

The advanced features include:
1. **Temporal Ratios**: Captures vegetative growth rates between sowing, peak vegetative growth, and harvest.
2. **Sobel Spatial Gradients**: Extracted using a Sobel filter on the aligned imagery to capture spatial edges and land cover fragmentation.
3. **Temporal Variance**: Village-wide backscatter variance across all four observation dates to capture crop lifecycle variability.
4. **IQR (Interquartile Range)**: Captures within-village backscatter spread per date.
5. **Shannon Entropy**: Measures the statistical complexity and diversity of backscatter values.
6. **Land Cover Fractions**: Calculated percentage of water, built-up, and vegetation pixels inside each village.

## Final Selected Features per Crop Target

Using Greedy Forward Feature Selection (FFS) evaluated under out-of-sample imputation conditions, we pruned the 65 candidate features down to the optimal subsets:

| Target Crop | Selected Feature Subset | Feature Rationale |
| :--- | :--- | :--- |
| **Rice_frac** | `['centroid_x', 'centroid_y', 'ratio_veg']` | Rice is highly concentrated in the central-western region. `ratio_veg` captures the large backscatter change during transplanting floods and peak growth. |
| **Cotton_frac** | `['centroid_y', 'centroid_x', 'diff_harvest', 'mean_20250606', 'ratio_harvest']` | Centroids capture geographic distribution; `diff_harvest` and `ratio_harvest` identify the post-monsoon cotton harvest signature. |
| **Maize_frac** | `['centroid_y', 'centroid_x', 'mean_sobel_20251013', 'mean_local_std_20250814', 'mean_local_std_20250606', 'mean_sobel_20250814']` | Maize is highly textured and fragmented. `mean_sobel` (spatial gradient magnitude) and `mean_local_std` (box-filtered std) capture this texture. |
| **Bajra_frac** | `['p25_20250619', 'centroid_x', 'centroid_y', 'temporal_variance', 'p75_20250619']` | Bajra is a short-duration crop. `temporal_variance` and June percentiles (`p25_20250619`, `p75_20250619`) capture sowing-to-harvest signatures. |
| **Groundnut_frac** | `['centroid_x', 'centroid_y', 'area_ha']` | Groundnut exhibits extremely strong spatial autocorrelation. Centroid coordinates and village area are sufficient to map the spatial distribution. |

## Feature Importance and Pruning Analysis

- **Pruning Redundancy**: Previous feature configurations relied on overlapping statistical summaries (`mean`, `p50`, `p25`, `p75` from the same date). Forward Feature Selection effectively stripped out these redundant columns, leaving only one or two representative backscatter statistics per date.
- **Geographic Centroids**: Centroids (`centroid_x`, `centroid_y`) emerged as the most critical features across all crops. Because Vadodara has distinct micro-climates and localized crop preferences, spatial coordinates act as a robust proxy for soil suitability, irrigation availability, and farming practices.
- **Texture over Backscatter**: For complex crops like Maize, spatial texture features (`mean_sobel` and `mean_local_std`) proved significantly more predictive than raw backscatter means, as Maize is planted in small, high-contrast plots.
