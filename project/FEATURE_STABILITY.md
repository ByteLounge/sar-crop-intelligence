# Feature Stability Report

This report evaluates the stability of our translation-invariant feature importances across the 17 Leave-One-Village-Out (LOVO) cross-validation folds.

## Measured Feature Stability (Sorted by Stability Index)

The table below lists each feature, its mean Extra Trees feature importance, its standard deviation, and its **Stability Index** (Coefficient of Variation: $\text{Std} / \text{Mean}$, lower is more stable/robust):

| Crop Target | Feature | Mean Importance | Std Importance | Stability Index | Feature Rationale & Physics |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Rice_frac** | `p50_20250619` | 0.7815 | 0.0165 | **0.0211** | Mid-June backscatter capture Rice transplanting flood dip. |
| **Cotton_frac** | `diff_harvest` | 0.4378 | 0.0149 | **0.0341** | Harvest backscatter change captures Cotton canopy opening/harvest. |
| **Bajra_frac** | `p25_20250619` | 0.5372 | 0.0194 | **0.0362** | Sowing percentile captures Bajra fields preparation. |
| **Cotton_frac** | `ratio_harvest` | 0.3439 | 0.0156 | **0.0454** | Ratio of peak vegetative growth to harvest. |
| **Bajra_frac** | `p75_20250619` | 0.3173 | 0.0162 | **0.0511** | Sowing percentile captures field texture. |
| **Groundnut_frac** | `mean_local_std_20250606` | 0.3205 | 0.0215 | **0.0672** | Early June local standard deviation captures soil roughness. |
| **Groundnut_frac** | `mean_local_std_20250814` | 0.3273 | 0.0252 | **0.0768** | Peak-veg local standard deviation captures canopy structure. |
| **Maize_frac** | `mean_local_std_20250814` | 0.2338 | 0.0210 | **0.0899** | Peak-veg local standard deviation captures Maize canopy height. |
| **Groundnut_frac** | `change_magnitude` | 0.2779 | 0.0262 | **0.0943** | Absolute backscatter difference between sowing and harvest. |
| **Rice_frac** | `ratio_veg` | 0.1735 | 0.0167 | **0.0965** | Ratio of transplanting flood to peak vegetative growth. |
| **Cotton_frac** | `mean_20250606` | 0.1749 | 0.0178 | **0.1018** | Sowing date mean backscatter. |
| **Maize_frac** | `mean_local_std_20250606` | 0.2192 | 0.0237 | **0.1081** | Sowing date local standard deviation. |
| **Maize_frac** | `mean_sobel_20250814` | 0.2294 | 0.0277 | **0.1208** | Spatial gradient magnitude capturing field texture. |
| **Maize_frac** | `mean_sobel_20251013` | 0.2684 | 0.0349 | **0.1302** | Spatial gradient magnitude capturing harvesting boundaries. |
| **Rice_frac** | `bbox_width` | 0.0173 | 0.0031 | **0.1789** | Geometry bounding box width (Rice is grown in wider basins). |
| **Rice_frac** | `area_ha` | 0.0277 | 0.0051 | **0.1849** | Village physical area. |
| **Groundnut_frac** | `area_ha` | 0.0333 | 0.0062 | **0.1877** | Village physical area. |
| **Bajra_frac** | `temporal_variance` | 0.1289 | 0.0244 | **0.1891** | Temporal backscatter variance. |
| **Maize_frac** | `area_ha` | 0.0492 | 0.0101 | **0.2064** | Village physical area. |
| **Cotton_frac** | `perimeter` | 0.0288 | 0.0063 | **0.2178** | Village boundary length. |
| **Groundnut_frac** | `perimeter` | 0.0411 | 0.0093 | **0.2258** | Village boundary length. |
| **Cotton_frac** | `area_ha` | 0.0145 | 0.0033 | **0.2263** | Village physical area. |
| **Bajra_frac** | `area_ha` | 0.0072 | 0.0024 | **0.3280** | Village physical area. |
| **Bajra_frac** | `mean_local_std_20250619` | 0.0094 | 0.0036 | **0.3845** | Local standard deviation during transplanting flood. |

## Feature Robustness Assessment

- **Extremely Stable Primary Predictors**: The primary features (`p50_20250619` for Rice, `diff_harvest` for Cotton, `p25_20250619` for Bajra, and local standard deviations for Groundnut/Maize) have stability indices below **0.10**. This means their importance is virtually independent of which village is excluded from training.
- **Biophysically Motivated Signatures**: These stable importances match known crop crop phenology. For example, Rice is highly dependent on June percentiles due to transplanting floods, whereas Cotton is dependent on the harvesting difference.
- **Safe Geometry Features**: Geometry features like `area_ha` and `perimeter` have higher stability indices (around 0.20), but they act as secondary shape regularizers to scale predictions stably across varying village sizes.
