# Pipeline Raster and Feature Report
**Lineage and Metadata Verification for SAR Crop Mapping**

This report programmatically lists all input datasets and intermediate generated rasters, including their spatial dimensions, exact data ranges (min/max), and the models that consumed them.

---

## 1. Input Datasets

| File Name | Type | Dimensions | Min Value | Max Value | Consumed By |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `CAPELLA_C14_SM_GEO_HH_20250606072501_20250606072506_preview.tif` | Input (Georeferenced Preview) | 26850 x 26678 | 0.0 | 255.0 | Reprojection & Alignment (align_rasters) |
| `CAPELLA_C14_SM_SLC_HH_20250606072501_20250606072506.tif` | Input (Slant-range SLC Complex) | 4682 x 27192 | Complex | Complex | Metadata reference (slant-range geometry) |
| `CAPELLA_C14_SM_GEO_HH_20250619021410_20250619021415_preview.tif` | Input (Georeferenced Preview) | 26965 x 26564 | 0.0 | 255.0 | Reprojection & Alignment (align_rasters) |
| `CAPELLA_C14_SM_SLC_HH_20250606072501_20250606072506.tif` | Input (Slant-range SLC Complex) | 4682 x 27192 | Complex | Complex | Metadata reference (slant-range geometry) |
| `CAPELLA_C14_SM_SLC_HH_20250619021410_20250619021415.tif` | Input (Slant-range SLC Complex) | 3910 x 27187 | Complex | Complex | Metadata reference (slant-range geometry) |
| `CAPELLA_C14_SM_GEO_HH_20250814031124_20250814031129_preview.tif` | Input (Georeferenced Preview) | 26976 x 26594 | 0.0 | 255.0 | Reprojection & Alignment (align_rasters) |
| `CAPELLA_C14_SM_SLC_HH_20250814031124_20250814031129.tif` | Input (Slant-range SLC Complex) | 3897 x 27219 | Complex | Complex | Metadata reference (slant-range geometry) |
| `CAPELLA_C14_SM_GEO_HH_20251013022643_20251013022648_preview.tif` | Input (Georeferenced Preview) | 26965 x 26636 | 0.0 | 255.0 | Reprojection & Alignment (align_rasters) |
| `CAPELLA_C14_SM_SLC_HH_20251013022643_20251013022648.tif` | Input (Slant-range SLC Complex) | 4244 x 27241 | Complex | Complex | Metadata reference (slant-range geometry) |

---

## 2. Generated Feature Rasters

| File Name | Dimensions | Min Value | Max Value | Consumed By |
| :--- | :---: | :---: | :---: | :--- |
| `august_db.tif` | 2447 x 2276 | 0.0000 | 216.0000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `august_entropy.tif` | 2447 x 2276 | 0.0000 | 4.1639 | Downstream Aggregation & Visualization |
| `closing_20250606.tif` | 2447 x 2276 | 0.0000 | 1.0000 | GMM Pixel Masking & Spatial KNN Fraction Estimator |
| `closing_20250619.tif` | 2447 x 2276 | 0.0000 | 1.0000 | GMM Pixel Masking & Spatial KNN Fraction Estimator |
| `closing_20250814.tif` | 2447 x 2276 | 0.0000 | 1.0000 | GMM Pixel Masking & Spatial KNN Fraction Estimator |
| `closing_20251013.tif` | 2447 x 2276 | 0.0000 | 1.0000 | GMM Pixel Masking & Spatial KNN Fraction Estimator |
| `connected_components_20250606.tif` | 2447 x 2276 | 0.0000 | 1.0000 | GMM Pixel Masking & Spatial KNN Fraction Estimator |
| `connected_components_20250619.tif` | 2447 x 2276 | 0.0000 | 1.0000 | GMM Pixel Masking & Spatial KNN Fraction Estimator |
| `connected_components_20250814.tif` | 2447 x 2276 | 0.0000 | 1.0000 | GMM Pixel Masking & Spatial KNN Fraction Estimator |
| `connected_components_20251013.tif` | 2447 x 2276 | 0.0000 | 1.0000 | GMM Pixel Masking & Spatial KNN Fraction Estimator |
| `cultivated_mask.tif` | 2447 x 2276 | 0.0000 | 1.0000 | GMM Pixel Masking & Spatial KNN Fraction Estimator |
| `glcm_asm_20250606.tif` | 2447 x 2276 | 0.0560 | 1.0000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `glcm_asm_20250619.tif` | 2447 x 2276 | 0.0560 | 1.0000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `glcm_asm_20250814.tif` | 2447 x 2276 | 0.0592 | 1.0000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `glcm_asm_20251013.tif` | 2447 x 2276 | 0.0464 | 1.0000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `glcm_contrast_20250606.tif` | 2447 x 2276 | 0.0000 | 10.7600 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `glcm_contrast_20250619.tif` | 2447 x 2276 | 0.0000 | 19.8000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `glcm_contrast_20250814.tif` | 2447 x 2276 | 0.0000 | 22.4400 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `glcm_contrast_20251013.tif` | 2447 x 2276 | 0.0000 | 9.0800 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `glcm_energy_20250606.tif` | 2447 x 2276 | 0.2366 | 1.0000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `glcm_energy_20250619.tif` | 2447 x 2276 | 0.2366 | 1.0000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `glcm_energy_20250814.tif` | 2447 x 2276 | 0.2433 | 1.0000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `glcm_energy_20251013.tif` | 2447 x 2276 | 0.2154 | 1.0000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `glcm_entropy_20250606.tif` | 2447 x 2276 | 0.0000 | 4.2439 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `glcm_entropy_20250619.tif` | 2447 x 2276 | 0.0000 | 4.2439 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `glcm_entropy_20250814.tif` | 2447 x 2276 | 0.0000 | 4.1639 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `glcm_entropy_20251013.tif` | 2447 x 2276 | 0.0000 | 4.4839 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `glcm_homogeneity_20250606.tif` | 2447 x 2276 | 0.2704 | 1.0000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `glcm_homogeneity_20250619.tif` | 2447 x 2276 | 0.2231 | 1.0000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `glcm_homogeneity_20250814.tif` | 2447 x 2276 | 0.1878 | 1.0000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `glcm_homogeneity_20251013.tif` | 2447 x 2276 | 0.2327 | 1.0000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `grad_mag_20250606.tif` | 2447 x 2276 | 0.0000 | 630.5696 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `grad_mag_20250619.tif` | 2447 x 2276 | 0.0000 | 919.9185 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `grad_mag_20250814.tif` | 2447 x 2276 | 0.0000 | 914.7732 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `grad_mag_20251013.tif` | 2447 x 2276 | 0.0000 | 582.8636 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `june_db.tif` | 2447 x 2276 | 0.0000 | 192.0000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `june_texture.tif` | 2447 x 2276 | 0.0000 | 10.7600 | Downstream Aggregation & Visualization |
| `laplacian_20250606.tif` | 2447 x 2276 | 0.0000 | 363.0000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `laplacian_20250619.tif` | 2447 x 2276 | 0.0000 | 484.0000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `laplacian_20250814.tif` | 2447 x 2276 | 0.0000 | 630.0000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `laplacian_20251013.tif` | 2447 x 2276 | 0.0000 | 310.0000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `local_variance_20250606.tif` | 2447 x 2276 | 0.0000 | 4945.9614 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `local_variance_20250619.tif` | 2447 x 2276 | 0.0000 | 10312.6143 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `local_variance_20250814.tif` | 2447 x 2276 | 0.0000 | 9120.2979 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `local_variance_20251013.tif` | 2447 x 2276 | 0.0000 | 4037.5264 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `mean_3x3_20250606.tif` | 2447 x 2276 | 0.0000 | 164.6667 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `mean_3x3_20250619.tif` | 2447 x 2276 | 0.0000 | 201.2222 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `mean_3x3_20250814.tif` | 2447 x 2276 | 0.0000 | 172.3333 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `mean_3x3_20251013.tif` | 2447 x 2276 | 0.0000 | 161.1111 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `mean_5x5_20250606.tif` | 2447 x 2276 | 0.0000 | 152.9200 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `mean_5x5_20250619.tif` | 2447 x 2276 | 0.0000 | 166.6400 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `mean_5x5_20250814.tif` | 2447 x 2276 | 0.0000 | 157.2000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `mean_5x5_20251013.tif` | 2447 x 2276 | 0.0000 | 149.5200 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `opening_20250606.tif` | 2447 x 2276 | 0.0000 | 1.0000 | GMM Pixel Masking & Spatial KNN Fraction Estimator |
| `opening_20250619.tif` | 2447 x 2276 | 0.0000 | 1.0000 | GMM Pixel Masking & Spatial KNN Fraction Estimator |
| `opening_20250814.tif` | 2447 x 2276 | 0.0000 | 1.0000 | GMM Pixel Masking & Spatial KNN Fraction Estimator |
| `opening_20251013.tif` | 2447 x 2276 | 0.0000 | 1.0000 | GMM Pixel Masking & Spatial KNN Fraction Estimator |
| `raw_db_20250606.tif` | 2447 x 2276 | 0.0000 | 192.0000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `raw_db_20250619.tif` | 2447 x 2276 | 0.0000 | 212.0000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `raw_db_20250814.tif` | 2447 x 2276 | 0.0000 | 216.0000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `raw_db_20251013.tif` | 2447 x 2276 | 0.0000 | 179.0000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `temporal_amplitude.tif` | 2447 x 2276 | 0.0000 | 216.0000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `temporal_cv.tif` | 2447 x 2276 | 0.0000 | 1.7321 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `temporal_diff_aug_oct.tif` | 2447 x 2276 | -118.0000 | 110.0000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `temporal_diff_july_aug.tif` | 2447 x 2276 | -172.0000 | 130.0000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `temporal_diff_june_july.tif` | 2447 x 2276 | -119.0000 | 172.0000 | Gaussian Mixture Model (GMM) Pixel Classifier |
| `temporal_slope.tif` | 2447 x 2276 | -1.0468 | 0.8861 | Gaussian Mixture Model (GMM) Pixel Classifier |

---

## 3. QA Alignment Check

The alignment of the georeferenced Capella GeoTIFFs against the village vector boundary shapefiles was visually verified. The resulting plot shows perfect overlap between the backscatter boundary features and the village polygons.

![QA Alignment Check](qa_alignment.png)
