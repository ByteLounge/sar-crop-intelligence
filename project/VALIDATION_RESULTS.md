# Validation Results Report

This document presents the detailed cross-validation results and residual analysis of our optimized pipeline.

## Standard vs. Imputed LOVO CV Comparison

To ensure robust generalization, we evaluated models using **Leave-One-Village-Out (LOVO) Cross Validation**. 

We report two metrics:
1. **Standard LOVO CV**: Validation village's SAR features are NOT masked. Evaluates model performance on covered villages.
2. **Imputed LOVO CV**: Validation village's SAR features are masked and imputed. Evaluates joint performance of imputer + model on zero-coverage villages.

The table below compares the **Imputed LOVO CV MSE** of the original pipeline versus our new optimized pipeline:

| Target Crop | Original Imputed MSE | New Tuned Imputed MSE | Improvement (%) | New Tuned Imputed RMSE |
| :--- | :---: | :---: | :---: | :---: |
| **Rice_frac** | 0.030429 | **0.001417** | **-95.34%** | 0.037649 |
| **Cotton_frac** | 0.006943 | **0.001632** | **-76.50%** | 0.040394 |
| **Maize_frac** | 0.003293 | **0.000618** | **-81.23%** | 0.024852 |
| **Bajra_frac** | 0.013806 | **0.002418** | **-82.49%** | 0.049177 |
| **Groundnut_frac**| 0.002737 | **0.001997** | **-27.03%** | 0.044692 |

*All crops experienced massive, validated reductions in cross-validation MSE, driven by robust linear models, spatial feature engineering, and the [0.0, 1.0] fraction clipping constraint.*

## Residual Analysis in Hectares

The final evaluation metric in the competition is Mean Squared Error (MSE) in **hectares** across all 5 crops. For the 17 covered villages, out-of-fold predictions were scaled by village area (`area_ha`) and normalized to sum to 0.99.

Here is the residual error (Predicted - True) in hectares for each village, sorted by total MSE:

| ID | Village Name | Area (ha) | Err_Rice | Err_Cotton | Err_Maize | Err_Bajra | Err_Groundnut | Total MSE ($ha^2$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 24 | Asoj | 1077.9 | -51.08 | 30.84 | 16.09 | -38.65 | 33.51 | 1287.27 |
| 9 | Sherkhi | 1224.7 | 54.44 | -43.39 | -0.72 | -19.77 | -1.44 | 1048.11 |
| 13 | Koyali | 1170.2 | 38.34 | -53.75 | -0.86 | -2.05 | 13.45 | 909.05 |
| 20 | Sisva | 417.6 | 2.61 | 10.04 | 8.19 | -56.69 | 32.21 | 885.28 |
| 22 | Sokhda | 1174.1 | 41.52 | 9.83 | -48.84 | -2.49 | -3.22 | 844.47 |
| 6 | Ankod | 510.9 | -3.90 | 49.08 | -7.51 | -30.92 | -9.88 | 706.82 |
| 4 | Ampad | 456.1 | -3.94 | -3.93 | -17.74 | 44.30 | -22.46 | 562.39 |
| 2 | Umeta | 471.1 | -19.52 | 12.33 | -7.62 | -14.64 | 38.96 | 464.67 |
| 16 | Dashrath | 919.1 | 17.57 | -27.20 | 28.55 | -4.20 | -19.11 | 449.19 |
| 14 | Dhanora | 398.3 | -39.90 | 16.64 | 5.12 | 4.16 | 11.17 | 407.46 |
| 18 | Ranoli | 704.0 | -36.14 | 25.79 | -1.54 | 6.99 | 1.00 | 404.66 |
| 21 | Padmala | 687.5 | -8.36 | 33.75 | -5.74 | -0.30 | -18.21 | 314.71 |
| 19 | Ajod | 402.9 | 22.74 | -17.32 | -24.50 | 7.14 | 9.61 | 312.23 |
| 15 | Karchiya | 491.3 | 8.31 | -27.28 | -4.37 | 5.10 | 13.95 | 210.60 |
| 7 | Singrot | 1085.9 | -2.96 | -26.31 | 12.28 | 7.99 | 11.04 | 207.50 |
| 10 | Bajwa | 217.6 | 3.07 | 0.57 | 2.06 | 9.76 | -16.09 | 73.66 |
| 28 | Kunpad | 507.1 | 6.03 | -4.33 | 0.46 | 9.79 | -3.82 | 33.16 |

### Overall Performance Metric
- **Overall Mean Squared Error in Hectares**: **536.5433 $ha^2$**

## Failure Case Analysis

1. **Area Scaling Effect**: The largest errors in hectares are heavily concentrated in villages with the largest physical areas:
   - **Asoj** (1077.9 ha): MSE 1287.27
   - **Sherkhi** (1224.7 ha): MSE 1048.11
   - **Koyali** (1170.2 ha): MSE 909.05
   - A small fraction error of 4% in a 1200 hectare village results in 48 hectares of error, which translates to a squared error of $2304 ha^2$.
2. **Crop Confusion**:
   - In **Asoj**, the model underpredicted Rice (-51 ha) and Bajra (-39 ha) while overpredicting Cotton (+31 ha) and Groundnut (+34 ha).
   - In **Sherkhi**, the model overpredicted Rice (+54 ha) and underpredicted Cotton (-43 ha).
   - Since Rice has a flood dip signal and Cotton/Groundnut have vegetated signatures, soil moisture variations or differences in planting dates can cause temporary backscatter profile overlaps, leading to fraction confusion.
3. **No-Weighting Generalization**: We evaluated area-weighted loss functions (`sample_weight=area_ha`) to prioritize large villages during training. However, this severely degraded out-of-fold generalization (increasing overall Hectares MSE from 536.54 to 1447.55 $ha^2$). This confirms that unweighted regularized models remain the most robust generalizers.
