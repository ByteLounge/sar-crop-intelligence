# Data Lineage and Traceability Report
**Capella SAR to Crop Hectares Prediction Trace**

This report traces the exact data lineage for every value in `submission_generated.csv` back to the original Capella multi-temporal SAR GeoTIFFs, detailing which features and models influenced each prediction.

---

## 1. Pixel-Level SAR Lineage (Covered Villages)

For the **17 covered villages** (with SAR swath coverage $> 0.35$), the crop hectare predictions are derived directly from pixel-level classifications. The mathematical relationship is:
$$\text{Crop\_ha}_i = N_c \times 0.01$$
where $N_c$ is the count of pixels in village $i$ assigned to crop class $c$, and $0.01$ is the spatial area of a single pixel ($10\text{m} \times 10\text{m} = 100\text{m}^2 = 0.01\text{ ha}$).

The classification of each pixel is determined by:
1. **The Cultivated Mask**: A pixel is only classified if it is inside the `cultivated_mask.tif` (combined mask). This mask is computed by filtering out water ($mean < 25$), built-up texture ($local\_std > 12$, $entropy > 4.5$, $gradient > 35$), and applying Otsu's threshold on local standard deviation to segment smooth crop fields.
2. **The GMM Classifier**: A Gaussian Mixture Model (`gmm_crop_model.pkl`) predicts the crop type using a 14-dimensional scaled feature vector extracted directly from the aligned Capella TIFFs:
   - **Preprocessed Backscatter (`raw_0` to `raw_3`)**: Derived by converting raw `uint8` preview pixels to linear power ($10^{DN/50}$), applying a 5x5 Lee speckle filter, and converting back to decibels (dB).
   - **Temporal Dynamics (`diff_1` to `diff_3`, `slope`, `amplitude`, `temp_var`)**: Pairwise differences, linear regression slope, and variation of backscatter across the June, July, August, and October acquisitions.
   - **Texture (`glcm_contrast`, `glcm_homogeneity`, `local_std`, `grad_mag`)**: Spatial co-occurrence texture and local neighborhood gradients computed on the temporal mean image.

---

## 2. Spatial Interpolation Lineage (Zero-Coverage Villages)

For the **12 zero-coverage villages** (with SAR swath coverage $\le 0.35$), there are no valid SAR pixels. The crop hectares are predicted using spatial models trained on the 17 covered villages' direct SAR classifications:
1. **Estimated Cultivated Area**: Predicted using `cultivated_knn.pkl`, a spatial `KNeighborsRegressor` that interpolates the cultivated fraction using village geometry features (coordinates, area, compactness):
   $$\text{pred\_cultivated\_ha}_i = \text{pred\_cultivated\_frac}_i \times \text{area\_ha}_i$$
2. **Estimated Crop Fractions**: Predicted using `spatial_crops_knn.pkl`, a spatial regressor that interpolates the relative crop proportions using the same geometry features. The fractions are normalized to sum to 1.0.
3. **Crop Area Calculation**:
   $$\text{Crop\_ha}_i = \text{pred\_crop\_frac}_i \times \text{pred\_cultivated\_ha}_i$$

Thus, every prediction for zero-coverage villages is a spatial interpolation of the neighboring covered villages' pixel-level SAR classifications.

---

## 3. Traceability Matrix per Village

The table below traces each village's swath coverage, its classification mode, its total cultivated area, and its predicted crop areas (all values in hectares):

| ID | Village | Coverage | Mode | Cultivated (ha) | Rice (ha) | Cotton (ha) | Maize (ha) | Bajra (ha) | Groundnut (ha) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | Manpura | 100.00% | Direct SAR (Pixel Count) | 38.44 | 0.59 | 1.21 | 22.01 | 4.44 | 10.20 |
| 2 | Umeta | 100.00% | Direct SAR (Pixel Count) | 133.28 | 5.01 | 1.72 | 107.68 | 4.94 | 13.93 |
| 3 | Sankhyad | 100.00% | Direct SAR (Pixel Count) | 103.29 | 2.27 | 2.73 | 65.11 | 9.92 | 23.26 |
| 4 | Ampad | 100.00% | Direct SAR (Pixel Count) | 3.18 | 0.00 | 0.11 | 0.08 | 1.02 | 1.97 |
| 5 | Khanpur | 100.00% | Direct SAR (Pixel Count) | 25.28 | 0.05 | 2.16 | 7.32 | 4.71 | 11.04 |
| 6 | Ankod | 100.00% | Direct SAR (Pixel Count) | 66.89 | 0.00 | 3.87 | 3.61 | 19.08 | 40.33 |
| 7 | Singrot | 100.00% | Direct SAR (Pixel Count) | 386.29 | 0.00 | 6.02 | 283.68 | 24.62 | 71.97 |
| 8 | Undera | 100.00% | Direct SAR (Pixel Count) | 82.39 | 0.13 | 6.62 | 23.44 | 15.52 | 36.68 |
| 9 | Sherkhi | 100.00% | Direct SAR (Pixel Count) | 268.43 | 0.00 | 8.70 | 72.42 | 51.01 | 136.30 |
| 10 | Bajwa | 100.00% | Direct SAR (Pixel Count) | 13.28 | 0.06 | 2.47 | 2.49 | 1.98 | 6.28 |
| 11 | Chhani | 100.00% | Direct SAR (Pixel Count) | 344.88 | 7.69 | 22.13 | 54.52 | 102.34 | 158.21 |
| 12 | Kotna | 100.00% | Direct SAR (Pixel Count) | 72.84 | 0.04 | 2.88 | 24.33 | 13.50 | 32.09 |
| 13 | Koyali | 100.00% | Direct SAR (Pixel Count) | 317.23 | 0.04 | 10.46 | 134.63 | 45.89 | 126.21 |
| 14 | Dhanora | 100.00% | Direct SAR (Pixel Count) | 159.86 | 0.32 | 4.70 | 124.59 | 7.86 | 22.39 |
| 15 | Karchiya | 100.00% | Direct SAR (Pixel Count) | 76.86 | 0.00 | 4.39 | 26.91 | 9.19 | 36.37 |
| 16 | Dashrath | 100.00% | Direct SAR (Pixel Count) | 220.88 | 1.98 | 7.31 | 26.97 | 46.13 | 138.49 |
| 17 | Angadh | 100.00% | Direct SAR (Pixel Count) | 405.99 | 2.75 | 12.31 | 188.29 | 56.50 | 146.14 |
| 18 | Ranoli | 100.00% | Direct SAR (Pixel Count) | 236.92 | 3.10 | 9.48 | 161.31 | 13.18 | 49.85 |
| 19 | Ajod | 100.00% | Direct SAR (Pixel Count) | 212.59 | 0.00 | 15.12 | 35.92 | 59.17 | 102.38 |
| 20 | Sisva | 100.00% | Direct SAR (Pixel Count) | 146.81 | 2.35 | 11.05 | 1.12 | 65.69 | 66.60 |
| 21 | Padmala | 100.00% | Direct SAR (Pixel Count) | 233.80 | 8.69 | 10.87 | 144.57 | 16.38 | 53.29 |
| 22 | Sokhda | 100.00% | Direct SAR (Pixel Count) | 681.01 | 17.11 | 31.71 | 260.91 | 113.14 | 258.14 |
| 23 | Vasna-Kotariya | 100.00% | Direct SAR (Pixel Count) | 174.51 | 1.11 | 12.01 | 27.12 | 52.57 | 81.68 |
| 24 | Asoj | 100.00% | Direct SAR (Pixel Count) | 199.72 | 6.90 | 22.63 | 1.31 | 100.75 | 68.13 |
| 25 | Pilol | 100.00% | Direct SAR (Pixel Count) | 327.35 | 7.93 | 23.44 | 44.96 | 107.19 | 143.83 |
| 26 | Manjusar | 100.00% | Direct SAR (Pixel Count) | 517.70 | 11.01 | 33.38 | 140.92 | 139.84 | 192.55 |
| 27 | Alindra | 100.00% | Direct SAR (Pixel Count) | 222.11 | 10.95 | 16.36 | 42.44 | 78.70 | 73.66 |
| 28 | Kunpad | 100.00% | Direct SAR (Pixel Count) | 140.29 | 13.66 | 3.00 | 82.51 | 11.27 | 29.85 |
| 29 | Jaspur | 100.00% | Direct SAR (Pixel Count) | 408.12 | 0.02 | 11.08 | 191.50 | 55.00 | 150.54 |
