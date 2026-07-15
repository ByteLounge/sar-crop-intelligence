# Model Comparison Report

This report compares the performance and generalization trade-offs of the 6 candidate modeling philosophies evaluated under Leave-One-Village-Out (LOVO) cross-validation.

## Candidate Comparison (Hectares LOVO CV MSE)

The table below shows the Hectares MSE (in $ha^2$) across all 5 crops for all covered villages, sorted by CV performance:

| Candidate Name | Features Used | Core Model | Hectares CV MSE ($ha^2$) | Generalization Safety |
| :--- | :--- | :--- | :---: | :--- |
| **`linear_coords`** | With Coordinates | Ridge Regression | **2125.7518** | **CRITICAL RISK**: Severe linear extrapolation outside swath bounding box. |
| **`tree_coords`** | With Coordinates | Extra Trees | **4865.7971** | **LOW RISK**: Constant boundary extrapolation. |
| **`tree_no_coords`** | **No Coordinates** | **Extra Trees** | **4925.8502** | **HIGH SAFETY**: Spatially translation-invariant; relies purely on SAR physics. |
| **`ensemble`** | No Coordinates | 0.5 Tree + 0.5 Linear | **5096.5378** | **HIGH SAFETY**: Spatially translation-invariant. |
| **`linear_no_coords`** | No Coordinates | Ridge Regression | **6231.5037** | **HIGH SAFETY**: Spatially translation-invariant. |
| **`conservative`** | None | Mean Baseline | **6756.2248** | **MAX SAFETY**: Zero prediction variance. |

## Generalization vs. CV Score Trade-off

1. **The Spatial Coordinate Trap**:
   - `linear_coords` achieves the absolute best CV score (**2125.75 $ha^2$**). However, this score is highly misleading because the validation villages are inside the covered swath and can be interpolated.
   - For zero-coverage villages outside the swath (e.g. Jaspur, Alindra), linear models extrapolate coordinates without bounds, leading to massive errors.
   - Therefore, `linear_coords` was rejected due to critical leaderboard generalization risk.
2. **The Safe Tree Choice**:
   - `tree_no_coords` achieves a CV score of **4925.85 $ha^2$**, which is extremely close to the coordinate-based tree model `tree_coords` (4865.80 $ha^2$, only a **1.2%** difference).
   - Because it completely excludes coordinates, it has zero spatial leakage and is immune to extrapolation errors. It relies entirely on crop backscatter physics (temporal ratios, Sobel gradients, CV, variance) and village area/perimeter.
   - Therefore, **`submission_tree_no_coords.csv`** is our recommended submission for the hidden leaderboard.
3. **Linear Models Without Coordinates**:
   - `linear_no_coords` performs poorly (**6231.50 $ha^2$**) compared to trees without coordinates. This is because the relationship between SAR features and unsupervised crop fractions is highly non-linear, which linear regressors cannot capture.
