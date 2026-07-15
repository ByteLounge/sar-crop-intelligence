# Generalization Audit Report

This report presents a comprehensive audit of the spatial-temporal machine learning pipeline, highlighting potential sources of overfitting and documenting our strategies for robust hidden leaderboard generalization.

## 1. Spatial Coordinate Leakage & Linear Extrapolation Audit

The most critical finding from this audit is **spatial coordinate leakage** coupled with **linear model extrapolation**:

- **The Overfitting Heuristic**: Centroid coordinates (`centroid_x`, `centroid_y`) were included in the features of Ridge, Bayesian Ridge, and ElasticNet models.
- **Why Local CV Was Misleading**: The 17 covered villages are clustered closely together inside the Capella SAR swath. Within this swath, spatial coordinates act as highly effective interpolators of regional crop variations, yielding extremely low out-of-fold validation MSE (e.g., Rice fraction CV MSE of `0.0016`).
- **Why it Failed on the Leaderboard**: The 12 zero-coverage test villages lie at the margins or entirely outside the swath. For instance, **Jaspur** (ID 29) lies 3,000 meters south of the nearest training village, and **Alindra** (ID 27) lies 4,000 meters east of the training boundary. 
- **Extrapolation Wildness**: Regularized linear models (Ridge/ElasticNet) fit coefficients on coordinates without feature scaling. When predicting on Jaspur or Alindra, they extrapolated these linear trends far outside the training bounding box, projecting fractions to unrealistic extremes (which either clipped to 0.0 or 1.0, completely erasing other physical and SAR features).
- **Tree-Based Safety**: Tree-based models (Random Forest / Extra Trees) do not extrapolate linearly. For out-of-bounds coordinates, they perform constant boundary extrapolation (predicting the same fraction as the nearest training village). While safer, coordinates still introduce geographic bias.

## 2. Generalization Strategies Implemented

To maximize generalization to the hidden leaderboard, we performed the following corrections:

1. **Complete Removal of Coordinates**: We developed a `safe_no_coords` feature set. By completely removing `centroid_x` and `centroid_y` from the features, we force the models to rely purely on:
   - **Village geometry** (area, perimeter, compactness, bbox dimensions)
   - **SAR backscatter signatures** (percentiles, temporal ratios, temporal variance, Shannon entropy, spatial gradients)
   This makes the model **spatially translation-invariant**, allowing it to recognize crop signatures anywhere in Vadodara, regardless of absolute coordinates.
2. **Transition to Stable Tree Models**: Linear models without coordinates struggle to map the non-linear relationship between SAR statistics and crop fractions. Tree-based models (specifically low-variance **ExtraTreesRegressor**) perform significantly better without coordinates, reducing Hectares CV MSE from **6231.50** (Linear) to **4925.85 $ha^2$** (Tree).
3. **Robust Feature Selection**: Retained only features with high, stable importance across all 17 folds.
4. **Fraction Clipping & Normalization**: Standardized clipping to `[0.0, 1.0]` and sum-normalization to target vegetation fractions, preventing unrealistic predictions.
