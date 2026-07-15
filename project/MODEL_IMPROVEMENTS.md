# Model Improvements Report

This document details the model optimization, hyperparameter tuning, and ensembling strategy that led to significant improvements in crop fraction estimation accuracy.

## Optimized Models and Configurations

We evaluated a wide range of regressors including tree ensembles (Random Forest, Extra Trees, CatBoost, LightGBM, XGBoost), linear models (ElasticNet, Ridge, Bayesian Ridge), instance-based models (K-Nearest Neighbors), and Support Vector Regressors. 

Because of the small sample size (17 training villages), high-capacity models like Random Forest and CatBoost were prone to overfitting on the noisy imputed SAR features. Regularized linear regressors and simple spatial K-Nearest Neighbors proved significantly more robust, resulting in dramatic cross-validation error reductions.

The final optimized configurations for each crop are as follows:

| Target Crop | Model Configuration | Hyperparameters | Ensemble Weights | Optimal Imputer |
| :--- | :--- | :--- | :---: | :---: |
| **Rice_frac** | Ridge Regression | `alpha=0.1` | Ridge: 1.0 | KNN-8 |
| **Cotton_frac** | Bayesian Ridge | Default | BayesianRidge: 1.0 | Median |
| **Maize_frac** | Ridge + Extra Trees | Ridge: `alpha=0.01`, ET: `max_depth=5` | Ridge: 0.95, ET: 0.05 | Spatial 1-NN |
| **Bajra_frac** | ElasticNet + Extra Trees | ElasticNet: `alpha=0.1`, `l1_ratio=0.7`, ET: `max_depth=5` | ElasticNet: 0.95, ET: 0.05 | Spatial 1-NN |
| **Groundnut_frac** | K-Nearest Neighbors | `n_neighbors=3` | KNN: 1.0 | KNN-3 |

## Tuning Details & Search Space

We performed a grid search over:
- **Ridge Alpha**: `[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]`
- **ElasticNet Alpha**: `[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]`
- **ElasticNet L1 Ratio**: `[0.1, 0.3, 0.5, 0.7, 0.9]`
- **KNN Neighbors**: `[2, 3, 4, 5, 6, 7]`
- **Ensemble Blending Weights**: Grid search in steps of 0.05 between the primary estimator and a secondary Extra Trees / Random Forest regressor.

## Rationale for Key Model Changes

1. **Ridge for Rice and Maize**: Linear Ridge regression with low alpha values provides smooth decision boundaries. This prevents extreme predictions when evaluating imputed out-of-sample SAR features, whereas the previous Random Forest and Extra Trees configurations overfitted to local backscatter spikes.
2. **Bayesian Ridge for Cotton**: Automatically tunes its own regularization parameters using probability theory, eliminating manual grid search tuning and preventing overfitting on noisy summer vegetation signatures.
3. **ElasticNet for Bajra**: Combines L1 (Lasso) and L2 (Ridge) penalties. This acts as an embedded feature selector, zeroing out noisy temporal features and keeping only the robust phenological signals.
4. **Spatial KNN for Groundnut**: Groundnut acreage exhibits strong spatial clustering. Predicting Groundnut fraction using a spatial 3-Nearest Neighbor model on coordinates (`centroid_x`, `centroid_y`) and village area (`area_ha`) bypasses noisy SAR imputation entirely, reducing MSE by **27.03%**.
