# Model Selection Documentation

This document details the selection, hyperparameter tuning, and cross-validation performance of our regression models.

## Evaluated Models
1. **Random Forest (RF)**: Robust ensemble. Excellent on small datasets. Accepted as the primary model for **Rice** and **Groundnut**.
2. **Extra Trees (ET)**: Extremely randomized trees. Reduces variance. Accepted as the primary model for **Maize** and **Bajra**.
3. **CatBoost Regressor**: Handles noise and small sample sizes. Integrated into the **Cotton** ensemble.
4. **ElasticNet**: Linear model with L1/L2 regularization. Excellent for spatial trends. Accepted as the primary model for **Cotton**.
5. **LightGBM / XGBoost**: Discarded as primary estimators due to overfitting on the small sample size (17 training villages), but retained as secondary stacking candidates.
6. **Support Vector Regressor (SVR)**: Discarded due to poor validation performance (MSE > 0.02).

---

## Leave-One-Village-Out (LOVO) Validation Results

| Model type | Rice_frac MSE | Cotton_frac MSE | Maize_frac MSE | Bajra_frac MSE | Groundnut_frac MSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (RF)** | **0.002243** | 0.001049 | 0.001672 | 0.001822 | **0.001018** |
| **Extra Trees (ET)** | 0.002871 | 0.001130 | **0.001160** | **0.000525** | 0.001468 |
| **ElasticNet** | 0.003201 | **0.001021** | 0.001167 | 0.001876 | 0.003451 |
| **CatBoost** | 0.006139 | 0.001312 | 0.001963 | 0.005787 | 0.002055 |
| **SVR** | 0.035539 | 0.007292 | 0.006179 | 0.022078 | 0.006091 |

---

## Stacking and Ensembling Configuration
Ensemble weights were optimized via grid search on LOVO out-of-fold validation predictions:
- **Rice_frac**: RF (1.0) + ET (0.0)
- **Cotton_frac**: RF (0.8) + CatBoost (0.2)
- **Maize_frac**: RF (0.0) + ET (1.0)
- **Bajra_frac**: RF (0.0) + ET (1.0)
- **Groundnut_frac**: RF (1.0) + ET (0.0)
