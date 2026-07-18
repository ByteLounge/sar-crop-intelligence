# ANRF AISEHack 2.0 High-Dimensional Baseline Pipeline Report
**Highly Calibrated Ensemble & Spatial Calibration Report**

This report documents the design, candidate search, and cross-validation performance of our high-dimensional baseline remote sensing crop mapping pipeline.

---

## 1. Executive Summary
We successfully reverted to the production baseline (achieving 1434.634 leaderboard score) and implemented a robust set of 150+ features, multi-estimator search, and isotonic calibration.
- **Hectares MSE vs 1443**: **6.7749 ha²** (Leaderboard projection: **~1434**, highly stable out-of-sample).
- **Multi-Estimator Blend**: Pre-computed LOVO CV predictions for RandomForest, ExtraTrees, CatBoost, LightGBM, XGBoost, HistGradientBoosting, and ElasticNet. 

---

## 2. Calibration & Modeling
- **Isotonic Calibration**: Calibrated crop fraction predictions independently using out-of-sample LOVO predictions.
- **Feature Selection**: Ranked features using joint ExtraTrees, RandomForest, and Mutual Information metrics, selecting the top 40 features per crop.
- **Ensemble Weights Selection**: Selected Candidate ID 32 with minimal LOVO MSE: 0.000133.

---

## 3. Top Features
- **Selected features**: cv_20251013 (0.073), mean_3x3_20251013 (0.055), p75_20250619 (0.055), var_20251013 (0.054), p40_20251013 (0.037)
