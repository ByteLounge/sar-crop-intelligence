# Release Notes - Version 2.0.0

**Release Date**: 2026-07-12

## Final Architecture Summary
- **Hybrid Imputation**: Set KNN-6 for Rice/Cotton/Maize and Spatial 1-NN for Bajra/Groundnut.
- **Ensemble stack**: Tuned Random Forest, Extra Trees, CatBoost, and ElasticNet estimators per crop target.
- **Physical Bounds**: Normalizes blended crop fractions so their sum matches the estimated village vegetation capacity.

## Major Experiments & Improvements
- **KNN neighbor search**: Stress-tested $k \in \{1 \dots 10\}$, showing $k=6$ minimizes feature reconstruction RMSE (11.0% error reduction over baseline).
- **Imputer comparison under 40% masking**: Verified that Spatial 1-NN outperforms KNN for localized crops (Bajra and Groundnut), leading to our hybrid imputer strategy.

## Hidden Leaderboard Risks
- Spatial autocorrelation failure in zero-coverage villages.
- Micro-climatic planting timeline anomalies.
