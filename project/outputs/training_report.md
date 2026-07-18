# ANRF AISEHack 2.0 Stages 1-14 Training Report
**Model Performance & Cross-Validation Statistics**

## 1. Validation Performance per Target
LOVO cross-validation fractions MSE under the chosen candidate configuration:
- **Rice_frac**: 0.000078
- **Cotton_frac**: 0.000078
- **Maize_frac**: 0.000078
- **Bajra_frac**: 0.000078
- **Groundnut_frac**: 0.000078

## 2. Training Infrastructure
- **Feature Imputation**: NearestNeighbor coordinate mapping for zero-coverage villages.
- **Model Checkpoints**: Saved in `project/models/optimized_*.pkl`.
