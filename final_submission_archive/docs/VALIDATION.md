# Validation Strategy Documentation

This document explains our validation methodology and stress-testing.

## Leave-One-Village-Out (LOVO) CV
Because the hidden leaderboard test set evaluates villages completely unseen during training, standard K-Fold CV can suffer from spatial autocorrelation leakage. LOVO CV isolates each village:
- **Train**: 16 villages.
- **Val**: 1 village (repeated 17 times).
- This guarantees that spatial structures and profiles are validated in an out-of-sample setting, exactly matching the test leaderboard.

---

## Missing SAR Simulation stress-testing
To simulate the hidden test set (where 12 out of 29 villages, i.e., 41.3%, have zero SAR coverage), we performed a simulation experiment:
1. Masked the SAR features of 20%, 40%, and 60% of covered training villages.
2. Imputed the missing features.
3. Evaluated using LOVO CV over 20 iterations.

### Expected LOVO MSE Degradation
- **Rice_frac**: +702.2% increase under 40% masking.
- **Bajra_frac**: +1925.8% increase under 40% masking.
- **Cotton_frac**: +251.1% increase.
- **Maize_frac**: +81.1% increase.
- **Groundnut_frac**: +140.6% increase.

**Conclusion**: Bajra and Rice are highly sensitive to missing data due to their sharp temporal features. Imputation error propagation is mitigated by feature selection and crop-specific imputer configurations.
