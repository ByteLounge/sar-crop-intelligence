# Validation Report
**Evaluation of the Image-Driven SAR Pipeline**

## 1. Feature Distributions & Clustering Quality
* **Agreement (ARI)**:
  - KMeans vs. GMM: 0.1241
  - KMeans vs. Agglomerative: 0.3093
  - GMM vs. Spectral: 0.3925
* **Silhouette Scores**:
  - KMeans: 0.1521
  - GMM: 0.0908

## 2. Prediction Sanity Checks
* **Physical Bound**: Verified. 100% of the villages satisfy:
  $$\sum \text{Crop\_ha} \leq \text{Village\_Area}$$
* **Zero Values**: Validated. All zero-coverage villages have non-zero crop hectares predicted via geometry-based spatial KNN interpolation.
* **Runtime & Memory**:
  - Preprocessing: ~4 seconds per acquisition.
  - GLCM Feature Extraction: ~26 seconds.
  - Clustering & Aggregation: ~15 seconds.
  - Peak Memory Usage: < 2 GB.
