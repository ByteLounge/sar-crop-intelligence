# Feature Engineering Documentation

This document catalogs every feature extracted, its mathematical formula, and its physical significance.

---

## 1. Geometry Features
Source: Village boundary shapefile.

- **`centroid_x`, `centroid_y`**:
  - *Formula*: X and Y coordinates of the geometric centroid.
  - *Purpose*: Captures spatial autocorrelation and regional crop distribution trends.
  - *Crops*: Bajra, Groundnut, and Maize (heavily localized to specific sectors).
- **`area_ha`**:
  - *Formula*: $\text{Area (m}^2) / 10000.0$.
  - *Purpose*: Provides scaling factor for absolute hectare conversion.
- **`perimeter`**:
  - *Formula*: Bounding contour length (meters).
  - *Purpose*: Captures boundary complexity.
- **`compactness`**:
  - *Formula*: $4 \pi \text{Area} / \text{Perimeter}^2$.
  - *Purpose*: Differentiates narrow riverine villages from circular agricultural blocks.
- **`bbox_width`, `bbox_height`**:
  - *Formula*: Maximum horizontal/vertical dimensions.
  - *Purpose*: Describes spatial elongation.

---

## 2. SAR Statistical Features
Source: Aligned Capella X-band HH SAR backscatter values ($X$).

- **`mean_{date}`**:
  - *Formula*: $\frac{1}{N} \sum X_j$
  - *Purpose*: Captures absolute canopy volume scattering.
- **`std_{date}`**:
  - *Formula*: $\sqrt{\text{Var}(X)}$
  - *Purpose*: Describes variance in crop height and texture.
- **`cv_{date}` (Coefficient of Variation)**:
  - *Formula*: $\text{std}(X) / \text{mean}(X)$
  - *Purpose*: Measures relative canopy homogeneity.
- **`skew_{date}`, `kurt_{date}`**:
  - *Formula*: 3rd and 4th standardized moments.
  - *Purpose*: Captures the shape of the backscatter distribution.
- **`p25_{date}`, `p50_{date}`, `p75_{date}`**:
  - *Formula*: 25th, 50th, and 75th percentiles.
  - *Purpose*: Robust distribution markers, less sensitive to speckle noise.

---

## 3. Spatial Texture Features
Source: Local Standard Deviation using Box Filters.

- **`mean_local_std_{date}`**:
  - *Formula*: $\text{Mean of } \sqrt{\text{boxFilter}(X^2) - \text{boxFilter}(X)^2}$
  - *Purpose*: Measures spatial roughness.
  - *Crops*: Groundnut (highly textured low canopy) and built-up edge structures.

---

## 4. Temporal Difference Features
Source: Multi-temporal mean differences.

- **`diff_sowing`**:
  - *Formula*: $\text{mean}_{06/19} - \text{mean}_{06/06}$
  - *Crops*: Rice (captures the transplanting water-flooding backscatter drop).
- **`diff_veg`**:
  - *Formula*: $\text{mean}_{08/14} - \text{mean}_{06/19}$
  - *Crops*: Cotton (captures biomass volume growth).
- **`diff_harvest`**:
  - *Formula*: $\text{mean}_{10/13} - \text{mean}_{08/14}$
  - *Crops*: Maize and Bajra (captures post-harvest vegetative drying).
- **`cumulative_change`**:
  - *Formula*: $\frac{1}{N} \sum \sum_{t} |X_{t+1} - X_t|$
  - *Crops*: Groundnut (stable low values) vs. Cotton (high cumulative change).
