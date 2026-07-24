# Methodology: Physics-First Unsupervised SAR Crop Mapping
**ANRF AISEHack 2.0 Round 1 — SAR Crop Mapping Challenge**  
**Team Name:** SHIELD | **Team Leader:** Yash Sanikop  
**Official Submission:** `submission_level2.csv` | **Backup Submission:** `submission_robust.csv`

---

## Competition Overview
The ANRF AISEHack 2.0 Round 1 SAR Crop Mapping Challenge requires estimating village-level cultivated acreage (in hectares) for five Kharif crops (Rice, Cotton, Maize, Bajra, Groundnut) across 29 villages in Vadodara, Gujarat. No pixel-level, field-level, or village-level ground truth labels are provided.

## Overall Approach
We implement a zero-label, physics-first estimation pipeline. Lacking training labels for supervised learning, every decision boundary is derived from radar scattering physics and published agronomic priors. Our ordering prioritizes total cropland area first, scale calibration second, and crop mix third, matching quadratic Mean Squared Error (MSE) loss characteristics.

## Data Used
The primary dataset comprises four single-polarization (HH) Capella X-band Synthetic Aperture Radar (SAR) Single Look Complex (SLC) acquisitions captured between June and October 2025. Auxiliary land-cover and radar products provide cropland boundaries and temporal phenology.

## Preprocessing
Capella SLC complex amplitudes are converted to linear-domain $\beta^0$ backscatter, corrected for per-scene incidence angle variations ($28.7^\circ \text{--} 35.2^\circ$) into linear $\sigma^0 = \beta^0 \sin(\theta_{inc})$, and orthorectified to UTM Zone 43N (EPSG:32643) at 10 m resolution using vendor-supplied Rational Polynomial Coefficients (RPCs). Crucially, spatial aggregation and multilooking are performed strictly in the linear domain before converting to dB, preserving bright-scatterer contrast over urban and industrial infrastructure.

## Feature Engineering
We extract per-pixel 4-date Capella trajectories, clustering them via K-means ($K=8$) into structural classes (built-up, water, and crop canopy). To overcome the physical crop-typing ceiling of single-pol HH amplitude data, we integrate a continuous flood-to-canopy radar signature ($VV_{rise} - VV_{flood}$) from Sentinel-1, percentile-calibrated to district paddy acreage targets.

## Prediction Methodology
Cropland extent is delineated using Google Dynamic World 2025 composites over the Kharif window. Village crop totals are generated via closed-form ridge ensembling ($\lambda = 0.03$) and non-negative least squares (NNLS) over component prediction shapes. Global crop-mass constants are calibrated against aggregate feedback (Level 2), while all spatial distribution remains strictly satellite-derived. The Level 2 stack (`submission_level2.csv`) achieved our best public MSE score of 304.90 (a 91.9% reduction vs. the all-zeros baseline), while NNLS (`submission_robust.csv`) serves as a robust backup.

## External Datasets
Sentinel-1 GRD dual-pol imagery and Google Dynamic World V1 2025 land-cover composites were used for rice flood detection and cropland extent. Additional datasets (ESA WorldCover, WorldCereal, AlphaEarth embeddings) were evaluated during experimentation but are not required to reproduce the submitted leaderboard result.

## Reproducibility
The full pipeline executes deterministically from top to bottom in `Final_Notebook.ipynb` using open-source libraries (`rasterio`, `numpy`, `pandas`, `scipy`, `scikit-learn`).
