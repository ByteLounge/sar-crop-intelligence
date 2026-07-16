# Pipeline Audit: Current Limitations & Discarded SAR Information
**SAR Crop Acreage Mapping Pipeline — Vadodara, Gujarat**

This report presents a forensic remote sensing and computer vision audit of the current pipeline, detailing how Synthetic Aperture Radar (SAR) imagery is processed, which information is discarded, and listing the missing features that should be extracted to build a robust, image-driven crop mapping model.

---

## 1. SAR Image Processing Limitations

### A. Linear vs. Decibel (dB) Domain Averaging
* **The Flaw**: The input Capella SAR images are log-compressed 8-bit previews (`uint8`), where pixel digital numbers (DN) are linearly proportional to the decibel (dB) scale. Currently, the pipeline computes means and standard deviations directly on these dB-like values:
  $$\text{mean\_dB} = \frac{1}{N} \sum_{i=1}^N DN_i$$
* **Why it is Incorrect**: In radar remote sensing, backscatter values represent power coefficients. Averaging in log space (dB) computes the **logarithm of the geometric mean**, which severely underestimates the true radar backscatter power due to radar speckle and the non-linear compression of log scales. 
* **The Correct Method**: SAR backscatter must be converted to the linear power domain, averaged/filtered, and then converted back to decibels:
  $$\beta^0_{\text{linear}} = 10^{\frac{DN - \text{offset}}{\text{scale}}}$$
  $$\text{mean\_linear} = \frac{1}{N} \sum_{i=1}^N \beta^0_{\text{linear}, i}$$
  $$\text{mean\_dB} = 10 \log_{10}(\text{mean\_linear})$$

### B. Speckle Filtering
* **The Flaw**: Coherent radar speckle noise is currently suppressed only by bilinear interpolation when resampling the 1m Capella imagery to a 10m grid. 
* **Why it is Incorrect**: Bilinear interpolation acts as a simple blur filter that degrades edge sharpness and fails to suppress speckle in homogeneous regions. 
* **The Correct Method**: Standard SAR adaptive filters (e.g., Lee, Refined Lee, Gamma MAP) must be applied to suppress speckle while preserving field edges and spatial detail.

---

## 2. Discarded Information & Features

The current pipeline reduces the entire multi-temporal SAR image stack to simple village-wide statistics (mean, std, p50, Sobel mean), completely discarding the following critical spatial, structural, and texture information:

### A. Spatial Field Texture (GLCM)
* **What is discarded**: Spatial crop canopy arrangement, row spacing, and crop soil surface roughness texture.
* **Missing Features**: Gray-Level Co-occurrence Matrix (GLCM) features:
  * **Contrast**: Measures local intensity variation. High in built-up and orchard regions; low in homogeneous crop fields.
  * **Homogeneity**: Measures spatial similarity. High in flat fields (water, flooded rice sowing); low in urban areas.
  * **Energy / Angular Second Moment (ASM)**: Measures orderliness. High in uniform croplands; low in complex textures.
  * **Entropy**: Measures randomness. High in cities; low in uniform fields.

### B. Local Neighborhood Features (Scale & Context)
* **What is discarded**: Spatial context of pixel clusters. Crop fields are contiguous regions of homogeneous backscatter, not isolated points.
* **Missing Features**: Multi-scale box filters (e.g., $3 \times 3$ and $5 \times 5$ means), local standard deviation, Laplacian (edge/roughness indicator), and spatial gradient magnitudes.

### C. Pixel-Level Spatial-Temporal Trajectories
* **What is discarded**: Currently, crop clustering is run on all "vegetated" pixels (defined as not water and not built-up), ignoring whether they are cultivated or not.
* **Missing Features**: Pixel-level change indicators, such as temporal amplitude ($\max - \min$), temporal variance, temporal slope, and maximum growth rate.

### D. Morphological & Structural Geometry
* **What is discarded**: Field boundaries and field distances.
* **Missing Features**: Morphological opening/closing, distance transform from boundaries, and connected components size filtering.

---

## 3. Comprehensive List of Missing SAR Features to Extract

To rebuild the pipeline around the actual SAR imagery, we should extract the following pixel-level features:

1. **Calibrated Backscatter (Linear Domain)**:
   * Calibrated linear backscatter coefficients ($\beta^0$) for June 6, June 19, August 14, and October 13.
2. **Temporal Features**:
   * Peak backscatter value, minimum backscatter, temporal amplitude ($\max - \min$), temporal variance, and slope across the season.
   * Pairwise temporal differences ($\Delta \text{June} \to \text{July}$, $\Delta \text{July} \to \text{August}$, $\Delta \text{August} \to \text{October}$) calculated in the linear domain.
3. **GLCM Textures (on Temporal Mean)**:
   * Contrast, Correlation, Energy, Homogeneity, Entropy, Dissimilarity, and ASM.
4. **Local Neighborhood Filters**:
   * Local mean ($3 \times 3$, $5 \times 5$), local standard deviation, Sobel gradient magnitude, Laplacian.
5. **Morphological Features**:
   * Morphological opening/closing of the mask, distance transform.
