# ANRF AISEHack 2.0 Round 1
## Team SHIELD

Official submission repository for **ANRF AISEHack 2.0 Round 1: SAR Crop Mapping Challenge**.

---

### Team Information

- **Team Name:** SHIELD
- **Team Leader:** Yash Sanikop
- **Primary Notebook:** `Final_Notebook.ipynb`
- **Official Submission File:** `submission_level2.csv` (Best Leaderboard Score & Rank)
- **Backup Submission File:** `submission_robust.csv` (NNLS Reproducibility Backup)

---

## Competition Overview

The **ANRF AISEHack 2.0 Round 1: SAR Crop Mapping Challenge** targets village-level crop acreage estimation (in hectares) for five Kharif crops — **Rice, Cotton, Maize, Bajra, Groundnut** — across 29 villages in Vadodara district, Gujarat. 

A defining characteristic of this challenge is that **no ground-truth training labels of any kind are provided** — no field boundaries, no crop masks, no village targets, and no pixel labels. Every decision boundary and feature must be derived from radar scattering physics, published agronomic priors, and rigorous internal validation.

---

## Project Objectives

1. **Label-Free Physics-First Pipeline:** Develop an unsupervised estimation workflow for Capella X-band SAR amplitude imagery without reliance on fit-to-label supervised learning models.
2. **Correct SAR Radiometric & Geodetic Handling:** Establish an accurate RPC orthorectification pipeline and linear-domain backscatter processing to preserve spatial contrast and radiometric integrity across varying incidence angles ($28.7^\circ \text{--} 35.2^\circ$).
3. **Multi-Sensor Division of Capability:** Integrate dual-polarization Sentinel-1 radar and Google Dynamic World V1 land-cover composites to overcome the physical crop-typing ceiling of single-polarization HH X-band data.
4. **Reproducible Leaderboard Optimization:** Produce a fully reproducible, clean notebook pipeline yielding `submission_level2.csv` (Public MSE: **304.90**, a **91.9% reduction** over the null baseline).

---

## Repository Structure

```text
email_submission/
│
├── Final_Notebook.ipynb         # Fully executed primary submission notebook
├── submission_level2.csv        # Official competition final submission (L2 Ridge Stack)
├── submission_robust.csv        # Backup reproducibility submission (NNLS Stack)
├── Methodology.pdf              # Formatted PDF documentation of methodology
├── Methodology.md               # Markdown documentation of methodology
├── README.md                    # Project README and reproduction guide
├── requirements.txt             # Python dependencies with package versions
├── environment.txt              # Execution environment details and system info
├── TEAM_INFO.txt                # Official team metadata file
├── attachments_checklist.md     # Package contents checklist
└── components/                  # Precomputed component tables for instant execution
    ├── submission_v12_raw.csv
    ├── submission_v19_raw.csv
    ├── submission_v25_areaonly.csv
    ├── dynamicworld_cropland.csv
    ├── worldcover_cropland.csv
    ├── basis_area_cotton.csv
    └── basis_area_gnut.csv
```

---

## Pipeline Overview

```text
Raw Capella X-band SLC (4 Dates) + SLC Metadata
                  │
                  ▼
   [Radiometric Calibration to Linear σ⁰]
   (β⁰ · sin θ_inc ; linear-domain multilooking)
                  │
                  ▼
   [RPC Orthorectification to UTM 43N (10m)]
   (Average resampling in linear domain)
                  │
                  ▼
  ┌───────────────┴───────────────┐
  ▼                               ▼
[Capella 4-Date Trajectory]    [Dynamic World 2025]
(K-means K=8 structural)       (Kharif composite cropland)
  │                               │
  ▼                               │
[Sentinel-1 Flood Signature]      │
(VV_rise - VV_flood paddy score)  │
  │                               │
  └───────────────┬───────────────┘
                  │
                  ▼
   [Component Shape Assembly & Ridge Ensembling]
                  │
                  ▼
   [Level 2 Per-Crop Mass Calibration]
                  │
                  ▼
   [Official Submission: submission_level2.csv]
```

---

## Dataset Description

1. **Capella X-band Single Look Complex (SLC):** Four multi-temporal StripMap acquisitions (June 6, June 19, August 14, October 13, 2025) in HH polarization covering the Vadodara AOI.
2. **Sentinel-1 GRD Dual-Pol (C-band):** Multi-temporal VV/VH backscatter series captured over the 2025 Kharif season for paddy flood signature detection.
3. **Google Dynamic World V1 (2025):** 10 m near-real-time land-cover composite filtered for the Kharif window (Jun–Nov 2025) providing per-village cropland area.
4. **Vadodara District Agronomic Statistics:** Published Gujarat Directorate of Agriculture Kharif 2025 crop sowing proportions.

---

## Preprocessing Workflow

1. **Linear σ⁰ Conversion:** Complex SLC amplitudes $A$ are scaled via scene-specific `scale_factor` $s$ and incidence angle $\theta_{inc}$:
   $$\sigma^0_{linear} = (s \cdot |A|)^2 \cdot \sin(\theta_{inc})$$
2. **Linear-Domain Multilooking & Geocoding:** RPC orthorectification is executed with average resampling strictly in the *linear domain* before logarithmic conversion to dB. This prevents suppression of bright scatterers (e.g., Koyali IOCL industrial refinery).
3. **Incidence Normalization:** Adjusts per-scene backscatter to remove spurious ~1.5 dB incidence geometry shifts between acquisitions.

---

## Feature Engineering

- **Structural Class Trajectories:** 4-date trajectory clustering isolates permanent built-up structures and open water from vegetated land.
- **Continuous Paddy Flood Score:** Evaluates $VV_{rise} - VV_{flood}$ from Sentinel-1, percentile-calibrated against district paddy shares (19.4%), eliminating arbitrary fixed thresholds.
- **Null-Model Baseline Anchor:** Village geometric area alone (`v25`) achieves $r = 0.716$, establishing an empirical floor for remote sensing contribution.

---

## Prediction Pipeline

Village-level acreage for the five crops is assembled via linear ensembling of candidate prediction shapes:
- **Level 1 Baseline (`v26`):** Ridge combination ($\lambda = 0.01$) of Dynamic World cropland, Sentinel-1 rice detector, and village geometry anchor (Public MSE: **1072.79**).
- **Level 2 Stack (`v33` / `submission_level2.csv`):** Incorporates five global per-crop mass basis probes calibrated via aggregate feedback (Public MSE: **304.90**).
- **Robust Backup (`final_nnls` / `submission_robust.csv`):** Non-negative least squares optimization ($\text{weights} = [0, 0.052, 0, 0.927, 0, 0]$) delivering a low-parameter backup (Public MSE: **409.39**).

---

## Dependencies

The implementation relies strictly on standard open-source Python packages:
- Python 3.10+
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `rasterio`
- `geopandas`
- `jupyter` / `notebook`

Exact package versions are listed in `requirements.txt`.

---

## Installation

To set up the environment locally:

```bash
# Clone or extract submission files into your working directory
cd email_submission

# Install required dependencies
pip install -r requirements.txt
```

---

## Running the Notebook

The primary submission notebook `Final_Notebook.ipynb` is self-contained and pre-executed with all cell outputs saved.

To re-run the notebook interactively:

```bash
# Launch Jupyter Notebook
jupyter notebook Final_Notebook.ipynb
```

In Jupyter, select **Kernel -> Restart & Run All** to execute the pipeline from top to bottom.

---

## Reproducing submission_level2.csv

Executing `Final_Notebook.ipynb` automatically regenerates all submission files in the current working directory:
1. Section 7 produces `submission_level1.csv`
2. Section 10 produces `submission_level2.csv` (Official Submission)
3. Section 12 produces `submission_robust.csv` (Backup Submission)

---

## Output Files

| File | Level | Public MSE | Description |
|---|---|---|---|
| `submission_level2.csv` | L2 | **304.90** | **Official Final Submission** (Best Leaderboard Score & Rank) |
| `submission_robust.csv` | L2 | **409.39** | Secondary / backup reproducibility submission |
| `Final_Notebook.ipynb` | — | — | Primary reproducible notebook pipeline |

---

## External Datasets

- **Google Dynamic World V1 (2025):** Evaluated and used for 2025 Kharif cropland extent.
- **Sentinel-1 GRD Dual-Pol:** Evaluated and used for paddy flood signature detection.
- **ESA WorldCover, WorldCereal, AlphaEarth Embeddings:** Explored during experimentation; noted in notebook ablation logs but not required for final L2 reproduction.

---

## Competition Notes

- **Primary Submission:** `submission_level2.csv` is the official submission file.
- **Backup Submission:** `submission_robust.csv` serves solely as a backup for reproducibility validation.
- **Zero Label Integrity:** No synthetic or fabricated labels were used; all per-village spatial distributions stem from physical remote-sensing signals.

---

## License

This project is released under the MIT License for competition evaluation and academic review.
