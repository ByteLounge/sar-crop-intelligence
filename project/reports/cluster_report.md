# Cluster Report: Physics-Based SAR Crop Assignment
**Vadodara, Gujarat — Capella X-band Multi-temporal Analysis**

This report documents the physical justification and temporal SAR signatures used to map the five unsupervised Gaussian Mixture Model (GMM) clusters to the five target crops: **Rice**, **Cotton**, **Maize**, **Bajra**, and **Groundnut**.

---

## 1. Summary of GMM Cluster Backscatter Profiles

The table below shows the mean backscatter values (in decibels) for each of the 5 clusters across the four acquisition dates:

| Cluster | Count | June 6 (dB) | June 19 (dB) | August 14 (dB) | October 13 (dB) | Assigned Crop |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | 16,370 | 52.97 | 54.67 | 63.01 | 65.23 | **Cotton** |
| **1** | 134,180 | 51.17 | 56.32 | 51.50 | 51.86 | **Groundnut** |
| **2** | 63,364 | 53.88 | 64.86 | 58.37 | 57.33 | **Bajra** |
| **3** | 7,880 | 30.07 | 48.26 | 39.81 | 40.36 | **Rice** |
| **4** | 162,375 | 44.53 | 37.09 | 43.43 | 50.04 | **Maize** |

---

## 2. Physics-Based Justification for Crop Assignments

### 🌾 Rice (Cluster 3)
* **SAR Signature**: Extremely low backscatter (**30.07 dB**) on June 6, followed by a sharp increase to **48.26 dB** on June 19, and staying around **40.0 dB** in August and October.
* **Physical Explanation**: Rice cultivation begins with field flooding (transplanting stage). Water acts as a specular reflector for radar waves, bouncing the signal away from the satellite sensor and resulting in extremely low backscatter. As the rice canopy grows (tillering and vegetative stages), the backscatter increases due to volume scattering from the vertical structure of the rice stems. This flooded signature is a unique, unmistakable physical marker of paddy rice.

### 🌱 Cotton (Cluster 0)
* **SAR Signature**: Moderate backscatter in June (~53.0 dB), increasing steadily to peak at **63.01 dB** in August and **65.23 dB** in October.
* **Physical Explanation**: Cotton is a long-duration crop that develops a dense, complex vegetative canopy with large leaves. As the canopy matures in late monsoon (August) and post-monsoon (October), it generates intense volume scattering and surface-canopy interactions, leading to the highest backscatter values among all crop types.

### 🥜 Groundnut (Cluster 1)
* **SAR Signature**: Very stable backscatter (ranging tightly between **51.17 dB** and **56.32 dB**) throughout the entire season.
* **Physical Explanation**: Groundnut is a low-growing crop with a dense, flat, and uniform canopy that remains close to the ground. This low vertical profile creates a stable surface scattering response that does not change drastically between growth stages, leading to a flat temporal backscatter signature.

### 🌽 Bajra / Pearl Millet (Cluster 2)
* **SAR Signature**: Early peaking on June 19 (**64.86 dB**) and then declining to **58.37 dB** in August and **57.33 dB** in October.
* **Physical Explanation**: Bajra is a short-duration crop that is sown early in the monsoon. It grows rapidly and reaches its peak biomass and height in June/July (producing very high backscatter due to the tall vertical stalks). It is harvested early in the monsoon (late August), causing the backscatter to drop as fields are cleared.

### 🌽 Maize (Cluster 4)
* **SAR Signature**: Low-to-moderate backscatter on June 19 (**37.09 dB**), rising steadily to peak in October (**50.04 dB**).
* **Physical Explanation**: Maize fields are prepared in June (lower backscatter due to bare soil / minimal cover). The crop undergoes vegetative growth throughout August and matures in late September/October, generating increased volume scattering from the large leaves and cobs before harvest.
