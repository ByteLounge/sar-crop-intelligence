import numpy as np
import pandas as pd
import geopandas as gpd
import cv2
from scipy.stats import skew, kurtosis

def extract_geometry_features(gdf_utm: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Extract geometric, shape, and centroid features for all villages.
    """
    features = []
    for idx, row in gdf_utm.iterrows():
        geom = row['geometry']
        centroid = geom.centroid
        bbox = geom.bounds
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        area = geom.area / 10000.0 # hectares
        perimeter = geom.length
        compactness = (4 * np.pi * geom.area) / (perimeter ** 2) if perimeter > 0 else 0
        
        features.append({
            'ID': row['ID'],
            'VILLAGE': row['VILLAGE'],
            'centroid_x': centroid.x,
            'centroid_y': centroid.y,
            'area_ha': area,
            'perimeter': perimeter,
            'compactness': compactness,
            'bbox_width': w,
            'bbox_height': h
        })
    return pd.DataFrame(features)

def extract_sar_features(
    gdf_utm: gpd.GeoDataFrame,
    flat_stack: np.ndarray,
    flat_mask: np.ndarray,
    meta_transform,
    H: int,
    W: int,
    dates: list
) -> pd.DataFrame:
    """
    Extract temporal backscatter statistics, textures, and growth metrics per village.
    """
    local_std_images = []
    sobel_images = []
    stack_3d = flat_stack.T.reshape(len(dates), H, W)
    for d_idx in range(len(dates)):
        img = stack_3d[d_idx].astype(float)
        
        # Local std (Texture Feature) using box filter
        local_mean = cv2.boxFilter(img, -1, (3, 3))
        local_sq_mean = cv2.boxFilter(img**2, -1, (3, 3))
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
        local_std_images.append(local_std)
        
        # Sobel spatial gradient magnitude
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        sobel_images.append(grad_mag)
        
    local_std_stack = np.stack(local_std_images, axis=0)
    flat_std_stack = local_std_stack.reshape(len(dates), -1).T
    
    sobel_stack = np.stack(sobel_images, axis=0)
    flat_sobel_stack = sobel_stack.reshape(len(dates), -1).T
    
    def calc_entropy(vals):
        hist, _ = np.histogram(vals, bins=10, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    sar_features = []
    for idx, row in gdf_utm.iterrows():
        v_id = row['ID']
        v_pixels = (flat_mask == v_id)
        X_v = flat_stack[v_pixels]
        X_std_v = flat_std_stack[v_pixels]
        X_sobel_v = flat_sobel_stack[v_pixels]
        
        is_nodata = (X_v == 0).all(axis=1)
        n_valid = np.sum(~is_nodata)
        
        v_sar = {'ID': v_id, 'valid_pixels': n_valid}
        
        if n_valid > 0:
            X_valid = X_v[~is_nodata]
            X_std_valid = X_std_v[~is_nodata]
            X_sobel_valid = X_sobel_v[~is_nodata]
            
            # Simple stats per date
            for d_idx, d in enumerate(dates):
                vals = X_valid[:, d_idx]
                v_sar[f'mean_{d}'] = np.mean(vals)
                v_sar[f'std_{d}'] = np.std(vals)
                v_sar[f'cv_{d}'] = np.std(vals) / (np.mean(vals) + 1e-5)
                v_sar[f'skew_{d}'] = skew(vals)
                v_sar[f'kurt_{d}'] = kurtosis(vals)
                v_sar[f'p25_{d}'] = np.percentile(vals, 25)
                v_sar[f'p50_{d}'] = np.percentile(vals, 50)
                v_sar[f'p75_{d}'] = np.percentile(vals, 75)
                v_sar[f'mean_local_std_{d}'] = np.mean(X_std_valid[:, d_idx])
                v_sar[f'mean_sobel_{d}'] = np.mean(X_sobel_valid[:, d_idx])
                v_sar[f'iqr_{d}'] = v_sar[f'p75_{d}'] - v_sar[f'p25_{d}']
                v_sar[f'entropy_{d}'] = calc_entropy(vals)
                
            # Temporal difference features
            v_sar['diff_sowing'] = v_sar['mean_20250619'] - v_sar['mean_20250606']
            v_sar['diff_veg'] = v_sar['mean_20250814'] - v_sar['mean_20250619']
            v_sar['diff_harvest'] = v_sar['mean_20251013'] - v_sar['mean_20250814']
            
            # Temporal ratios
            v_sar['ratio_sowing'] = (v_sar['mean_20250619'] + 1e-5) / (v_sar['mean_20250606'] + 1e-5)
            v_sar['ratio_veg'] = (v_sar['mean_20250814'] + 1e-5) / (v_sar['mean_20250619'] + 1e-5)
            v_sar['ratio_harvest'] = (v_sar['mean_20251013'] + 1e-5) / (v_sar['mean_20250814'] + 1e-5)
            
            # Growth rates and change magnitude
            v_sar['growth_rate'] = (v_sar['mean_20250814'] - v_sar['mean_20250606']) / 2.0
            v_sar['change_magnitude'] = np.mean(np.max(X_valid, axis=1) - np.min(X_valid, axis=1))
            v_sar['cumulative_change'] = np.sum(np.abs(np.diff(X_valid, axis=1)), axis=1).mean()
            v_sar['temporal_variance'] = np.var([v_sar[f'mean_{d}'] for d in dates])
            v_sar['slope'] = (3.0*v_sar['mean_20251013'] + v_sar['mean_20250814'] - v_sar['mean_20250619'] - 3.0*v_sar['mean_20250606']) / 10.0
            
            # Land cover fractions
            mean_vals_v = X_v.mean(axis=1)
            min_vals_v = X_v.min(axis=1)
            max_vals_v = X_v.max(axis=1)
            is_water_v = (mean_vals_v < 20) & (max_vals_v < 40)
            is_builtup_v = (mean_vals_v > 160) & (min_vals_v > 80)
            is_veg_v = ~is_water_v & ~is_builtup_v
            v_sar['water_fraction'] = np.mean(is_water_v)
            v_sar['builtup_fraction'] = np.mean(is_builtup_v)
            v_sar['veg_fraction'] = np.mean(is_veg_v)
            
        else:
            for d in dates:
                v_sar[f'mean_{d}'] = np.nan
                v_sar[f'std_{d}'] = np.nan
                v_sar[f'cv_{d}'] = np.nan
                v_sar[f'skew_{d}'] = np.nan
                v_sar[f'kurt_{d}'] = np.nan
                v_sar[f'p25_{d}'] = np.nan
                v_sar[f'p50_{d}'] = np.nan
                v_sar[f'p75_{d}'] = np.nan
                v_sar[f'mean_local_std_{d}'] = np.nan
                v_sar[f'mean_sobel_{d}'] = np.nan
                v_sar[f'iqr_{d}'] = np.nan
                v_sar[f'entropy_{d}'] = np.nan
            v_sar['diff_sowing'] = np.nan
            v_sar['diff_veg'] = np.nan
            v_sar['diff_harvest'] = np.nan
            v_sar['ratio_sowing'] = np.nan
            v_sar['ratio_veg'] = np.nan
            v_sar['ratio_harvest'] = np.nan
            v_sar['growth_rate'] = np.nan
            v_sar['change_magnitude'] = np.nan
            v_sar['cumulative_change'] = np.nan
            v_sar['temporal_variance'] = np.nan
            v_sar['slope'] = np.nan
            v_sar['water_fraction'] = np.nan
            v_sar['builtup_fraction'] = np.nan
            v_sar['veg_fraction'] = np.nan
            
        sar_features.append(v_sar)
        
    return pd.DataFrame(sar_features)
