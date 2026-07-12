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
    # Compute Local Standard Deviation (Texture Feature) for each date using box filter
    local_std_images = []
    stack_3d = flat_stack.T.reshape(len(dates), H, W)
    for d_idx in range(len(dates)):
        img = stack_3d[d_idx].astype(float)
        local_mean = cv2.boxFilter(img, -1, (3, 3))
        local_sq_mean = cv2.boxFilter(img**2, -1, (3, 3))
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
        local_std_images.append(local_std)
    local_std_stack = np.stack(local_std_images, axis=0)
    flat_std_stack = local_std_stack.reshape(len(dates), -1).T

    sar_features = []
    for idx, row in gdf_utm.iterrows():
        v_id = row['ID']
        v_pixels = (flat_mask == v_id)
        X_v = flat_stack[v_pixels]
        X_std_v = flat_std_stack[v_pixels]
        
        is_nodata = (X_v == 0).all(axis=1)
        n_valid = np.sum(~is_nodata)
        
        v_sar = {'ID': v_id, 'valid_pixels': n_valid}
        
        if n_valid > 0:
            X_valid = X_v[~is_nodata]
            X_std_valid = X_std_v[~is_nodata]
            
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
                
            # Temporal difference features
            v_sar['diff_sowing'] = v_sar['mean_20250619'] - v_sar['mean_20250606']
            v_sar['diff_veg'] = v_sar['mean_20250814'] - v_sar['mean_20250619']
            v_sar['diff_harvest'] = v_sar['mean_20251013'] - v_sar['mean_20250814']
            v_sar['growth_rate'] = (v_sar['mean_20250814'] - v_sar['mean_20250606']) / 2.0
            v_sar['cumulative_change'] = np.sum(np.abs(np.diff(X_valid, axis=1)), axis=1).mean()
            
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
            v_sar['diff_sowing'] = np.nan
            v_sar['diff_veg'] = np.nan
            v_sar['diff_harvest'] = np.nan
            v_sar['growth_rate'] = np.nan
            v_sar['cumulative_change'] = np.nan
            
        sar_features.append(v_sar)
        
    return pd.DataFrame(sar_features)
