import numpy as np
import pandas as pd
import geopandas as gpd
import cv2
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import warnings
warnings.filterwarnings('ignore')

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

def compute_glcm_texture(image_data, mask):
    # Scale valid pixels to 0-7 integers
    valid_vals = image_data[mask]
    if len(valid_vals) == 0:
        return 0, 0, 0, 0, 0
    
    # Normalize to 0-7 range
    min_v, max_v = valid_vals.min(), valid_vals.max()
    if max_v > min_v:
        scaled = np.clip(((image_data - min_v) / (max_v - min_v) * 8).astype(int), 0, 7)
    else:
        scaled = np.zeros_like(image_data, dtype=int)
        
    scaled[~mask] = 0
    
    glcm = graycomatrix(scaled, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=8, symmetric=True, normed=True)
    
    contrast = float(graycoprops(glcm, 'contrast').mean())
    homogeneity = float(graycoprops(glcm, 'homogeneity').mean())
    asm = float(graycoprops(glcm, 'ASM').mean())
    energy = float(graycoprops(glcm, 'energy').mean())
    
    # Entropy calculation
    glcm_flat = glcm.flatten()
    glcm_flat = glcm_flat[glcm_flat > 0]
    entropy = float(-np.sum(glcm_flat * np.log2(glcm_flat)))
    
    return contrast, homogeneity, entropy, energy, asm

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
    local_var_images = []
    sobel_images = []
    laplacian_images = []
    lbp_images = []
    edge_density_images = []
    
    stack_3d = flat_stack.T.reshape(len(dates), H, W)
    for d_idx in range(len(dates)):
        img = stack_3d[d_idx].astype(float)
        
        # Local std & local var
        local_mean = cv2.boxFilter(img, -1, (3, 3))
        local_sq_mean = cv2.boxFilter(img**2, -1, (3, 3))
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
        local_std_images.append(local_std)
        local_var_images.append(local_std**2)
        
        # Sobel spatial gradient magnitude
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        sobel_images.append(grad_mag)
        
        # Laplacian (texture/edge roughness)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        laplacian_images.append(laplacian)
        
        # Edge density using Canny edge detector
        # Capella DN is typically uint8 in range [0, 255]
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        edges = cv2.Canny(img_uint8, 30, 100)
        edge_density_images.append((edges > 0).astype(float))
        
        # Local Binary Pattern (LBP)
        lbp = local_binary_pattern(img_uint8, P=8, R=1, method='uniform')
        lbp_images.append(lbp)
        
    local_std_stack = np.stack(local_std_images, axis=0)
    flat_std_stack = local_std_stack.reshape(len(dates), -1).T
    
    local_var_stack = np.stack(local_var_images, axis=0)
    flat_var_stack = local_var_stack.reshape(len(dates), -1).T
    
    sobel_stack = np.stack(sobel_images, axis=0)
    flat_sobel_stack = sobel_stack.reshape(len(dates), -1).T
    
    laplacian_stack = np.stack(laplacian_images, axis=0)
    flat_laplacian_stack = laplacian_stack.reshape(len(dates), -1).T
    
    edge_density_stack = np.stack(edge_density_images, axis=0)
    flat_edge_density_stack = edge_density_stack.reshape(len(dates), -1).T
    
    lbp_stack = np.stack(lbp_images, axis=0)
    flat_lbp_stack = lbp_stack.reshape(len(dates), -1).T
    
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
        X_var_v = flat_var_stack[v_pixels]
        X_sobel_v = flat_sobel_stack[v_pixels]
        X_laplacian_v = flat_laplacian_stack[v_pixels]
        X_edge_density_v = flat_edge_density_stack[v_pixels]
        X_lbp_v = flat_lbp_stack[v_pixels]
        
        is_nodata = (X_v == 0).all(axis=1)
        n_valid = np.sum(~is_nodata)
        
        v_sar = {'ID': v_id, 'valid_pixels': n_valid}
        
        if n_valid > 0:
            X_valid = X_v[~is_nodata]
            X_std_valid = X_std_v[~is_nodata]
            X_var_valid = X_var_v[~is_nodata]
            X_sobel_valid = X_sobel_v[~is_nodata]
            X_laplacian_valid = X_laplacian_v[~is_nodata]
            X_edge_density_valid = X_edge_density_v[~is_nodata]
            X_lbp_valid = X_lbp_v[~is_nodata]
            
            # Statistics per date
            for d_idx, d in enumerate(dates):
                vals = X_valid[:, d_idx]
                v_sar[f'mean_{d}'] = np.mean(vals)
                v_sar[f'std_{d}'] = np.std(vals)
                v_sar[f'min_{d}'] = np.min(vals)
                v_sar[f'max_{d}'] = np.max(vals)
                v_sar[f'cv_{d}'] = np.std(vals) / (np.mean(vals) + 1e-5)
                v_sar[f'skew_{d}'] = skew(vals)
                v_sar[f'kurt_{d}'] = kurtosis(vals)
                v_sar[f'p10_{d}'] = np.percentile(vals, 10)
                v_sar[f'p25_{d}'] = np.percentile(vals, 25)
                v_sar[f'p50_{d}'] = np.percentile(vals, 50)
                v_sar[f'p75_{d}'] = np.percentile(vals, 75)
                v_sar[f'p90_{d}'] = np.percentile(vals, 90)
                v_sar[f'mean_local_std_{d}'] = np.mean(X_std_valid[:, d_idx])
                v_sar[f'local_variance_{d}'] = np.mean(X_var_valid[:, d_idx])
                v_sar[f'mean_sobel_{d}'] = np.mean(X_sobel_valid[:, d_idx])
                v_sar[f'laplacian_var_{d}'] = np.var(X_laplacian_valid[:, d_idx])
                v_sar[f'edge_density_{d}'] = np.mean(X_edge_density_valid[:, d_idx])
                v_sar[f'lbp_mean_{d}'] = np.mean(X_lbp_valid[:, d_idx])
                v_sar[f'iqr_{d}'] = v_sar[f'p75_{d}'] - v_sar[f'p25_{d}']
                v_sar[f'entropy_{d}'] = calc_entropy(vals)
                
                # GLCM texture on Capella image for this date
                # We can construct the bounding box grid of the village, but for speed,
                # we can compute GLCM on the 2D slice directly
                img_slice = stack_3d[d_idx]
                v_mask_2d = (flat_mask.reshape(H, W) == v_id)
                contrast_c, homogeneity_c, entropy_c, energy_c, asm_c = compute_glcm_texture(img_slice, v_mask_2d)
                v_sar[f'glcm_contrast_{d}'] = contrast_c
                v_sar[f'glcm_homogeneity_{d}'] = homogeneity_c
                v_sar[f'glcm_entropy_{d}'] = entropy_c
                v_sar[f'glcm_energy_{d}'] = energy_c
                v_sar[f'glcm_asm_{d}'] = asm_c
                
            # Temporal difference features
            v_sar['diff_sowing'] = v_sar['mean_20250619'] - v_sar['mean_20250606']
            v_sar['diff_veg'] = v_sar['mean_20250814'] - v_sar['mean_20250619']
            v_sar['diff_harvest'] = v_sar['mean_20251013'] - v_sar['mean_20250814']
            
            # Second differences
            v_sar['sec_diff_1'] = v_sar['diff_veg'] - v_sar['diff_sowing']
            v_sar['sec_diff_2'] = v_sar['diff_harvest'] - v_sar['diff_veg']
            
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
                v_sar[f'min_{d}'] = np.nan
                v_sar[f'max_{d}'] = np.nan
                v_sar[f'cv_{d}'] = np.nan
                v_sar[f'skew_{d}'] = np.nan
                v_sar[f'kurt_{d}'] = np.nan
                v_sar[f'p10_{d}'] = np.nan
                v_sar[f'p25_{d}'] = np.nan
                v_sar[f'p50_{d}'] = np.nan
                v_sar[f'p75_{d}'] = np.nan
                v_sar[f'p90_{d}'] = np.nan
                v_sar[f'mean_local_std_{d}'] = np.nan
                v_sar[f'local_variance_{d}'] = np.nan
                v_sar[f'mean_sobel_{d}'] = np.nan
                v_sar[f'laplacian_var_{d}'] = np.nan
                v_sar[f'edge_density_{d}'] = np.nan
                v_sar[f'lbp_mean_{d}'] = np.nan
                v_sar[f'iqr_{d}'] = np.nan
                v_sar[f'entropy_{d}'] = np.nan
                
                v_sar[f'glcm_contrast_{d}'] = np.nan
                v_sar[f'glcm_homogeneity_{d}'] = np.nan
                v_sar[f'glcm_entropy_{d}'] = np.nan
                v_sar[f'glcm_energy_{d}'] = np.nan
                v_sar[f'glcm_asm_{d}'] = np.nan
                
            v_sar['diff_sowing'] = np.nan
            v_sar['diff_veg'] = np.nan
            v_sar['diff_harvest'] = np.nan
            v_sar['sec_diff_1'] = np.nan
            v_sar['sec_diff_2'] = np.nan
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
