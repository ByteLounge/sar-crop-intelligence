import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import cv2
import pickle
import json
from skimage.filters.rank import entropy as skimage_entropy
from skimage.morphology import disk
from skimage.filters import threshold_otsu
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsRegressor

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.preprocess import align_rasters
from features.extract import extract_geometry_features

def lee_filter(img, size=5, sigma_v=0.25):
    """
    Apply standard Lee filter for speckle reduction in linear power domain.
    """
    img_mean = cv2.boxFilter(img, -1, (size, size))
    img_sqr_mean = cv2.boxFilter(img**2, -1, (size, size))
    img_variance = np.maximum(img_sqr_mean - img_mean**2, 0)
    
    # Noise variance based on relative standard deviation of speckle (sigma_v)
    noise_variance = (img_mean * sigma_v)**2
    
    # Calculate weight K
    img_weights = np.maximum(0.0, (img_variance - noise_variance) / (img_variance + 1e-10))
    img_weights = np.minimum(img_weights, 1.0)
    
    # Filter image
    img_filtered = img_mean + img_weights * (img - img_mean)
    return img_filtered

def run_inference():
    print("Running inference from image-driven GMM models...")
    
    workspace_dir = r"D:\PC\resources"
    shp_path = os.path.join(workspace_dir, "villages_clean", "villages_clean.shp")
    gdf = gpd.read_file(shp_path)
    gdf_utm = gdf.to_crs("EPSG:32643")
    
    aligned_dir = os.path.join(workspace_dir, "aligned_images")
    dates = ["20250606", "20250619", "20250814", "20251013"]
    tif_paths = [os.path.join(aligned_dir, f"capella_hh_{d}_10m.tif") for d in dates]
    
    # Load and preprocess stack
    images = []
    for date, path in zip(dates, tif_paths):
        with rasterio.open(path) as src:
            dn_data = src.read(1)
            meta = src.meta.copy()
            
        valid_mask = dn_data > 0
        linear_data = np.zeros_like(dn_data, dtype=float)
        linear_data[valid_mask] = 10.0 ** (dn_data[valid_mask] / 50.0)
        
        filtered_linear = lee_filter(linear_data, size=5, sigma_v=0.25)
        
        db_data = np.zeros_like(dn_data, dtype=np.uint8)
        db_data[valid_mask] = np.clip(50.0 * np.log10(np.maximum(filtered_linear[valid_mask], 1e-5)), 0, 255).astype(np.uint8)
        images.append(db_data)
        
    stack = np.stack(images, axis=0)
    H, W = stack.shape[1], stack.shape[2]
    
    # Create village mask and valid SAR swath mask
    shapes = [(row['geometry'], row['ID']) for idx, row in gdf_utm.iterrows()]
    village_mask = rasterize(
        shapes,
        out_shape=(H, W),
        transform=meta['transform'],
        fill=0,
        all_touched=True,
        dtype='int32'
    )
    valid_sar = (stack > 0).any(axis=0)
    
    # Load combined cultivated mask
    with rasterio.open(os.path.join(workspace_dir, "cultivated_mask.tif")) as src:
        combined_mask = src.read(1)
        
    # Extract Feature Stack
    diff_1 = np.zeros((H, W))
    diff_2 = np.zeros((H, W))
    diff_3 = np.zeros((H, W))
    valid_all = (stack > 0).all(axis=0)
    diff_1[valid_all] = stack[1, valid_all].astype(float) - stack[0, valid_all].astype(float)
    diff_2[valid_all] = stack[2, valid_all].astype(float) - stack[1, valid_all].astype(float)
    diff_3[valid_all] = stack[3, valid_all].astype(float) - stack[2, valid_all].astype(float)
    
    days = np.array([0, 13, 69, 129])
    mean_days = days.mean()
    denom = np.sum((days - mean_days)**2)
    slope = np.zeros((H, W))
    for idx in range(4):
        slope += (days[idx] - mean_days) * stack[idx].astype(float)
    slope /= denom
    
    peak_val = np.max(stack, axis=0)
    min_val = np.min(stack, axis=0)
    max_val = np.max(stack, axis=0)
    amplitude = max_val - min_val
    temp_var = np.var(stack, axis=0)
    
    stack_linear = np.zeros_like(stack, dtype=float)
    for i in range(4):
        valid_m = stack[i] > 0
        stack_linear[i, valid_m] = 10.0 ** (stack[i, valid_m] / 50.0)
    sum_linear = np.sum(stack_linear, axis=0) + 1e-10
    p_temp = stack_linear / sum_linear
    temp_entropy = -np.sum(p_temp * np.log2(p_temp + 1e-10), axis=0)
    
    # GLCM (7)
    mean_img = np.mean(stack, axis=0)
    num_levels = 8
    img_min, img_max = float(mean_img.min()), float(mean_img.max())
    mean_img_quant = np.clip(((mean_img - img_min) / (img_max - img_min + 1e-5) * num_levels).astype(int), 0, num_levels - 1)
    shifted = np.roll(np.roll(mean_img_quant, 1, axis=0), 1, axis=1)
    mean_1 = cv2.boxFilter(mean_img_quant.astype(float), -1, (5, 5))
    mean_2 = cv2.boxFilter(shifted.astype(float), -1, (5, 5))
    var_1 = cv2.boxFilter(mean_img_quant.astype(float)**2, -1, (5, 5)) - mean_1**2
    var_2 = cv2.boxFilter(shifted.astype(float)**2, -1, (5, 5)) - mean_2**2
    std_1 = np.sqrt(np.maximum(var_1, 1e-5))
    std_2 = np.sqrt(np.maximum(var_2, 1e-5))
    
    asm = np.zeros((H, W))
    entropy = np.zeros((H, W))
    contrast = np.zeros((H, W))
    dissimilarity = np.zeros((H, W))
    homogeneity = np.zeros((H, W))
    correlation = np.zeros((H, W))
    for g1 in range(num_levels):
        mask1 = (mean_img_quant == g1)
        for g2 in range(num_levels):
            mask2 = (shifted == g2)
            pair_mask = mask1 & mask2
            p = cv2.boxFilter(pair_mask.astype(float), -1, (5, 5))
            asm += p ** 2
            p_safe = np.maximum(p, 1e-10)
            entropy -= p_safe * np.log2(p_safe)
            contrast += (g1 - g2) ** 2 * p
            dissimilarity += np.abs(g1 - g2) * p
            homogeneity += p / (1.0 + (g1 - g2) ** 2)
            correlation += (g1 - mean_1) * (g2 - mean_2) * p / (std_1 * std_2)
    energy = np.sqrt(asm)
    
    # Neighborhood (5)
    mean_3x3 = cv2.boxFilter(mean_img.astype(float), -1, (3, 3))
    mean_5x5 = cv2.boxFilter(mean_img.astype(float), -1, (5, 5))
    local_sq_mean = cv2.boxFilter(mean_img.astype(float)**2, -1, (5, 5))
    local_std = np.sqrt(np.maximum(local_sq_mean - mean_5x5**2, 0))
    grad_x = cv2.Sobel(mean_img.astype(float), cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(mean_img.astype(float), cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    edges = cv2.Canny(mean_img.astype(np.uint8), 30, 100)
    edge_density = cv2.boxFilter((edges > 0).astype(float), -1, (7, 7))
    
    # Morphology (4)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    dist_transform = cv2.distanceTransform(combined_mask.astype(np.uint8), cv2.DIST_L2, 5)
    village_boundary_mask = (village_mask > 0).astype(np.uint8)
    boundaries = cv2.Canny(village_boundary_mask, 0.5, 1.5)
    boundary_dist = cv2.distanceTransform(255 - boundaries, cv2.DIST_L2, 5)
    
    feature_layers = [
        stack[0], stack[1], stack[2], stack[3],
        diff_1, diff_2, diff_3, slope, peak_val, min_val, max_val, amplitude, temp_var, temp_entropy,
        contrast, correlation, energy, homogeneity, entropy, dissimilarity, asm,
        mean_3x3, mean_5x5, local_std, grad_mag, edge_density,
        opening, closing, dist_transform, boundary_dist
    ]
    feature_names = [
        'raw_0', 'raw_1', 'raw_2', 'raw_3',
        'diff_1', 'diff_2', 'diff_3', 'slope', 'peak_val', 'min_val', 'max_val', 'amplitude', 'temp_var', 'temp_entropy',
        'glcm_contrast', 'glcm_correlation', 'glcm_energy', 'glcm_homogeneity', 'glcm_entropy', 'glcm_dissimilarity', 'glcm_asm',
        'mean_3x3', 'mean_5x5', 'local_std', 'grad_mag', 'edge_density',
        'opening', 'closing', 'dist_transform', 'boundary_dist'
    ]
    
    in_villages = (village_mask > 0) & valid_sar
    pixel_indices = np.where(in_villages)
    N_pixels = len(pixel_indices[0])
    
    feature_matrix = np.zeros((N_pixels, len(feature_layers)), dtype=np.float32)
    for idx, layer in enumerate(feature_layers):
        feature_matrix[:, idx] = layer[in_villages]
        
    # Load Models
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    with open(os.path.join(models_dir, "cluster_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(models_dir, "gmm_crop_model.pkl"), "rb") as f:
        gmm_model = pickle.load(f)
    with open(os.path.join(models_dir, "scaler_geom.pkl"), "rb") as f:
        scaler_geom = pickle.load(f)
    with open(os.path.join(models_dir, "cultivated_knn.pkl"), "rb") as f:
        model_cult = pickle.load(f)
    with open(os.path.join(models_dir, "spatial_crops_knn.pkl"), "rb") as f:
        model_crops = pickle.load(f)
        
    # GMM clustering predictions
    cult_indices = np.where(combined_mask[in_villages] > 0)[0]
    X_cult = feature_matrix[cult_indices]
    features_to_use = [
        'raw_0', 'raw_1', 'raw_2', 'raw_3',
        'diff_1', 'diff_2', 'diff_3', 'slope', 'amplitude', 'temp_var',
        'glcm_contrast', 'glcm_homogeneity', 'local_std', 'grad_mag'
    ]
    indices_to_use = [feature_names.index(f) for f in features_to_use]
    X_cluster = X_cult[:, indices_to_use].astype(np.float64)
    X_scaled = scaler.transform(X_cluster)
    labels_all = gmm_model.predict(X_scaled)
    
    pixel_village_ids_cult = village_mask[in_villages][cult_indices]
    crop_mapping = {
        3: 'Rice_ha',
        0: 'Cotton_ha',
        4: 'Maize_ha',
        2: 'Bajra_ha',
        1: 'Groundnut_ha'
    }
    crop_cols = ['Rice_ha', 'Cotton_ha', 'Maize_ha', 'Bajra_ha', 'Groundnut_ha']
    
    village_data = []
    for idx, row in gdf_utm.iterrows():
        v_id = row['ID']
        geom = row['geometry']
        v_pixels = (village_mask == v_id)
        n_total_px = np.sum(v_pixels)
        n_valid_px = np.sum(v_pixels & valid_sar)
        coverage = n_valid_px / n_total_px if n_total_px > 0 else 0.0
        
        centroid = geom.centroid
        bounds = geom.bounds
        compactness = 4 * np.pi * geom.area / (geom.length ** 2) if geom.length > 0 else 0.0
        
        record = {
            'ID': v_id,
            'VILLAGE': row['VILLAGE'],
            'area_ha': geom.area / 10000.0,
            'centroid_x': centroid.x,
            'centroid_y': centroid.y,
            'perimeter': geom.length,
            'compactness': compactness,
            'bbox_width': bounds[2] - bounds[0],
            'bbox_height': bounds[3] - bounds[1],
            'coverage': coverage,
            'valid_pixels': n_valid_px,
            'total_pixels': n_total_px
        }
        n_cultivated = np.sum(v_pixels & (combined_mask > 0))
        record['cultivated_mask_ha'] = n_cultivated * 0.01
        
        v_cult_labels = labels_all[(pixel_village_ids_cult == v_id)]
        for c_id, c_name in crop_mapping.items():
            record[f'pixel_{c_name}'] = np.sum(v_cult_labels == c_id) * 0.01
        village_data.append(record)
        
    df_villages = pd.DataFrame(village_data)
    
    covered_df = df_villages[df_villages['coverage'] > 0.35].copy().reset_index(drop=True)
    zero_df = df_villages[df_villages['coverage'] <= 0.35].copy().reset_index(drop=True)
    
    geom_features = ['centroid_x', 'centroid_y', 'area_ha', 'perimeter', 'compactness', 'bbox_width', 'bbox_height']
    X_zero_geom = scaler_geom.transform(zero_df[geom_features])
    zero_df['pred_cultivated_frac'] = np.clip(model_cult.predict(X_zero_geom), 0.0, 1.0)
    zero_df['pred_cultivated_ha'] = zero_df['pred_cultivated_frac'] * zero_df['area_ha']
    
    pred_crop_fracs = np.clip(model_crops.predict(X_zero_geom), 0.0, 1.0)
    sum_pred_fracs = np.sum(pred_crop_fracs, axis=1, keepdims=True)
    pred_crop_fracs = np.where(sum_pred_fracs > 0, pred_crop_fracs / sum_pred_fracs, 0.2)
    for idx, c_name in enumerate(crop_cols):
        zero_df[f'pred_{c_name}'] = pred_crop_fracs[:, idx] * zero_df['pred_cultivated_ha']
        
    final_rows = []
    for idx, row in df_villages.iterrows():
        v_id = row['ID']
        final_record = {'ID': v_id, 'VILLAGE': row['VILLAGE']}
        if row['coverage'] > 0.35:
            final_record['cultivated_area_ha'] = row['cultivated_mask_ha']
            for c_name in crop_cols:
                final_record[c_name] = row[f'pixel_{c_name}']
        else:
            zero_row = zero_df[zero_df['ID'] == v_id].iloc[0]
            final_record['cultivated_area_ha'] = zero_row['pred_cultivated_ha']
            for c_name in crop_cols:
                final_record[c_name] = zero_row[f'pred_{c_name}']
        final_rows.append(final_record)
        
    df_final = pd.DataFrame(final_rows)
    df_sub = df_final[['ID'] + crop_cols].sort_values('ID').reset_index(drop=True)
    
    # Save both submission_final.csv and submission_generated.csv
    for name in ["submission_final.csv", "submission_generated.csv"]:
        out_sub_root = os.path.join(workspace_dir, name)
        out_sub_proj = os.path.join(workspace_dir, "project", name)
        df_sub.to_csv(out_sub_root, index=False)
        df_sub.to_csv(out_sub_proj, index=False)
        print(f"Inference successfully written to: {out_sub_root}")

if __name__ == '__main__':
    run_inference()
