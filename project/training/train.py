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

def run_pipeline():
    print("Starting image-driven crop mapping pipeline...")
    
    workspace_dir = r"D:\PC\resources"
    shp_path = os.path.join(workspace_dir, "villages_clean", "villages_clean.shp")
    gdf = gpd.read_file(shp_path)
    gdf_utm = gdf.to_crs("EPSG:32643")
    
    # 1. Align images if needed
    aligned_dir = os.path.join(workspace_dir, "aligned_images")
    dates = ["20250606", "20250619", "20250814", "20251013"]
    tif_paths = [os.path.join(aligned_dir, f"capella_hh_{d}_10m.tif") for d in dates]
    
    if not all(os.path.exists(p) for p in tif_paths):
        raw_tifs = sorted(glob.glob(os.path.join(workspace_dir, "CAPELLA_*", "*_preview.tif")))
        align_rasters(raw_tifs, gdf_utm, aligned_dir)
        
    # 2. SAR Preprocessing (linear scale conversion & Lee despeckling)
    # Why: Direct dB averaging is mathematically incorrect. Despeckling is required to suppress coherent noise.
    images = []
    preprocessed_dir = os.path.join(workspace_dir, "preprocessed_images")
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    for date, path in zip(dates, tif_paths):
        with rasterio.open(path) as src:
            dn_data = src.read(1)
            meta = src.meta.copy()
            
        valid_mask = dn_data > 0
        linear_data = np.zeros_like(dn_data, dtype=float)
        linear_data[valid_mask] = 10.0 ** (dn_data[valid_mask] / 50.0) # Map log preview to linear
        
        filtered_linear = lee_filter(linear_data, size=5, sigma_v=0.25)
        
        db_data = np.zeros_like(dn_data, dtype=np.uint8)
        db_data[valid_mask] = np.clip(50.0 * np.log10(np.maximum(filtered_linear[valid_mask], 1e-5)), 0, 255).astype(np.uint8)
        
        images.append(db_data)
        
    stack = np.stack(images, axis=0)
    H, W = stack.shape[1], stack.shape[2]
    
    # 3. Create spatial village mask and valid SAR swath mask
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
    
    # 4. Cultivated Land Masking (Phase 3)
    # Why: We must isolate agricultural cropland from forests, urban areas, and water bodies before clustering.
    masks = []
    for idx, date in enumerate(dates):
        img = stack[idx]
        local_mean = cv2.boxFilter(img.astype(float), -1, (5, 5))
        local_sq_mean = cv2.boxFilter(img.astype(float)**2, -1, (5, 5))
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
        
        # Local Shannon entropy to detect built-up structures
        img_uint8 = img.astype(np.uint8)
        entropy_img = skimage_entropy(img_uint8, disk(3))
        
        # Gradients & Laplacian
        grad_x = cv2.Sobel(img.astype(float), cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img.astype(float), cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        laplacian = np.abs(cv2.Laplacian(img.astype(float), cv2.CV_64F))
        
        # Vectorized GLCM homogeneity
        num_levels = 8
        img_min, img_max = float(img.min()), float(img.max())
        img_quantized = np.clip(((img.astype(float) - img_min) / (img_max - img_min + 1e-5) * num_levels).astype(int), 0, num_levels - 1)
        shifted = np.roll(np.roll(img_quantized, 1, axis=0), 1, axis=1)
        glcm_homogeneity = np.zeros_like(img, dtype=float)
        for g1 in range(num_levels):
            mask1 = (img_quantized == g1)
            for g2 in range(num_levels):
                mask2 = (shifted == g2)
                pair_mask = mask1 & mask2
                p = cv2.boxFilter(pair_mask.astype(float), -1, (5, 5))
                glcm_homogeneity += p / (1.0 + (g1 - g2) ** 2)
                
        is_water = (img < 25)
        is_builtup = (img > 130) | (local_std > 12.0) | (entropy_img > 4.5) | (grad_mag > 35.0) | (laplacian > 25.0)
        potential_cropland = valid_sar & (~is_water) & (~is_builtup)
        
        # Otsu thresholding on local standard deviation
        local_std_vals = local_std[potential_cropland]
        std_thresh = threshold_otsu(local_std_vals) if len(local_std_vals) > 0 else 8.0
        
        is_smooth = (local_std < std_thresh)
        is_homogeneous = (glcm_homogeneity > 0.4)
        
        cultivated_raw = potential_cropland & is_smooth & is_homogeneous
        
        # Morphological opening/closing and size filtering
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(cultivated_raw.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
        mask_filtered = np.zeros_like(closed)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 5:
                mask_filtered[labels == i] = 1
        masks.append(mask_filtered)
        
    combined_mask = np.any(masks, axis=0)
    
    # Save combined cultivated mask
    out_mask_path = os.path.join(workspace_dir, "cultivated_mask.tif")
    meta.update({'dtype': 'uint8', 'count': 1, 'nodata': 0})
    with rasterio.open(out_mask_path, 'w', **meta) as dst:
        dst.write(combined_mask.astype(np.uint8), 1)
    print(f"Cultivated mask saved to: {out_mask_path}")
    
    # 5. Extract Feature Stack (Phase 4)
    # Why: Rich feature representation enables highly accurate spatial-temporal crop segmentation.
    # Raw (4)
    # Temporal (9)
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
    
    # GLCM (7) on temporal mean
    mean_img = np.mean(stack, axis=0)
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
        
    pixel_features_dir = os.path.join(workspace_dir, "pixel_features")
    os.makedirs(pixel_features_dir, exist_ok=True)
    np.save(os.path.join(pixel_features_dir, "feature_matrix.npy"), feature_matrix)
    np.save(os.path.join(pixel_features_dir, "pixel_y.npy"), pixel_indices[0])
    np.save(os.path.join(pixel_features_dir, "pixel_x.npy"), pixel_indices[1])
    np.save(os.path.join(pixel_features_dir, "pixel_village_ids.npy"), village_mask[in_villages])
    np.save(os.path.join(pixel_features_dir, "pixel_cultivated.npy"), combined_mask[in_villages])
    with open(os.path.join(pixel_features_dir, "feature_names.json"), 'w') as f:
        json.dump(feature_names, f)
        
    # 6. Unsupervised Clustering on Cultivated Pixels (Phase 5)
    cult_indices = np.where(combined_mask[in_villages] > 0)[0]
    X_cult = feature_matrix[cult_indices]
    
    features_to_use = [
        'raw_0', 'raw_1', 'raw_2', 'raw_3',
        'diff_1', 'diff_2', 'diff_3', 'slope', 'amplitude', 'temp_var',
        'glcm_contrast', 'glcm_homogeneity', 'local_std', 'grad_mag'
    ]
    indices_to_use = [feature_names.index(f) for f in features_to_use]
    X_cluster = X_cult[:, indices_to_use].astype(np.float64)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    models_out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(models_out_dir, exist_ok=True)
    with open(os.path.join(models_out_dir, "cluster_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
        
    gmm_final = GaussianMixture(n_components=5, random_state=42, max_iter=200, reg_covar=1e-3)
    gmm_final.fit(X_scaled)
    labels_all = gmm_final.predict(X_scaled)
    
    with open(os.path.join(models_out_dir, "gmm_crop_model.pkl"), "wb") as f:
        pickle.dump(gmm_final, f)
    print("Saved GMM crop model to gmm_crop_model.pkl")
    
    np.save(os.path.join(pixel_features_dir, "pixel_crop_labels.npy"), labels_all)
    
    # 7. Village Aggregation & Spatial Imputation (Phase 7)
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
    
    target_cols = [c.replace('_ha', '_frac') for c in crop_cols]
    for c_name in crop_cols:
        frac_name = c_name.replace('_ha', '_frac')
        covered_df[frac_name] = covered_df[f'pixel_{c_name}'] / (covered_df['cultivated_mask_ha'] + 1e-10)
        
    geom_features = ['centroid_x', 'centroid_y', 'area_ha', 'perimeter', 'compactness', 'bbox_width', 'bbox_height']
    covered_df['cultivated_frac'] = covered_df['cultivated_mask_ha'] / covered_df['area_ha']
    
    scaler_geom = StandardScaler()
    X_train_geom = scaler_geom.fit_transform(covered_df[geom_features])
    
    with open(os.path.join(models_out_dir, "scaler_geom.pkl"), "wb") as f:
        pickle.dump(scaler_geom, f)
        
    model_cult = KNeighborsRegressor(n_neighbors=3, weights='distance')
    model_cult.fit(X_train_geom, covered_df['cultivated_frac'].values)
    with open(os.path.join(models_out_dir, "cultivated_knn.pkl"), "wb") as f:
        pickle.dump(model_cult, f)
        
    model_crops = KNeighborsRegressor(n_neighbors=3, weights='distance')
    model_crops.fit(X_train_geom, covered_df[target_cols].values)
    with open(os.path.join(models_out_dir, "spatial_crops_knn.pkl"), "wb") as f:
        pickle.dump(model_crops, f)
        
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
        print(f"Image-driven submission written to: {out_sub_root}")

if __name__ == '__main__':
    run_pipeline()
