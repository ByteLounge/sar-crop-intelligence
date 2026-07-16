import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import pickle
import cv2

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.preprocess import align_rasters
from features.extract import extract_geometry_features, extract_sar_features

def run_inference():
    print("Running inference from serialized checkpoints...")
    
    workspace_dir = r"D:\PC\resources"
    shp_path = os.path.join(workspace_dir, "villages_clean", "villages_clean.shp")
    gdf = gpd.read_file(shp_path)
    gdf_utm = gdf.to_crs("EPSG:32643")
    
    df_geom = extract_geometry_features(gdf_utm)
    
    aligned_dir = os.path.join(workspace_dir, "aligned_images")
    dates = ["20250606", "20250619", "20250814", "20251013"]
    tif_paths = [os.path.join(aligned_dir, f"capella_hh_{d}_10m.tif") for d in dates]
    
    images = []
    for p in tif_paths:
        with rasterio.open(p) as src:
            images.append(src.read(1))
            meta = src.meta.copy()
            
    stack = np.stack(images, axis=0)
    H, W = stack.shape[1], stack.shape[2]
    
    shapes = [(row['geometry'], row['ID']) for idx, row in gdf_utm.iterrows()]
    village_mask = rasterize(
        shapes,
        out_shape=(H, W),
        transform=meta['transform'],
        fill=0,
        all_touched=True,
        dtype='int32'
    )
    
    flat_stack = stack.reshape(4, -1).T.astype(float)
    flat_mask = village_mask.flatten()
    
    df_sar = extract_sar_features(gdf_utm, flat_stack, flat_mask, meta['transform'], H, W, dates)
    
    # Target label extraction is only needed to obtain observed crop fractions for the covered portions
    in_village = flat_mask > 0
    X_village = flat_stack[in_village]
    village_ids = flat_mask[in_village]
    
    mean_vals = X_village.mean(axis=1)
    min_vals = X_village.min(axis=1)
    max_vals = X_village.max(axis=1)
    is_water = (mean_vals < 20) & (max_vals < 40)
    is_builtup = (mean_vals > 160) & (min_vals > 80)
    is_veg = ~is_water & ~is_builtup
    
    X_veg = X_village[is_veg]
    X_veg_mean = X_veg.mean(axis=1, keepdims=True)
    X_veg_std = X_veg.std(axis=1, keepdims=True) + 1e-5
    X_veg_norm = (X_veg - X_veg_mean) / X_veg_std
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_veg_norm)
    
    crop_mapping = {
        0: 'Cotton_frac',
        1: 'Groundnut_frac',
        2: 'Maize_frac',
        3: 'Rice_frac',
        4: 'Bajra_frac'
    }
    
    pixel_crops = np.full(len(X_village), -1, dtype=int)
    pixel_crops[is_veg] = labels
    
    labels_list = []
    for idx, row in gdf_utm.iterrows():
        v_id = row['ID']
        v_pixels = (village_ids == v_id)
        v_crop_pixels = pixel_crops[v_pixels]
        X_v = X_village[v_pixels]
        is_nodata = (X_v == 0).all(axis=1)
        n_valid = np.sum(~is_nodata)
        
        crop_fracs = {'ID': v_id}
        for c_id, crop_name in crop_mapping.items():
            if n_valid > 0:
                count = np.sum(v_crop_pixels == c_id)
                crop_fracs[crop_name] = count / n_valid
            else:
                crop_fracs[crop_name] = 0.0
        labels_list.append(crop_fracs)
        
    df_labels = pd.DataFrame(labels_list)
    df_data = pd.merge(df_geom, df_sar, on='ID')
    df_data = pd.merge(df_data, df_labels, on='ID')
    
    total_px = [{'ID': row['ID'], 'total_pixels': np.sum(flat_mask == row['ID'])} for idx, row in gdf_utm.iterrows()]
    df_total_px = pd.DataFrame(total_px)
    df_data = pd.merge(df_data, df_total_px, on='ID')
    df_data['coverage'] = df_data['valid_pixels'] / df_data['total_pixels']
    
    # Load Serialized Objects
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    
    with open(os.path.join(models_dir, "imputer_knn.pkl"), "rb") as f:
        imputer_knn = pickle.load(f)
    with open(os.path.join(models_dir, "nn_spatial.pkl"), "rb") as f:
        nn = pickle.load(f)
    with open(os.path.join(models_dir, "selected_features.pkl"), "rb") as f:
        selected_features = pickle.load(f)
        
    geom_cols = ['centroid_x', 'centroid_y', 'area_ha', 'perimeter', 'compactness', 'bbox_width', 'bbox_height']
    sar_cols = [c for c in df_sar.columns if c not in ['ID', 'valid_pixels']]
    all_cols = geom_cols + sar_cols
    
    # Impute KNN (for Rice, Cotton, Maize)
    df_final_knn = df_data.copy()
    df_final_knn[all_cols] = imputer_knn.transform(df_data[all_cols])
    
    # Impute Spatial 1-NN (for Bajra, Groundnut)
    df_final_spatial = df_data.copy()
    train_indices = df_data[df_data['coverage'] > 0.35].index
    zero_cov_indices = df_data[df_data['coverage'] <= 0.35].index
    for idx in zero_cov_indices:
        coord = df_data.loc[idx, ['centroid_x', 'centroid_y']].values.reshape(1, -1)
        neighbor_idx = train_indices[nn.kneighbors(coord, return_distance=False)[0][0]]
        df_final_spatial.loc[idx, sar_cols] = df_data.loc[neighbor_idx, sar_cols]
        
    target_cols = ['Rice_frac', 'Cotton_frac', 'Maize_frac', 'Bajra_frac', 'Groundnut_frac']
    crop_names_ha = ['Rice_ha', 'Cotton_ha', 'Maize_ha', 'Bajra_ha', 'Groundnut_ha']
    
    final_predictions = {}
    for target in target_cols:
        features_t = selected_features[target]
        
        if target in ['Rice_frac', 'Cotton_frac', 'Maize_frac']:
            df_all = df_final_knn
        else:
            df_all = df_final_spatial
            
        X_all_t = df_all[features_t].values
        
        # Load serialized ensemble checkpoint
        with open(os.path.join(models_dir, f"ensemble_{target}.pkl"), "rb") as f:
            ensemble = pickle.load(f)
            
        preds = ensemble.predict(X_all_t)
        preds = np.clip(preds, 0.0, 1.0)
        final_predictions[target] = preds
        
    # Calibrate predictions using estimated cultivated land area (added to fix the 5x overestimation flaw)
    df_final = df_final_knn.copy()
    cov = df_final['coverage'].values
    
    # 8a. Build the cultivated land mask from the stack (swath-covered areas)
    mean_img = np.mean(stack, axis=0)
    max_img = np.max(stack, axis=0)
    var_img = np.var(stack, axis=0)
    
    # Local std of the temporal mean (texture filter)
    local_mean = cv2.boxFilter(mean_img, -1, (3, 3))
    local_sq_mean = cv2.boxFilter(mean_img**2, -1, (3, 3))
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
    
    # Local Shannon entropy of temporal mean
    mean_uint8 = np.clip(mean_img, 0, 255).astype(np.uint8)
    from skimage.filters.rank import entropy
    from skimage.morphology import disk
    entropy_img = entropy(mean_uint8, disk(3))
    
    # Swath valid mask
    valid_sar = (stack > 0).any(axis=0)
    
    is_water = (mean_img < 25) & (max_img < 45)
    is_builtup = (mean_img > 130) | (local_std > 12) | (entropy_img > 4.5)
    potential_cropland = valid_sar & (~is_water) & (~is_builtup)
    
    # Variance threshold of 30.0 isolates active agricultural fields
    is_active = (var_img >= 30)
    cultivated_raw = potential_cropland & is_active
    
    # Morphological processing
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(cultivated_raw.astype(np.uint8), cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
    
    # Filter out small connected components (noise)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    min_size = 5
    mask_filtered = np.zeros_like(closed)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            mask_filtered[labels == i] = 1
            
    # Calculate observed cultivated fraction per village
    df_final['Obs_cultivated_frac'] = 0.0
    for idx, row in gdf_utm.iterrows():
        v_id = row['ID']
        v_pixels = (village_mask == v_id)
        n_cultivated = np.sum(v_pixels & (mask_filtered > 0))
        area_ha = df_final.loc[df_final['ID'] == v_id, 'area_ha'].values[0]
        df_final.loc[df_final['ID'] == v_id, 'Obs_cultivated_frac'] = (n_cultivated * 0.01) / area_ha if area_ha > 0 else 0.0
        
    # Load serialized cultivated KNN model
    with open(os.path.join(models_dir, "cultivated_knn.pkl"), "rb") as f:
        knn_cult = pickle.load(f)
        
    geom_features = ['centroid_x', 'centroid_y', 'area_ha', 'perimeter', 'compactness', 'bbox_width', 'bbox_height']
    X_all_c = df_final[geom_features].values
    
    # Predict cultivated fraction
    df_final['Pred_cultivated_frac'] = knn_cult.predict(X_all_c)
    
    # Blend observed mask fractions and spatial model predictions
    estimated_cultivated_frac = cov * df_final['Obs_cultivated_frac'].values + (1.0 - cov) * df_final['Pred_cultivated_frac'].values
    
    # Blend crop fractions from models
    blended_fracs = {}
    for target in target_cols:
        obs_val = df_final[target].values
        pred_val = final_predictions[target]
        blended_fracs[target] = cov * obs_val + (1.0 - cov) * pred_val
        
    sum_blended = np.zeros(len(df_final))
    for target in target_cols:
        sum_blended += blended_fracs[target]
        
    # Apply calibrated post-processing normalization to the crop area predictions
    for target, ha_name in zip(target_cols, crop_names_ha):
        norm_frac = np.where(sum_blended > 0, blended_fracs[target] * estimated_cultivated_frac / sum_blended, 0.0)
        df_final[ha_name] = norm_frac * df_final['area_ha']
        
    df_sub = df_final[['ID'] + crop_names_ha].sort_values('ID').reset_index(drop=True)
    
    # Output calibrated submission to submission_updated.csv
    sub_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'submission_updated.csv'))
    df_sub.to_csv(sub_path, index=False)
    print(f"Regenerated calibrated submission written to: {sub_path}")

if __name__ == '__main__':
    run_inference()
