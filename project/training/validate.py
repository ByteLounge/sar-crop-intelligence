import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
import pickle

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.preprocess import align_rasters
from features.extract import extract_geometry_features, extract_sar_features
from models.ensemble import OptimizedCropEnsemble

def get_pipeline_data():
    workspace_dir = r"D:\PC\resources"
    shp_path = os.path.join(workspace_dir, "villages_clean", "villages_clean.shp")
    gdf = gpd.read_file(shp_path)
    gdf_utm = gdf.to_crs("EPSG:32643")
    
    # 1. Extract geometry shape features
    df_geom = extract_geometry_features(gdf_utm)
    
    # 2. Align images
    aligned_dir = os.path.join(workspace_dir, "aligned_images")
    dates = ["20250606", "20250619", "20250814", "20251013"]
    tif_paths = [os.path.join(aligned_dir, f"capella_hh_{d}_10m.tif") for d in dates]
    
    # 3. Load aligned rasters
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
    
    # 4. Extract advanced SAR-derived features
    df_sar = extract_sar_features(gdf_utm, flat_stack, flat_mask, meta['transform'], H, W, dates)
    
    # 5. Extract target crop fractions (unsupervised pixel classification)
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
    
    return df_data, df_sar

def evaluate_lovo_cv(df_data, df_sar, selected_features, models_dict=None):
    geom_cols = ['centroid_x', 'centroid_y', 'area_ha', 'perimeter', 'compactness', 'bbox_width', 'bbox_height']
    sar_cols = [c for c in df_sar.columns if c not in ['ID', 'valid_pixels']]
    all_cols = geom_cols + sar_cols
    
    # Identify covered villages for training/validation (coverage > 0.35)
    covered_df = df_data[df_data['coverage'] > 0.35].copy().reset_index(drop=True)
    n_villages = len(covered_df)
    print(f"Number of covered villages for validation: {n_villages}")
    
    target_cols = ['Rice_frac', 'Cotton_frac', 'Maize_frac', 'Bajra_frac', 'Groundnut_frac']
    
    # Store errors
    errors_standard = {t: [] for t in target_cols}
    errors_imputed = {t: [] for t in target_cols}
    
    for val_idx in range(n_villages):
        train_df = covered_df.drop(val_idx).reset_index(drop=True)
        val_df = covered_df.iloc[[val_idx]].reset_index(drop=True)
        
        # We need two versions of the val set: unmasked (standard) and masked (for imputed)
        # To compute imputed features, we combine train_df and a masked val_df, and run the imputer.
        # Let's construct a df containing all train_df and masked val_df.
        combined_df = pd.concat([train_df, val_df], ignore_index=True)
        # Mask the last row (validation village)
        val_row_idx = len(combined_df) - 1
        combined_df.loc[val_row_idx, sar_cols] = np.nan
        
        # 1. KNN Imputer (k=6)
        imputer_knn = KNNImputer(n_neighbors=min(6, len(train_df)))
        combined_imputed_knn = imputer_knn.fit_transform(combined_df[all_cols])
        df_imputed_knn = combined_df.copy()
        df_imputed_knn[all_cols] = combined_imputed_knn
        
        # 2. Spatial 1-NN Imputer
        df_imputed_spatial = combined_df.copy()
        train_coords = train_df[['centroid_x', 'centroid_y']].values
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(train_coords)
        val_coord = val_df[['centroid_x', 'centroid_y']].values.reshape(1, -1)
        neighbor_idx = nn.kneighbors(val_coord, return_distance=False)[0][0]
        df_imputed_spatial.loc[val_row_idx, sar_cols] = train_df.loc[neighbor_idx, sar_cols]
        
        for target in target_cols:
            features_t = selected_features[target]
            
            # Select correct dataframes based on crop
            is_knn = target in ['Rice_frac', 'Cotton_frac', 'Maize_frac']
            
            if is_knn:
                # Standard (no masking of validation)
                # Training on train_df, predicting on val_df
                X_train = train_df[features_t].values
                y_train = train_df[target].values
                X_val = val_df[features_t].values
                
                # Imputed
                X_train_imp = df_imputed_knn.iloc[:-1][features_t].values
                X_val_imp = df_imputed_knn.iloc[[val_row_idx]][features_t].values
            else:
                X_train = train_df[features_t].values
                y_train = train_df[target].values
                X_val = val_df[features_t].values
                
                X_train_imp = df_imputed_spatial.iloc[:-1][features_t].values
                X_val_imp = df_imputed_spatial.iloc[[val_row_idx]][features_t].values
                
            y_val = val_df[target].values[0]
            
            # Initialize model
            if models_dict is not None and target in models_dict:
                model_standard = models_dict[target](target)
                model_imputed = models_dict[target](target)
            else:
                model_standard = OptimizedCropEnsemble(target=target)
                model_imputed = OptimizedCropEnsemble(target=target)
                
            # Fit & Predict Standard
            model_standard.fit(X_train, y_train)
            pred_standard = model_standard.predict(X_val)[0]
            errors_standard[target].append((pred_standard - y_val)**2)
            
            # Fit & Predict Imputed
            model_imputed.fit(X_train_imp, y_train)
            pred_imputed = model_imputed.predict(X_val_imp)[0]
            errors_imputed[target].append((pred_imputed - y_val)**2)
            
    # Compute mean errors
    results = []
    for target in target_cols:
        mse_std = np.mean(errors_standard[target])
        rmse_std = np.sqrt(mse_std)
        mse_imp = np.mean(errors_imputed[target])
        rmse_imp = np.sqrt(mse_imp)
        results.append({
            'Crop': target,
            'Standard_MSE': mse_std,
            'Standard_RMSE': rmse_std,
            'Imputed_MSE': mse_imp,
            'Imputed_RMSE': rmse_imp
        })
        
    df_res = pd.DataFrame(results)
    return df_res

if __name__ == '__main__':
    print("Running baseline LOVO CV validation...")
    df_data, df_sar = get_pipeline_data()
    
    selected_features = {
        'Rice_frac': ['bbox_width', 'area_ha', 'p50_20250619', 'p75_20250619', 'p25_20250619', 'mean_20250619'],
        'Cotton_frac': ['centroid_y', 'perimeter', 'diff_harvest', 'p75_20250814', 'mean_20250814', 'p50_20250814'],
        'Maize_frac': ['centroid_y', 'centroid_x', 'diff_harvest', 'mean_local_std_20251013', 'p50_20250814', 'p75_20250606'],
        'Bajra_frac': ['centroid_y', 'centroid_x', 'p25_20250619', 'p50_20250619', 'mean_20250619', 'p75_20250619'],
        'Groundnut_frac': ['centroid_y', 'centroid_x', 'mean_local_std_20250606', 'mean_local_std_20250814', 'mean_local_std_20251013', 'cumulative_change']
    }
    
    df_res = evaluate_lovo_cv(df_data, df_sar, selected_features)
    print("\nLOVO CV Baseline Performance:")
    print(df_res.to_string(index=False))
