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
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from catboost import CatBoostRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')

class OptimizedCropEnsemble:
    """
    Kaggle-optimized ensemble incorporating specific tuned models and weights per crop.
    """
    def __init__(self, target: str, random_state: int = 42):
        self.target = target
        if target == 'Rice_frac':
            self.model1 = RandomForestRegressor(n_estimators=100, random_state=random_state)
            self.model2 = ExtraTreesRegressor(n_estimators=100, random_state=random_state)
            self.w1, self.w2 = 1.0, 0.0
        elif target == 'Cotton_frac':
            self.model1 = RandomForestRegressor(n_estimators=100, random_state=random_state)
            self.model2 = CatBoostRegressor(iterations=50, depth=3, learning_rate=0.05, random_seed=random_state, verbose=0)
            self.w1, self.w2 = 0.8, 0.2
        elif target == 'Maize_frac':
            self.model1 = RandomForestRegressor(n_estimators=100, random_state=random_state)
            self.model2 = ExtraTreesRegressor(n_estimators=100, random_state=random_state)
            self.w1, self.w2 = 0.0, 1.0
        elif target == 'Bajra_frac':
            self.model1 = RandomForestRegressor(n_estimators=100, random_state=random_state)
            self.model2 = ExtraTreesRegressor(n_estimators=100, random_state=random_state)
            self.w1, self.w2 = 0.0, 1.0
        elif target == 'Groundnut_frac':
            self.model1 = RandomForestRegressor(n_estimators=100, random_state=random_state)
            self.model2 = ExtraTreesRegressor(n_estimators=100, random_state=random_state)
            self.w1, self.w2 = 1.0, 0.0
            
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model1.fit(X, y)
        self.model2.fit(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        p1 = self.model1.predict(X)
        p2 = self.model2.predict(X)
        return self.w1 * p1 + self.w2 * p2

def run_rich_feature_pipeline():
    print("========================================================================")
    print("RUNNING ROLLED-BACK OPTIMIZED CROP PIPELINE...")
    print("========================================================================")
    
    workspace_dir = r"D:\PC\resources"
    project_dir = os.path.join(workspace_dir, "project")
    
    shp_path = os.path.join(workspace_dir, "villages_clean", "villages_clean.shp")
    gdf = gpd.read_file(shp_path)
    gdf_utm = gdf.to_crs("EPSG:32643")
    gdf_utm['area_ha'] = gdf_utm.geometry.area / 10000.0
    
    # 1. Align preview TIFFs
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
    
    # Rasterize village boundaries
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
    
    # 2. Extract geometry and SAR statistics (Reference features)
    features_geom = []
    for idx, row in gdf_utm.iterrows():
        geom = row['geometry']
        centroid = geom.centroid
        bbox = geom.bounds
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        area = row['area_ha']
        perimeter = geom.length
        compactness = (4 * np.pi * geom.area) / (perimeter ** 2) if perimeter > 0 else 0
        
        features_geom.append({
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
    df_geom = pd.DataFrame(features_geom)
    
    # Compute Local Standard Deviation (Texture Feature) using cv2 box filter
    import cv2
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

    # Extract SAR features per village
    from scipy.stats import skew, kurtosis
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
    df_sar = pd.DataFrame(sar_features)

    # 3. Target Generation via KMeans (Reference)
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
    
    # 4. Hybrid Imputation
    geom_cols = ['centroid_x', 'centroid_y', 'area_ha', 'perimeter', 'compactness', 'bbox_width', 'bbox_height']
    sar_cols = [c for c in df_sar.columns if c not in ['ID', 'valid_pixels']]
    all_cols = geom_cols + sar_cols
    
    imputer_knn = KNNImputer(n_neighbors=6)
    X_imputed_knn = imputer_knn.fit_transform(df_data[all_cols])
    df_final_knn = df_data.copy()
    df_final_knn[all_cols] = X_imputed_knn
    
    df_final_spatial = df_data.copy()
    train_indices = df_data[df_data['coverage'] > 0.35].index
    zero_cov_indices = df_data[df_data['coverage'] <= 0.35].index
    train_coords = df_data.loc[train_indices, ['centroid_x', 'centroid_y']].values
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(train_coords)
    
    for idx in zero_cov_indices:
        coord = df_data.loc[idx, ['centroid_x', 'centroid_y']].values.reshape(1, -1)
        neighbor_idx = train_indices[nn.kneighbors(coord, return_distance=False)[0][0]]
        df_final_spatial.loc[idx, sar_cols] = df_data.loc[neighbor_idx, sar_cols]
        
    df_train_knn = df_final_knn[df_final_knn['coverage'] > 0.35].copy()
    df_train_spatial = df_final_spatial[df_final_spatial['coverage'] > 0.35].copy()
    
    # 5. Model Training & Prediction using Reference Ensemble & Features
    selected_features = {
        'Rice_frac': ['bbox_width', 'area_ha', 'p50_20250619', 'p75_20250619', 'p25_20250619', 'mean_20250619'],
        'Cotton_frac': ['centroid_y', 'perimeter', 'diff_harvest', 'p75_20250814', 'mean_20250814', 'p50_20250814'],
        'Maize_frac': ['centroid_y', 'centroid_x', 'diff_harvest', 'mean_local_std_20251013', 'p50_20250814', 'p75_20250606'],
        'Bajra_frac': ['centroid_y', 'centroid_x', 'p25_20250619', 'p50_20250619', 'mean_20250619', 'p75_20250619'],
        'Groundnut_frac': ['centroid_y', 'centroid_x', 'mean_local_std_20250606', 'mean_local_std_20250814', 'mean_local_std_20251013', 'cumulative_change']
    }
    
    target_cols = ['Rice_frac', 'Cotton_frac', 'Maize_frac', 'Bajra_frac', 'Groundnut_frac']
    crop_names_ha = ['Rice_ha', 'Cotton_ha', 'Maize_ha', 'Bajra_ha', 'Groundnut_ha']
    
    # Models folder
    models_out_dir = os.path.join(project_dir, "models")
    os.makedirs(models_out_dir, exist_ok=True)
    
    with open(os.path.join(models_out_dir, "imputer_knn.pkl"), "wb") as f:
        pickle.dump(imputer_knn, f)
    with open(os.path.join(models_out_dir, "nn_spatial.pkl"), "wb") as f:
        pickle.dump(nn, f)
    with open(os.path.join(models_out_dir, "selected_features.pkl"), "wb") as f:
        pickle.dump(selected_features, f)
        
    final_predictions = {}
    for target in target_cols:
        features_t = selected_features[target]
        if target in ['Rice_frac', 'Cotton_frac', 'Maize_frac']:
            df_tr = df_train_knn
            df_all = df_final_knn
        else:
            df_tr = df_train_spatial
            df_all = df_final_spatial
            
        X_train_t = df_tr[features_t].values
        y_train_t = df_tr[target].values
        X_all_t = df_all[features_t].values
        
        ensemble = OptimizedCropEnsemble(target=target)
        ensemble.fit(X_train_t, y_train_t)
        final_predictions[target] = ensemble.predict(X_all_t)
        
        with open(os.path.join(models_out_dir, f"ensemble_{target}.pkl"), "wb") as f:
            pickle.dump(ensemble, f)
            
    # 6. Blending & Enforcing Physical Constraints using Cultivated Land Mask Calibration
    df_final = df_final_knn.copy()
    cov = df_final['coverage'].values
    blended_fracs = {}
    for target in target_cols:
        obs_val = df_final[target].values
        pred_val = final_predictions[target]
        blended_fracs[target] = cov * obs_val + (1.0 - cov) * pred_val
        
    # Load physical cultivated mask fraction per village
    cult_stats = pd.read_csv(os.path.join(workspace_dir, "preprocessed_images", "village_cultivated_stats.csv"))
    cult_stats['cultivated_fraction'] = cult_stats['cultivated_combined_ha'] / (cult_stats['Area_ha'] + 1e-10)
    df_final = pd.merge(df_final, cult_stats[['ID', 'cultivated_fraction']], on='ID')
    
    target_sum = df_final['cultivated_fraction'].values
    
    sum_blended = np.zeros(len(df_final))
    for target in target_cols:
        sum_blended += blended_fracs[target]
        
    for target, ha_name in zip(target_cols, crop_names_ha):
        norm_frac = np.where(sum_blended > 0, blended_fracs[target] * target_sum / sum_blended, 0.0)
        df_final[ha_name] = norm_frac * df_final['area_ha']
        
    # 7. Output final submission
    df_sub = df_final[['ID'] + crop_names_ha].sort_values('ID').reset_index(drop=True)
    
    out_root = os.path.join(workspace_dir, "submission.csv")
    out_proj = os.path.join(project_dir, "submission.csv")
    df_sub.to_csv(out_root, index=False)
    df_sub.to_csv(out_proj, index=False)
    print(f"\nFinal rolled-back calibrated submission.csv saved to:\n  {out_root}\n  {out_proj}")
    
    # Difference check
    ref_sub_path = os.path.join(workspace_dir, "submission_rank_82.csv")
    if os.path.exists(ref_sub_path):
        df_ref = pd.read_csv(ref_sub_path)
        diff = np.abs(df_sub[crop_names_ha].values - df_ref[crop_names_ha].values)
        mean_abs_change = np.mean(diff)
        print(f"Hectares MSE vs Rank 82: {np.mean(diff**2):.4f} | Mean Absolute Change: {mean_abs_change:.4f} ha")

if __name__ == '__main__':
    run_rich_feature_pipeline()
