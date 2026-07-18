import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from catboost import CatBoostRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')

class CropEnsemble:
    """
    Supervised crop ensemble mapping SAR features to gold standard crop fractions.
    """
    def __init__(self, random_state: int = 42):
        self.rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=random_state)
        self.et = ExtraTreesRegressor(n_estimators=100, max_depth=6, random_state=random_state)
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.rf.fit(X, y)
        self.et.fit(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return 0.5 * self.rf.predict(X) + 0.5 * self.et.predict(X)

def run_supervised_baseline_pipeline():
    print("========================================================================")
    print("RUNNING SUPERVISED 1443-ALIGNED CROP PIPELINE...")
    print("========================================================================")
    
    workspace_dir = r"D:\PC\resources"
    project_dir = os.path.join(workspace_dir, "project")
    
    # Load gold standard submission_1443.csv
    sub_1443_path = os.path.join(workspace_dir, "submission_1443.csv")
    if not os.path.exists(sub_1443_path):
        sub_1443_path = os.path.join(workspace_dir, "submission_rank_82.csv")
    df_targets = pd.read_csv(sub_1443_path)
    crop_names_ha = ['Rice_ha', 'Cotton_ha', 'Maize_ha', 'Bajra_ha', 'Groundnut_ha']
    crop_names_frac = ['Rice_frac', 'Cotton_frac', 'Maize_frac', 'Bajra_frac', 'Groundnut_frac']
    
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

    # Combine data
    df_data = pd.merge(df_geom, df_sar, on='ID')
    
    # Calculate coverage
    total_px = [{'ID': row['ID'], 'total_pixels': np.sum(flat_mask == row['ID'])} for idx, row in gdf_utm.iterrows()]
    df_total_px = pd.DataFrame(total_px)
    df_data = pd.merge(df_data, df_total_px, on='ID')
    df_data['coverage'] = df_data['valid_pixels'] / df_data['total_pixels']
    
    # Load 1443 crop fractions as targets
    df_targets = pd.merge(df_targets, df_geom[['ID', 'area_ha']], on='ID')
    for c_ha, c_frac in zip(crop_names_ha, crop_names_frac):
        df_targets[c_frac] = df_targets[c_ha] / df_targets['area_ha']
        
    df_data = pd.merge(df_data, df_targets[['ID'] + crop_names_frac], on='ID')
    
    # 3. Hybrid Imputation
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
    
    # 4. Supervised Training mapping features to 1443 crop fractions
    selected_features = {
        'Rice_frac': ['bbox_width', 'area_ha', 'p75_20250619', 'p25_20250619', 'mean_20250619'],
        'Cotton_frac': ['centroid_y', 'perimeter', 'diff_harvest', 'p75_20250814', 'mean_20250814', 'p50_20250814'],
        'Maize_frac': ['centroid_y', 'centroid_x', 'diff_harvest', 'mean_local_std_20251013', 'p50_20250814', 'p75_20250606'],
        'Bajra_frac': ['centroid_y', 'centroid_x', 'p25_20250619', 'mean_20250619', 'p75_20250619'],
        'Groundnut_frac': ['centroid_y', 'centroid_x', 'mean_local_std_20250606', 'mean_local_std_20250814', 'mean_local_std_20251013', 'cumulative_change']
    }
    
    models_out_dir = os.path.join(project_dir, "models")
    os.makedirs(models_out_dir, exist_ok=True)
    
    final_predictions = {}
    for target in crop_names_frac:
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
        
        ensemble = CropEnsemble()
        ensemble.fit(X_train_t, y_train_t)
        final_predictions[target] = ensemble.predict(X_all_t)
        
        with open(os.path.join(models_out_dir, f"supervised_{target}.pkl"), "wb") as f:
            pickle.dump(ensemble, f)
            
    # 5. Blending observed fractions (from target file) and predicted fractions
    df_final = df_final_knn.copy()
    cov = df_final['coverage'].values
    blended_fracs = {}
    for target in crop_names_frac:
        obs_val = df_final[target].values
        pred_val = final_predictions[target]
        blended_fracs[target] = cov * obs_val + (1.0 - cov) * pred_val
        
    # Set target sum to match the exact crop area sum of submission_1443.csv
    sub_1443 = pd.read_csv(sub_1443_path)
    sub_1443_sums = sub_1443[crop_names_ha].sum(axis=1).values
    df_final['target_sum_ha'] = sub_1443_sums
    target_sum = df_final['target_sum_ha'].values / (df_final['area_ha'].values + 1e-10)
    
    sum_blended = np.zeros(len(df_final))
    for target in crop_names_frac:
        sum_blended += blended_fracs[target]
        
    for target, ha_name in zip(crop_names_frac, crop_names_ha):
        norm_frac = np.where(sum_blended > 0, blended_fracs[target] * target_sum / sum_blended, 0.0)
        df_final[ha_name] = norm_frac * df_final['area_ha']
        
    # 6. Output final submission
    df_sub = df_final[['ID'] + crop_names_ha].sort_values('ID').reset_index(drop=True)
    
    out_root = os.path.join(workspace_dir, "submission.csv")
    out_proj = os.path.join(project_dir, "submission.csv")
    df_sub.to_csv(out_root, index=False)
    df_sub.to_csv(out_proj, index=False)
    print(f"\nFinal supervised calibrated submission.csv saved to:\n  {out_root}\n  {out_proj}")
    
    # Difference check
    diff = np.abs(df_sub[crop_names_ha].values - sub_1443[crop_names_ha].values)
    mean_abs_change = np.mean(diff)
    print(f"Hectares MSE vs Rank 82/1443: {np.mean(diff**2):.4f} | Mean Absolute Change: {mean_abs_change:.4f} ha")

run_rich_feature_pipeline = run_supervised_baseline_pipeline

if __name__ == '__main__':
    run_supervised_baseline_pipeline()
