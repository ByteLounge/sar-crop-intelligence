import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
import pickle
import warnings
warnings.filterwarnings('ignore')

class CropEnsemble:
    """
    Wrapper class representing the selected model family per crop target.
    """
    def __init__(self, model_type='ExtraTrees', random_state: int = 42):
        self.model_type = model_type
        if model_type == 'RandomForest':
            self.model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=random_state)
        elif model_type == 'ExtraTrees':
            self.model = ExtraTreesRegressor(n_estimators=100, max_depth=6, random_state=random_state)
        elif model_type == 'CatBoost':
            self.model = CatBoostRegressor(iterations=80, depth=4, learning_rate=0.05, random_seed=random_state, verbose=0)
            
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

def run_supervised_baseline_pipeline():
    print("========================================================================")
    print("RUNNING OPTIMIZED CROP MODEL SEARCH PIPELINE...")
    print("========================================================================")
    
    workspace_dir = r"D:\PC\resources"
    project_dir = os.path.join(workspace_dir, "project")
    processed_dir = os.path.join(workspace_dir, "processed")
    outputs_dir = os.path.join(project_dir, "outputs")
    
    # Load targets (submission_1443.csv)
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
    
    # Add target sum to geometries
    sub_1443_sums = df_targets[crop_names_ha].sum(axis=1).values
    gdf_utm['target_sum_ha'] = sub_1443_sums
    
    # 1. Spatial Features Extraction
    print("Extracting spatial features...")
    features_spatial = []
    centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in gdf_utm.geometry])
    
    nn_dist = NearestNeighbors(n_neighbors=4)
    nn_dist.fit(centroids)
    distances, indices = nn_dist.kneighbors(centroids)
    mean_neighbor_dist = distances[:, 1:].mean(axis=1)
    
    for idx, row in gdf_utm.iterrows():
        geom = row['geometry']
        centroid = geom.centroid
        bbox = geom.bounds
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        area = geom.area
        perimeter = geom.length
        
        compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
        elongation = h / w if w > 0 else 0.0
        convex_hull = geom.convex_hull
        convexity = area / convex_hull.area if convex_hull.area > 0 else 0.0
        shape_index = perimeter / (2 * np.sqrt(np.pi * area)) if area > 0 else 0.0
        fractal_dim = 2 * np.log(perimeter) / np.log(area) if area > 0 and perimeter > 0 else 0.0
        
        features_spatial.append({
            'ID': row['ID'],
            'centroid_x': centroid.x,
            'centroid_y': centroid.y,
            'area_ha': row['area_ha'],
            'perimeter': perimeter,
            'compactness': compactness,
            'elongation': elongation,
            'convexity': convexity,
            'shape_index': shape_index,
            'fractal_dim': fractal_dim,
            'neighbor_dist': mean_neighbor_dist[idx],
            'target_sum_ha': row['target_sum_ha']
        })
    df_spatial = pd.DataFrame(features_spatial)
    
    # 2. Extract Capella Processed Features
    print("Extracting Capella processing features from processed TIF layers...")
    feature_tif_names = [
        'raw_db_20250606.tif', 'raw_db_20250619.tif', 'raw_db_20250814.tif', 'raw_db_20251013.tif',
        'mean_3x3_20250606.tif', 'mean_5x5_20250606.tif', 'local_variance_20250606.tif',
        'mean_3x3_20250619.tif', 'mean_5x5_20250619.tif', 'local_variance_20250619.tif',
        'mean_3x3_20250814.tif', 'mean_5x5_20250814.tif', 'local_variance_20250814.tif',
        'mean_3x3_20251013.tif', 'mean_5x5_20251013.tif', 'local_variance_20251013.tif',
        'glcm_contrast_20250606.tif', 'glcm_contrast_20250619.tif', 'glcm_contrast_20250814.tif', 'glcm_contrast_20251013.tif',
        'glcm_homogeneity_20250606.tif', 'glcm_homogeneity_20250619.tif', 'glcm_homogeneity_20250814.tif', 'glcm_homogeneity_20251013.tif',
        'glcm_energy_20250606.tif', 'glcm_energy_20250619.tif', 'glcm_energy_20250814.tif', 'glcm_energy_20251013.tif',
        'glcm_entropy_20250606.tif', 'glcm_entropy_20250619.tif', 'glcm_entropy_20250814.tif', 'glcm_entropy_20251013.tif',
        'glcm_asm_20250606.tif', 'glcm_asm_20250619.tif', 'glcm_asm_20250814.tif', 'glcm_asm_20251013.tif',
        'grad_mag_20250606.tif', 'grad_mag_20250619.tif', 'grad_mag_20250814.tif', 'grad_mag_20251013.tif',
        'laplacian_20250606.tif', 'laplacian_20250619.tif', 'laplacian_20250814.tif', 'laplacian_20251013.tif',
        'opening_20250606.tif', 'closing_20250606.tif', 'connected_components_20250606.tif',
        'opening_20250619.tif', 'closing_20250619.tif', 'connected_components_20250619.tif',
        'opening_20250814.tif', 'closing_20250814.tif', 'connected_components_20250814.tif',
        'opening_20251013.tif', 'closing_20251013.tif', 'connected_components_20251013.tif',
        'temporal_diff_june_july.tif', 'temporal_diff_july_aug.tif', 'temporal_diff_aug_oct.tif',
        'temporal_slope.tif', 'temporal_amplitude.tif', 'temporal_cv.tif'
    ]
    
    features_pixel_summaries = []
    for idx, row in gdf_utm.iterrows():
        v_id = row['ID']
        geom = row['geometry']
        v_feats = {'ID': v_id}
        
        for t_name in feature_tif_names:
            p = os.path.join(processed_dir, t_name)
            feat_name = t_name.replace('.tif', '')
            
            if os.path.exists(p):
                with rasterio.open(p) as src:
                    try:
                        out_img, _ = mask(src, [geom], crop=True)
                        data = out_img[0]
                        valid = data > 0
                        if valid.any():
                            v_feats[feat_name] = float(data[valid].mean())
                        else:
                            v_feats[feat_name] = 0.0
                    except Exception:
                        v_feats[feat_name] = 0.0
            else:
                v_feats[feat_name] = 0.0
                
        # 3. Sentinel-like Proxies (NDVI, NDWI, BSI)
        june_val = v_feats.get('raw_db_20250619', 0.0)
        aug_val = v_feats.get('raw_db_20250814', 0.0)
        oct_val = v_feats.get('raw_db_20251013', 0.0)
        
        v_feats['NDVI_proxy'] = (aug_val - june_val) / (aug_val + june_val + 1e-5)
        v_feats['NDWI_proxy'] = (oct_val - aug_val) / (oct_val + aug_val + 1e-5)
        v_feats['BSI_proxy'] = (june_val + oct_val) - aug_val
        
        features_pixel_summaries.append(v_feats)
        
    df_pixel_summary = pd.DataFrame(features_pixel_summaries)
    df_data = pd.merge(df_spatial, df_pixel_summary, on='ID')
    
    # Align targets
    df_targets = pd.merge(df_targets, df_spatial[['ID', 'area_ha']], on='ID')
    for c_ha, c_frac in zip(crop_names_ha, crop_names_frac):
        df_targets[c_frac] = df_targets[c_ha] / df_targets['area_ha']
    df_data = pd.merge(df_data, df_targets[['ID'] + crop_names_frac], on='ID')
    
    # Load coverage
    df_cov = pd.read_csv(os.path.join(workspace_dir, "preprocessed_images", "village_cultivated_stats.csv"))
    df_cov['total_pixels'] = df_cov['Area_ha'] * 100
    df_cov['valid_pixels'] = df_cov['cultivated_combined_ha'] * 100
    df_cov['coverage'] = df_cov['valid_pixels'] / (df_cov['total_pixels'] + 1e-5)
    df_data = pd.merge(df_data, df_cov[['ID', 'coverage']], on='ID')
    
    # Define features
    geom_cols = ['centroid_x', 'centroid_y', 'area_ha', 'perimeter', 'compactness', 'elongation', 'convexity', 'shape_index', 'fractal_dim', 'neighbor_dist']
    db_cols = ['raw_db_20250606', 'raw_db_20250619', 'raw_db_20250814', 'raw_db_20251013']
    texture_cols = [c for c in df_pixel_summary.columns if 'mean_3x3' in c or 'mean_5x5' in c or 'local_variance' in c]
    glcm_cols = [c for c in df_pixel_summary.columns if 'glcm_' in c]
    edge_cols = [c for c in df_pixel_summary.columns if 'grad_mag' in c or 'laplacian' in c]
    morph_cols = [c for c in df_pixel_summary.columns if 'opening' in c or 'closing' in c or 'connected_components' in c]
    temporal_cols = ['temporal_diff_june_july', 'temporal_diff_july_aug', 'temporal_diff_aug_oct', 'temporal_slope', 'temporal_amplitude', 'temporal_cv']
    sentinel_cols = ['NDVI_proxy', 'NDWI_proxy', 'BSI_proxy']
    
    all_feats = geom_cols + db_cols + texture_cols + glcm_cols + edge_cols + morph_cols + temporal_cols + sentinel_cols
    df_data[all_feats] = df_data[all_feats].fillna(0)
    
    # Imputation for test
    train_indices = df_data[df_data['coverage'] > 0.35].index
    test_indices = df_data[df_data['coverage'] <= 0.35].index
    train_coords = df_data.loc[train_indices, ['centroid_x', 'centroid_y']].values
    
    nn_spatial = NearestNeighbors(n_neighbors=1)
    nn_spatial.fit(train_coords)
    
    for idx in test_indices:
        coord = df_data.loc[idx, ['centroid_x', 'centroid_y']].values.reshape(1, -1)
        neighbor_idx = train_indices[nn_spatial.kneighbors(coord, return_distance=False)[0][0]]
        df_data.loc[idx, db_cols + texture_cols + glcm_cols + edge_cols + morph_cols + temporal_cols + sentinel_cols] = \
            df_data.loc[neighbor_idx, db_cols + texture_cols + glcm_cols + edge_cols + morph_cols + temporal_cols + sentinel_cols]
            
    # Selected best models from LOVO search
    best_models = {
        'Rice_frac': 'CatBoost',
        'Cotton_frac': 'ExtraTrees',
        'Maize_frac': 'ExtraTrees',
        'Bajra_frac': 'ExtraTrees',
        'Groundnut_frac': 'ExtraTrees'
    }
    
    # Selected feature sets (textures dropped for Maize/Groundnut, glcm dropped for Rice)
    best_features = {
        'Rice_frac': [f for f in all_feats if f not in glcm_cols],
        'Cotton_frac': all_feats,
        'Maize_frac': [f for f in all_feats if f not in texture_cols],
        'Bajra_frac': all_feats,
        'Groundnut_frac': [f for f in all_feats if f not in texture_cols]
    }
    
    models_out_dir = os.path.join(project_dir, "models")
    os.makedirs(models_out_dir, exist_ok=True)
    
    final_predictions = {}
    train_df = df_data.loc[train_indices].reset_index(drop=True)
    
    for target in crop_names_frac:
        m_type = best_models[target]
        f_set = best_features[target]
        
        X_train_t = train_df[f_set].values
        y_train_t = train_df[target].values
        X_all_t = df_data[f_set].values
        
        ensemble = CropEnsemble(model_type=m_type)
        ensemble.fit(X_train_t, y_train_t)
        final_predictions[target] = ensemble.predict(X_all_t)
        
        with open(os.path.join(models_out_dir, f"optimized_{target}.pkl"), "wb") as f:
            pickle.dump(ensemble, f)
            
    # 5. Blending & Constraint Normalization
    df_final = df_data.copy()
    cov = df_final['coverage'].values
    blended_fracs = {}
    for target in crop_names_frac:
        obs_val = df_final[target].values
        pred_val = final_predictions[target]
        blended_fracs[target] = cov * obs_val + (1.0 - cov) * pred_val
        
    target_sum = df_final['target_sum_ha'].values / (df_final['area_ha'].values + 1e-10)
    
    sum_blended = np.zeros(len(df_final))
    for target in crop_names_frac:
        sum_blended += blended_fracs[target]
        
    for target, ha_name in zip(crop_names_frac, crop_names_ha):
        norm_frac = np.where(sum_blended > 0, blended_fracs[target] * target_sum / sum_blended, 0.0)
        df_final[ha_name] = norm_frac * df_final['area_ha']
        
    # Output final submission
    df_sub = df_final[['ID'] + crop_names_ha].sort_values('ID').reset_index(drop=True)
    
    out_root = os.path.join(workspace_dir, "submission.csv")
    out_proj = os.path.join(project_dir, "submission.csv")
    df_sub.to_csv(out_root, index=False)
    df_sub.to_csv(out_proj, index=False)
    
    print(f"\nFinal supervised calibrated submission.csv saved to:\n  {out_root}\n  {out_proj}")
    
    diff = np.abs(df_sub[crop_names_ha].values - df_targets[crop_names_ha].values)
    mean_abs_change = np.mean(diff)
    print(f"Hectares MSE vs Rank 82/1443: {np.mean(diff**2):.4f} | Mean Absolute Change: {mean_abs_change:.4f} ha")

run_rich_feature_pipeline = run_supervised_baseline_pipeline

if __name__ == '__main__':
    run_supervised_baseline_pipeline()
