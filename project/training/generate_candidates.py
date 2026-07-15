import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Ridge, BayesianRidge
from sklearn.metrics import mean_squared_error
import cv2
import warnings
warnings.filterwarnings('ignore')

from search import get_base_data, extract_advanced_features

def run_candidate_generation():
    print("Loading data...", flush=True)
    gdf_utm, df_geom, df_labels, df_total_px, flat_stack, flat_mask, H, W, dates = get_base_data()
    
    print("Extracting advanced features...", flush=True)
    df_sar = extract_advanced_features(gdf_utm, flat_stack, flat_mask, H, W, dates)
    
    df_data = pd.merge(df_geom, df_sar, on='ID')
    df_data = pd.merge(df_data, df_labels, on='ID')
    df_data = pd.merge(df_data, df_total_px, on='ID')
    df_data['coverage'] = df_data['valid_pixels'] / df_data['total_pixels']
    
    covered_df = df_data[df_data['coverage'] > 0.35].copy().reset_index(drop=True)
    n_villages = len(covered_df)
    
    geom_cols = ['centroid_x', 'centroid_y', 'area_ha', 'perimeter', 'compactness', 'bbox_width', 'bbox_height']
    sar_cols = [c for c in df_sar.columns if c not in ['ID', 'valid_pixels']]
    all_cols = geom_cols + sar_cols
    
    target_cols = ['Rice_frac', 'Cotton_frac', 'Maize_frac', 'Bajra_frac', 'Groundnut_frac']
    crop_names_ha = ['Rice_ha', 'Cotton_ha', 'Maize_ha', 'Bajra_ha', 'Groundnut_ha']
    
    # Define configurations for each candidate
    feature_configs = {
        'with_coords': {
            'Rice_frac': ['centroid_x', 'centroid_y', 'ratio_veg'],
            'Cotton_frac': ['centroid_y', 'centroid_x', 'diff_harvest', 'mean_20250606', 'ratio_harvest'],
            'Maize_frac': ['centroid_y', 'centroid_x', 'mean_sobel_20251013', 'mean_local_std_20250814', 'mean_local_std_20250606', 'mean_sobel_20250814'],
            'Bajra_frac': ['p25_20250619', 'centroid_x', 'centroid_y', 'temporal_variance', 'p75_20250619'],
            'Groundnut_frac': ['centroid_x', 'centroid_y', 'area_ha']
        },
        'no_coords': {
            'Rice_frac': ['area_ha', 'bbox_width', 'ratio_veg', 'p50_20250619'],
            'Cotton_frac': ['area_ha', 'perimeter', 'diff_harvest', 'mean_20250606', 'ratio_harvest'],
            'Maize_frac': ['area_ha', 'mean_sobel_20251013', 'mean_local_std_20250814', 'mean_local_std_20250606', 'mean_sobel_20250814'],
            'Bajra_frac': ['area_ha', 'p25_20250619', 'temporal_variance', 'p75_20250619', 'mean_local_std_20250619'],
            'Groundnut_frac': ['area_ha', 'perimeter', 'mean_local_std_20250814', 'mean_local_std_20250606', 'change_magnitude']
        }
    }
    
    # We will compute LOVO CV Hectare MSE for all candidates
    candidate_names = [
        'tree_coords',
        'tree_no_coords',
        'linear_coords',
        'linear_no_coords',
        'conservative',
        'ensemble'
    ]
    
    cv_predictions = {name: {t: np.zeros(n_villages) for t in target_cols} for name in candidate_names}
    
    print("\nStarting LOVO CV evaluation for all candidates...", flush=True)
    
    # 1. LOVO CV Loop
    for val_idx in range(n_villages):
        train_df = covered_df.drop(val_idx).reset_index(drop=True)
        val_df = covered_df.iloc[[val_idx]].reset_index(drop=True)
        
        combined_df = pd.concat([train_df, val_df], ignore_index=True)
        val_row_idx = len(combined_df) - 1
        combined_df.loc[val_row_idx, sar_cols] = np.nan
        
        # Imputers
        # KNN-6 for Rice/Cotton/Maize (with/without coords)
        imputer_knn = KNNImputer(n_neighbors=min(6, len(train_df)))
        df_imputed_knn = combined_df.copy()
        df_imputed_knn[all_cols] = imputer_knn.fit_transform(combined_df[all_cols])
        
        # Spatial 1-NN for Bajra/Groundnut
        df_imputed_spatial = combined_df.copy()
        train_coords = train_df[['centroid_x', 'centroid_y']].values
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(train_coords)
        val_coord = val_df[['centroid_x', 'centroid_y']].values.reshape(1, -1)
        neighbor_idx = nn.kneighbors(val_coord, return_distance=False)[0][0]
        df_imputed_spatial.loc[val_row_idx, sar_cols] = train_df.loc[neighbor_idx, sar_cols]
        
        # Train and Predict for each target and each candidate
        for target in target_cols:
            is_knn = target in ['Rice_frac', 'Cotton_frac', 'Maize_frac']
            df_imp = df_imputed_knn if is_knn else df_imputed_spatial
            
            # 1. tree_coords
            feats_wc = feature_configs['with_coords'][target]
            X_tr_wc = df_imp.iloc[:-1][feats_wc].values
            y_tr = train_df[target].values
            X_val_wc = df_imp.iloc[[val_row_idx]][feats_wc].values
            
            model_tree = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
            model_tree.fit(X_tr_wc, y_tr)
            cv_predictions['tree_coords'][target][val_idx] = np.clip(model_tree.predict(X_val_wc)[0], 0.0, 1.0)
            
            # 2. tree_no_coords
            feats_nc = feature_configs['no_coords'][target]
            X_tr_nc = df_imp.iloc[:-1][feats_nc].values
            X_val_nc = df_imp.iloc[[val_row_idx]][feats_nc].values
            
            model_tree_nc = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
            model_tree_nc.fit(X_tr_nc, y_tr)
            cv_predictions['tree_no_coords'][target][val_idx] = np.clip(model_tree_nc.predict(X_val_nc)[0], 0.0, 1.0)
            
            # 3. linear_coords
            model_lin = Ridge(alpha=1.0, random_state=42)
            model_lin.fit(X_tr_wc, y_tr)
            cv_predictions['linear_coords'][target][val_idx] = np.clip(model_lin.predict(X_val_wc)[0], 0.0, 1.0)
            
            # 4. linear_no_coords
            model_lin_nc = Ridge(alpha=1.0, random_state=42)
            model_lin_nc.fit(X_tr_nc, y_tr)
            cv_predictions['linear_no_coords'][target][val_idx] = np.clip(model_lin_nc.predict(X_val_nc)[0], 0.0, 1.0)
            
            # 5. conservative (mean fraction of train set)
            cv_predictions['conservative'][target][val_idx] = np.mean(y_tr)
            
            # 6. ensemble (blend of tree_no_coords and linear_no_coords)
            cv_predictions['ensemble'][target][val_idx] = 0.5 * cv_predictions['tree_no_coords'][target][val_idx] + \
                                                          0.5 * cv_predictions['linear_no_coords'][target][val_idx]
            
    # Calculate Hectare MSE for each candidate
    print("\nComputing Hectare MSE for each candidate...", flush=True)
    candidate_mses = {}
    
    for name in candidate_names:
        village_sq_errors = []
        for idx, row in covered_df.iterrows():
            area = row['area_ha']
            pred_fracs = np.array([cv_predictions[name][t][idx] for t in target_cols])
            
            # Apply normalisation
            sum_pred = np.sum(pred_fracs)
            if sum_pred > 0:
                norm_fracs = pred_fracs * 0.99 / sum_pred
            else:
                norm_fracs = np.zeros(5)
                
            pred_ha = norm_fracs * area
            true_ha = np.array([covered_df[t].values[idx] for t in target_cols]) * area
            village_sq_errors.append(np.mean((pred_ha - true_ha)**2))
            
        candidate_mses[name] = np.mean(village_sq_errors)
        print(f"Candidate '{name}' Hectares LOVO CV MSE: {candidate_mses[name]:.4f} ha^2", flush=True)
        
    # 2. Train on ALL covered villages & generate submissions
    print("\nTraining final models on all covered villages and generating submissions...", flush=True)
    
    # Pre-impute all villages
    imputer_all_knn = KNNImputer(n_neighbors=6)
    df_imputed_all_knn = df_data.copy()
    df_imputed_all_knn[all_cols] = imputer_all_knn.fit_transform(df_data[all_cols])
    
    df_imputed_all_spatial = df_data.copy()
    train_indices = df_data[df_data['coverage'] > 0.35].index
    zero_cov_indices = df_data[df_data['coverage'] <= 0.35].index
    train_coords = df_data.loc[train_indices, ['centroid_x', 'centroid_y']].values
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(train_coords)
    for idx in zero_cov_indices:
        coord = df_data.loc[idx, ['centroid_x', 'centroid_y']].values.reshape(1, -1)
        neighbor_idx = train_indices[nn.kneighbors(coord, return_distance=False)[0][0]]
        df_imputed_all_spatial.loc[idx, sar_cols] = df_data.loc[neighbor_idx, sar_cols]
        
    # Fit and Predict dictionaries
    test_predictions = {name: {t: np.zeros(len(df_data)) for t in target_cols} for name in candidate_names}
    
    for target in target_cols:
        is_knn = target in ['Rice_frac', 'Cotton_frac', 'Maize_frac']
        df_imp_tr = df_imputed_all_knn[df_imputed_all_knn['coverage'] > 0.35]
        df_imp_all = df_imputed_all_knn if is_knn else df_imputed_all_spatial
        
        # 1. tree_coords
        feats_wc = feature_configs['with_coords'][target]
        X_tr_wc = df_imp_tr[feats_wc].values
        y_tr = covered_df[target].values
        X_all_wc = df_imp_all[feats_wc].values
        
        model_tree = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        model_tree.fit(X_tr_wc, y_tr)
        test_predictions['tree_coords'][target] = np.clip(model_tree.predict(X_all_wc), 0.0, 1.0)
        
        # 2. tree_no_coords
        feats_nc = feature_configs['no_coords'][target]
        X_tr_nc = df_imp_tr[feats_nc].values
        X_all_nc = df_imp_all[feats_nc].values
        
        model_tree_nc = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        model_tree_nc.fit(X_tr_nc, y_tr)
        test_predictions['tree_no_coords'][target] = np.clip(model_tree_nc.predict(X_all_nc), 0.0, 1.0)
        
        # 3. linear_coords
        model_lin = Ridge(alpha=1.0, random_state=42)
        model_lin.fit(X_tr_wc, y_tr)
        test_predictions['linear_coords'][target] = np.clip(model_lin.predict(X_all_wc), 0.0, 1.0)
        
        # 4. linear_no_coords
        model_lin_nc = Ridge(alpha=1.0, random_state=42)
        model_lin_nc.fit(X_tr_nc, y_tr)
        test_predictions['linear_no_coords'][target] = np.clip(model_lin_nc.predict(X_all_nc), 0.0, 1.0)
        
        # 5. conservative
        test_predictions['conservative'][target] = np.full(len(df_data), np.mean(y_tr))
        
        # 6. ensemble
        test_predictions['ensemble'][target] = 0.5 * test_predictions['tree_no_coords'][target] + \
                                               0.5 * test_predictions['linear_no_coords'][target]
                                               
    # Format and save final predictions
    cov = df_data['coverage'].values
    output_dir = r"D:\PC\resources\project\outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    for name in candidate_names:
        df_final = df_data.copy()
        
        # Blend observations with model predictions
        blended_fracs = {}
        for target in target_cols:
            obs_val = df_final[target].values
            pred_val = test_predictions[name][target]
            blended_fracs[target] = cov * obs_val + (1.0 - cov) * pred_val
            
        obs_veg_frac = df_final[target_cols].sum(axis=1).values
        obs_veg_frac = np.where(obs_veg_frac > 0, obs_veg_frac, 0.99)
        target_sum = cov * obs_veg_frac + (1.0 - cov) * 0.99
        
        sum_blended = np.zeros(len(df_final))
        for target in target_cols:
            sum_blended += blended_fracs[target]
            
        for target, ha_name in zip(target_cols, crop_names_ha):
            norm_frac = np.where(sum_blended > 0, blended_fracs[target] * target_sum / sum_blended, 0.0)
            df_final[ha_name] = norm_frac * df_final['area_ha']
            
        df_sub = df_final[['ID'] + crop_names_ha].sort_values('ID').reset_index(drop=True)
        
        # Save candidates to outputs folder and workspace root (for user's testing)
        candidate_sub_path = os.path.join(output_dir, f"submission_{name}.csv")
        df_sub.to_csv(candidate_sub_path, index=False)
        
        workspace_sub_path = os.path.join(r"D:\PC\resources", f"submission_{name}.csv")
        df_sub.to_csv(workspace_sub_path, index=False)
        print(f"Generated submission candidate: {candidate_sub_path}", flush=True)

if __name__ == '__main__':
    run_candidate_generation()
