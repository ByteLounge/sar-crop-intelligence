import os
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import ExtraTreesRegressor
import warnings
warnings.filterwarnings('ignore')

from search import get_base_data, extract_advanced_features

def run_physical_evaluation():
    print("Loading data...")
    gdf_utm, df_geom, df_labels, df_total_px, flat_stack, flat_mask, H, W, dates = get_base_data()
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
    
    # 1. Feature Set: safe_no_coords (previous)
    feats_previous = {
        'Rice_frac': ['area_ha', 'bbox_width', 'ratio_veg', 'p50_20250619'],
        'Cotton_frac': ['area_ha', 'perimeter', 'diff_harvest', 'mean_20250606', 'ratio_harvest'],
        'Maize_frac': ['area_ha', 'mean_sobel_20251013', 'mean_local_std_20250814', 'mean_local_std_20250606', 'mean_sobel_20250814'],
        'Bajra_frac': ['area_ha', 'p25_20250619', 'temporal_variance', 'p75_20250619', 'mean_local_std_20250619'],
        'Groundnut_frac': ['area_ha', 'perimeter', 'mean_local_std_20250814', 'mean_local_std_20250606', 'change_magnitude']
    }
    
    # 2. Feature Set: safe_physical (new, replacing ratios with difference log-ratios)
    feats_physical = {
        'Rice_frac': ['area_ha', 'bbox_width', 'diff_veg', 'p50_20250619'],
        'Cotton_frac': ['area_ha', 'perimeter', 'diff_harvest', 'mean_20250606', 'mean_20250814'],
        'Maize_frac': ['area_ha', 'mean_sobel_20251013', 'mean_local_std_20250814', 'mean_local_std_20250606', 'mean_sobel_20250814'],
        'Bajra_frac': ['area_ha', 'p25_20250619', 'temporal_variance', 'p75_20250619', 'mean_local_std_20250619'],
        'Groundnut_frac': ['area_ha', 'perimeter', 'mean_local_std_20250814', 'mean_local_std_20250606', 'change_magnitude']
    }
    
    feature_sets = {
        'Previous_Ratios': feats_previous,
        'Physical_Differences': feats_physical
    }
    
    for set_name, feat_config in feature_sets.items():
        oof_preds = {t: np.zeros(n_villages) for t in target_cols}
        
        for val_idx in range(n_villages):
            train_df = covered_df.drop(val_idx).reset_index(drop=True)
            val_df = covered_df.iloc[[val_idx]].reset_index(drop=True)
            
            combined_df = pd.concat([train_df, val_df], ignore_index=True)
            val_row_idx = len(combined_df) - 1
            combined_df.loc[val_row_idx, sar_cols] = np.nan
            
            # Hybrid imputation
            imputer = KNNImputer(n_neighbors=min(6, len(train_df)))
            df_imputed_knn = combined_df.copy()
            df_imputed_knn[all_cols] = imputer.fit_transform(combined_df[all_cols])
            
            df_imputed_spatial = combined_df.copy()
            train_coords = train_df[['centroid_x', 'centroid_y']].values
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(train_coords)
            val_coord = val_df[['centroid_x', 'centroid_y']].values.reshape(1, -1)
            neighbor_idx = nn.kneighbors(val_coord, return_distance=False)[0][0]
            df_imputed_spatial.loc[val_row_idx, sar_cols] = train_df.loc[neighbor_idx, sar_cols]
            
            for target in target_cols:
                is_knn = target in ['Rice_frac', 'Cotton_frac', 'Maize_frac']
                df_imp = df_imputed_knn if is_knn else df_imputed_spatial
                
                feats = feat_config[target]
                X_train = df_imp.iloc[:-1][feats].values
                y_train = train_df[target].values
                X_val = df_imp.iloc[[val_row_idx]][feats].values
                
                model = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                pred = model.predict(X_val)[0]
                oof_preds[target][val_idx] = np.clip(pred, 0.0, 1.0)
                
        # Hectare MSE calculation
        errors = []
        for idx, row in covered_df.iterrows():
            area = row['area_ha']
            pred_fracs = np.array([oof_preds[t][idx] for t in target_cols])
            
            # Normalization
            sum_pred = np.sum(pred_fracs)
            if sum_pred > 0:
                norm_fracs = pred_fracs * 0.99 / sum_pred
            else:
                norm_fracs = np.zeros(5)
                
            pred_ha = norm_fracs * area
            true_ha = np.array([covered_df[t].values[idx] for t in target_cols]) * area
            errors.append(np.mean((pred_ha - true_ha)**2))
            
        print(f"FeatureSet '{set_name}' Hectares CV MSE: {np.mean(errors):.4f} ha^2")

if __name__ == '__main__':
    run_physical_evaluation()
