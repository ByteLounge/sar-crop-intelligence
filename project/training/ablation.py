import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from search import get_base_data, extract_advanced_features

def run_ablation():
    print("Loading data...")
    gdf_utm, df_geom, df_labels, df_total_px, flat_stack, flat_mask, H, W, dates = get_base_data()
    
    print("Extracting advanced features...")
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
    
    # 36 basic SAR features (excluding ratios, gradients, variance, entropy, fractions, etc.)
    basic_sar_cols = []
    for d in dates:
        basic_sar_cols += [f'mean_{d}', f'std_{d}', f'cv_{d}', f'skew_{d}', f'kurt_{d}', f'p25_{d}', f'p50_{d}', f'p75_{d}', f'mean_local_std_{d}']
    basic_sar_cols += ['diff_sowing', 'diff_veg', 'diff_harvest', 'growth_rate', 'cumulative_change']
    
    target_cols = ['Rice_frac', 'Cotton_frac', 'Maize_frac', 'Bajra_frac', 'Groundnut_frac']
    crop_names_ha = ['Rice_ha', 'Cotton_ha', 'Maize_ha', 'Bajra_ha', 'Groundnut_ha']
    
    feature_configs = {
        'no_coords': {
            'Rice_frac': ['area_ha', 'bbox_width', 'ratio_veg', 'p50_20250619'],
            'Cotton_frac': ['area_ha', 'perimeter', 'diff_harvest', 'mean_20250606', 'ratio_harvest'],
            'Maize_frac': ['area_ha', 'mean_sobel_20251013', 'mean_local_std_20250814', 'mean_local_std_20250606', 'mean_sobel_20250814'],
            'Bajra_frac': ['area_ha', 'p25_20250619', 'temporal_variance', 'p75_20250619', 'mean_local_std_20250619'],
            'Groundnut_frac': ['area_ha', 'perimeter', 'mean_local_std_20250814', 'mean_local_std_20250606', 'change_magnitude']
        },
        'with_coords': {
            'Rice_frac': ['centroid_x', 'centroid_y', 'ratio_veg', 'area_ha', 'bbox_width', 'p50_20250619'],
            'Cotton_frac': ['centroid_x', 'centroid_y', 'area_ha', 'perimeter', 'diff_harvest', 'mean_20250606', 'ratio_harvest'],
            'Maize_frac': ['centroid_x', 'centroid_y', 'area_ha', 'mean_sobel_20251013', 'mean_local_std_20250814', 'mean_local_std_20250606', 'mean_sobel_20250814'],
            'Bajra_frac': ['centroid_x', 'centroid_y', 'area_ha', 'p25_20250619', 'temporal_variance', 'p75_20250619', 'mean_local_std_20250619'],
            'Groundnut_frac': ['centroid_x', 'centroid_y', 'area_ha', 'perimeter', 'mean_local_std_20250814', 'mean_local_std_20250606', 'change_magnitude']
        },
        'basic_only': {
            'Rice_frac': ['area_ha', 'bbox_width', 'p50_20250619', 'mean_20250619'],
            'Cotton_frac': ['area_ha', 'perimeter', 'diff_harvest', 'mean_20250606', 'p50_20250814'],
            'Maize_frac': ['area_ha', 'mean_local_std_20251013', 'mean_local_std_20250814', 'mean_local_std_20250606', 'diff_harvest'],
            'Bajra_frac': ['area_ha', 'p25_20250619', 'p50_20250619', 'p75_20250619', 'mean_local_std_20250619'],
            'Groundnut_frac': ['area_ha', 'perimeter', 'mean_local_std_20250814', 'mean_local_std_20250606', 'cumulative_change']
        }
    }
    
    ablations = [
        'Full_Pipeline',
        'Remove_Hybrid_Imputation',
        'Remove_KNN_Imputation',
        'Remove_Spatial_Imputation',
        'Remove_Advanced_Features',
        'Remove_Clipping',
        'Remove_Normalisation',
        'Add_Coordinates'
    ]
    
    ablation_results = {}
    
    for ab_name in ablations:
        print(f"Running Ablation: {ab_name}...", flush=True)
        oof_preds = {t: np.zeros(n_villages) for t in target_cols}
        
        for val_idx in range(n_villages):
            train_df = covered_df.drop(val_idx).reset_index(drop=True)
            val_df = covered_df.iloc[[val_idx]].reset_index(drop=True)
            
            combined_df = pd.concat([train_df, val_df], ignore_index=True)
            val_row_idx = len(combined_df) - 1
            combined_df.loc[val_row_idx, sar_cols] = np.nan
            
            # Setup Imputation based on ablation
            if ab_name == 'Remove_Hybrid_Imputation':
                imputer = SimpleImputer(strategy='median')
                df_imputed_knn = combined_df.copy()
                df_imputed_knn[all_cols] = imputer.fit_transform(combined_df[all_cols])
                df_imputed_spatial = df_imputed_knn
            elif ab_name == 'Remove_KNN_Imputation':
                # Median for all instead of KNN
                imputer = SimpleImputer(strategy='median')
                df_imputed_knn = combined_df.copy()
                df_imputed_knn[all_cols] = imputer.fit_transform(combined_df[all_cols])
                # Keep spatial for Bajra/Groundnut
                df_imputed_spatial = combined_df.copy()
                train_coords = train_df[['centroid_x', 'centroid_y']].values
                nn = NearestNeighbors(n_neighbors=1)
                nn.fit(train_coords)
                val_coord = val_df[['centroid_x', 'centroid_y']].values.reshape(1, -1)
                neighbor_idx = nn.kneighbors(val_coord, return_distance=False)[0][0]
                df_imputed_spatial.loc[val_row_idx, sar_cols] = train_df.loc[neighbor_idx, sar_cols]
            elif ab_name == 'Remove_Spatial_Imputation':
                # KNN for all
                imputer = KNNImputer(n_neighbors=min(6, len(train_df)))
                df_imputed_knn = combined_df.copy()
                df_imputed_knn[all_cols] = imputer.fit_transform(combined_df[all_cols])
                df_imputed_spatial = df_imputed_knn
            else:
                # Baseline Hybrid Imputation: KNN-6 for Rice/Cotton/Maize, Spatial 1-NN for Bajra/Groundnut
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
                
            # Train & Predict
            for target in target_cols:
                is_knn = target in ['Rice_frac', 'Cotton_frac', 'Maize_frac']
                df_imp = df_imputed_knn if is_knn else df_imputed_spatial
                
                # Feature config based on ablation
                if ab_name == 'Remove_Advanced_Features':
                    feats = feature_configs['basic_only'][target]
                elif ab_name == 'Add_Coordinates':
                    feats = feature_configs['with_coords'][target]
                else:
                    feats = feature_configs['no_coords'][target]
                    
                X_train = df_imp.iloc[:-1][feats].values
                y_train = train_df[target].values
                X_val = df_imp.iloc[[val_row_idx]][feats].values
                
                model = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                pred = model.predict(X_val)[0]
                
                if ab_name != 'Remove_Clipping':
                    pred = np.clip(pred, 0.0, 1.0)
                    
                oof_preds[target][val_idx] = pred
                
        # Compute Hectare MSE
        village_sq_errors = []
        for idx, row in covered_df.iterrows():
            area = row['area_ha']
            pred_fracs = np.array([oof_preds[t][idx] for t in target_cols])
            
            if ab_name != 'Remove_Normalisation':
                sum_pred = np.sum(pred_fracs)
                if sum_pred > 0:
                    norm_fracs = pred_fracs * 0.99 / sum_pred
                else:
                    norm_fracs = np.zeros(5)
            else:
                norm_fracs = pred_fracs
                
            pred_ha = norm_fracs * area
            true_ha = np.array([covered_df[t].values[idx] for t in target_cols]) * area
            village_sq_errors.append(np.mean((pred_ha - true_ha)**2))
            
        ablation_results[ab_name] = np.mean(village_sq_errors)
        
    print("\n========================================")
    print("Ablation Study Results (Hectares LOVO CV MSE):")
    print("========================================")
    sorted_ablations = sorted(ablation_results.items(), key=lambda x: x[1])
    for name, score in sorted_ablations:
        print(f"Ablation '{name}': {score:.4f} ha^2")
        
    # Write ablation results to CSV
    df_ab = pd.DataFrame(sorted_ablations, columns=['Ablation_Component', 'Hectares_MSE'])
    output_path = r"D:\PC\resources\project\outputs\ablation_results.csv"
    df_ab.to_csv(output_path, index=False)
    print(f"\nAblation results written to: {output_path}")

if __name__ == '__main__':
    run_ablation()
