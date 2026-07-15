import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge, BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import cv2
import warnings
warnings.filterwarnings('ignore')

from search import get_base_data, extract_advanced_features

def run_evaluation():
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
    
    target_cols = ['Rice_frac', 'Cotton_frac', 'Maize_frac', 'Bajra_frac', 'Groundnut_frac']
    
    # Feature sets to evaluate
    feature_sets = {
        # 1. Baseline features (used in the original Rank 119 submission)
        'original': {
            'Rice_frac': ['bbox_width', 'area_ha', 'p50_20250619', 'p75_20250619', 'p25_20250619', 'mean_20250619'],
            'Cotton_frac': ['centroid_y', 'perimeter', 'diff_harvest', 'p75_20250814', 'mean_20250814', 'p50_20250814'],
            'Maize_frac': ['centroid_y', 'centroid_x', 'diff_harvest', 'mean_local_std_20251013', 'p50_20250814', 'p75_20250606'],
            'Bajra_frac': ['centroid_y', 'centroid_x', 'p25_20250619', 'p50_20250619', 'mean_20250619', 'p75_20250619'],
            'Groundnut_frac': ['centroid_y', 'centroid_x', 'mean_local_std_20250606', 'mean_local_std_20250814', 'mean_local_std_20251013', 'cumulative_change']
        },
        # 2. Optimized features from our previous search (with coordinates)
        'previous_optimized': {
            'Rice_frac': ['centroid_x', 'centroid_y', 'ratio_veg'],
            'Cotton_frac': ['centroid_y', 'centroid_x', 'diff_harvest', 'mean_20250606', 'ratio_harvest'],
            'Maize_frac': ['centroid_y', 'centroid_x', 'mean_sobel_20251013', 'mean_local_std_20250814', 'mean_local_std_20250606', 'mean_sobel_20250814'],
            'Bajra_frac': ['p25_20250619', 'centroid_x', 'centroid_y', 'temporal_variance', 'p75_20250619'],
            'Groundnut_frac': ['centroid_x', 'centroid_y', 'area_ha']
        },
        # 3. Safe features (NO COORDINATES at all to prevent spatial extrapolation wildness!)
        'safe_no_coords': {
            'Rice_frac': ['area_ha', 'bbox_width', 'ratio_veg', 'p50_20250619'],
            'Cotton_frac': ['area_ha', 'perimeter', 'diff_harvest', 'mean_20250606', 'ratio_harvest'],
            'Maize_frac': ['area_ha', 'mean_sobel_20251013', 'mean_local_std_20250814', 'mean_local_std_20250606', 'mean_sobel_20250814'],
            'Bajra_frac': ['area_ha', 'p25_20250619', 'temporal_variance', 'p75_20250619', 'mean_local_std_20250619'],
            'Groundnut_frac': ['area_ha', 'perimeter', 'mean_local_std_20250814', 'mean_local_std_20250606', 'change_magnitude']
        }
    }
    
    # Models to evaluate
    models_pool = {
        'RF_100': lambda: RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
        'ET_100': lambda: ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
        'Ridge': lambda: Ridge(alpha=1.0, random_state=42),
        'BayesianRidge': lambda: BayesianRidge(),
        'ElasticNet': lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        'KNeighbors': lambda: KNeighborsRegressor(n_neighbors=3)
    }
    
    # We will run LOVO CV with out-of-sample imputation
    results = []
    
    for feat_name, feat_config in feature_sets.items():
        for model_name, model_fn in models_pool.items():
            print(f"Evaluating FeatureSet={feat_name}, Model={model_name}...", flush=True)
            
            # For each target crop, compute LOVO CV MSE under imputation
            crop_mses = {t: [] for t in target_cols}
            
            for val_idx in range(n_villages):
                train_df = covered_df.drop(val_idx).reset_index(drop=True)
                val_df = covered_df.iloc[[val_idx]].reset_index(drop=True)
                
                combined_df = pd.concat([train_df, val_df], ignore_index=True)
                val_row_idx = len(combined_df) - 1
                combined_df.loc[val_row_idx, sar_cols] = np.nan
                
                # We use simple KNN-6 imputer for this evaluation to compare fairly
                imputer = KNNImputer(n_neighbors=min(6, len(train_df)))
                df_imputed = combined_df.copy()
                df_imputed[all_cols] = imputer.fit_transform(combined_df[all_cols])
                
                for target in target_cols:
                    feats = feat_config[target]
                    
                    X_train = df_imputed.iloc[:-1][feats].values
                    y_train = train_df[target].values
                    X_val = df_imputed.iloc[[val_row_idx]][feats].values
                    y_val = val_df[target].values[0]
                    
                    # KNeighbors fails if features include coordinates with no scaling
                    # Skip KNeighbors on coordinates to prevent errors
                    if model_name == 'KNeighbors' and ('centroid_x' in feats or 'centroid_y' in feats):
                        crop_mses[target].append(np.nan)
                        continue
                        
                    model = model_fn()
                    model.fit(X_train, y_train)
                    pred = model.predict(X_val)[0]
                    pred = np.clip(pred, 0.0, 1.0)
                    crop_mses[target].append((pred - y_val)**2)
                    
            # Compute summary stats across the 17 folds
            for target in target_cols:
                errs = [e for e in crop_mses[target] if not np.isnan(e)]
                if len(errs) > 0:
                    mean_mse = np.mean(errs)
                    median_mse = np.median(errs)
                    std_mse = np.std(errs)
                    worst_fold = np.max(errs)
                else:
                    mean_mse, median_mse, std_mse, worst_fold = np.nan, np.nan, np.nan, np.nan
                    
                results.append({
                    'FeatureSet': feat_name,
                    'Model': model_name,
                    'Crop': target,
                    'LOVO_Mean_MSE': mean_mse,
                    'LOVO_Median_MSE': median_mse,
                    'LOVO_Std_MSE': std_mse,
                    'LOVO_Worst_MSE': worst_fold
                })
                
    df_res = pd.DataFrame(results)
    
    # Save the evaluation results
    output_path = r"D:\PC\resources\project\outputs\generalization_eval.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_res.to_csv(output_path, index=False)
    print(f"Results written to: {output_path}")
    
    # Display top performing models per crop for the 'safe_no_coords' set
    print("\n========================================")
    print("Safe (No Coordinates) Feature Set Performance:")
    print("========================================")
    df_safe = df_res[df_res['FeatureSet'] == 'safe_no_coords']
    for target in target_cols:
        print(f"\nCrop: {target}")
        df_target = df_safe[df_safe['Crop'] == target].sort_values('LOVO_Mean_MSE')
        print(df_target[['Model', 'LOVO_Mean_MSE', 'LOVO_Median_MSE', 'LOVO_Std_MSE', 'LOVO_Worst_MSE']].to_string(index=False))

if __name__ == '__main__':
    run_evaluation()
