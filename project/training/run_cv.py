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
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

from search import get_base_data, extract_advanced_features

def run_validation_pipeline():
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
    
    # Tuned pipeline configurations
    pipeline_configs = {
        'Rice_frac': {
            'imputer': 'knn_8',
            'features': ['centroid_x', 'centroid_y', 'ratio_veg'],
            'model_fn': lambda: Ridge(alpha=0.1, random_state=42)
        },
        'Cotton_frac': {
            'imputer': 'median',
            'features': ['centroid_y', 'centroid_x', 'diff_harvest', 'mean_20250606', 'ratio_harvest'],
            'model_fn': lambda: BayesianRidge()
        },
        'Maize_frac': {
            'imputer': 'spatial_1nn',
            'features': ['centroid_y', 'centroid_x', 'mean_sobel_20251013', 'mean_local_std_20250814', 'mean_local_std_20250606', 'mean_sobel_20250814'],
            # Blend Ridge(alpha=0.01) (0.95) + ExtraTrees (0.05)
            'model_fn': lambda: Ridge(alpha=0.01, random_state=42) # we can implement the blend directly in loop
        },
        'Bajra_frac': {
            'imputer': 'spatial_1nn',
            'features': ['p25_20250619', 'centroid_x', 'centroid_y', 'temporal_variance', 'p75_20250619'],
            'model_fn': lambda: ElasticNet(alpha=0.1, l1_ratio=0.7, random_state=42)
        },
        'Groundnut_frac': {
            'imputer': 'knn_3',
            'features': ['centroid_x', 'centroid_y', 'area_ha'],
            'model_fn': lambda: KNeighborsRegressor(n_neighbors=3)
        }
    }
    
    target_cols = ['Rice_frac', 'Cotton_frac', 'Maize_frac', 'Bajra_frac', 'Groundnut_frac']
    crop_names_ha = ['Rice_ha', 'Cotton_ha', 'Maize_ha', 'Bajra_ha', 'Groundnut_ha']
    
    # Store out-of-fold predictions
    oof_predictions = {t: np.zeros(n_villages) for t in target_cols}
    true_fractions = {t: covered_df[t].values for t in target_cols}
    
    # Folds
    for val_idx in range(n_villages):
        train_df = covered_df.drop(val_idx).reset_index(drop=True)
        val_df = covered_df.iloc[[val_idx]].reset_index(drop=True)
        
        combined_df = pd.concat([train_df, val_df], ignore_index=True)
        val_row_idx = len(combined_df) - 1
        combined_df.loc[val_row_idx, sar_cols] = np.nan
        
        # 1. Impute KNN-8 (for Rice)
        imputer_knn8 = KNNImputer(n_neighbors=min(8, len(train_df)))
        combined_knn8 = imputer_knn8.fit_transform(combined_df[all_cols])
        df_knn8 = combined_df.copy()
        df_knn8[all_cols] = combined_knn8
        
        # 2. Impute Median (for Cotton)
        imputer_med = SimpleImputer(strategy='median')
        combined_med = imputer_med.fit_transform(combined_df[all_cols])
        df_med = combined_df.copy()
        df_med[all_cols] = combined_med
        
        # 3. Impute Spatial 1-NN (for Maize, Bajra)
        df_spatial = combined_df.copy()
        train_coords = train_df[['centroid_x', 'centroid_y']].values
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(train_coords)
        val_coord = val_df[['centroid_x', 'centroid_y']].values.reshape(1, -1)
        neighbor_idx = nn.kneighbors(val_coord, return_distance=False)[0][0]
        df_spatial.loc[val_row_idx, sar_cols] = train_df.loc[neighbor_idx, sar_cols]
        
        # 4. Impute KNN-3 (for Groundnut)
        imputer_knn3 = KNNImputer(n_neighbors=min(3, len(train_df)))
        combined_knn3 = imputer_knn3.fit_transform(combined_df[all_cols])
        df_knn3 = combined_df.copy()
        df_knn3[all_cols] = combined_knn3
        
        # Train & Predict each target
        for target in target_cols:
            cfg = pipeline_configs[target]
            feats = cfg['features']
            
            # Select correct imputed df
            if target == 'Rice_frac':
                df_tr = df_knn8.iloc[:-1]
                df_val = df_knn8.iloc[[val_row_idx]]
            elif target == 'Cotton_frac':
                df_tr = df_med.iloc[:-1]
                df_val = df_med.iloc[[val_row_idx]]
            elif target in ['Maize_frac', 'Bajra_frac']:
                df_tr = df_spatial.iloc[:-1]
                df_val = df_spatial.iloc[[val_row_idx]]
            else:
                df_tr = df_knn3.iloc[:-1]
                df_val = df_knn3.iloc[[val_row_idx]]
                
            X_train = df_tr[feats].values
            y_train = train_df[target].values
            X_val = df_val[feats].values
            
            # Fit models
            if target == 'Maize_frac':
                # Blend: 95% Ridge + 5% ET
                m1 = Ridge(alpha=0.01, random_state=42)
                m2 = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
                m1.fit(X_train, y_train)
                m2.fit(X_train, y_train)
                pred = 0.95 * m1.predict(X_val)[0] + 0.05 * m2.predict(X_val)[0]
            elif target == 'Bajra_frac':
                # Blend: 95% ElasticNet + 5% ET
                m1 = ElasticNet(alpha=0.1, l1_ratio=0.7, random_state=42)
                m2 = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
                m1.fit(X_train, y_train)
                m2.fit(X_train, y_train)
                pred = 0.95 * m1.predict(X_val)[0] + 0.05 * m2.predict(X_val)[0]
            else:
                model = cfg['model_fn']()
                model.fit(X_train, y_train)
                pred = model.predict(X_val)[0]
                
            # Constraint: Clip predictions to [0.0, 1.0]
            pred = np.clip(pred, 0.0, 1.0)
            oof_predictions[target][val_idx] = pred

    # Perform residual analysis
    # Compute error metrics on fractions
    frac_metrics = []
    for target in target_cols:
        mse = mean_squared_error(true_fractions[target], oof_predictions[target])
        rmse = np.sqrt(mse)
        frac_metrics.append({
            'Crop': target,
            'LOVO_CV_MSE': mse,
            'LOVO_CV_RMSE': rmse
        })
    print("\n========================================\nLOVO CV Fraction Performance:\n========================================")
    print(pd.DataFrame(frac_metrics).to_string(index=False))
    
    # Calculate Hectares residuals
    # Blended hectares:
    # Final Frac_i = C_i * Obs_i + (1 - C_i) * Pred_i
    # For out-of-fold validation, we treat the village as zero-coverage (C_i = 0), so blended fraction is exactly the OOF prediction.
    # We then apply constraint normalization over the 5 crops:
    # Target Sum = 0.99 (simulating zero coverage)
    # Norm Frac = pred * 0.99 / sum(pred)
    # Pred Hectares = Norm Frac * area_ha
    # True Hectares = True Frac * area_ha (using the K-Means fraction * area)
    
    village_results = []
    for idx, row in covered_df.iterrows():
        v_id = row['ID']
        v_name = row['VILLAGE']
        area = row['area_ha']
        
        # Predicted fractions for this village
        pred_fracs = np.array([oof_predictions[t][idx] for t in target_cols])
        sum_pred = np.sum(pred_fracs)
        if sum_pred > 0:
            norm_fracs = pred_fracs * 0.99 / sum_pred
        else:
            norm_fracs = np.zeros(len(target_cols))
            
        pred_ha = norm_fracs * area
        
        # True fractions and hectares
        true_fracs = np.array([true_fractions[t][idx] for t in target_cols])
        true_ha = true_fracs * area
        
        residuals_ha = pred_ha - true_ha
        sq_errs = residuals_ha ** 2
        
        village_results.append({
            'ID': v_id,
            'VILLAGE': v_name,
            'Area_ha': area,
            'True_Rice': true_ha[0], 'Pred_Rice': pred_ha[0], 'Err_Rice': residuals_ha[0],
            'True_Cotton': true_ha[1], 'Pred_Cotton': pred_ha[1], 'Err_Cotton': residuals_ha[1],
            'True_Maize': true_ha[2], 'Pred_Maize': pred_ha[2], 'Err_Maize': residuals_ha[2],
            'True_Bajra': true_ha[3], 'Pred_Bajra': pred_ha[3], 'Err_Bajra': residuals_ha[3],
            'True_Groundnut': true_ha[4], 'Pred_Groundnut': pred_ha[4], 'Err_Groundnut': residuals_ha[4],
            'Total_MSE_ha': np.mean(sq_errs)
        })
        
    df_villages = pd.DataFrame(village_results)
    
    print("\n========================================\nResidual Analysis per Covered Village (Hectares):\n========================================")
    cols_to_show = ['ID', 'VILLAGE', 'Area_ha', 'Err_Rice', 'Err_Cotton', 'Err_Maize', 'Err_Bajra', 'Err_Groundnut', 'Total_MSE_ha']
    print(df_villages[cols_to_show].sort_values('Total_MSE_ha', ascending=False).to_string(index=False))
    
    overall_mse_ha = np.mean(df_villages['Total_MSE_ha'])
    print(f"\nOverall Mean Squared Error (MSE) in Hectares across all 5 crops: {overall_mse_ha:.4f} ha^2")

if __name__ == '__main__':
    run_validation_pipeline()
