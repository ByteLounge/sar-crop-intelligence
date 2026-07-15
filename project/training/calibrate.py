import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from search import get_base_data, extract_advanced_features

def run_calibration_test():
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
    
    safe_features = {
        'Rice_frac': ['area_ha', 'bbox_width', 'ratio_veg', 'p50_20250619'],
        'Cotton_frac': ['area_ha', 'perimeter', 'diff_harvest', 'mean_20250606', 'ratio_harvest'],
        'Maize_frac': ['area_ha', 'mean_sobel_20251013', 'mean_local_std_20250814', 'mean_local_std_20250606', 'mean_sobel_20250814'],
        'Bajra_frac': ['area_ha', 'p25_20250619', 'temporal_variance', 'p75_20250619', 'mean_local_std_20250619'],
        'Groundnut_frac': ['area_ha', 'perimeter', 'mean_local_std_20250814', 'mean_local_std_20250606', 'change_magnitude']
    }
    
    oof_predictions = {t: np.zeros(n_villages) for t in target_cols}
    true_values = {t: covered_df[t].values for t in target_cols}
    
    for val_idx in range(n_villages):
        train_df = covered_df.drop(val_idx).reset_index(drop=True)
        val_df = covered_df.iloc[[val_idx]].reset_index(drop=True)
        
        combined_df = pd.concat([train_df, val_df], ignore_index=True)
        val_row_idx = len(combined_df) - 1
        combined_df.loc[val_row_idx, sar_cols] = np.nan
        
        imputer = KNNImputer(n_neighbors=min(6, len(train_df)))
        df_imputed = combined_df.copy()
        df_imputed[all_cols] = imputer.fit_transform(combined_df[all_cols])
        
        for target in target_cols:
            feats = safe_features[target]
            X_train = df_imputed.iloc[:-1][feats].values
            y_train = train_df[target].values
            X_val = df_imputed.iloc[[val_row_idx]][feats].values
            
            model = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            oof_predictions[target][val_idx] = np.clip(model.predict(X_val)[0], 0.0, 1.0)
            
    print("\n========================================")
    print("Residual Bias & Variance Compression Audit:")
    print("========================================")
    for target in target_cols:
        preds = oof_predictions[target]
        trues = true_values[target]
        bias = np.mean(preds) - np.mean(trues)
        pred_std = np.std(preds)
        true_std = np.std(trues)
        
        # Uncalibrated MSE
        base_mse = mean_squared_error(trues, preds)
        
        # Test Linear Calibration (Fit LinearRegression on OOF predictions)
        # To evaluate calibration out-of-fold, we fit linear calibration in a nested loop
        calibrated_preds = np.zeros(n_villages)
        for fold in range(n_villages):
            train_preds = np.delete(preds, fold).reshape(-1, 1)
            train_trues = np.delete(trues, fold)
            val_pred = preds[fold].reshape(1, -1)
            
            calibrator = LinearRegression()
            calibrator.fit(train_preds, train_trues)
            cal_pred = calibrator.predict(val_pred)[0]
            calibrated_preds[fold] = np.clip(cal_pred, 0.0, 1.0)
            
        cal_mse = mean_squared_error(trues, calibrated_preds)
        print(f"Crop: {target}")
        print(f"  Mean True: {np.mean(trues):.4f} | Mean Pred: {np.mean(preds):.4f} | Bias: {bias:+.4f}")
        print(f"  Std True : {true_std:.4f} | Std Pred : {pred_std:.4f}")
        print(f"  Base MSE : {base_mse:.6f}")
        print(f"  Calibrated MSE: {cal_mse:.6f} | Change: {(cal_mse - base_mse)/base_mse*100:+.2f}%")
        print("-" * 40)

if __name__ == '__main__':
    run_calibration_test()
