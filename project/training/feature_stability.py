import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import ExtraTreesRegressor
import warnings
warnings.filterwarnings('ignore')

from search import get_base_data, extract_advanced_features

def run_feature_stability():
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
    
    # Safe features list (no coords)
    safe_features = {
        'Rice_frac': ['area_ha', 'bbox_width', 'ratio_veg', 'p50_20250619'],
        'Cotton_frac': ['area_ha', 'perimeter', 'diff_harvest', 'mean_20250606', 'ratio_harvest'],
        'Maize_frac': ['area_ha', 'mean_sobel_20251013', 'mean_local_std_20250814', 'mean_local_std_20250606', 'mean_sobel_20250814'],
        'Bajra_frac': ['area_ha', 'p25_20250619', 'temporal_variance', 'p75_20250619', 'mean_local_std_20250619'],
        'Groundnut_frac': ['area_ha', 'perimeter', 'mean_local_std_20250814', 'mean_local_std_20250606', 'change_magnitude']
    }
    
    stability_records = []
    
    for target in target_cols:
        feats = safe_features[target]
        fold_importances = {f: [] for f in feats}
        
        for val_idx in range(n_villages):
            train_df = covered_df.drop(val_idx).reset_index(drop=True)
            val_df = covered_df.iloc[[val_idx]].reset_index(drop=True)
            
            combined_df = pd.concat([train_df, val_df], ignore_index=True)
            val_row_idx = len(combined_df) - 1
            combined_df.loc[val_row_idx, sar_cols] = np.nan
            
            # Simple KNN Imputer for feature importance stability calculation
            imputer = KNNImputer(n_neighbors=min(6, len(train_df)))
            df_imputed = combined_df.copy()
            df_imputed[all_cols] = imputer.fit_transform(combined_df[all_cols])
            
            X_train = df_imputed.iloc[:-1][feats].values
            y_train = train_df[target].values
            
            model = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            for f_idx, f in enumerate(feats):
                fold_importances[f].append(model.feature_importances_[f_idx])
                
        for f in feats:
            imp_vals = fold_importances[f]
            mean_imp = np.mean(imp_vals)
            std_imp = np.std(imp_vals)
            cov_imp = std_imp / (mean_imp + 1e-5) # Coefficient of variation (stability index, lower is more stable)
            
            stability_records.append({
                'Crop': target,
                'Feature': f,
                'Mean_Importance': mean_imp,
                'Std_Importance': std_imp,
                'Stability_Index': cov_imp
            })
            
    df_stab = pd.DataFrame(stability_records)
    output_path = r"D:\PC\resources\project\outputs\feature_stability.csv"
    df_stab.to_csv(output_path, index=False)
    print(f"Feature stability analysis saved to: {output_path}")
    
    print("\n========================================")
    print("Feature Stability Analysis (Sorted by Crop & Stability Index):")
    print("========================================")
    print(df_stab.sort_values(['Crop', 'Stability_Index']).to_string(index=False))

if __name__ == '__main__':
    run_feature_stability()
