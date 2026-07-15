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
import warnings
warnings.filterwarnings('ignore')

from search import get_base_data, extract_advanced_features

def run_test():
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
            'model_fn': lambda: Ridge(alpha=0.01, random_state=42)
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
    
    for use_weight in [False, True]:
        oof_predictions = {t: np.zeros(n_villages) for t in target_cols}
        true_fractions = {t: covered_df[t].values for t in target_cols}
        
        for val_idx in range(n_villages):
            train_df = covered_df.drop(val_idx).reset_index(drop=True)
            val_df = covered_df.iloc[[val_idx]].reset_index(drop=True)
            
            combined_df = pd.concat([train_df, val_df], ignore_index=True)
            val_row_idx = len(combined_df) - 1
            combined_df.loc[val_row_idx, sar_cols] = np.nan
            
            # Impute
            imputer_knn8 = KNNImputer(n_neighbors=min(8, len(train_df)))
            df_knn8 = combined_df.copy()
            df_knn8[all_cols] = imputer_knn8.fit_transform(combined_df[all_cols])
            
            imputer_med = SimpleImputer(strategy='median')
            df_med = combined_df.copy()
            df_med[all_cols] = imputer_med.fit_transform(combined_df[all_cols])
            
            df_spatial = combined_df.copy()
            train_coords = train_df[['centroid_x', 'centroid_y']].values
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(train_coords)
            val_coord = val_df[['centroid_x', 'centroid_y']].values.reshape(1, -1)
            neighbor_idx = nn.kneighbors(val_coord, return_distance=False)[0][0]
            df_spatial.loc[val_row_idx, sar_cols] = train_df.loc[neighbor_idx, sar_cols]
            
            imputer_knn3 = KNNImputer(n_neighbors=min(3, len(train_df)))
            df_knn3 = combined_df.copy()
            df_knn3[all_cols] = imputer_knn3.fit_transform(combined_df[all_cols])
            
            for target in target_cols:
                cfg = pipeline_configs[target]
                feats = cfg['features']
                
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
                
                weights = train_df['area_ha'].values if use_weight else None
                
                if target == 'Maize_frac':
                    # Blend
                    m1 = Ridge(alpha=0.01, random_state=42)
                    m2 = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
                    if use_weight:
                        m1.fit(X_train, y_train, sample_weight=weights)
                        m2.fit(X_train, y_train, sample_weight=weights)
                    else:
                        m1.fit(X_train, y_train)
                        m2.fit(X_train, y_train)
                    pred = 0.95 * m1.predict(X_val)[0] + 0.05 * m2.predict(X_val)[0]
                elif target == 'Bajra_frac':
                    # Blend
                    m1 = ElasticNet(alpha=0.1, l1_ratio=0.7, random_state=42)
                    m2 = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
                    if use_weight:
                        m1.fit(X_train, y_train, sample_weight=weights)
                        m2.fit(X_train, y_train, sample_weight=weights)
                    else:
                        m1.fit(X_train, y_train)
                        m2.fit(X_train, y_train)
                    pred = 0.95 * m1.predict(X_val)[0] + 0.05 * m2.predict(X_val)[0]
                elif target == 'Groundnut_frac':
                    # KNeighborsRegressor doesn't support sample_weight
                    model = cfg['model_fn']()
                    model.fit(X_train, y_train)
                    pred = model.predict(X_val)[0]
                else:
                    model = cfg['model_fn']()
                    if use_weight:
                        model.fit(X_train, y_train, sample_weight=weights)
                    else:
                        model.fit(X_train, y_train)
                    pred = model.predict(X_val)[0]
                    
                pred = np.clip(pred, 0.0, 1.0)
                oof_predictions[target][val_idx] = pred

        # Compute MSE in Hectares
        village_results = []
        for idx, row in covered_df.iterrows():
            area = row['area_ha']
            pred_fracs = np.array([oof_predictions[t][idx] for t in target_cols])
            sum_pred = np.sum(pred_fracs)
            if sum_pred > 0:
                norm_fracs = pred_fracs * 0.99 / sum_pred
            else:
                norm_fracs = np.zeros(5)
            pred_ha = norm_fracs * area
            true_ha = np.array([true_fractions[t][idx] for t in target_cols]) * area
            village_results.append(np.mean((pred_ha - true_ha)**2))
            
        print(f"Sample Weighting = {use_weight} | Hectares MSE: {np.mean(village_results):.4f} ha^2")

if __name__ == '__main__':
    run_test()
