import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Ridge, BayesianRidge
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import cv2
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

from search import get_base_data, extract_advanced_features, evaluate_lovo_cv_imputed

def run_tuning():
    print("Loading data...")
    gdf_utm, df_geom, df_labels, df_total_px, flat_stack, flat_mask, H, W, dates = get_base_data()
    
    print("Extracting advanced features...")
    df_sar = extract_advanced_features(gdf_utm, flat_stack, flat_mask, H, W, dates)
    
    df_data = pd.merge(df_geom, df_sar, on='ID')
    df_data = pd.merge(df_data, df_labels, on='ID')
    df_data = pd.merge(df_data, df_total_px, on='ID')
    df_data['coverage'] = df_data['valid_pixels'] / df_data['total_pixels']
    
    covered_df = df_data[df_data['coverage'] > 0.35].copy().reset_index(drop=True)
    
    geom_cols = ['centroid_x', 'centroid_y', 'area_ha', 'perimeter', 'compactness', 'bbox_width', 'bbox_height']
    sar_cols = [c for c in df_sar.columns if c not in ['ID', 'valid_pixels']]
    all_cols = geom_cols + sar_cols
    
    # Best configs from search.py
    best_configs = {
        'Rice_frac': {
            'imputer': 'knn_8',
            'features': ['centroid_x', 'centroid_y', 'ratio_veg']
        },
        'Cotton_frac': {
            'imputer': 'median',
            'features': ['centroid_y', 'centroid_x', 'diff_harvest', 'mean_20250606', 'ratio_harvest']
        },
        'Maize_frac': {
            'imputer': 'spatial_1nn',
            'features': ['centroid_y', 'centroid_x', 'mean_sobel_20251013', 'mean_local_std_20250814', 'mean_local_std_20250606', 'mean_sobel_20250814']
        },
        'Bajra_frac': {
            'imputer': 'spatial_1nn',
            'features': ['p25_20250619', 'centroid_x', 'centroid_y', 'temporal_variance', 'p75_20250619']
        },
        'Groundnut_frac': {
            'imputer': 'knn_3',
            'features': ['centroid_x', 'centroid_y', 'area_ha']
        }
    }
    
    tuned_configs = {}
    
    # Let's tune each crop
    
    # 1. RICE FRAC
    print("\n--- Tuning Rice_frac ---")
    rice_feats = best_configs['Rice_frac']['features']
    rice_imp = best_configs['Rice_frac']['imputer']
    
    best_rice_mse = float('inf')
    best_rice_alpha = None
    for alpha in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
        model_fn = lambda: Ridge(alpha=alpha, random_state=42)
        mse = evaluate_lovo_cv_imputed(covered_df, 'Rice_frac', rice_feats, model_fn, rice_imp, geom_cols, sar_cols, all_cols)
        if mse < best_rice_mse:
            best_rice_mse = mse
            best_rice_alpha = alpha
    print(f"Best Ridge Alpha for Rice: {best_rice_alpha} | MSE: {best_rice_mse:.6f}")
    
    # Check if ensembling with ExtraTrees helps
    best_rice_blend_mse = best_rice_mse
    best_rice_w = 1.0
    for w in np.linspace(0.5, 1.0, 11):
        class RiceBlend:
            def __init__(self):
                self.m1 = Ridge(alpha=best_rice_alpha, random_state=42)
                self.m2 = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
            def fit(self, X, y):
                self.m1.fit(X, y)
                self.m2.fit(X, y)
            def predict(self, X):
                return w * self.m1.predict(X) + (1.0 - w) * self.m2.predict(X)
        mse = evaluate_lovo_cv_imputed(covered_df, 'Rice_frac', rice_feats, RiceBlend, rice_imp, geom_cols, sar_cols, all_cols)
        if mse < best_rice_blend_mse:
            best_rice_blend_mse = mse
            best_rice_w = w
    print(f"Rice Blend (Ridge={best_rice_w:.2f}, ET={1.0-best_rice_w:.2f}) | MSE: {best_rice_blend_mse:.6f}")
    tuned_configs['Rice_frac'] = {
        'model_name': f"Ridge(alpha={best_rice_alpha})",
        'imputer': rice_imp,
        'features': rice_feats,
        'mse': best_rice_blend_mse,
        'weights': (best_rice_w, 1.0 - best_rice_w)
    }
    
    # 2. COTTON FRAC
    print("\n--- Tuning Cotton_frac ---")
    cotton_feats = best_configs['Cotton_frac']['features']
    cotton_imp = best_configs['Cotton_frac']['imputer']
    
    # Tune BayesianRidge (no main hyperparameters to tune other than defaults, but let's test ensembling with RF)
    model_fn = lambda: BayesianRidge()
    best_cotton_mse = evaluate_lovo_cv_imputed(covered_df, 'Cotton_frac', cotton_feats, model_fn, cotton_imp, geom_cols, sar_cols, all_cols)
    print(f"BayesianRidge baseline Cotton MSE: {best_cotton_mse:.6f}")
    
    best_cotton_blend_mse = best_cotton_mse
    best_cotton_w = 1.0
    for w in np.linspace(0.5, 1.0, 11):
        class CottonBlend:
            def __init__(self):
                self.m1 = BayesianRidge()
                self.m2 = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
            def fit(self, X, y):
                self.m1.fit(X, y)
                self.m2.fit(X, y)
            def predict(self, X):
                return w * self.m1.predict(X) + (1.0 - w) * self.m2.predict(X)
        mse = evaluate_lovo_cv_imputed(covered_df, 'Cotton_frac', cotton_feats, CottonBlend, cotton_imp, geom_cols, sar_cols, all_cols)
        if mse < best_cotton_blend_mse:
            best_cotton_blend_mse = mse
            best_cotton_w = w
    print(f"Cotton Blend (BayesianRidge={best_cotton_w:.2f}, RF={1.0-best_cotton_w:.2f}) | MSE: {best_cotton_blend_mse:.6f}")
    tuned_configs['Cotton_frac'] = {
        'model_name': "BayesianRidge",
        'imputer': cotton_imp,
        'features': cotton_feats,
        'mse': best_cotton_blend_mse,
        'weights': (best_cotton_w, 1.0 - best_cotton_w)
    }
    
    # 3. MAIZE FRAC
    print("\n--- Tuning Maize_frac ---")
    maize_feats = best_configs['Maize_frac']['features']
    maize_imp = best_configs['Maize_frac']['imputer']
    
    best_maize_mse = float('inf')
    best_maize_alpha = None
    for alpha in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
        model_fn = lambda: Ridge(alpha=alpha, random_state=42)
        mse = evaluate_lovo_cv_imputed(covered_df, 'Maize_frac', maize_feats, model_fn, maize_imp, geom_cols, sar_cols, all_cols)
        if mse < best_maize_mse:
            best_maize_mse = mse
            best_maize_alpha = alpha
    print(f"Best Ridge Alpha for Maize: {best_maize_alpha} | MSE: {best_maize_mse:.6f}")
    
    best_maize_blend_mse = best_maize_mse
    best_maize_w = 1.0
    for w in np.linspace(0.5, 1.0, 11):
        class MaizeBlend:
            def __init__(self):
                self.m1 = Ridge(alpha=best_maize_alpha, random_state=42)
                self.m2 = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
            def fit(self, X, y):
                self.m1.fit(X, y)
                self.m2.fit(X, y)
            def predict(self, X):
                return w * self.m1.predict(X) + (1.0 - w) * self.m2.predict(X)
        mse = evaluate_lovo_cv_imputed(covered_df, 'Maize_frac', maize_feats, MaizeBlend, maize_imp, geom_cols, sar_cols, all_cols)
        if mse < best_maize_blend_mse:
            best_maize_blend_mse = mse
            best_maize_w = w
    print(f"Maize Blend (Ridge={best_maize_w:.2f}, ET={1.0-best_maize_w:.2f}) | MSE: {best_maize_blend_mse:.6f}")
    tuned_configs['Maize_frac'] = {
        'model_name': f"Ridge(alpha={best_maize_alpha})",
        'imputer': maize_imp,
        'features': maize_feats,
        'mse': best_maize_blend_mse,
        'weights': (best_maize_w, 1.0 - best_maize_w)
    }
    
    # 4. BAJRA FRAC
    print("\n--- Tuning Bajra_frac ---")
    bajra_feats = best_configs['Bajra_frac']['features']
    bajra_imp = best_configs['Bajra_frac']['imputer']
    
    best_bajra_mse = float('inf')
    best_bajra_alpha = None
    best_bajra_l1 = None
    for alpha in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        for l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
            model_fn = lambda: ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            mse = evaluate_lovo_cv_imputed(covered_df, 'Bajra_frac', bajra_feats, model_fn, bajra_imp, geom_cols, sar_cols, all_cols)
            if mse < best_bajra_mse:
                best_bajra_mse = mse
                best_bajra_alpha = alpha
                best_bajra_l1 = l1_ratio
    print(f"Best ElasticNet Alpha: {best_bajra_alpha}, L1: {best_bajra_l1} | MSE: {best_bajra_mse:.6f}")
    
    best_bajra_blend_mse = best_bajra_mse
    best_bajra_w = 1.0
    for w in np.linspace(0.5, 1.0, 11):
        class BajraBlend:
            def __init__(self):
                self.m1 = ElasticNet(alpha=best_bajra_alpha, l1_ratio=best_bajra_l1, random_state=42)
                self.m2 = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
            def fit(self, X, y):
                self.m1.fit(X, y)
                self.m2.fit(X, y)
            def predict(self, X):
                return w * self.m1.predict(X) + (1.0 - w) * self.m2.predict(X)
        mse = evaluate_lovo_cv_imputed(covered_df, 'Bajra_frac', bajra_feats, BajraBlend, bajra_imp, geom_cols, sar_cols, all_cols)
        if mse < best_bajra_blend_mse:
            best_bajra_blend_mse = mse
            best_bajra_w = w
    print(f"Bajra Blend (ElasticNet={best_bajra_w:.2f}, ET={1.0-best_bajra_w:.2f}) | MSE: {best_bajra_blend_mse:.6f}")
    tuned_configs['Bajra_frac'] = {
        'model_name': f"ElasticNet(alpha={best_bajra_alpha},l1={best_bajra_l1})",
        'imputer': bajra_imp,
        'features': bajra_feats,
        'mse': best_bajra_blend_mse,
        'weights': (best_bajra_w, 1.0 - best_bajra_w)
    }
    
    # 5. GROUNDNUT FRAC
    print("\n--- Tuning Groundnut_frac ---")
    gn_feats = best_configs['Groundnut_frac']['features']
    gn_imp = best_configs['Groundnut_frac']['imputer']
    
    best_gn_mse = float('inf')
    best_gn_k = None
    for k in [2, 3, 4, 5, 6, 7]:
        model_fn = lambda: KNeighborsRegressor(n_neighbors=k)
        mse = evaluate_lovo_cv_imputed(covered_df, 'Groundnut_frac', gn_feats, model_fn, gn_imp, geom_cols, sar_cols, all_cols)
        if mse < best_gn_mse:
            best_gn_mse = mse
            best_gn_k = k
    print(f"Best KNeighbors K: {best_gn_k} | MSE: {best_gn_mse:.6f}")
    
    best_gn_blend_mse = best_gn_mse
    best_gn_w = 1.0
    for w in np.linspace(0.5, 1.0, 11):
        class GnBlend:
            def __init__(self):
                self.m1 = KNeighborsRegressor(n_neighbors=best_gn_k)
                self.m2 = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
            def fit(self, X, y):
                self.m1.fit(X, y)
                self.m2.fit(X, y)
            def predict(self, X):
                return w * self.m1.predict(X) + (1.0 - w) * self.m2.predict(X)
        mse = evaluate_lovo_cv_imputed(covered_df, 'Groundnut_frac', gn_feats, GnBlend, gn_imp, geom_cols, sar_cols, all_cols)
        if mse < best_gn_blend_mse:
            best_gn_blend_mse = mse
            best_gn_w = w
    print(f"Groundnut Blend (KNeighbors={best_gn_w:.2f}, RF={1.0-best_gn_w:.2f}) | MSE: {best_gn_blend_mse:.6f}")
    tuned_configs['Groundnut_frac'] = {
        'model_name': f"KNeighbors(k={best_gn_k})",
        'imputer': gn_imp,
        'features': gn_feats,
        'mse': best_gn_blend_mse,
        'weights': (best_gn_w, 1.0 - best_gn_w)
    }
    
    print("\n\n========================================\nTUNED CONFIGURATIONS SUMMARY\n========================================")
    for target in target_cols:
        cfg = tuned_configs[target]
        print(f"Crop: {target}")
        print(f"  Best Model: {cfg['model_name']}")
        print(f"  Imputer: {cfg['imputer']}")
        print(f"  Features: {cfg['features']}")
        print(f"  Ensemble Weights: {cfg['model_name']}={cfg['weights'][0]:.2f}, TreeModel={cfg['weights'][1]:.2f}")
        print(f"  Tuned LOVO CV MSE: {cfg['mse']:.6f}")
        print("-" * 40)

if __name__ == '__main__':
    run_tuning()
