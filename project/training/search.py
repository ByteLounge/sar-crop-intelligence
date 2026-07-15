import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge, BayesianRidge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import cv2
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

def get_base_data():
    workspace_dir = r"D:\PC\resources"
    shp_path = os.path.join(workspace_dir, "villages_clean", "villages_clean.shp")
    gdf = gpd.read_file(shp_path)
    gdf_utm = gdf.to_crs("EPSG:32643")
    
    df_geom = []
    for idx, row in gdf_utm.iterrows():
        geom = row['geometry']
        centroid = geom.centroid
        bbox = geom.bounds
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        area = geom.area / 10000.0
        perimeter = geom.length
        compactness = (4 * np.pi * geom.area) / (perimeter ** 2) if perimeter > 0 else 0
        df_geom.append({
            'ID': row['ID'],
            'VILLAGE': row['VILLAGE'],
            'centroid_x': centroid.x,
            'centroid_y': centroid.y,
            'area_ha': area,
            'perimeter': perimeter,
            'compactness': compactness,
            'bbox_width': w,
            'bbox_height': h
        })
    df_geom = pd.DataFrame(df_geom)
    
    aligned_dir = os.path.join(workspace_dir, "aligned_images")
    dates = ["20250606", "20250619", "20250814", "20251013"]
    tif_paths = [os.path.join(aligned_dir, f"capella_hh_{d}_10m.tif") for d in dates]
    
    images = []
    for p in tif_paths:
        with rasterio.open(p) as src:
            images.append(src.read(1))
            meta = src.meta.copy()
            
    stack = np.stack(images, axis=0)
    H, W = stack.shape[1], stack.shape[2]
    
    shapes = [(row['geometry'], row['ID']) for idx, row in gdf_utm.iterrows()]
    village_mask = rasterize(
        shapes,
        out_shape=(H, W),
        transform=meta['transform'],
        fill=0,
        all_touched=True,
        dtype='int32'
    )
    
    flat_stack = stack.reshape(4, -1).T.astype(float)
    flat_mask = village_mask.flatten()
    
    # Target crop fractions
    in_village = flat_mask > 0
    X_village = flat_stack[in_village]
    village_ids = flat_mask[in_village]
    
    mean_vals = X_village.mean(axis=1)
    min_vals = X_village.min(axis=1)
    max_vals = X_village.max(axis=1)
    is_water = (mean_vals < 20) & (max_vals < 40)
    is_builtup = (mean_vals > 160) & (min_vals > 80)
    is_veg = ~is_water & ~is_builtup
    
    X_veg = X_village[is_veg]
    X_veg_mean = X_veg.mean(axis=1, keepdims=True)
    X_veg_std = X_veg.std(axis=1, keepdims=True) + 1e-5
    X_veg_norm = (X_veg - X_veg_mean) / X_veg_std
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_veg_norm)
    
    crop_mapping = {
        0: 'Cotton_frac',
        1: 'Groundnut_frac',
        2: 'Maize_frac',
        3: 'Rice_frac',
        4: 'Bajra_frac'
    }
    
    pixel_crops = np.full(len(X_village), -1, dtype=int)
    pixel_crops[is_veg] = labels
    
    labels_list = []
    for idx, row in gdf_utm.iterrows():
        v_id = row['ID']
        v_pixels = (village_ids == v_id)
        v_crop_pixels = pixel_crops[v_pixels]
        X_v = X_village[v_pixels]
        is_nodata = (X_v == 0).all(axis=1)
        n_valid = np.sum(~is_nodata)
        
        crop_fracs = {'ID': v_id}
        for c_id, crop_name in crop_mapping.items():
            if n_valid > 0:
                count = np.sum(v_crop_pixels == c_id)
                crop_fracs[crop_name] = count / n_valid
            else:
                crop_fracs[crop_name] = 0.0
        labels_list.append(crop_fracs)
    df_labels = pd.DataFrame(labels_list)
    
    total_px = [{'ID': row['ID'], 'total_pixels': np.sum(flat_mask == row['ID'])} for idx, row in gdf_utm.iterrows()]
    df_total_px = pd.DataFrame(total_px)
    
    return gdf_utm, df_geom, df_labels, df_total_px, flat_stack, flat_mask, H, W, dates

def extract_advanced_features(gdf_utm, flat_stack, flat_mask, H, W, dates):
    # Compute Local Standard Deviation (Texture Feature) using box filter
    local_std_images = []
    sobel_images = []
    stack_3d = flat_stack.T.reshape(len(dates), H, W)
    for d_idx in range(len(dates)):
        img = stack_3d[d_idx].astype(float)
        
        # Local std
        local_mean = cv2.boxFilter(img, -1, (3, 3))
        local_sq_mean = cv2.boxFilter(img**2, -1, (3, 3))
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
        local_std_images.append(local_std)
        
        # Sobel gradient magnitude
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        sobel_images.append(grad_mag)
        
    local_std_stack = np.stack(local_std_images, axis=0)
    flat_std_stack = local_std_stack.reshape(len(dates), -1).T
    
    sobel_stack = np.stack(sobel_images, axis=0)
    flat_sobel_stack = sobel_stack.reshape(len(dates), -1).T
    
    def calc_entropy(vals):
        hist, _ = np.histogram(vals, bins=10, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
        
    sar_features = []
    for idx, row in gdf_utm.iterrows():
        v_id = row['ID']
        v_pixels = (flat_mask == v_id)
        X_v = flat_stack[v_pixels]
        X_std_v = flat_std_stack[v_pixels]
        X_sobel_v = flat_sobel_stack[v_pixels]
        
        is_nodata = (X_v == 0).all(axis=1)
        n_valid = np.sum(~is_nodata)
        
        v_sar = {'ID': v_id, 'valid_pixels': n_valid}
        
        if n_valid > 0:
            X_valid = X_v[~is_nodata]
            X_std_valid = X_std_v[~is_nodata]
            X_sobel_valid = X_sobel_v[~is_nodata]
            
            # Simple stats per date
            for d_idx, d in enumerate(dates):
                vals = X_valid[:, d_idx]
                v_sar[f'mean_{d}'] = np.mean(vals)
                v_sar[f'std_{d}'] = np.std(vals)
                v_sar[f'cv_{d}'] = np.std(vals) / (np.mean(vals) + 1e-5)
                v_sar[f'skew_{d}'] = skew(vals)
                v_sar[f'kurt_{d}'] = kurtosis(vals)
                v_sar[f'p25_{d}'] = np.percentile(vals, 25)
                v_sar[f'p50_{d}'] = np.percentile(vals, 50)
                v_sar[f'p75_{d}'] = np.percentile(vals, 75)
                v_sar[f'mean_local_std_{d}'] = np.mean(X_std_valid[:, d_idx])
                v_sar[f'mean_sobel_{d}'] = np.mean(X_sobel_valid[:, d_idx])
                v_sar[f'iqr_{d}'] = v_sar[f'p75_{d}'] - v_sar[f'p25_{d}']
                v_sar[f'entropy_{d}'] = calc_entropy(vals)
                
            # Temporal difference features
            v_sar['diff_sowing'] = v_sar['mean_20250619'] - v_sar['mean_20250606']
            v_sar['diff_veg'] = v_sar['mean_20250814'] - v_sar['mean_20250619']
            v_sar['diff_harvest'] = v_sar['mean_20251013'] - v_sar['mean_20250814']
            
            # Temporal ratios
            v_sar['ratio_sowing'] = (v_sar['mean_20250619'] + 1e-5) / (v_sar['mean_20250606'] + 1e-5)
            v_sar['ratio_veg'] = (v_sar['mean_20250814'] + 1e-5) / (v_sar['mean_20250619'] + 1e-5)
            v_sar['ratio_harvest'] = (v_sar['mean_20251013'] + 1e-5) / (v_sar['mean_20250814'] + 1e-5)
            
            # Growth rates and change magnitude
            v_sar['growth_rate'] = (v_sar['mean_20250814'] - v_sar['mean_20250606']) / 2.0
            v_sar['change_magnitude'] = np.mean(np.max(X_valid, axis=1) - np.min(X_valid, axis=1))
            v_sar['cumulative_change'] = np.sum(np.abs(np.diff(X_valid, axis=1)), axis=1).mean()
            v_sar['temporal_variance'] = np.var([v_sar[f'mean_{d}'] for d in dates])
            v_sar['slope'] = (3.0*v_sar['mean_20251013'] + v_sar['mean_20250814'] - v_sar['mean_20250619'] - 3.0*v_sar['mean_20250606']) / 10.0
            
            # Land cover fractions
            mean_vals_v = X_v.mean(axis=1)
            min_vals_v = X_v.min(axis=1)
            max_vals_v = X_v.max(axis=1)
            is_water_v = (mean_vals_v < 20) & (max_vals_v < 40)
            is_builtup_v = (mean_vals_v > 160) & (min_vals_v > 80)
            is_veg_v = ~is_water_v & ~is_builtup_v
            v_sar['water_fraction'] = np.mean(is_water_v)
            v_sar['builtup_fraction'] = np.mean(is_builtup_v)
            v_sar['veg_fraction'] = np.mean(is_veg_v)
            
        else:
            for d in dates:
                v_sar[f'mean_{d}'] = np.nan
                v_sar[f'std_{d}'] = np.nan
                v_sar[f'cv_{d}'] = np.nan
                v_sar[f'skew_{d}'] = np.nan
                v_sar[f'kurt_{d}'] = np.nan
                v_sar[f'p25_{d}'] = np.nan
                v_sar[f'p50_{d}'] = np.nan
                v_sar[f'p75_{d}'] = np.nan
                v_sar[f'mean_local_std_{d}'] = np.nan
                v_sar[f'mean_sobel_{d}'] = np.nan
                v_sar[f'iqr_{d}'] = np.nan
                v_sar[f'entropy_{d}'] = np.nan
            v_sar['diff_sowing'] = np.nan
            v_sar['diff_veg'] = np.nan
            v_sar['diff_harvest'] = np.nan
            v_sar['ratio_sowing'] = np.nan
            v_sar['ratio_veg'] = np.nan
            v_sar['ratio_harvest'] = np.nan
            v_sar['growth_rate'] = np.nan
            v_sar['change_magnitude'] = np.nan
            v_sar['cumulative_change'] = np.nan
            v_sar['temporal_variance'] = np.nan
            v_sar['slope'] = np.nan
            v_sar['water_fraction'] = np.nan
            v_sar['builtup_fraction'] = np.nan
            v_sar['veg_fraction'] = np.nan
            
        sar_features.append(v_sar)
    return pd.DataFrame(sar_features)

def evaluate_lovo_cv_imputed(covered_df, target, features, model_fn, imputer_type, geom_cols, sar_cols, all_cols):
    n_villages = len(covered_df)
    errors = []
    
    for val_idx in range(n_villages):
        train_df = covered_df.drop(val_idx).reset_index(drop=True)
        val_df = covered_df.iloc[[val_idx]].reset_index(drop=True)
        
        # Combine train and masked val
        combined_df = pd.concat([train_df, val_df], ignore_index=True)
        val_row_idx = len(combined_df) - 1
        combined_df.loc[val_row_idx, sar_cols] = np.nan
        
        # Apply imputer
        if imputer_type.startswith('knn_'):
            k = int(imputer_type.split('_')[1])
            imputer = KNNImputer(n_neighbors=min(k, len(train_df)))
            combined_imputed = imputer.fit_transform(combined_df[all_cols])
            df_imputed = combined_df.copy()
            df_imputed[all_cols] = combined_imputed
        elif imputer_type == 'spatial_1nn':
            df_imputed = combined_df.copy()
            train_coords = train_df[['centroid_x', 'centroid_y']].values
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(train_coords)
            val_coord = val_df[['centroid_x', 'centroid_y']].values.reshape(1, -1)
            neighbor_idx = nn.kneighbors(val_coord, return_distance=False)[0][0]
            df_imputed.loc[val_row_idx, sar_cols] = train_df.loc[neighbor_idx, sar_cols]
        elif imputer_type == 'iterative_rf':
            # MissForest simulation using IterativeImputer with RandomForestRegressor
            estimator = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42, n_jobs=-1)
            imputer = IterativeImputer(estimator=estimator, max_iter=5, random_state=42)
            combined_imputed = imputer.fit_transform(combined_df[all_cols])
            df_imputed = combined_df.copy()
            df_imputed[all_cols] = combined_imputed
        elif imputer_type == 'median':
            imputer = SimpleImputer(strategy='median')
            combined_imputed = imputer.fit_transform(combined_df[all_cols])
            df_imputed = combined_df.copy()
            df_imputed[all_cols] = combined_imputed
        else:
            raise ValueError(f"Unknown imputer: {imputer_type}")
            
        X_train = df_imputed.iloc[:-1][features].values
        y_train = train_df[target].values
        X_val = df_imputed.iloc[[val_row_idx]][features].values
        y_val = val_df[target].values[0]
        
        # Fit model & predict
        model = model_fn()
        model.fit(X_train, y_train)
        pred = model.predict(X_val)[0]
        errors.append((pred - y_val)**2)
        
    return np.mean(errors)

def run_search():
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
    
    target_cols = ['Rice_frac', 'Cotton_frac', 'Maize_frac', 'Bajra_frac', 'Groundnut_frac']
    
    # Models list
    models_pool = {
        'RF_100': lambda: RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
        'ET_100': lambda: ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
        'CatBoost_50': lambda: CatBoostRegressor(iterations=50, depth=3, learning_rate=0.05, random_seed=42, verbose=0),
        'LGBM_50': lambda: LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42, verbose=-1, n_jobs=-1),
        'XGB_50': lambda: XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1),
        'HGBR_50': lambda: HistGradientBoostingRegressor(max_iter=50, max_depth=3, random_state=42),
        'ElasticNet': lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        'Ridge': lambda: Ridge(alpha=1.0, random_state=42),
        'BayesianRidge': lambda: BayesianRidge(),
        'SVR': lambda: SVR(C=1.0, epsilon=0.05),
        'KNeighbors': lambda: KNeighborsRegressor(n_neighbors=3)
    }
    
    imputers_pool = ['knn_3', 'knn_5', 'knn_6', 'knn_8', 'spatial_1nn', 'iterative_rf', 'median']
    
    # We will do feature selection for each target crop
    best_configs = {}
    
    for target in target_cols:
        print(f"\n========================================\nTarget Crop: {target}\n========================================")
        
        # Step 1: Pre-select top 15 features using RF importance on all features (using standard KNN imputed data)
        # Impute temporarily to run feature importance
        temp_imputer = KNNImputer(n_neighbors=6)
        X_temp = temp_imputer.fit_transform(covered_df[all_cols])
        df_temp = pd.DataFrame(X_temp, columns=all_cols)
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(df_temp.values, covered_df[target].values)
        importances = pd.Series(rf.feature_importances_, index=all_cols).sort_values(ascending=False)
        
        # Include centroid_x, centroid_y, area_ha always because spatial coordinates and area are crucial
        mandatory_cols = ['centroid_x', 'centroid_y', 'area_ha']
        candidate_cols = [c for c in importances.index if c not in mandatory_cols][:12]
        features_pool = mandatory_cols + candidate_cols
        print("Candidate features:")
        print(features_pool)
        
        # Step 2: Search for best model & imputer using a baseline set of features (e.g. top 5 features)
        baseline_feats = features_pool[:5]
        best_mse = float('inf')
        best_model_name = None
        best_imputer = None
        
        for model_name, model_fn in models_pool.items():
            for imp in imputers_pool:
                mse = evaluate_lovo_cv_imputed(covered_df, target, baseline_feats, model_fn, imp, geom_cols, sar_cols, all_cols)
                if mse < best_mse:
                    best_mse = mse
                    best_model_name = model_name
                    best_imputer = imp
                    
        print(f"Top Candidate from baseline search: Model={best_model_name}, Imputer={best_imputer}, MSE={best_mse:.6f}")
        
        # Step 3: Run Greedy Forward Feature Selection for the best candidate model + imputer
        # Start with empty feature set, but allow mandatory features or start completely greedy
        selected_feats = []
        best_ffs_mse = float('inf')
        
        model_fn = models_pool[best_model_name]
        
        # Limit to max 6 features to avoid overfitting
        for step in range(6):
            step_best_mse = float('inf')
            step_best_feat = None
            
            for feat in features_pool:
                if feat in selected_feats:
                    continue
                test_feats = selected_feats + [feat]
                mse = evaluate_lovo_cv_imputed(covered_df, target, test_feats, model_fn, best_imputer, geom_cols, sar_cols, all_cols)
                if mse < step_best_mse:
                    step_best_mse = mse
                    step_best_feat = feat
            
            if step_best_mse < best_ffs_mse:
                best_ffs_mse = step_best_mse
                selected_feats.append(step_best_feat)
                print(f"  FFS Step {step+1}: Added '{step_best_feat}' | MSE: {step_best_mse:.6f}")
            else:
                print(f"  FFS Step {step+1}: No improvement. Stopping FFS.")
                break
                
        # Step 4: Let's also evaluate the baseline features (using the current features in the pipeline) to compare
        baseline_selected = {
            'Rice_frac': ['bbox_width', 'area_ha', 'p50_20250619', 'p75_20250619', 'p25_20250619', 'mean_20250619'],
            'Cotton_frac': ['centroid_y', 'perimeter', 'diff_harvest', 'p75_20250814', 'mean_20250814', 'p50_20250814'],
            'Maize_frac': ['centroid_y', 'centroid_x', 'diff_harvest', 'mean_local_std_20251013', 'p50_20250814', 'p75_20250606'],
            'Bajra_frac': ['centroid_y', 'centroid_x', 'p25_20250619', 'p50_20250619', 'mean_20250619', 'p75_20250619'],
            'Groundnut_frac': ['centroid_y', 'centroid_x', 'mean_local_std_20250606', 'mean_local_std_20250814', 'mean_local_std_20251013', 'cumulative_change']
        }
        
        # Compare with the original configuration:
        # Original configuration for Rice: RF_100, KNN_6, original features
        # Cotton: RF_100 + CatBoost_50 (0.8 + 0.2), KNN_6, original features
        # Maize: ET_100, KNN_6, original features
        # Bajra: ET_100, spatial_1nn, original features
        # Groundnut: RF_100, spatial_1nn, original features
        
        orig_model_pool = {
            'Rice_frac': lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Cotton_frac': lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), # simplified
            'Maize_frac': lambda: ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Bajra_frac': lambda: ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Groundnut_frac': lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        }
        orig_imputer = {
            'Rice_frac': 'knn_6', 'Cotton_frac': 'knn_6', 'Maize_frac': 'knn_6',
            'Bajra_frac': 'spatial_1nn', 'Groundnut_frac': 'spatial_1nn'
        }
        
        orig_mse = evaluate_lovo_cv_imputed(covered_df, target, baseline_selected[target], orig_model_pool[target], orig_imputer[target], geom_cols, sar_cols, all_cols)
        print(f"Original Configuration MSE: {orig_mse:.6f}")
        
        best_configs[target] = {
            'model_name': best_model_name,
            'imputer': best_imputer,
            'features': selected_feats,
            'best_mse': best_ffs_mse,
            'orig_mse': orig_mse
        }
        
    print("\n\n========================================\nSUMMARY OF IMPROVEMENTS\n========================================")
    for target in target_cols:
        cfg = best_configs[target]
        print(f"Crop: {target}")
        print(f"  Original MSE: {cfg['orig_mse']:.6f}")
        print(f"  New Best MSE: {cfg['best_mse']:.6f} (Change: {(cfg['best_mse'] - cfg['orig_mse'])/cfg['orig_mse']*100:+.2f}%)")
        print(f"  Best Model: {cfg['model_name']}")
        print(f"  Best Imputer: {cfg['imputer']}")
        print(f"  Best Features: {cfg['features']}")
        print("-" * 40)

if __name__ == '__main__':
    run_search()
