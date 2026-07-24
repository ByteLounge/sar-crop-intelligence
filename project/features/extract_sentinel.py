import os
import requests
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.features import geometry_mask
from rasterio.enums import Resampling
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from scipy.interpolate import griddata
from skimage.feature import graycomatrix, graycoprops
import warnings
warnings.filterwarnings('ignore')

# API URL
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"

def sign_url(url):
    sign_api = f"https://planetarycomputer.microsoft.com/api/sas/v1/sign?href={url}"
    res = requests.get(sign_api)
    if res.status_code == 200:
        return res.json()["href"]
    return url

def fetch_s2_assets(bbox, date_range):
    payload = {
        "bbox": bbox,
        "datetime": date_range,
        "collections": ["sentinel-2-l2a"],
        "limit": 20
    }
    res = requests.post(STAC_URL, json=payload)
    if res.status_code == 200:
        features = res.json().get("features", [])
        # Filter out S2C scenes
        features = [f for f in features if not f["id"].startswith("S2C_")]
        return features
    return []

def fetch_s1_assets(bbox, date_range):
    payload = {
        "bbox": bbox,
        "datetime": date_range,
        "collections": ["sentinel-1-grd"],
        "limit": 20
    }
    res = requests.post(STAC_URL, json=payload)
    if res.status_code == 200:
        features = res.json().get("features", [])
        return features
    return []

def compute_glcm_texture(image_data, mask):
    valid_vals = image_data[mask]
    if len(valid_vals) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    min_v, max_v = valid_vals.min(), valid_vals.max()
    if max_v > min_v:
        scaled = np.clip(((image_data - min_v) / (max_v - min_v) * 8).astype(int), 0, 7)
    else:
        scaled = np.zeros_like(image_data, dtype=int)
        
    scaled[~mask] = 0
    
    glcm = graycomatrix(scaled, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=8, symmetric=True, normed=True)
    
    contrast = float(graycoprops(glcm, 'contrast').mean())
    homogeneity = float(graycoprops(glcm, 'homogeneity').mean())
    asm = float(graycoprops(glcm, 'ASM').mean())
    energy = float(graycoprops(glcm, 'energy').mean())
    
    glcm_flat = glcm.flatten()
    glcm_flat = glcm_flat[glcm_flat > 0]
    entropy = float(-np.sum(glcm_flat * np.log2(glcm_flat)))
    
    return contrast, homogeneity, entropy, energy, asm

def get_best_s2_item_for_village(v_geom, best_s2):
    best_tile = None
    max_intersection_area = -1
    for tile, item in best_s2.items():
        scene_geom = gpd.GeoDataFrame.from_features([item], crs="EPSG:4326").geometry.iloc[0]
        intersection = scene_geom.intersection(v_geom)
        if not intersection.is_empty:
            area = intersection.area
            if area > max_intersection_area:
                max_intersection_area = area
                best_tile = tile
    if best_tile is None and best_s2:
        return list(best_s2.keys())[0]
    return best_tile

def parse_s1_incidence_points(base_url):
    prod_xml_url = f"{base_url}/annotation/iw-vv.xml"
    signed_url = sign_url(prod_xml_url)
    res = requests.get(signed_url)
    if res.status_code == 200:
        try:
            root = ET.fromstring(res.content)
            grid_list = root.find(".//geolocationGridPointList")
            if grid_list is not None:
                pts = grid_list.findall("geolocationGridPoint")
                lats, lons, angles = [], [], []
                for p in pts:
                    lats.append(float(p.find("latitude").text))
                    lons.append(float(p.find("longitude").text))
                    angles.append(float(p.find("incidenceAngle").text))
                return np.column_stack((lats, lons)), np.array(angles)
        except Exception as e:
            print(f"Error parsing XML: {e}")
    return None, None

def run_extraction():
    print("Starting Sentinel-1 and Sentinel-2 feature extraction pipeline...")
    
    workspace_dir = r"D:\PC\resources"
    project_dir = os.path.join(workspace_dir, "project")
    features_dir = os.path.join(project_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    
    shp_path = os.path.join(workspace_dir, "villages_clean", "villages_clean.shp")
    gdf = gpd.read_file(shp_path)
    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    bbox = list(gdf_wgs84.total_bounds)
    
    # Target periods
    target_periods = [
        {"name": "June06", "s2_range": "2025-06-01T00:00:00Z/2025-06-15T23:59:59Z", "s1_range": "2025-05-28T00:00:00Z/2025-06-15T23:59:59Z", "capella_date": "20250606"},
        {"name": "June19", "s2_range": "2025-06-01T00:00:00Z/2025-06-15T23:59:59Z", "s1_range": "2025-06-12T00:00:00Z/2025-06-30T23:59:59Z", "capella_date": "20250619"},
        {"name": "Aug14",  "s2_range": "2025-08-30T00:00:00Z/2025-09-15T23:59:59Z", "s1_range": "2025-08-10T00:00:00Z/2025-08-30T23:59:59Z", "capella_date": "20250814"},
        {"name": "Oct13",  "s2_range": "2025-10-10T00:00:00Z/2025-10-15T23:59:59Z", "s1_range": "2025-10-05T00:00:00Z/2025-10-18T23:59:59Z", "capella_date": "20251013"}
    ]
    
    metadata_records = []
    village_records = {row['ID']: {'ID': row['ID'], 'VILLAGE': row['VILLAGE']} for idx, row in gdf_wgs84.iterrows()}
    
    for period in target_periods:
        p_name = period["name"]
        print(f"\nProcessing period: {p_name}...")
        
        # S2 Scene Search
        s2_scenes = fetch_s2_assets(bbox, period["s2_range"])
        s2_tiles = {}
        for f in s2_scenes:
            tile = f["properties"]["s2:mgrs_tile"]
            if tile not in s2_tiles:
                s2_tiles[tile] = []
            s2_tiles[tile].append(f)
            
        best_s2 = {}
        for tile, scenes in s2_tiles.items():
            scenes_sorted = sorted(scenes, key=lambda x: x["properties"]["eo:cloud_cover"])
            best_s2[tile] = scenes_sorted[0]
            print(f"  S2 Tile {tile}: Best Scene ID={best_s2[tile]['id']} | Clouds={best_s2[tile]['properties']['eo:cloud_cover']:.2f}%")
            metadata_records.append({
                "Period": p_name,
                "Sensor": "Sentinel-2",
                "Tile": tile,
                "SceneID": best_s2[tile]['id'],
                "Date": best_s2[tile]['properties']['datetime'],
                "CloudCover": best_s2[tile]['properties']['eo:cloud_cover']
            })
            
        # S1 Scene Search
        s1_scenes = fetch_s1_assets(bbox, period["s1_range"])
        desc_scenes = [f for f in s1_scenes if f["properties"].get("sat:orbit_state") == "descending" and f["properties"].get("sat:relative_orbit") == 34]
        if not desc_scenes:
            desc_scenes = [f for f in s1_scenes if f["properties"].get("sat:orbit_state") == "descending"]
        if not desc_scenes:
            desc_scenes = s1_scenes
            
        desc_scenes = sorted(desc_scenes, key=lambda x: x["properties"]["datetime"])
        best_s1 = desc_scenes[0] if desc_scenes else None
        
        # 1. Open Sentinel-2 rasters ONCE per period
        opened_s2 = {}
        s2_bands_list = ["B02", "B03", "B04", "B08", "B05", "B11", "B12", "SCL"]
        for tile, item in best_s2.items():
            opened_s2[tile] = {}
            for b in s2_bands_list:
                try:
                    href = sign_url(item["assets"][b]["href"])
                    opened_s2[tile][b] = rasterio.open(href)
                except Exception as e:
                    print(f"    Failed to open S2 tile {tile} band {b}: {e}")
                    
        # 2. Open Sentinel-1 rasters ONCE per period
        opened_s1 = {}
        s1_pts, s1_angles = None, None
        if best_s1:
            print(f"  S1: Best Scene ID={best_s1['id']} | Date={best_s1['properties']['datetime']}")
            metadata_records.append({
                "Period": p_name,
                "Sensor": "Sentinel-1",
                "Tile": "N/A",
                "SceneID": best_s1['id'],
                "Date": best_s1['properties']['datetime'],
                "CloudCover": 0.0
            })
            try:
                vv_href = sign_url(best_s1["assets"]["vv"]["href"])
                vh_href = sign_url(best_s1["assets"]["vh"]["href"])
                opened_s1["vv"] = rasterio.open(vv_href)
                opened_s1["vh"] = rasterio.open(vh_href)
                
                vv_raw_href = best_s1["assets"]["vv"]["href"]
                parts = vv_raw_href.split("/")
                base_scene_url = "/".join(parts[:-2])
                s1_pts, s1_angles = parse_s1_incidence_points(base_scene_url)
            except Exception as e:
                print(f"    Failed to open S1 rasters: {e}")
                
        # Process each village shape
        for idx, row in gdf_wgs84.iterrows():
            v_id = row['ID']
            v_name = row['VILLAGE']
            v_geom_wgs = row['geometry']
            v_centroid = v_geom_wgs.centroid
            
            s2_tile = get_best_s2_item_for_village(v_geom_wgs, best_s2)
            if s2_tile is None or s2_tile not in opened_s2:
                continue
                
            s2_bands = {}
            try:
                if "B08" not in opened_s2[s2_tile]:
                    continue
                ref_src = opened_s2[s2_tile]["B08"]
                v_utm_geom = gdf.to_crs(ref_src.crs)[gdf['ID'] == v_id].iloc[0]['geometry']
                minx, miny, maxx, maxy = v_utm_geom.bounds
                window_10m = from_bounds(minx, miny, maxx, maxy, transform=ref_src.transform).round_shape()
                h_win_10m, w_win_10m = int(window_10m.height), int(window_10m.width)
                win_transform_10m = ref_src.window_transform(window_10m)
                
                v_mask = geometry_mask([v_utm_geom], out_shape=(h_win_10m, w_win_10m), transform=win_transform_10m, invert=True)
                
                # Read 10m bands from pre-opened files
                for b in ["B02", "B03", "B04", "B08"]:
                    if b not in opened_s2[s2_tile]:
                        raise KeyError(b)
                    src = opened_s2[s2_tile][b]
                    win = from_bounds(minx, miny, maxx, maxy, transform=src.transform).round_shape()
                    s2_bands[b] = src.read(1, window=win).astype(float)
                    
                # Read 20m bands from pre-opened files and resample to 10m window shape
                for b in ["B05", "B11", "B12", "SCL"]:
                    if b not in opened_s2[s2_tile]:
                        raise KeyError(b)
                    src = opened_s2[s2_tile][b]
                    win = from_bounds(minx, miny, maxx, maxy, transform=src.transform).round_shape()
                    s2_bands[b] = src.read(1, window=win, out_shape=(h_win_10m, w_win_10m), resampling=Resampling.bilinear).astype(float)
            except Exception as e:
                print(f"  Error reading S2 bands for village {v_name}: {e}")
                continue
                
            B2 = s2_bands["B02"]
            B3 = s2_bands["B03"]
            B4 = s2_bands["B04"]
            B5 = s2_bands["B05"]
            B8 = s2_bands["B08"]
            B11 = s2_bands["B11"]
            B12 = s2_bands["B12"]
            SCL = s2_bands["SCL"].astype(int)
            
            clean_mask = v_mask & (~np.isin(SCL, [0, 1, 3, 8, 9, 10]))
            if np.sum(clean_mask) == 0:
                clean_mask = v_mask
                
            eps = 1e-10
            NDVI = (B8 - B4) / (B8 + B4 + eps)
            EVI = 2.5 * (B8 - B4) / (B8 + 6.0 * B4 - 7.5 * B2 + 1.0 + eps)
            SAVI = 1.5 * (B8 - B4) / (B8 + B4 + 0.5 + eps)
            MSAVI = (2.0 * B8 + 1.0 - np.sqrt(np.maximum((2.0 * B8 + 1.0)**2 - 8.0 * (B8 - B4), 0.0))) / 2.0
            NDRE = (B8 - B5) / (B8 + B5 + eps)
            GCI = (B8 / (B3 + eps)) - 1.0
            NDWI = (B3 - B8) / (B3 + B8 + eps)
            MNDWI = (B3 - B11) / (B3 + B11 + eps)
            LSWI = (B8 - B11) / (B8 + B11 + eps)
            BSI = ((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2) + eps)
            NDBSI = (B11 - B12) / (B11 + B12 + eps)
            
            v_rec = village_records[v_id]
            v_rec[f"S2_NDVI_{p_name}"] = float(np.mean(NDVI[clean_mask]))
            v_rec[f"S2_EVI_{p_name}"] = float(np.mean(EVI[clean_mask]))
            v_rec[f"S2_SAVI_{p_name}"] = float(np.mean(SAVI[clean_mask]))
            v_rec[f"S2_MSAVI_{p_name}"] = float(np.mean(MSAVI[clean_mask]))
            v_rec[f"S2_NDRE_{p_name}"] = float(np.mean(NDRE[clean_mask]))
            v_rec[f"S2_GCI_{p_name}"] = float(np.mean(GCI[clean_mask]))
            v_rec[f"S2_NDWI_{p_name}"] = float(np.mean(NDWI[clean_mask]))
            v_rec[f"S2_MNDWI_{p_name}"] = float(np.mean(MNDWI[clean_mask]))
            v_rec[f"S2_LSWI_{p_name}"] = float(np.mean(LSWI[clean_mask]))
            v_rec[f"S2_BSI_{p_name}"] = float(np.mean(BSI[clean_mask]))
            v_rec[f"S2_NDBSI_{p_name}"] = float(np.mean(NDBSI[clean_mask]))
            
            contrast, homogeneity, entropy, energy, asm = compute_glcm_texture(B8, clean_mask)
            v_rec[f"S2_Contrast_{p_name}"] = contrast
            v_rec[f"S2_Homogeneity_{p_name}"] = homogeneity
            v_rec[f"S2_Entropy_{p_name}"] = entropy
            v_rec[f"S2_Energy_{p_name}"] = energy
            v_rec[f"S2_ASM_{p_name}"] = asm
            
            # Sentinel-1 Feature Extraction with robust src.index windowing
            if best_s1 and "vv" in opened_s1 and "vh" in opened_s1:
                try:
                    src_vv = opened_s1["vv"]
                    src_vh = opened_s1["vh"]
                    
                    minx_wgs, miny_wgs, maxx_wgs, maxy_wgs = v_geom_wgs.bounds
                    r1, c1 = src_vv.index(minx_wgs, maxy_wgs)
                    r2, c2 = src_vv.index(maxx_wgs, miny_wgs)
                    r_min, r_max = min(r1, r2), max(r1, r2)
                    c_min, c_max = min(c1, c2), max(c1, c2)
                    
                    s1_window = Window(c_min, r_min, max(c_max - c_min, 1), max(r_max - r_min, 1))
                    h_win_s1, w_win_s1 = int(s1_window.height), int(s1_window.width)
                    s1_win_transform = src_vv.window_transform(s1_window)
                    
                    s1_v_mask = geometry_mask([v_geom_wgs], out_shape=(h_win_s1, w_win_s1), transform=s1_win_transform, invert=True)
                    vv_data = src_vv.read(1, window=s1_window).astype(float)
                    vh_data = src_vh.read(1, window=s1_window).astype(float)
                    
                    vv_db = 10.0 * np.log10(np.maximum(vv_data, eps))
                    vh_db = 10.0 * np.log10(np.maximum(vh_data, eps))
                    ratio_db = vv_db - vh_db
                    
                    v_rec[f"S1_VV_{p_name}"] = float(np.mean(vv_db[s1_v_mask]))
                    v_rec[f"S1_VH_{p_name}"] = float(np.mean(vh_db[s1_v_mask]))
                    v_rec[f"S1_Ratio_{p_name}"] = float(np.mean(ratio_db[s1_v_mask]))
                    
                    contrast_s1, homogeneity_s1, entropy_s1, energy_s1, asm_s1 = compute_glcm_texture(vv_db, s1_v_mask)
                    v_rec[f"S1_Contrast_{p_name}"] = contrast_s1
                    v_rec[f"S1_Homogeneity_{p_name}"] = homogeneity_s1
                    v_rec[f"S1_Entropy_{p_name}"] = entropy_s1
                    v_rec[f"S1_Energy_{p_name}"] = energy_s1
                    v_rec[f"S1_ASM_{p_name}"] = asm_s1
                    
                    # Incidence Angle
                    if s1_pts is not None:
                        point = np.array([[v_centroid.y, v_centroid.x]])
                        angle_interp = griddata(s1_pts, s1_angles, point, method='linear')[0]
                        if np.isnan(angle_interp):
                            angle_interp = griddata(s1_pts, s1_angles, point, method='nearest')[0]
                        v_rec[f"S1_Incidence_{p_name}"] = float(angle_interp)
                    else:
                        v_rec[f"S1_Incidence_{p_name}"] = 37.5
                        
                except Exception as e:
                    print(f"  Error reading S1 bands for village {v_name}: {e}")
                    v_rec[f"S1_VV_{p_name}"] = np.nan
                    v_rec[f"S1_VH_{p_name}"] = np.nan
                    v_rec[f"S1_Ratio_{p_name}"] = np.nan
                    v_rec[f"S1_Contrast_{p_name}"] = np.nan
                    v_rec[f"S1_Homogeneity_{p_name}"] = np.nan
                    v_rec[f"S1_Entropy_{p_name}"] = np.nan
                    v_rec[f"S1_Energy_{p_name}"] = np.nan
                    v_rec[f"S1_ASM_{p_name}"] = np.nan
                    v_rec[f"S1_Incidence_{p_name}"] = 37.5
            else:
                v_rec[f"S1_VV_{p_name}"] = np.nan
                v_rec[f"S1_VH_{p_name}"] = np.nan
                v_rec[f"S1_Ratio_{p_name}"] = np.nan
                v_rec[f"S1_Contrast_{p_name}"] = np.nan
                v_rec[f"S1_Homogeneity_{p_name}"] = np.nan
                v_rec[f"S1_Entropy_{p_name}"] = np.nan
                v_rec[f"S1_Energy_{p_name}"] = np.nan
                v_rec[f"S1_ASM_{p_name}"] = np.nan
                v_rec[f"S1_Incidence_{p_name}"] = 37.5

        # Close S2 rasters
        for tile in opened_s2:
            for b in opened_s2[tile]:
                try:
                    opened_s2[tile][b].close()
                except:
                    pass
        # Close S1 rasters
        for b in opened_s1:
            try:
                opened_s1[b].close()
            except:
                pass

    df_sentinel = pd.DataFrame(list(village_records.values()))
    
    print("Computing S2 temporal features...")
    periods = [p["name"] for p in target_periods]
    ndvi_cols = [f"S2_NDVI_{p}" for p in periods]
    
    for col in ndvi_cols:
        if col not in df_sentinel.columns:
            df_sentinel[col] = np.nan
            
    df_sentinel["peak_NDVI"] = df_sentinel[ndvi_cols].max(axis=1)
    
    def safe_sub(col1, col2):
        if col1 in df_sentinel.columns and col2 in df_sentinel.columns:
            return df_sentinel[col1] - df_sentinel[col2]
        return np.nan
        
    df_sentinel["NDVI_growth"] = safe_sub("S2_NDVI_Aug14", "S2_NDVI_June06")
    df_sentinel["NDVI_decline"] = safe_sub("S2_NDVI_Aug14", "S2_NDVI_Oct13")
    df_sentinel["vegetation_integral"] = df_sentinel[ndvi_cols].sum(axis=1)
    df_sentinel["vegetation_amplitude"] = df_sentinel["peak_NDVI"] - df_sentinel[ndvi_cols].min(axis=1)
    
    # Save features
    features_csv = os.path.join(features_dir, "sentinel_features.csv")
    df_sentinel.to_csv(features_csv, index=False)
    print(f"DEBUG: File exists after to_csv? {os.path.exists(features_csv)} | Absolute path: {os.path.abspath(features_csv)}")
    print(f"Sentinel features successfully written to: {features_csv}")
    
    # Save metadata report
    df_metadata = pd.DataFrame(metadata_records)
    metadata_csv = os.path.join(project_dir, "outputs", "sentinel_coverage_report.csv")
    df_metadata.to_csv(metadata_csv, index=False)
    print(f"Sentinel coverage metadata written to: {metadata_csv}")
    
    print("Feature extraction finished successfully!")

if __name__ == "__main__":
    run_extraction()
