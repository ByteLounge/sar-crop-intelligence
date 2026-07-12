import os
from typing import Tuple, List
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import geopandas as gpd

def align_rasters(
    tif_paths: List[str], 
    gdf_utm: gpd.GeoDataFrame, 
    output_dir: str, 
    resolution: float = 10.0
) -> Tuple[int, int, rasterio.Affine]:
    """
    Reproject and align all input GeoTIFFs to a common grid matching the shapefile's extent.
    """
    os.makedirs(output_dir, exist_ok=True)
    xmin, ymin, xmax, ymax = gdf_utm.total_bounds
    width = int(np.ceil((xmax - xmin) / resolution))
    height = int(np.ceil((ymax - ymin) / resolution))
    xmax = xmin + width * resolution
    ymax = ymin + height * resolution
    
    dst_transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, width, height)
    
    for path in tif_paths:
        fn = os.path.basename(path)
        parts = fn.split("_")
        date_str = parts[6][:8]
        out_path = os.path.join(output_dir, f"capella_hh_{date_str}_10m.tif")
        
        with rasterio.open(path) as src:
            profile = src.profile.copy()
            profile.update({
                'crs': 'EPSG:32643',
                'transform': dst_transform,
                'width': width,
                'height': height,
                'nodata': 0,
                'dtype': 'uint8'
            })
            
            dst_data = np.zeros((height, width), dtype='uint8')
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs='EPSG:32643',
                resampling=Resampling.bilinear
            )
            
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(dst_data, 1)
                
    return width, height, dst_transform
