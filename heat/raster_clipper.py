import numpy as np
import pandas as pd
import random
#import matplotlib.pyplot as plt
import xarray as xr
import time, sys
from datetime import datetime, timedelta
import rioxarray
import rasterio
from rasterio.mask import mask
import fiona
import matplotlib as mpl
from shapely.geometry import Polygon
from shapely.geometry import box
from fiona.crs import from_epsg

###===================================================================================================================

class RasterClipper:
    """
    This class clips a raster based on the extent of another raster using rasterio.
    """

    def __init__(self):
        pass

    def clip(self, input_ras, template, israster=True):
        match = rioxarray.open_rasterio(template)
        match = match.rio.write_crs(4326)

        # Read and align the input raster
        if israster:
            ds = rioxarray.open_rasterio(input_ras)
            ds = ds.rio.write_crs(4326)
            ds = ds.rio.reproject_match(match)[0]
        else:
            ds = input_ras
            ds = ds.rio.write_crs(4326)
            ds = ds.rename({'lon': 'x', 'lat': 'y'})
            ds = ds.rio.reproject_match(match)
        return ds

    def clip_old(self, raster_path, study_area, out_path=None, save_output=False):
        """
        Clips the source raster (src_path) using the extent of the clip raster (clip_path) 
        and saves the clipped data to a new file (dst_path).
        
        Args:
          src_path (str): Path to the source raster file.
          clip_path (str): Path to the clip raster file.
          dst_path (str): Path to save the clipped raster file.
        """

        minx, miny, maxx, maxy = list(study_area)

        # Create a polygon from the bounding box
        bbox_polygon = box(minx, miny, maxx, maxy)
        
        # Convert the polygon to GeoJSON format
        #bbox_gdf = gpd.GeoDataFrame({"geometry": [bbox_polygon]}, crs=from_epsg(4326)) # Assuming the bbox is in EPSG:4326
        shapes = [bbox_polygon.__geo_interface__]

        with rasterio.open(raster_path) as src:
            out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
            out_meta = src.meta

        if save_output==True:
            if out_path!=None:
                out_meta.update({"driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})
        
                with rasterio.open(out_path, "w", **out_meta) as dest:
                    dest.write(out_image)
            else:
                print('out_path should not be None. Provide path where clipped raster should be saved')
        return out_image.astype('float32')
   
