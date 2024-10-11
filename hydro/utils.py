import numpy as np
import xarray as xr
import os
from pathlib import Path
import rasterio
import rasterio.warp
from rasterio.crs import CRS
import rioxarray
import matplotlib
from operator import itemgetter
import geopandas as gpd
import fiona
import glob
from shapely.geometry import shape
from shapely.geometry import box
from functools import lru_cache
#from raster_clipper import RasterClipper
#from dem import DEM
import warnings
import dask
from rasterio.warp import calculate_default_transform, reproject, Resampling
warnings.filterwarnings("ignore", category=rasterio.errors.RasterioDeprecationWarning)


class Utils:
    def __init__(self, project_name, study_area):
        #self.orig_dem = f'./projects/{project_name}/elevation/dem_volta.tif'
        self.study_area = study_area
        self.project_name = project_name
        
        # self.clipped_dem ='./input_data/elevation/clipped_dem.tif'
        # filetest = self.process_existing_file(self.clipped_dem)
        # if filetest is False:
        #     self.clip(orig_dem, 'EPSG:4326', self.clipped_dem, True)

    def process_existing_file(self, file_path):
        directory, filename = os.path.split(file_path)
        if os.path.exists(file_path):
            #print(f"     - The file {filename} already exists in the directory {directory}. Skipping further processing.")
            # You can add your specific processing here if the file exists
            return True
        else:
            return False

    # Write output to a new GeoTIFF file
    def save_to_scratch(self,output_file_path, array_to_save):
        with rasterio.open(f'./projects/{self.project_name}/elevation/dem_{self.project_name}.tif') as lc_src: 
            luc = lc_src.profile
        lc_meta = lc_src.meta.copy()
        lc_meta.update({
            "height": array_to_save.shape[0],
            "width": array_to_save.shape[1],
            "compress": "lzw"  
        })
    
        #output_file ='./input_data/scratch/cn2.tif'
        with rasterio.open(output_file_path, 'w', **lc_meta) as dst:
            dst.write(array_to_save, 1)
            
    def reproject_raster(self, input_ras, out_ras):
        dst_crs = 'EPSG:4326'

        with rasterio.open(input_ras) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            with rasterio.open(out_ras, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest)
    
    def align_rasters_30m(self, input_ras, elev, israster=True):
        match = rioxarray.open_rasterio(elev)
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
    
    def align_rasters_1km(self, input_ras, clipped_nc, israster=True):
        #print('     - Aligning raster in terms of extent, resolution and projection')

        # Open the DEM raster with caching to avoid reopening it multiple times
        match = clipped_nc.pr.sel()[0]
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
    
    def get_bbox(self, dst_crs):
        shp = gpd.read_file(self.study_area)
        #dst_crs = 'EPSG:4326'
        dst_crs = dst_crs

        if shp.crs.equals(dst_crs):
            prj_shp = shp
        else:
            geometry = rasterio.warp.transform_geom(
                src_crs=shp.crs,
                dst_crs=dst_crs,
                geom=shp.geometry.values,
            )
            prj_shp = shp.set_geometry(
                [shape(geom) for geom in geometry],
                crs=dst_crs,
            )
        bounds = prj_shp.geometry.apply(lambda x: x.bounds).tolist()
        self.minx, self.miny, self.maxx, self.maxy = min(bounds, key=itemgetter(0))[0], min(bounds, key=itemgetter(1))[1], max(bounds, key=itemgetter(2))[2], max(bounds, key=itemgetter(3))[3]
        #self.minx = self.minx - 1
        #self.miny = self.miny - 1
        #self.maxx = self.maxx + 1
        #self.maxy = self.maxy + 1
        
    def clip_nc(self, files):
        #ds = xr.open_mfdataset(files, combine='nested', concat_dim='time', join='override')
        ds = xr.open_dataset(files)
        #ds2 = ds.sortby('time')
        data_var = ds.sel()
        ds3 = data_var.rio.write_crs(4326)  # Ensure consistent CRS            
        ds3 = data_var.rio.clip_box(self.study_area[0], self.study_area[1], 
                                        self.study_area[2], self.study_area[3])
        #ds3.to_netcdf('/lustre/backup/WUR/ESG/duku002/NBAT/hydro/input_data/clim/' + out_nc, mode='w')
        return ds3
        

    def clip(self, raster_path,  dst_crs='EPSG:4326', out_path=None, save_output=False):
        
        """
        Clips the source raster (src_path) using the extent of the clip raster (clip_path) 
        and saves the clipped data to a new file (dst_path).
        
        Args:
          src_path (str): Path to the source raster file.
          clip_path (str): Path to the clip raster file.
          dst_path (str): Path to save the clipped raster file.
        """
        #shp = gpd.read_file(self.study_area)
        #dst_crs = 'EPSG:4326'
        dst_crs = dst_crs

        minx, miny, maxx, maxy = self.study_area

        # Create a polygon from the bounding box
        bbox_polygon = box(minx, miny, maxx, maxy)
    
        # Convert the polygon to a GeoDataFrame
        gdf = gpd.GeoDataFrame({'geometry': bbox_polygon}, index=[0], crs=dst_crs)
    
        # Extract the geometry in GeoJSON format
        shapes = [bbox_polygon.__geo_interface__]
        
        with rasterio.open(raster_path) as src:
            out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
            out_meta = src.meta
            out_image = np.where(out_image == src.nodata, np.nan, out_image)

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
