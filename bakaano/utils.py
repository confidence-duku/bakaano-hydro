import numpy as np
import xarray as xr
import os
from rasterio.enums import Resampling
import rasterio
import rasterio.warp
import rioxarray
from operator import itemgetter
import geopandas as gpd
import fiona
from shapely.geometry import shape
from rasterio.windows import from_bounds
import warnings
from rasterio.warp import calculate_default_transform, reproject, Resampling
warnings.filterwarnings("ignore", category=rasterio.errors.RasterioDeprecationWarning)


class Utils:
    def __init__(self, working_dir, study_area):
        """_summary_

        Args:
            working_dir (str): The parent working directory where files and outputs will be stored.
            study_area (str): The path to the shapefile of the river basin or watershed.

        """

        self.study_area = study_area
        self.working_dir = working_dir
        # reference_data = f'{self.working_dir}/elevation/dem_clipped.tif'
        # self.match = rioxarray.open_rasterio(reference_data)
        # self.match = self.match.rio.write_crs(4326)
        
    def process_existing_file(self, file_path):
        directory, filename = os.path.split(file_path)
        if os.path.exists(file_path):
            #print(f"     - The file {filename} already exists in the directory {directory}. Skipping further processing.")
            return True
        else:
            return False

    # Write output to a new GeoTIFF file
    def save_to_scratch(self,output_file_path, array_to_save):
        with rasterio.open(f'{self.working_dir}/elevation/dem_clipped.tif') as lc_src: 
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
    
    def align_rasters(self, input_ras, israster=True):
        #print('     - Aligning raster in terms of extent, resolution and projection')
        reference_data = f'{self.working_dir}/elevation/dem_clipped.tif'
        self.match = rioxarray.open_rasterio(reference_data)
        self.match = self.match.rio.write_crs(4326)
        # Read and align the input raster
        if israster:
            ds = rioxarray.open_rasterio(input_ras)
            ds = ds.rio.write_crs(4326)
            ds = ds.rio.reproject_match(self.match, resampling=Resampling.nearest)
        else:
            ds = input_ras
            ds = ds.rio.write_crs(4326)
            if 'lat' in ds.coords and 'lon' in ds.coords:
                ds = ds.rename({'lon': 'x', 'lat': 'y'})
            # ds = ds.chunk({'x': 100, 'y': 100})
            ds = ds.rio.reproject_match(self.match, resampling=Resampling.average)
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

        
    def concat_nc(self, clim_dir, dataset_str):
        #nc_list = []
        self.get_bbox('EPSG:4326')
        files = list(map(str, clim_dir.glob(dataset_str)))
        #ds = xr.open_mfdataset(files, combine='nested', concat_dim='time', join='override')
        ds = xr.open_mfdataset(files, combine='nested', concat_dim='time', join='override', chunks={'time': 100})
        ds2 = ds.sortby('time')
        if 'lat' in ds2.coords:
            ds2 = ds2.assign_coords(lat=ds2['lat'].astype('float32'))
        if 'lon' in ds2.coords:
            ds2 = ds2.assign_coords(lon=ds2['lon'].astype('float32'))
        data_var = ds2.sel()
        data_var = data_var.rio.write_crs(4326)  # Ensure consistent CRS            
        ds3 = data_var.rio.clip_box(self.minx, self.miny, self.maxx, self.maxy)
        return ds3
        

    def clip(self, raster_path,  dst_crs='EPSG:4326', out_path=None, save_output=False, crop_type=False):
        
        """
        Clips the source raster (src_path) using the extent of the clip raster (clip_path) 
        and saves the clipped data to a new file (dst_path).
        
        Args:
          src_path (str): Path to the source raster file.
          clip_path (str): Path to the clip raster file.
          dst_path (str): Path to save the clipped raster file.
        """
        shp = gpd.read_file(self.study_area)
        #dst_crs = 'EPSG:4326'
        dst_crs = dst_crs
        if shp.crs.equals(dst_crs):
            with fiona.open(self.study_area, "r") as shapefile:
                shapes = [feature["geometry"] for feature in shapefile]
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
            prj_shp_path = f'{self.working_dir}/shapes/prj_study_area.shp'
            prj_shp.to_file(prj_shp_path)
        
            with fiona.open(prj_shp_path, "r") as shapefile:
                shapes = [feature["geometry"] for feature in shapefile]

        if crop_type == False:  
            geom_bounds = shape(shapes[0]).bounds  # (minx, miny, maxx, maxy)
            with rasterio.open(raster_path) as src:
                window = from_bounds(*geom_bounds, transform=src.transform)
                window = window.round_offsets().round_lengths()
                
                # Read the rectangular window
                out_image = src.read(window=window).astype('float32')
                out_image = np.where(out_image == src.nodata, -9999, out_image)
                out_transform = src.window_transform(window)
                
                out_meta = src.meta.copy()

        else:
            with rasterio.open(raster_path) as src:
                out_image, out_transform = rasterio.mask.mask(src, shapes, crop=crop_type)
                out_meta = src.meta
                out_image = out_image.astype('float32')
                out_image = np.where(out_image == src.nodata, -9999, out_image)

        if save_output==True:
            if out_path!=None:
                #out_image = np.where(np.isnan(out_image) | np.isinf(out_image), 0, out_image)  # Handle NaN/inf
                #out_image = out_image.astype(np.float32)  # Ensure consistent dtype
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "dtype": "float32",
                    "nodata": -9999.0
                })
        
                with rasterio.open(out_path, "w", **out_meta) as dest:
                    dest.write(out_image)
            else:
                print('out_path should not be None. Provide path where clipped raster should be saved')
        return out_image
