import ee
import numpy as np
import geemap
import datetime
import rioxarray
import os
import glob
import rasterio
from bakaano.utils import Utils

class TreeCover:
    def __init__(self, working_dir, study_area):
        """
        A class used to download and preprocess fractional vegetation cover data

        Args:
            working_dir (str): The parent working directory where files and outputs will be stored.
            study_area (str): The path to the shapefile of the river basin or watershed.
        
        Methods
        -------
        __init__(working_dir, study_area):
            Initializes the TreeCover object with project details.
        download_tree_cover():
            Download fractional vegetation cover data. 
        preprocess_tree_cover():
            Preprocess downloaded data.
        plot_tree_cover():
            Plot mean tree cover data
        """
        self.study_area = study_area
        self.working_dir = working_dir
        os.makedirs(f'{self.working_dir}/vcf', exist_ok=True)
        self.uw = Utils(self.working_dir, self.study_area)
        self.uw.get_bbox('EPSG:4326')

    def download_tree_cover(self):
        """Download fractional vegetation cover data.
        """
        vcf_check = f'{self.working_dir}/vcf/mean_tree_cover.tif'
        if not os.path.exists(vcf_check):
            ee.Authenticate()
            ee.Initialize()

            vcf = ee.ImageCollection("MODIS/006/MOD44B")

            i_date = str(2001)+'-01-01'
            f_date = str(2021)+'-01-01'
            df = vcf.select('Percent_Tree_Cover', 'Percent_NonTree_Vegetation').filterDate(i_date, f_date)

            area = ee.Geometry.BBox(self.uw.minx, self.uw.miny, self.uw.maxx, self.uw.maxy) 
            out_path = f'{self.working_dir}/vcf'
            geemap.ee_export_image_collection(ee_object=df, out_dir=out_path, scale=1000, region=area, crs='EPSG:4326', file_per_band=True) 
            print('Download completed')
        else:
            print(f"     - Tree cover data already exists in {self.working_dir}/vcf/mean_tree_cover.tif; skipping download.")

    def preprocess_tree_cover(self):
        """Preprocess downloaded data
        """
        vcf_check = f'{self.working_dir}/vcf/mean_tree_cover.tif'
        if not os.path.exists(vcf_check):
            vcf_path= f'{self.working_dir}/vcf'
            tree_cover_list = glob.glob(f'{vcf_path}/*Percent_Tree_Cover*.tif')
            herb_cover_list = glob.glob(f'{vcf_path}/*Percent_NonTree_Vegetation.tif')

            with rasterio.open(tree_cover_list[0]) as src:
                meta = src.meta

            tree_cover_rasters = []
            herb_cover_rasters = []

            # Read all rasters and stack them into a list
            for file in tree_cover_list:
                with rasterio.open(file) as src:
                    # Read the raster and append to the list
                    tree_cover_rasters.append(src.read(1))

            tree_rasters_stack = np.stack(tree_cover_rasters)
            mean_tree_raster = np.mean(tree_rasters_stack, axis=0)

            mean_tree_path= f'{self.working_dir}/vcf/mean_tree_cover.tif'
            meta.update(dtype=rasterio.float32, count=1)
            with rasterio.open(mean_tree_path, 'w', **meta) as dst:
                dst.write(mean_tree_raster.astype(rasterio.float32), 1)

            for file in herb_cover_list:
                with rasterio.open(file) as src:
                    # Read the raster and append to the list
                    herb_cover_rasters.append(src.read(1))

            herb_rasters_stack = np.stack(herb_cover_rasters)
            mean_herb_raster = np.mean(herb_rasters_stack, axis=0)

            mean_herb_path= f'{self.working_dir}/vcf/mean_herb_cover.tif'
            meta.update(dtype=rasterio.float32, count=1)
            with rasterio.open(mean_herb_path, 'w', **meta) as dst:
                dst.write(mean_herb_raster.astype(rasterio.float32), 1)
        else:
            print(f"     - Tree cover data already exists in {self.working_dir}/vcf/mean_tree_cover.tif; skipping preprocessing.")

    def plot_tree_cover(self):
        """Plot mean tree cover data
        """
        tc_data = rioxarray.open_rasterio(f'{self.working_dir}/vcf/mean_tree_cover.tif')
        tc_data = tc_data.where(tc_data > 0)
        tc_data = tc_data.where(tc_data < 100)
        tc_data.plot(cmap='terrain_r')