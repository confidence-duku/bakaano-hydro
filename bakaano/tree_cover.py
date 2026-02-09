"""Fractional vegetation cover preparation from MODIS VCF.

Role: Download and preprocess tree/herbaceous cover rasters.
"""

import ee
import numpy as np
import geemap
import os
import glob
import re
import rasterio
from bakaano.utils import Utils
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class TreeCover:
    def __init__(self, working_dir, study_area, start_date, end_date):
        """
        Role: Prepare tree and herbaceous cover inputs.

        A class used to download and preprocess fractional vegetation cover data

        Args:
            working_dir (str): The parent working directory where files and outputs will be stored.
            study_area (str): The path to the shapefile of the river basin or watershed.
            start_date (str): The start date for the simulation period in 'YYYY-MM-DD' format.
            end_date (str): The end date for the simulation period in 'YYYY-MM-DD' format.
        
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
        self.start_date = start_date
        self.end_date = end_date

    def _extract_file_dates(self, files):
        """Extract datetime stamps from exported GeoTIFF filenames."""
        dates = []
        for file in files:
            name = os.path.basename(file)
            match = re.search(r"(\d{4})[_-]?(\d{2})[_-]?(\d{2})", name)
            if not match:
                continue
            year, month, day = match.groups()
            try:
                dates.append(datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d"))
            except ValueError:
                continue
        return dates

    def _raw_tree_cover_has_requested_span(self, raw_tree, raw_herb):
        """Check raw VCF files cover the requested yearly span."""
        if not raw_tree or not raw_herb:
            return False
        tree_dates = self._extract_file_dates(raw_tree)
        herb_dates = self._extract_file_dates(raw_herb)
        if not tree_dates or not herb_dates:
            return False

        required_years = set(range(
            datetime.strptime(self.start_date, "%Y-%m-%d").year,
            datetime.strptime(self.end_date, "%Y-%m-%d").year + 1,
        ))
        tree_years = {d.year for d in tree_dates}
        herb_years = {d.year for d in herb_dates}
        return required_years.issubset(tree_years) and required_years.issubset(herb_years)

    def _download_tree_cover(self):
        """Download fractional vegetation cover data.

        Returns:
            None. Downloads GeoTIFFs to ``{working_dir}/vcf``.
        """
        mean_tree_check = f'{self.working_dir}/vcf/mean_tree_cover.tif'
        mean_herb_check = f'{self.working_dir}/vcf/mean_herb_cover.tif'
        raw_tree = glob.glob(f'{self.working_dir}/vcf/*Percent_Tree_Cover*.tif')
        raw_herb = glob.glob(f'{self.working_dir}/vcf/*Percent_NonTree_Vegetation*.tif')
        if os.path.exists(mean_tree_check) and os.path.exists(mean_herb_check):
            print("     - Processed tree/herb cover rasters already exist; skipping download.")
            return

        ee.Authenticate()
        ee.Initialize()

        vcf = ee.ImageCollection("MODIS/061/MOD44B")
        i_date = self.start_date
        f_date = datetime.strptime(self.end_date, "%Y-%m-%d") + timedelta(days=1)
        df = vcf.select('Percent_Tree_Cover', 'Percent_NonTree_Vegetation').filterDate(i_date, f_date)

        # Build expected acquisition dates from Earth Engine metadata.
        ts_list = df.aggregate_array('system:time_start').getInfo() or []
        expected_dates = sorted({
            datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
            for ts in ts_list
        })
        if not expected_dates:
            print("     - No VCF images found for requested period.")
            return

        local_tree_dates = {
            d.strftime("%Y-%m-%d") for d in self._extract_file_dates(raw_tree)
        }
        local_herb_dates = {
            d.strftime("%Y-%m-%d") for d in self._extract_file_dates(raw_herb)
        }

        missing_tree = [d for d in expected_dates if d not in local_tree_dates]
        missing_herb = [d for d in expected_dates if d not in local_herb_dates]
        if not missing_tree and not missing_herb:
            print(
                f"     - Raw VCF GeoTIFFs in {self.working_dir}/vcf already cover requested dates; "
                "skipping download and proceeding to preprocessing."
            )
            return

        area = ee.Geometry.BBox(self.uw.minx, self.uw.miny, self.uw.maxx, self.uw.maxy)
        missing_dates = sorted(set(missing_tree + missing_herb))
        for date_str in missing_dates:
            next_day = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            img = df.filterDate(date_str, next_day).first()
            if img is None:
                continue

            if date_str in missing_tree:
                geemap.ee_export_image(
                    ee_object=img.select('Percent_Tree_Cover'),
                    filename=f"{self.working_dir}/vcf/{date_str}.Percent_Tree_Cover.tif",
                    scale=1000,
                    region=area,
                    crs='EPSG:4326'
                )
            if date_str in missing_herb:
                geemap.ee_export_image(
                    ee_object=img.select('Percent_NonTree_Vegetation'),
                    filename=f"{self.working_dir}/vcf/{date_str}.Percent_NonTree_Vegetation.tif",
                    scale=1000,
                    region=area,
                    crs='EPSG:4326'
                )
        print("Download completed (missing VCF files only).")

    def _preprocess_tree_cover(self):
        """Preprocess downloaded data

        Returns:
            None. Writes mean tree/herb cover rasters.
        """
        mean_tree_check = f'{self.working_dir}/vcf/mean_tree_cover.tif'
        mean_herb_check = f'{self.working_dir}/vcf/mean_herb_cover.tif'
        if not (os.path.exists(mean_tree_check) and os.path.exists(mean_herb_check)):
            vcf_path= f'{self.working_dir}/vcf'
            tree_cover_list = glob.glob(f'{vcf_path}/*Percent_Tree_Cover*.tif')
            herb_cover_list = glob.glob(f'{vcf_path}/*Percent_NonTree_Vegetation*.tif')

            if not tree_cover_list or not herb_cover_list:
                raise FileNotFoundError(
                    "Missing raw VCF GeoTIFFs required for preprocessing. "
                    "Expected files matching '*Percent_Tree_Cover*.tif' and "
                    "'*Percent_NonTree_Vegetation*.tif' in the vcf directory."
                )

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
            print("     - Processed tree/herb cover rasters already exist; skipping preprocessing.")
    
    def get_tree_cover_data(self):
        """Download and preprocess tree and herbaceous cover rasters.

        Returns:
            None. Writes mean cover rasters to ``{working_dir}/vcf``.
        """
        self._download_tree_cover()
        self._preprocess_tree_cover()


    def plot_tree_cover(self, variable='tree_cover'):
        """Plot mean vegetation cover.

        Args:
            variable (str): ``'tree_cover'`` or ``'herb_cover'``.

        Returns:
            None. Displays a matplotlib plot.
        """
   
        if variable == 'tree_cover':
            this_tc = self.uw.clip(raster_path=f'{self.working_dir}/vcf/mean_tree_cover.tif', 
                                   out_path=None, save_output=False, crop_type=True)[0]
            plt.title('Mean Tree Cover')
            this_tc = np.where(this_tc<=0, np.nan, this_tc)
            this_tc = np.where(this_tc>100, np.nan, this_tc)
            plt.imshow(this_tc, cmap='viridis_r', vmax=50)
            plt.colorbar()
            plt.show()
        elif variable == 'herb_cover':
            this_tc = self.uw.clip(raster_path=f'{self.working_dir}/vcf/mean_herb_cover.tif', 
                                   out_path=None, save_output=False, crop_type=True)[0]
            plt.title('Mean Herbacous Cover')
            this_tc = np.where(this_tc<=0, np.nan, this_tc)
            this_tc = np.where(this_tc>100, np.nan, this_tc)
            plt.imshow(this_tc, cmap='viridis')
            plt.colorbar()
            plt.show()
        else:
            raise ValueError("Invalid variable. Choose 'tree_cover' or 'herb_cover'.")
        
