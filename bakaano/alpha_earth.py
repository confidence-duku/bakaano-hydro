import ee
import numpy as np
import geemap
import geopandas as gpd
import os
import glob
import rasterio
from bakaano.utils import Utils
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class AlphaEarth:
    """
    AlphaEarth Class:
    -----------------
    This class facilitates the downloading and preprocessing of Alpha Earth satellite embedding datasets. It includes methods for checking existing data, downloading missing bands, and ensuring data integrity.

    Key Methods:
    - `get_alpha_earth`: Downloads Alpha Earth satellite embeddings. It checks for existing bands and downloads only the missing ones. The data is saved in GeoTIFF format.

    Dependencies:
    - Google Earth Engine (GEE) for data retrieval.
    - Geopandas for shapefile handling.
    - Geemap for GEE integration.

    Usage:
    ```python
    alpha_earth = AlphaEarth(
        working_dir="/path/to/working_dir",
        study_area="/path/to/shapefile.shp",
        start_date="YYYY-MM-DD",
        end_date="YYYY-MM-DD"
    )
    alpha_earth.get_alpha_earth()
    ```
    """

    def __init__(self, working_dir, study_area, start_date, end_date):
        """
        A class used to download and preprocess Alpha Earth satellite embedding dataset

        Args:
            working_dir (str): The parent working directory where files and outputs will be stored.
            study_area (str): The path to the shapefile of the river basin or watershed.
            start_date (str): The start date for the simulation period in 'YYYY-MM-DD' format.
            end_date (str): The end date for the simulation period in 'YYYY-MM-DD' format.

        Attributes:
            study_area (str): Path to the shapefile defining the study area.
            working_dir (str): Directory for storing outputs and intermediate files.
            start_date (str): Start date for data processing.
            end_date (str): End date for data processing.
            uw (Utils): Utility object for handling spatial operations.
        """
        # Initialize class attributes
        self.study_area = study_area  # Path to the shapefile defining the study area
        self.working_dir = working_dir  # Directory for storing outputs and intermediate files
        os.makedirs(f'{self.working_dir}/alpha_earth', exist_ok=True)  # Ensure output directory exists

        # Create a utility object for spatial operations
        self.uw = Utils(self.working_dir, self.study_area)
        self.uw.get_bbox('EPSG:4326')  # Get bounding box in EPSG:4326 coordinate system

        # Set the start and end dates for data processing
        self.start_date = start_date  # Start date for the simulation period
        self.end_date = end_date  # End date for the simulation period



    def get_alpha_earth(self):
        """Download alpha earth satellite embeddings data."""
        
        # List of required AlphaEarth band names
        bandlist = [
            f"A{str(i).zfill(2)}" for i in range(64)
        ]
    
        # Folder path
        outdir = f"{self.working_dir}/alpha_earth"
        os.makedirs(outdir, exist_ok=True)
    
        # Check which bands already exist
        existing_bands = []
        missing_bands = []
    
        for band in bandlist:
            out_path = f"{outdir}/band_{band}.tif"
            if os.path.exists(out_path):
                existing_bands.append(band)
            else:
                missing_bands.append(band)
    
        # --- Case 1: All bands exist → skip download ---
        if len(missing_bands) == 0:
            print(f"✓ All {len(bandlist)} AlphaEarth bands already downloaded. Skipping.")
            return
    
        # --- Case 2: Some bands missing → download only missing ones ---
        print(f"⚠ {len(missing_bands)} bands missing. Downloading:")
        print("   " + ", ".join(missing_bands))
    
        # Authenticate only if needed
        ee.Authenticate()
        ee.Initialize()
    
        embeddings = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    
        gdf = gpd.read_file(self.study_area)
        fc = geemap.gdf_to_ee(gdf)
        region = fc.geometry()
    
        i_date = self.start_date
        f_date = datetime.strptime(self.end_date, "%Y-%m-%d") + timedelta(days=1)
    
        for band in missing_bands:
    
            df = (
                embeddings
                .filterBounds(region)
                .filterDate(i_date, f_date)
                .select(band)
            )
    
            emb_mosaic = df.mean().clip(region)
    
            out_path = f"{outdir}/band_{band}.tif"
    
            geemap.ee_export_image(
                ee_object=emb_mosaic.unmask(0),
                filename=out_path,
                scale=1000,
                region=region,
                crs="EPSG:4326"
            )
    
        print("✓ Missing AlphaEarth bands downloaded successfully.")
    
    def plot_alpha_earth(self, variable='A00'):
        """Plot alpha earth data
        """

        this_tc = self.uw.clip(raster_path=f'{self.working_dir}/alpha_earth/band_{variable}.tif', 
                                out_path=None, save_output=False, crop_type=True)[0]
        plt.title(f'Alpha Earth Vector {variable}')
        this_tc = np.where(this_tc==0, np.nan, this_tc)
        this_tc = np.where(this_tc>1, np.nan, this_tc)
        plt.imshow(this_tc, cmap='viridis')
        plt.colorbar()
        plt.show()

