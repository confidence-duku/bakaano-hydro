
import requests as r
import os
import numpy as np
from bakaano.utils import Utils
import zipfile
import matplotlib.pyplot as plt
import rasterio
from scipy.ndimage import convolve

class DEM:
    def __init__(self, working_dir, study_area, local_data=False, local_data_path=None):
        """
        Initialize a DEM (Digital Elevation Model) object.

        Args:
            working_dir (str): The parent working directory where files and outputs will be stored.
            study_area (str): The path to the shapefile of the river basin or watershed.
            local_data (bool, optional): Flag indicating whether to use local data instead of downloading new data. Defaults to False.
            local_data_path (str, optional): Path to the local DEM geotiff tile if `local_data` is True. Defaults to None. Local DEM provided should be in the GCS WGS84 or EPSG:4326 coordinate system
        Methods
        -------
        __init__(working_dir, study_area, local_data=False, local_data_path=None):
            Initializes the DEM object with project details.
        get_dem_data():
            Download DEM data. 
        preprocess():
            Preprocess downloaded data.
        plot_dem():
            Plot DEM data

        Returns:
            A DEM geotiff clipped to the study area extent to be stored in "{working_dir}/elevation" directory
        """
        
        self.study_area = study_area
        self.working_dir = working_dir
        os.makedirs(f'{self.working_dir}/elevation', exist_ok=True)
        self.uw = Utils(self.working_dir, self.study_area)
        self.out_path = f'{self.working_dir}/elevation/dem_clipped.tif'
        self.local_data = local_data
        self.local_data_path = local_data_path
        
    def get_dem_data(self):
        """Download DEM data.
        """
        if self.local_data is False:
            if not os.path.exists(self.out_path):
                url = 'https://data.hydrosheds.org/file/hydrosheds-v1-dem/hyd_glo_dem_30s.zip'
                local_filename = f'{self.working_dir}/elevation/hyd_glo_dem_30s.zip'
                uw = Utils(self.working_dir, self.study_area)
                uw.get_bbox('EPSG:4326')
                response = r.get(url, stream=True)
                if response.status_code == 200:
                    with open(local_filename, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"File downloaded successfully and saved as '{local_filename}'")
                else:
                    print(f"Failed to download the file. HTTP status code: {response.status_code}")

                
                extraction_path = f'{self.working_dir}/elevation'  # Directory where files will be extracted

                # Open and extract the zip file
                with zipfile.ZipFile(local_filename, 'r') as zip_ref:
                    zip_ref.extractall(extraction_path)
                    print(f"Files extracted to '{extraction_path}'")

                self.preprocess()

            else:
                print(f"     - DEM data already exists in {self.working_dir}/elevation; skipping download.")
                

        else:
            #print(f"     - Local DEM data already provided")
            try:
                if not self.local_data_path:
                    raise ValueError("Local data path must be provided when 'local_data' is set to True.")
                if not os.path.exists(self.local_data_path):
                    raise FileNotFoundError(f"The specified local DEM file '{self.local_data_path}' does not exist.")
                if not self.local_data_path.endswith('.tif'):
                    raise ValueError("The local DEM file must be a GeoTIFF (.tif) file.")
                self.uw.clip(raster_path=self.local_data_path, out_path=self.out_path, save_output=True)
            except (ValueError, FileNotFoundError) as e:
                print(f"Error: {e}")

    def preprocess(self):
        """Preprocess DEM data.
        """
        dem = f'{self.working_dir}/elevation/hyd_glo_dem_30s.tif'   
        self.uw.clip(raster_path=dem, out_path=self.out_path, save_output=True, crop_type=False)
        #self.uw.clip(raster_path=dem, out_path=self.out_path_uncropped, save_output=True, crop_type=False)

        slope_name = f'{self.working_dir}/elevation/slope_clipped.tif'
        if not os.path.exists(slope_name):
            self.compute_slope_percent_riserun()

    def compute_slope_percent_riserun(self):
        with rasterio.open(self.out_path) as src:
            elevation = src.read(1).astype(float)
            profile = src.profile.copy()
            res_x, res_y = src.res
            cellsize = np.mean([abs(res_x), abs(res_y)]) *100000
            nodata = src.nodata

        # Handle NoData
        if nodata is not None:
            elevation[elevation == nodata] = np.nan

        # Fill NaNs for convolution
        elevation_filled = np.where(np.isnan(elevation), np.nanmean(elevation), elevation)

        # Horn kernels (adjusted for cellsize in meters)
        kernel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]]) / (8 * cellsize)

        kernel_y = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]]) / (8 * cellsize)

        # Compute gradients
        dzdx = convolve(elevation_filled, kernel_x, mode='nearest')
        dzdy = convolve(elevation_filled, kernel_y, mode='nearest')

        # Slope in percent rise/run
        slope_percent = np.sqrt(dzdx**2 + dzdy**2) * 100
        slope_percent[np.isnan(elevation)] = np.nan

        # Update metadata
        profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)

        # Save to GeoTIFF
        output_path = f'{self.working_dir}/elevation/slope_clipped.tif'
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(slope_percent.astype(np.float32), 1)

        print(f"Slope saved to: {output_path}")

    def plot_dem(self):
        """Plot DEM data.
        """
        dem_data = self.uw.clip(raster_path=self.out_path, out_path=None, save_output=False, crop_type=True)[0]
        dem_data = np.where(dem_data > 0, dem_data, np.nan)
        dem_data = np.where(dem_data < 32000, dem_data, np.nan)
        plt.imshow(dem_data, cmap='terrain')
        plt.colorbar()
        
        