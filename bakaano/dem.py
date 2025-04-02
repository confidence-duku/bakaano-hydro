
import requests as r
import os
import rasterio
import rioxarray
from bakaano.utils import Utils
import zipfile
import richdem as rd
import matplotlib.pyplot as plt

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
        self.uw.clip(raster_path=dem, out_path=self.out_path, save_output=True)

        slope_name = f'{self.working_dir}/elevation/slope_clipped.tif'
        if not os.path.exists(slope_name):
            dem_array = rasterio.open(self.out_path).read(1)
            rd_dem = rd.rdarray(dem_array, no_data=-9999)
            slope = rd.TerrainAttribute(rd_dem, attrib='slope_riserun')
            self.uw.save_to_scratch(slope_name, slope)

    
    def plot_dem(self):
        """Plot DEM data.
        """
        dem_data = rioxarray.open_rasterio(self.out_path)
        dem_data = dem_data.where(dem_data > 0)
        dem_data = dem_data.where(dem_data < 32000)
        #dem_data.plot(cmap='terrain')
        plt.imshow(dem_data[0], cmap='terrain')
        plt.colorbar()
        
        