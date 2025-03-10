
import requests as r
import os
import rioxarray
from bakaano.utils import Utils

class Soil:
    def __init__(self, working_dir, study_area):
        """
        Initialize a DEM (Digital Elevation Model) object.

        Args:
            working_dir (str): The working directory where files and outputs will be stored.
            study_area (str): The path to the shapefile of the river basin or watershed.
            local_data (bool, optional): Flag indicating whether to use local data instead of downloading new data. Defaults to False.
            local_data_path (str, optional): Path to the local DEM geotiff tile if `local_data` is True. Defaults to None. Local DEM provided should be in the GCS WGS84 or EPSG:4326 coordinate system
        
        Returns:
            A DEM geotiff clipped to the study area extent to be stored in "{working_dir}/elevation" directory
        """
        
        self.study_area = study_area
        self.working_dir = working_dir
        os.makedirs(f'{self.working_dir}/soil', exist_ok=True)
        self.uw = Utils(self.working_dir, self.study_area)
        
    def get_soil_data(self):
        soil_check = f'{self.working_dir}/soil/clipped_AWCtS_M_sl6_1km_ll.tif'
        if not os.path.exists(soil_check):
            urls = ['https://files.isric.org/soilgrids/former/2017-03-10/aggregated/1km/AWCh3_M_sl6_1km_ll.tif',
                    'https://files.isric.org/soilgrids/former/2017-03-10/aggregated/1km/WWP_M_sl6_1km_ll.tif', 
                    'https://files.isric.org/soilgrids/former/2017-03-10/aggregated/1km/AWCtS_M_sl6_1km_ll.tif']
            
            local_filenames = ['AWCh3_M_sl6_1km_ll.tif', 'WWP_M_sl6_1km_ll.tif', 'AWCtS_M_sl6_1km_ll.tif']
            
            for url, filename in zip(urls, local_filenames):
                local_filename = f'{self.working_dir}/soil/{filename}'
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

                
                #extraction_path = f'{self.working_dir}/soil'  # Directory where files will be extracted
                out_path = f'{self.working_dir}/soil/clipped_{filename}'
                self.preprocess(local_filename, out_path)
        else:
            print(f"     - Soil data already exists in {self.working_dir}/soil; skipping download.")
            

    def preprocess(self, raster_dir, out_path):  
        self.uw.clip(raster_path=raster_dir, out_path=out_path, save_output=True)
    
    def plot_soil(self, vmax):
        soil_data = rioxarray.open_rasterio(f'{self.working_dir}/soil/clipped_AWCtS_M_sl6_1km_ll.tif')
        soil_data = soil_data.where(soil_data > 0)
        soil_data.plot(cmap='copper', vmax=vmax)
        