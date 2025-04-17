
import requests as r
import os
import numpy as np
from bakaano.utils import Utils
import matplotlib.pyplot as plt

class Soil:
    def __init__(self, working_dir, study_area):
        """
        Initialize a Soil object.

        Args:
            working_dir (str): The working directory where files and outputs will be stored.
            study_area (str): The path to the shapefile of the river basin or watershed.
        Methods
        -------
        __init__(working_dir, study_area):
            Initializes the Soil object with project details.
        get_soil_data():
            Download soil data.
        preprocess():
            Preprocess downloaded data.
        plot_soil():
            Plot soil data.
        """
        
        self.study_area = study_area
        self.working_dir = working_dir
        os.makedirs(f'{self.working_dir}/soil', exist_ok=True)
        self.uw = Utils(self.working_dir, self.study_area)
        
    def get_soil_data(self):
        """Download soil data.
        """
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
                self._preprocess(local_filename, out_path)
        else:
            print(f"     - Soil data already exists in {self.working_dir}/soil; skipping download.")
            

    def _preprocess(self, raster_dir, out_path):  
        """Preprocess downloaded data.
        """
        self.uw.clip(raster_path=raster_dir, out_path=out_path, save_output=True, crop_type=False)
    
    def plot_soil(self, variable):
        """Plot soil data.
        """
        if variable=='available_water_content':
            soil_data = self.uw.clip(raster_path=f'{self.working_dir}/soil/clipped_AWCh3_M_sl6_1km_ll.tif', 
                                     out_path=None, save_output=False, crop_type=True)[0]
            soil_data = np.where(soil_data > 0, soil_data, np.nan)
            soil_data = np.where(soil_data < 100, soil_data, np.nan)
            plt.title('Available_water_content')
            plt.imshow(soil_data, cmap='copper')
            plt.colorbar()
        elif variable == 'wilting_point':
            soil_data = self.uw.clip(raster_path=f'{self.working_dir}/soil/clipped_WWP_M_sl6_1km_ll.tif', 
                                     out_path=None, save_output=False, crop_type=True)[0]
            soil_data = np.where(soil_data > 0, soil_data, np.nan)
            soil_data = np.where(soil_data < 100, soil_data, np.nan)
            plt.title('Wilting_point')
            plt.imshow(soil_data, cmap='copper')
            plt.colorbar()
        elif variable == 'saturation_point':
            soil_data = self.uw.clip(raster_path=f'{self.working_dir}/soil/clipped_AWCtS_M_sl6_1km_ll.tif', 
                                     out_path=None, save_output=False, crop_type=True)[0]
            soil_data = np.where(soil_data > 0, soil_data, np.nan)
            soil_data = np.where(soil_data < 100, soil_data, np.nan)
            plt.title('Saturation_point')
            plt.imshow(soil_data, cmap='copper')
            plt.colorbar()
        else:
            raise ValueError("Invalid variable. Select one of 'available_water_content', 'wilting_point', 'saturation_point'")
        
        