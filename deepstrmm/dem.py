
import requests as r
import geopandas as gpd
import glob
from shapely.geometry import Polygon
import json, time, os
import uuid
from deepstrmm.utils import Utils
import zipfile

class DEM:
    def __init__(self, working_dir, study_area, local_data=False, local_data_path=None):
        self.study_area = study_area
        #self.username = earthexplorer_username
        #self.password = earthexplorer_password
        self.working_dir = working_dir
        self.uw = Utils(self.working_dir, self.study_area)
        self.out_path = f'{self.working_dir}/elevation/dem_clipped.tif'
        self.local_data = local_data
        self.local_data_path = local_data_path
        
    def get_dem_data(self):
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
            print(f"     - Local DEM data already provided")
            self.uw.clip(raster_path=self.local_data_path, out_path=self.out_path, save_output=True)

    def preprocess(self):
        dem = f'{self.working_dir}/elevation/hyd_glo_dem_30s.tif'
        
        self.uw.clip(raster_path=dem, out_path=self.out_path, save_output=True)
    
        