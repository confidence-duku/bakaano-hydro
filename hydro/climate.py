
import os
from isimip_client.client import ISIMIPClient
from utils import Utils
import os
from concurrent.futures import ThreadPoolExecutor


class ChelsaDataDownloader:
    def __init__(self, project_name, study_area):
        self.study_area = study_area
        self.project_name = project_name
        
        self.uw = Utils(self.project_name, self.study_area)
        self.uw.get_bbox('EPSG:4326')
        self.client = ISIMIPClient()

    def _download_data(self, climate_variable, output_folder):
        response = self.client.datasets(
            simulation_round='ISIMIP3a',
            product='InputData',
            climate_forcing='chelsa-w5e5',
            climate_scenario='obsclim',
            resolution='30arcsec',
            time_step='daily',
            climate_variable=climate_variable
        )
        
        dataset = response["results"][0]
        paths = [file['path'] for file in dataset['files']]
        ds = self.client.cutout(
            paths,
            bbox=[self.uw.miny, self.uw.maxy, self.uw.minx, self.uw.maxx],
            poll=10
        )
        
        download_path = f'../{self.project_name}/{output_folder}'
        os.makedirs(download_path, exist_ok=True)
        self.client.download(ds['file_url'], path=download_path, validate=False, extract=True)

    def get_chelsa_clim_data(self):
        climate_variables = {
            'tasmax': 'tasmax',
            'tasmin': 'tasmin',
            'tas': 'tmean',
            'pr': 'prep'
        }
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._download_data, variable, folder): variable
                for variable, folder in climate_variables.items()
            }
            
            for future in futures:
                variable = futures[future]
                try:
                    future.result()  # Raises exception if download fails
                    print(f"Download completed for {variable}")
                except Exception as e:
                    print(f"An error occurred while downloading {variable}: {e}")
        
        
                    


        

        
        
                    

