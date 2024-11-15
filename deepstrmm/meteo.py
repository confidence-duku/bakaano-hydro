
import os
from isimip_client.client import ISIMIPClient
from deepstrmm.utils import Utils
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class Meteo:
    def __init__(self, working_dir, study_area, local_data=False, local_data_path=None):
        self.study_area = study_area
        self.working_dir = working_dir
        
        self.uw = Utils(self.working_dir, self.study_area)
        self.uw.get_bbox('EPSG:4326')
        self.client = ISIMIPClient()
        self.local_data = local_data
        self.local_data_path = local_data_path

    def _download_data(self, climate_variable, output_folder):
        tasmax_path = Path(f'{self.working_dir}/tasmax/')
        tasmin_path = Path(f'{self.working_dir}/tasmin/')
        tmean_path = Path(f'{self.working_dir}/tmean/')
        prep_path = Path(f'{self.working_dir}/prep/')

        if not any(folder.exists() and any(folder.iterdir()) for folder in [tasmax_path, tasmin_path, tmean_path, prep_path]):
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
            
            download_path = f'{self.working_dir}/{output_folder}'
            os.makedirs(download_path, exist_ok=True)
            self.client.download(ds['file_url'], path=download_path, validate=False, extract=True)
        else:
            print(f"     - Climate data already exists in {tasmax_path}, {tasmin_path}, {tmean_path} and {prep_path}; skipping download.")

    def get_meteo_data(self):
        if self.local_data is False:
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

        
        
                    


        

        
        
                    

