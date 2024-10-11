import numpy as np
import os
import xarray as xr
from pathlib import Path
from isimip_client.client import ISIMIPClient
import requests
from datetime import datetime, timedelta
import os
from concurrent.futures import ThreadPoolExecutor

# class ChelsaDataDownloader:
#     def __init__(self, project_name, study_area):
#         self.study_area = study_area
#         self.project_name = project_name
    
#     def get_chelsa_clim_data(self):
#         client = ISIMIPClient()
        
#         #download tasmax
#         response = client.datasets(simulation_round='ISIMIP3a', product='InputData', climate_forcing='chelsa-w5e5',
#                                   climate_scenario='obsclim', resolution='30arcsec', 
#                                   time_step='daily', climate_variable='tasmax')

#         dataset = response["results"][0]
#         paths = [file['path'] for file in dataset['files']]
#         ds = client.cutout(paths, bbox=[self.study_area[0], self.study_area[2], 
#                                         self.study_area[1], self.study_area[3]], poll=10)
#         client.download(ds['file_url'], path=f'./projects/{self.project_name}/input_data/tasmax', validate=False, extract=True)

#         #downloading tasmin
#         response = client.datasets(simulation_round='ISIMIP3a', product='InputData', climate_forcing='chelsa-w5e5',
#                                   climate_scenario='obsclim', resolution='30arcsec', 
#                                   time_step='daily', climate_variable='tasmax')

#         dataset = response["results"][0]
#         paths = [file['path'] for file in dataset['files']]
#         ds = client.cutout(paths, bbox=[self.study_area[0], self.study_area[2], 
#                                         self.study_area[1], self.study_area[3]], poll=10)
#         client.download(ds['file_url'], path=f'./projects/{self.project_name}/input_data/tasmin', validate=False, extract=True)

#         #downloading srad
#         response = client.datasets(simulation_round='ISIMIP3a', product='InputData', climate_forcing='chelsa-w5e5',
#                                   climate_scenario='obsclim', resolution='30arcsec', 
#                                   time_step='daily', climate_variable='rsds')

#         dataset = response["results"][0]
#         paths = [file['path'] for file in dataset['files']]
#         ds = client.cutout(paths, bbox=[self.study_area[0], self.study_area[2], 
#                                         self.study_area[1], self.study_area[3]], poll=10)
#         client.download(ds['file_url'], path=f'./projects/{self.project_name}/input_data/srad', validate=False, extract=True)


class ChelsaDataDownloader:
    def __init__(self, project_name, study_area):
        self.study_area = study_area
        self.project_name = project_name
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
            bbox=[self.study_area[1], self.study_area[3], 
                  self.study_area[0], self.study_area[2]],
            poll=10
        )
        
        download_path = f'./projects/{self.project_name}/input_data/{output_folder}'
        os.makedirs(download_path, exist_ok=True)
        self.client.download(ds['file_url'], path=download_path, validate=False, extract=True)

    def get_chelsa_clim_data(self):
        climate_variables = {
            'tasmax': 'tasmax',
            'tasmin': 'tasmin',
            'rsds': 'srad'
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


        
                    

