import numpy as np
import os
import xarray as xr
from pathlib import Path
from isimip_client.client import ISIMIPClient
import requests
from datetime import datetime, timedelta
import os
#from concurrent.futures import ThreadPoolExecutor, as_completed

class ChelsaDataDownloader:
    def __init__(self, project_name, study_area):
        self.study_area = study_area
        self.project_name = project_name
    
    def get_chelsa_clim_data(self):
        client = ISIMIPClient()

        response = client.datasets(simulation_round='ISIMIP3a', product='InputData', climate_forcing='chelsa-w5e5',
                                  climate_scenario='obsclim', resolution='30arcsec', 
                                  time_step='daily', climate_variable='pr')

        dataset = response["results"][0]
        paths = [file['path'] for file in dataset['files']]
        ds = client.cutout(paths, bbox=[self.study_area[0], self.study_area[2], 
                                        self.study_area[1], self.study_area[3]], poll=10)
        client.download(ds['file_url'], path=f'./projects/{self.project_name}/clim', validate=False, extract=True)
        

        
        
                    

