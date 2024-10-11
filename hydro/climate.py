import numpy as np
import os
import xarray as xr
from pathlib import Path
from io import BytesIO
from isimip_client.client import ISIMIPClient
import requests
from utils import Utils
from datetime import datetime, timedelta
import os
#from concurrent.futures import ThreadPoolExecutor, as_completed

class ChelsaDataDownloader:
    def __init__(self, project_name, study_area, start_date, end_date):
        self.study_area = study_area
        self.start_date = start_date
        self.end_date = end_date
        self.project_name = project_name
        
        self.uw = Utils(self.project_name, self.study_area)
        self.uw.get_bbox('EPSG:4326')
        
        os.makedirs(f'./{self.project_name}/clim', exist_ok=True)
        self.clim_dir = Path(f'./{self.project_name}/clim')
        
    def download_and_process(self, cv, uw, minx, miny, maxx, maxy):
        clim_match = requests.get(cv)
        fp, fn = os.path.split(cv)
        with xr.open_dataset(BytesIO(clim_match.content)) as ds:
            ds = ds.chunk({'time': 1})
            ds = ds.rio.write_crs(4326)  # Ensure consistent CRS
            ds = ds.rio.clip_box(minx, miny, maxx, maxy)
            ds.to_netcdf(f'./{self.project_name}/clim/' + fn, mode='w')

    def get_chelsa_clim_data_multi(self):
        client = ISIMIPClient()

        # Fetch URLs for different climate variables
        tasmax_urls = client.datasets(simulation_round='ISIMIP3a', product='InputData', climate_forcing='chelsa-w5e5',
                                      climate_scenario='obsclim', resolution='30arcsec', 
                                      time_step='daily', climate_variable='tasmax')['results'][0]['filelist_url']

        tasmin_urls = client.datasets(simulation_round='ISIMIP3a', product='InputData', climate_forcing='chelsa-w5e5',
                                      climate_scenario='obsclim', resolution='30arcsec', 
                                      time_step='daily', climate_variable='tasmin')['results'][0]['filelist_url']

        tas_urls = client.datasets(simulation_round='ISIMIP3a', product='InputData', climate_forcing='chelsa-w5e5',
                                   climate_scenario='obsclim', resolution='30arcsec', 
                                   time_step='daily', climate_variable='tas')['results'][0]['filelist_url']

        pr_urls = client.datasets(simulation_round='ISIMIP3a', product='InputData', climate_forcing='chelsa-w5e5',
                                  climate_scenario='obsclim', resolution='30arcsec', 
                                  time_step='daily', climate_variable='pr')['results'][0]['filelist_url']

        reference = '1979-01-01'
        refdate = datetime.strptime(reference, '%Y-%m-%d').date()
        startdate = datetime.strptime(self.start_date, '%Y-%m-%d').date()
        enddate = datetime.strptime(self.end_date, '%Y-%m-%d').date()

        start_calc = (startdate.year - refdate.year) * 12 + startdate.month - refdate.month
        calc_period = (enddate.year - startdate.year) * 12 + enddate.month - startdate.month

        tasmax_response = requests.get(tasmax_urls)
        tasmin_response = requests.get(tasmin_urls)
        tas_response = requests.get(tas_urls)
        pr_response = requests.get(pr_urls)

        response_list = [tasmax_response, tasmin_response, tas_response, pr_response]
        varnames = ['tasmax', 'tasmin', 'tas', 'pr']

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []

            for r, n in zip(response_list, varnames):
                print("     - Downloading CHELSA " + n + ' data')
                self.filename = n + '_' + self.start_date[:-3] + '_' + self.end_date[:-3] + '.nc'
                filepath = os.path.join(self.clim_dir, self.filename)
                check = self.uw.process_existing_file(filepath)
                if r.status_code == 200 and not check:
                    clim_datalist = [line.strip() for line in r.text.split('\n')]
                    clim_datalist = clim_datalist[start_calc:(start_calc + calc_period)]

                    for cv in clim_datalist:
                        futures.append(executor.submit(self.download_and_process, cv, self.uw, self.uw.minx, 
                                                       self.uw.miny, self.uw.maxx, self.uw.maxy))

            # Wait for all futures to complete
            for future in as_completed(futures):
                future.result()

    
    def get_chelsa_clim_data(self):
        client = ISIMIPClient()

        # Fetch URLs for different climate variables
        tasmax_urls = client.datasets(simulation_round='ISIMIP3a', product='InputData', climate_forcing='chelsa-w5e5',
                                      climate_scenario='obsclim', resolution='30arcsec', 
                                      time_step='daily', climate_variable='tasmax')['results'][0]['filelist_url']

        tasmin_urls = client.datasets(simulation_round='ISIMIP3a', product='InputData', climate_forcing='chelsa-w5e5',
                                      climate_scenario='obsclim', resolution='30arcsec', 
                                      time_step='daily', climate_variable='tasmin')['results'][0]['filelist_url']

        tas_urls = client.datasets(simulation_round='ISIMIP3a', product='InputData', climate_forcing='chelsa-w5e5',
                                   climate_scenario='obsclim', resolution='30arcsec', 
                                   time_step='daily', climate_variable='tas')['results'][0]['filelist_url']

        pr_urls = client.datasets(simulation_round='ISIMIP3a', product='InputData', climate_forcing='chelsa-w5e5',
                                  climate_scenario='obsclim', resolution='30arcsec', 
                                  time_step='daily', climate_variable='pr')['results'][0]['filelist_url']

        reference = '1979-01-01'
        refdate = datetime.strptime(reference, '%Y-%m-%d').date()
        startdate = datetime.strptime(self.start_date, '%Y-%m-%d').date()
        enddate = datetime.strptime(self.end_date, '%Y-%m-%d').date()

        start_calc = (startdate.year - refdate.year) * 12 + startdate.month - refdate.month
        calc_period = (enddate.year - startdate.year) * 12 + enddate.month - startdate.month

        tasmax_response = requests.get(tasmax_urls)
        tasmin_response = requests.get(tasmin_urls)
        tas_response = requests.get(tas_urls)
        pr_response = requests.get(pr_urls)

        response_list = [tasmax_response, tasmin_response, tas_response, pr_response]
        varnames = ['tasmax', 'tasmin', 'tas', 'pr']

        for r, n in zip(response_list, varnames):
            print("     - Downloading CHELSA " + n + ' data')
            self.filename = n + '_' + self.start_date[:-3] + '_' + self.end_date[:-3] + '.nc'
            filepath = os.path.join(self.clim_dir, self.filename)
            check = self.uw.process_existing_file(filepath)
            if r.status_code == 200 and not check:
                clim_datalist = [line.strip() for line in r.text.split('\n')]
                clim_datalist = clim_datalist[start_calc:(start_calc + calc_period)]
                for cv in clim_datalist:
                    self.download_and_process(cv, self.uw, self.uw.minx, self.uw.miny, self.uw.maxx, self.uw.maxy)

