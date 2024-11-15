import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np
import fiona
import glob
from shapely.geometry import shape
import rasterio
import rasterio.warp
from rasterio.crs import CRS
import rioxarray
#import matplotlib.pyplot as plt
import os
from operator import itemgetter
import glob
from datetime import datetime, timedelta
from pathlib import Path
from io import BytesIO
from isimip_client.client import ISIMIPClient
import requests
import tempfile
import warnings
from rasterio.errors import RasterioDeprecationWarning

# Suppress the RasterioDeprecationWarning
warnings.filterwarnings("ignore", category=RasterioDeprecationWarning)

class Process_ClimateData:
    
    """
    A class for processing climate data including clipping and calculating climate indices.
    
    Args:
      data_path (str): Path to the directory containing climate data files.
      clip_bbox (dict): Dictionary containing bounding box coordinates for clipping (minx, miny, maxx, maxy).
      year_start (str): Starting year for climate index calculations (inclusive).
      year_end (str): Ending year for climate index calculations (inclusive).
    """
    
    def __init__(self, project_name, study_area):
        self.working_dir = project_name
        self.ras_template = glob.glob(f'./projects/{self.working_dir}/input_data/lst/*')[0]
        self.match = rioxarray.open_rasterio(self.ras_template)
        self.client = ISIMIPClient()
        self.study_area = study_area
   #     test = 0
        
    def concat_nc(self):
        #nc_list = []
        match = rioxarray.open_rasterio(self.ras_template)
        
        files = list(map(str, self.clim_dir.glob('*.nc')))
        concatenated_nc_list = []
        for nc in files:
            #current_directory = os.getcwd()  # Get the current working directory
            #file_path = os.path.join(current_directory, nc)
            dataset = xr.open_dataset(nc)
            data_var = dataset.sel()  # Select the desired variable
            concatenated_nc_list.append(data_var)
        concatenated_nc = xr.concat(concatenated_nc_list, dim='time')
        ds = concatenated_nc.rio.write_crs(4326)  # Ensure consistent CRS
        ds = ds.rename({'lon': 'x', 'lat':'y'})
        ds = ds.rio.reproject_match(match)
        #data_var = data_var.rio.clip_box(self.minx, self.miny, self.maxx, self.maxy)
        #ds = data_var.sel(time=slice(self.time_start, self.time_end))
        return ds

    
    def select_daily_clim(self,  var_name, lst_date):
        
        year = lst_date[:4]
        month = lst_date[4:6]
        day = lst_date[-2:]
        year_month = lst_date[:-2]
        date = year + '-' + month + '-' + day
        thisdate = datetime.strptime(date, '%Y-%m-%d').date()
        thisdate = str(thisdate.year) + '-' + str(thisdate.month) + '-'+str(thisdate.day)

        clim_urls = self.client.datasets(simulation_round='ISIMIP3a',
                           product='InputData',
                           climate_forcing='chelsa-w5e5',
                           climate_scenario='obsclim',
                           resolution='30arcsec',
                           time_step='daily',
                           climate_variable=var_name)['results'][0]['filelist_url']

        response = requests.get(clim_urls)

        if response.status_code == 200:
            clim_datalist = [line.strip() for line in response.text.split('\n')]
            clim_match =  [s for s in clim_datalist if year_month in s][0]
            clim_match = requests.get(clim_match)
            with xr.open_dataset(BytesIO(clim_match.content)) as ds:
                day_clim = ds.sel(time=thisdate)
                ds = day_clim.rio.write_crs(4326)  # Ensure consistent CRS
                ds = ds.rio.clip_box(self.study_area[0], self.study_area[1], 
                                        self.study_area[2], self.study_area[3])
                ds = ds.rename({'lon': 'x', 'lat':'y'})
                ds = ds.rio.reproject_match(self.match)
        return ds.to_array()

        
        with xr.open_dataset(clim_filepath + '/' + clim_match[0]) as ds:
            day_clim = ds.sel(time=thisdate)
            ds = day_clim.rio.write_crs(4326)  # Ensure consistent CRS
            ds = ds.rename({'lon': 'x', 'lat':'y'})
            ds = ds.rio.reproject_match(self.match)
        return ds.to_array()

    def select_daily_clim_local(self,  clim_dir, lst_date):
        
        year = lst_date[:4]
        month = lst_date[4:6]
        day = lst_date[-2:]
        year_month = lst_date[:-2]
        date = year + '-' + month + '-' + day
        thisdate = datetime.strptime(date, '%Y-%m-%d').date()
        thisdate = str(thisdate.year) + '-' + str(thisdate.month) + '-'+str(thisdate.day)
        nc_files = list(map(str, clim_dir.glob('*.nc')))
        clim_match =  [s for s in nc_files if year_month in s][0]
        with xr.open_dataset(clim_match) as ds:
            day_clim = ds.sel(time=thisdate)
            ds = day_clim.rio.write_crs(4326)  # Ensure consistent CRS
            #ds = ds.rio.clip_box(self.study_area[0], self.study_area[1], 
            #                            self.study_area[2], self.study_area[3])
            ds = ds.rename({'lon': 'x', 'lat':'y'})
            ds = ds.rio.reproject_match(self.match)
        return ds.to_array()
        

    
        
