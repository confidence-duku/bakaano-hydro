import numpy as np
import xarray as xr
import os
from pathlib import Path
import rasterio
import rasterio.warp
from rasterio.crs import CRS
import rioxarray
import matplotlib
from operator import itemgetter
import geopandas as gpd
import fiona
import glob
from shapely.geometry import shape
from utils import Utils
from io import BytesIO
from isimip_client.client import ISIMIPClient
import requests
import tempfile
from datetime import datetime, timedelta
# import dask.array as da
# from dask.array import store
# from dask.distributed import Client
#import cupy as cp
from numba import jit
import time
from datetime import datetime, timedelta



class PotentialEvapotranspiration:
    def __init__(self, project_name, study_area):
        self.study_area = study_area
        os.makedirs(f'./{project_name}/clim', exist_ok=True)
        self.clim_dir = Path(f'./{project_name}/clim')

#         os.makedirs('./input_data/tasmin', exist_ok=True)
#         self.tasmin_dir = Path('./input_data/tasmin')

#         os.makedirs('./input_data/tmean', exist_ok=True)
#         self.tmean_dir = Path('./input_data/tmean')
        self.client = ISIMIPClient()

        #self.start_date = start_date
        #self.end_date = end_date
        self.uw = Utils(project_name, self.study_area)
        #self.uw.get_bbox('EPSG:4326')
        
        
    def compute_PET_single(self, tasmax, tasmin, tmean):

        #print('     - Calculating Hargreaves PET')
        lats = tmean['lat']
        latgrids = lats.expand_dims(lon=tmean['lon'], axis=[1])
        
        p1=np.radians(latgrids.values)
        #p1 = np.tile(p1, (tmean_period.values.shape[0], 1, 1))
        dayNum = tmean['time'].dt.dayofyear
        dayNum = dayNum.values
        p2 = 1 + 0.033 *np.cos((2*np.pi*dayNum)/365)
        p3 = 0.409 * np.sin(((2*np.pi*dayNum)/365) - 1.39) #sol_dec
        p4 = np.arccos(-1*np.tan(p1)*np.tan(p3))
        p5 = np.sin(p1) * np.sin(p3)
        p6 = np.cos(p1) * np.cos(p3)
        Ra = ((24 * 60)/np.pi)*0.0820*p2*((p4*p5)+p6*np.sin(p4))

        #computing hargreaves evapotranspiration
        td = tasmax.values - tasmin.values
        b1 = np.sqrt(td)
        
        #b2 = (np.where(b1<0,0,b1))**0.5
        pet = 0.408 * 0.0023 * (tmean.values + 17.8) * b1 * Ra #units is mm/day
        return pet
    
    def compute_PET_numba(self, tasmax, tasmin, tmean):
        
        @jit(nopython=True)
        def hargreaves(tmean_values, tasmax_values, tasmin_values, dayNum_values, lat_values):
            p1 = np.radians(lat_values)
            p2 = 1 + 0.033 * np.cos((2 * np.pi * dayNum_values) / 365)
            p3 = 0.409 * np.sin(((2 * np.pi * dayNum_values) / 365) - 1.39)
            p4 = np.arccos(-1 * np.tan(p1) * np.tan(p3))
            p5 = np.sin(p1) * np.sin(p3)
            p6 = np.cos(p1) * np.cos(p3)
            Ra = ((24 * 60) / np.pi) * 0.0820 * p2 * ((p4 * p5) + p6 * np.sin(p4))
            
            td = tasmax_values - tasmin_values
            b1 = np.sqrt(td)
            pet = 0.408 * 0.0023 * (tmean_values + 17.8) * b1 * Ra
            return pet
        
        lats = tmean['lat']
        latgrids = lats.expand_dims(lon=tmean['lon'], axis=[1])
        latgrids = latgrids.values
        
        dayNum = tmean['time'].dt.dayofyear.values
        
        pet = hargreaves(tmean.values, tasmax.values, tasmin.values, dayNum, latgrids)
        return pet

    

    def compute_PET_dask(self, tasmax, tasmin, tmean):
        """
        This function computes the Hargreaves PET using Dask arrays.

        Args:
          tasmax (da.Array): Dask array of maximum daily temperature.
          tasmin (da.Array): Dask array of minimum daily temperature.
          tmean (da.Array): Dask array of mean daily temperature.

        Returns:
          da.Array: Dask array of potential evapotranspiration (PET) in mm/day.
        """

        # Extract latitudes from tmean
        lats = tmean['lat']

        # Expand latitudes to match data dimensions (avoid tiling)
        latgrids = lats.expand_dims(lon=tmean['lon'], axis=[1])

        

        # Dask operations on arrays
        p1 = da.radians(latgrids.values)
        dayNum = tmean['time'].dt.dayofyear.values
        p2 = 1 + 0.033 * da.cos((2 * np.pi * dayNum) / 365)
        p3 = 0.409 * da.sin(((2 * np.pi * dayNum) / 365) - 1.39)  # sol_dec
        p4 = da.arccos(-1 * da.tan(p1) * da.tan(p3))
        p5 = da.sin(p1) * da.sin(p3)
        p6 = da.cos(p1) * da.cos(p3)
        Ra = ((24 * 60) / np.pi) * 0.0820 * p2 * ((p4 * p5) + p6 * da.sin(p4))

        # PET calculation with Dask arrays
        td = tasmax - tasmin
        b1 = da.sqrt(td)
        pet = 0.408 * 0.0023 * (tmean + 17.8) * b1 * Ra  # units are mm/day
        return pet

    
    def compute_PET(self, day_pet_params, latgrids, tmean):
        #start_time = time.perf_counter()
        
        # Convert latgrids to CuPy array and calculate radians
        p1 = np.radians(latgrids)

        # Extract day of the year and convert to CuPy array
        
        dayNum = tmean['time'].dt.dayofyear
        dayNum = dayNum.values

        # Calculate solar radiation components using CuPy
        p2 = 1 + 0.033 * np.cos((2 * np.pi * dayNum) / 365)
        p3 = 0.409 * np.sin(((2 * np.pi * dayNum) / 365) - 1.39)  # sol_dec
        p4 = np.arccos(-1 * np.tan(p1) * np.tan(p3))
        p5 = np.sin(p1) * np.sin(p3)
        p6 = np.cos(p1) * np.cos(p3)
        Ra = ((24 * 60) / np.pi) * 0.0820 * p2 * ((p4 * p5) + p6 * np.sin(p4))

        pet = day_pet_params * Ra  # units in mm/day

        #end_time = time.perf_counter()
        #print(f"     Execution time: {end_time - start_time} seconds")
        return pet
        
    
    

    
    
    def compute_PET_online(self, tasmax, tasmin, tmean):
       
        #lg = './input_data/pet/latgrids.tif'
        print('     - Calculating Hargreaves PET')
        #converting to .to_array() stacks all variables along a new dimension in this case instead of 2d shape creates a 3d shape.
        # this code just is to select the variable
        tasmax = tasmax[0]
        tasmin = tasmin[0]
        tmean = tmean[0]
        
        lats = tmean['lat']
        latgrids = lats.expand_dims(lon=tmean['lon'], axis=[1])
        
        p1=np.radians(latgrids)
        #p1 = np.tile(p1, (tmean_period.values.shape[0], 1, 1))
        doy = tmean['time'].dt.dayofyear
        dayNum = int(doy.values)
        #dayNum = doy.expand_dims(lat=tmean['lat'], lon=tmean['lon'], axis=[1,2])
        p2 = 1 + 0.033 *np.cos((2*np.pi*dayNum)/365)
        p3 = 0.409 * np.sin(((2*np.pi*dayNum)/365) - 1.39) #sol_dec
        p4 = np.arccos(-1*np.tan(p1)*np.tan(p3))
        p5 = np.sin(p1) * np.sin(p3)
        p6 = np.cos(p1) * np.cos(p3)
        Ra = ((24 * 60)/np.pi)*0.0820*p2*((p4*p5)+p6*np.sin(p4))

        #computing hargreaves evapotranspiration
        td = tasmax.values - tasmin.values
        b1 = np.sqrt(td)
        
        #b2 = (np.where(b1<0,0,b1))**0.5
        pet = 0.408 * 0.0023 * (tmean.values + 17.8) * b1 * Ra #units is mm/day
        return pet

    def get_chelsa_clim_data(self,  start_date):
        year, month, day = start_date.split('-')
        year_month = year + month
        date = year + '-' + month + '-' + day        
        
        start_date = datetime.strptime(date, '%Y-%m-%d').date()
        thisdate = str(thisdate.year) + '-' + str(thisdate.month) + '-'+str(thisdate.day)

        tasmax_response = requests.get(self.tasmax_urls)
        tasmin_response = requests.get(self.tasmin_urls)
        tas_response = requests.get(self.tas_urls)
        pr_response = requests.get(self.pr_urls)

        response_list = [tasmax_response, tasmin_response, tas_response, pr_response]
        output_ds = []

        for r in response_list:
            if r.status_code == 200:
                self.clim_datalist = [line.strip() for line in r.text.split('\n')]
                clim_match =  [s for s in self.clim_datalist if year_month in s][0]
                clim_match = requests.get(clim_match)
                with xr.open_dataset(BytesIO(clim_match.content)) as ds:
                    day_clim = ds.sel(time=thisdate)
                    ds = day_clim.rio.write_crs(4326)  # Ensure consistent CRS
                    ds = ds.rio.clip_box(self.uw.minx, self.uw.miny, self.uw.maxx, self.uw.maxy)
                output_ds.append(ds.to_array())
                #ds = ds.rename({'lon': 'x', 'lat':'y'})
                #ds = ds.rio.reproject_match(self.match)
        return output_ds
    
        


