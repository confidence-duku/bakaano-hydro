
import os
import ee
import geemap
import glob
import numpy as np
import rasterio
import rioxarray
from isimip_client.client import ISIMIPClient
from bakaano.utils import Utils
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import xarray as xr
from datetime import datetime


class Meteo:
    def __init__(self, working_dir, study_area, start_date, end_date, local_data=False, data_source='CHELSA', local_prep_path=None, 
                 local_tasmax_path=None, local_tasmin_path=None, local_tmean_path=None):
        
        """
        Initialize a Meteo object.

        Args:
            working_dir (str): The working directory where files and outputs will be stored.
            study_area (str): The path to the shapefile of the river basin or watershed.
            start_date (str): The start date for the data in 'YYYY-MM-DD' format.
            end_date (str): The end date for the data in 'YYYY-MM-DD' format.
            local_data (bool, optional): Flag indicating whether to use local data instead of downloading new data. Defaults to False.
            data_source (str, optional): The source of the data. Options are 'CHELSA', 'ERA5', or 'CHIRPS'. Defaults to 'CHELSA'.
            local_prep_path (str, optional): Path to the local NetCDF file containing daily rainfall data. Required if `local_data` is True.
            local_tasmax_path (str, optional): Path to the local NetCDF file containing daily maximum temperature data (in Kelvin). Required if `local_data` is True.
            local_tasmin_path (str, optional): Path to the local NetCDF file containing daily minimum temperature data (in Kelvin). Required if `local_data` is True.
            local_tmean_path (str, optional): Path to the local NetCDF file containing daily mean temperature data (in Kelvin). Required if `local_data` is True.
        Methods
        -------
        __init__(working_dir, study_area, start_date, end_date, local_data=False, data_source='CHELSA', local_prep_path=None,
                    local_tasmax_path=None, local_tasmin_path=None, local_tmean_path=None):
                Initializes the Meteo object with project details.
        get_meteo_data():
                Downloads and processes meteorological data from the specified source.
        get_era5_land_meteo_data():
                Downloads and processes ERA5 Land meteorological data.
        get_chirps_prep_meteo_data():
                Downloads and processes CHIRPS precipitation data.
        get_chelsa_meteo_data():
                Downloads and processes CHELSA meteorological data.
        export_urls_for_download_manager():
                Exports URLs for downloading CHELSA meteorological data.
        _download_chelsa_data():
                Downloads CHELSA meteorological data.
        _download_era5_land_data():
                Downloads ERA5 Land meteorological data.
        _download_chirps_prep_data():
                Downloads CHIRPS precipitation data.
        _download_chelsa_data():
                Downloads CHELSA meteorological data.
        _download_era5_land_data():
                Downloads ERA5 Land meteorological data.
        
        Returns:
            Dataarrays clipped to the study area extent, reprojected to the correct CRS, resampled to match DEM resolution
        """
        self.study_area = study_area
        self.working_dir = working_dir
        os.makedirs(f'{self.working_dir}/{data_source}/tasmax', exist_ok=True)
        os.makedirs(f'{self.working_dir}/{data_source}/tasmin', exist_ok=True)
        os.makedirs(f'{self.working_dir}/{data_source}/prep', exist_ok=True)
        os.makedirs(f'{self.working_dir}/{data_source}/tmean', exist_ok=True)
        self.uw = Utils(self.working_dir, self.study_area)
        self.uw.get_bbox('EPSG:4326')
        self.client = ISIMIPClient()
        self.local_data = local_data
        self.data_source = data_source
        self.start_date = start_date
        self.end_date = end_date

        if local_data is True:
            self.prep_path = local_prep_path
            self.tasmax_path = local_tasmax_path
            self.tasmin_path = local_tasmin_path
            self.tmean_path = local_tmean_path
        else:
            if self.data_source == 'chirps':

                self.tasmax_path = Path(f'{self.working_dir}/era5_land/tasmax/')
                self.tasmin_path = Path(f'{self.working_dir}/era5_land/tasmin/')
                self.tmean_path = Path(f'{self.working_dir}/era5_land/tmean/')
                self.prep_path = Path(f'{self.working_dir}/era5_land/prep/')
                self.era5_scratch = Path(f'{self.working_dir}/era5_scratch/')
                self.chirps_scratch = Path(f'{self.working_dir}/chirps_scratch/')
            else:
                self.tasmax_path = Path(f'{self.working_dir}/{self.data_source}/tasmax/')
                self.tasmin_path = Path(f'{self.working_dir}/{self.data_source}/tasmin/')
                self.tmean_path = Path(f'{self.working_dir}/{self.data_source}/tmean/')
                self.prep_path = Path(f'{self.working_dir}/{self.data_source}/prep/')
                self.era5_scratch = Path(f'{self.working_dir}/era5_scratch/')

    def _download_chelsa_data(self, climate_variable, output_folder):

        if not any(folder.exists() and any(folder.iterdir()) for folder in [self.tasmax_path, self.tasmin_path, self.tmean_path, self.prep_path]):
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
            print(f"     - Climate data already exists in {self.tasmax_path}, {self.tasmin_path}, {self.tmean_path} and {self.prep_path}; skipping download.")
    
    def _download_era5_land_data(self):
       
        #ee.Authenticate()
        ee.Initialize()

        era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")

        start_year = int(self.start_date[:4])
        end_year = int(self.end_date[:4])

        for year in range(start_year, end_year + 1):
            i_date = f"{year}-01-01"
            f_date = f"{year + 1}-01-01" if year < end_year else self.end_date  # Final year may be partial
            df = era5.select('total_precipitation_sum', 'temperature_2m_min', 'temperature_2m_max', 'temperature_2m').filterDate(i_date, f_date)
    
            area = ee.Geometry.BBox(self.uw.minx, self.uw.miny, self.uw.maxx, self.uw.maxy) 
            geemap.ee_export_image_collection(ee_object=df, out_dir=self.era5_scratch, scale=10000, region=area, crs='EPSG:4326', file_per_band=True)  
        print('Download completed')

    def _download_chirps_prep_data(self):
        
        ee.Authenticate()
        ee.Initialize()

        chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")

        start_year = int(self.start_date[:4])
        end_year = int(self.end_date[:4])

        for year in range(start_year, end_year + 1):
            i_date = f"{year}-01-01"
            f_date = f"{year + 1}-01-01" if year < end_year else self.end_date  # Final year may be partial
            df = chirps.select('precipitation').filterDate(i_date, f_date)
            area = ee.Geometry.BBox(self.uw.minx, self.uw.miny, self.uw.maxx, self.uw.maxy) 
            geemap.ee_export_image_collection(ee_object=df, out_dir=self.chirps_scratch, scale=5000, region=area, crs='EPSG:4326', file_per_band=True) 
            
           
            df2 = era5.select('temperature_2m_min', 'temperature_2m_max', 'temperature_2m').filterDate(i_date, f_date)
            geemap.ee_export_image_collection(ee_object=df2, out_dir=self.era5_scratch, scale=10000, region=area, crs='EPSG:4326', file_per_band=True)
        print('Download completed')
        
    def get_era5_land_meteo_data(self):
        if self.local_data is False:
            data_check = f'{self.working_dir}/ERA5/prep/pr.nc'
            if not os.path.exists(data_check):
                self._download_era5_land_data()
                variable_dirs = {
                    'pr': os.path.join(self.era5_scratch, '*total_precipitation_sum*.tif'),
                    'tasmax': os.path.join(self.era5_scratch, '*temperature_2m_max*.tif'),
                    'tasmin': os.path.join(self.era5_scratch, '*temperature_2m_min*.tif'),
                    'tmean': os.path.join(self.era5_scratch, '*temperature_2m.tif')
                }

                output_dir_map = {
                    'pr': self.prep_path,
                    'tasmax': self.tasmax_path,
                    'tasmin': self.tasmin_path,
                    'tmean': self.tmean_path
                }

                for var_name, tif_pattern in variable_dirs.items():
                    self.pattern = tif_pattern
                    output_dir = output_dir_map[var_name]
                    tif_files = sorted(glob.glob(tif_pattern))

                    # 🕒 Extract timestamps from filenames (e.g., '20210101.tif')
                    try:
                        timestamps = [
                            datetime.strptime(os.path.basename(f).split('.')[0], "%Y%m%d")
                            for f in tif_files
                        ]
                    except ValueError:
                        raise ValueError(f"Could not parse timestamps from filenames in: {tif_pattern}")
                    self.timestamps = timestamps
                    # 📚 Load and stack GeoTIFFs
                    data_list = []
                    for i, file in enumerate(tif_files):
                        da = rioxarray.open_rasterio(file, masked=True).squeeze()  # Remove band dim if present
                        da = da.expand_dims(dim={"time": [timestamps[i]]})         # ⬅️ Make time a proper dim
                        da = da.assign_coords(time=("time", [timestamps[i]]))      # ⬅️ Make it a coordinate too
                        data_list.append(da)

                    # 📈 Combine into a single xarray DataArray
                    data = xr.concat(data_list, dim="time")
                    data.name = var_name
                    data = data.rename({"y": "lat", "x": "lon"})

                    # 💾 Save to NetCDF
                    os.makedirs(output_dir, exist_ok=True)
                    nc_path = os.path.join(output_dir, f"{var_name}.nc")
                    data.to_netcdf(nc_path)
                    print(f"✅ Saved NetCDF for '{var_name}': {nc_path}")

            else:
                print(f"     - ERA5 Land daily data already exists in {self.working_dir}/era5_land; skipping download.")


            # 🔄 Load datasets for return
            prep_nc = xr.open_dataset(os.path.join(self.prep_path, "pr.nc"))
            tasmax_nc = xr.open_dataset(os.path.join(self.tasmax_path, "tasmax.nc"))
            tasmin_nc = xr.open_dataset(os.path.join(self.tasmin_path, "tasmin.nc"))
            tmean_nc = xr.open_dataset(os.path.join(self.tmean_path, "tmean.nc"))

        return prep_nc, tasmax_nc, tasmin_nc, tmean_nc

            

    def get_chirps_prep_meteo_data(self):
        if self.local_data is False:
            data_check = f'{self.working_dir}/CHIRPS/prep/pr.nc'
            if not os.path.exists(data_check):
                self._download_chirps_prep_data()
                variable_dirs = {
                    'pr': os.path.join(self.chirps_scratch, '*precipitation*.tif'),
                    'tasmax': os.path.join(self.era5_scratch, '*temperature_2m_max*.tif'),
                    'tasmin': os.path.join(self.era5_scratch, '*temperature_2m_min*.tif'),
                    'tmean': os.path.join(self.era5_scratch, '*temperature_2m*.tif')
                }

                output_dir_map = {
                    'pr': self.prep_path,
                    'tasmax': self.tasmax_path,
                    'tasmin': self.tasmin_path,
                    'tmean': self.tmean_path
                }

                for var_name, tif_pattern in variable_dirs.items():
                    self.pattern = tif_pattern
                    output_dir = output_dir_map[var_name]
                    tif_files = sorted(glob.glob(tif_pattern))

                    # 🕒 Extract timestamps from filenames (e.g., '20210101.tif')
                    try:
                        timestamps = [
                            datetime.strptime(os.path.basename(f).split('.')[0], "%Y%m%d")
                            for f in tif_files
                        ]
                    except ValueError:
                        raise ValueError(f"Could not parse timestamps from filenames in: {tif_pattern}")

                    # 📚 Load and stack GeoTIFFs
                    data_list = []
                    for i, file in enumerate(tif_files):
                        da = rioxarray.open_rasterio(file, masked=True).squeeze()  # Remove band dim if present
                        da = da.expand_dims(dim={"time": [timestamps[i]]})         # ⬅️ Make time a proper dim
                        da = da.assign_coords(time=("time", [timestamps[i]]))      # ⬅️ Make it a coordinate too
                        data_list.append(da)

                    # 📈 Combine into a single xarray DataArray
                    data = xr.concat(data_list, dim="time")
                    data.name = var_name
                    data = data.rename({"y": "lat", "x": "lon"})

                    # 💾 Save to NetCDF
                    os.makedirs(output_dir, exist_ok=True)
                    nc_path = os.path.join(output_dir, f"{var_name}.nc")
                    data.to_netcdf(nc_path)
                    print(f"✅ Saved NetCDF for '{var_name}': {nc_path}")

            else:
                print(f"     - CHIRPS daily data already exists in {self.working_dir}/CHIRPS; skipping download.")

            # 🔄 Load datasets for return
            prep_nc = xr.open_dataset(os.path.join(self.prep_path, "pr.nc"))
            tasmax_nc = xr.open_dataset(os.path.join(self.tasmax_path, "tasmax.nc"))
            tasmin_nc = xr.open_dataset(os.path.join(self.tasmin_path, "tasmin.nc"))
            tmean_nc = xr.open_dataset(os.path.join(self.tmean_path, "tmean.nc"))

        return prep_nc, tasmax_nc, tasmin_nc, tmean_nc

    def get_chelsa_meteo_data(self):
        if self.local_data is False:
            climate_variables = {
                'tasmax': 'tasmax',
                'tasmin': 'tasmin',
                'tas': 'tmean',
                'pr': 'prep'
            }
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self._download_chelsa_data, variable, folder): variable
                    for variable, folder in climate_variables.items()
                }
                
                for future in futures:
                    variable = futures[future]
                    try:
                        future.result()  # Raises exception if download fails
                        print(f"Download completed for {variable}")
                    except Exception as e:
                        print(f"An error occurred while downloading {variable}: {e}")

            tasmax_nc = self.uw.concat_nc(self.tasmax_path, '*tasmax*.nc')
            tasmin_nc = self.uw.concat_nc(self.tasmin_path, '*tasmin*.nc')   
            tmean_nc = self.uw.concat_nc(self.tmean_path, '*tas_*.nc')
            prep_nc = self.uw.concat_nc(self.prep_path, '*pr_*.nc')

        else:
            try:
                if not all([self.prep_path, self.tasmax_path, self.tasmin_path, self.tmean_path]):
                    raise ValueError("All paths to local NetCDF files must be provided if 'local_data' is True.")

                for path, variable in zip(
                    [self.prep_path, self.tasmax_path, self.tasmin_path, self.tmean_path],
                    ['local_prep_path', 'local_tasmax_path', 'local_tasmin_path', 'local_tmean_path']
                ):
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"The specified local data file for {variable} at '{path}' does not exist.")
                    if not path.endswith('.nc'):
                        raise ValueError(f"The file for {variable} at '{path}' is not a NetCDF file (.nc).")
                
                print("Local data paths validated. Proceeding with processing.")
            except ValueError as e:
                print(f"Configuration error: {e}")
            except FileNotFoundError as e:
                print(f"File error: {e}")
            except Exception as e:
                print(f"An unexpected error occurred while processing local data: {e}")

            # tasmax_nc = self.uw.align_rasters(self.tasmax_path, israster=False)
            # tasmin_nc = self.uw.align_rasters(self.tasmin_path, israster=False)
            # tmean_nc = self.uw.align_rasters(self.tmean_path, israster=False)
            # prep_nc = self.uw.align_rasters(self.prep_path, israster=False)
            
            tasmax_nc = xr.open_dataset(self.tasmax_path)
            tasmin_nc = xr.open_dataset(self.tasmin_path)
            tmean_nc = xr.open_dataset(self.tmean_path)
            prep_nc = xr.open_dataset(self.prep_path)

        return prep_nc, tasmax_nc, tasmin_nc, tmean_nc
    
    def export_urls_for_download_manager(self):
        climate_variables = ['tasmax', 'tasmin', 'tas', 'pr']
        all_urls = []

        for climate_variable in climate_variables:
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
            urls = [file['file_url'] for file in dataset['files']]
            #all_urls.extend(urls)

            with open(os.path.join(f'{self.working_dir}/', f"{climate_variable}_download_urls.txt"), "w") as f:
                f.write("\n".join(urls))
    
    def get_meteo_data(self):
        '''
        Get meteo data from the specified data source: 'CHELSA', 'ERA5' or 'CHIRPS'
        '''
        if self.data_source == 'CHELSA':
            prep_nc, tasmax_nc, tasmin_nc, tmean_nc = self.get_chelsa_meteo_data()
        elif self.data_source == 'ERA5':
            prep_nc, tasmax_nc, tasmin_nc, tmean_nc = self.get_era5_land_meteo_data()
        elif self.data_source == 'CHIRPS':
            prep_nc, tasmax_nc, tasmin_nc, tmean_nc= self.get_chirps_prep_meteo_data()
        return prep_nc, tasmax_nc, tasmin_nc, tmean_nc

        
        
                    


        

        
        
                    

