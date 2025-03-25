
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
    def __init__(self, working_dir, study_area, local_data=False, data_source='chelsa', local_prep_path=None, 
                 local_tasmax_path=None, local_tasmin_path=None, local_tmean_path=None):
        """
        Initialize a Meteo object.

        Args:
            working_dir (str): The working directory where files and outputs will be stored.
            study_area (str): The path to the shapefile of the river basin or watershed.
            local_data (bool, optional): Flag indicating whether to use local data instead of downloading new data. Defaults to False.
            local_prep_path (str, optional): Path to the local NetCDF file containing daily rainfall data. Required if `local_data` is True.
            local_tasmax_path (str, optional): Path to the local NetCDF file containing daily maximum temperature data (in Kelvin). Required if `local_data` is True.
            local_tasmin_path (str, optional): Path to the local NetCDF file containing daily minimum temperature data (in Kelvin). Required if `local_data` is True.
            local_tmean_path (str, optional): Path to the local NetCDF file containing daily mean temperature data (in Kelvin). Required if `local_data` is True.
        Returns:
            Dataarrays clipped to the study area extent, reprojected to the correct CRS, resampled to match DEM resolution
        """
        self.study_area = study_area
        self.working_dir = working_dir
        os.makedirs(f'{self.working_dir}/tasmax', exist_ok=True)
        os.makedirs(f'{self.working_dir}/tasmin', exist_ok=True)
        os.makedirs(f'{self.working_dir}/prep', exist_ok=True)
        os.makedirs(f'{self.working_dir}/tmean', exist_ok=True)
        self.uw = Utils(self.working_dir, self.study_area)
        self.uw.get_bbox('EPSG:4326')
        self.client = ISIMIPClient()
        self.local_data = local_data
        self.data_source = data_source

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
        data_check = f'{self.working_dir}/era5_land/daily_prep.nc'
        if not os.path.exists(data_check):
            ee.Authenticate()
            ee.Initialize()

            era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")

            i_date = str(1981)+'-01-01'
            f_date = str(2021)+'-01-01'
            df = era5.select('total_precipitation_sum', 'temperature_2m_min', 'temperature_2m_max', 'temperature_2m').filterDate(i_date, f_date)

            area = ee.Geometry.BBox(self.uw.minx, self.uw.miny, self.uw.maxx, self.uw.maxy) 
            geemap.ee_export_image_collection(ee_object=df, out_dir=self.era5_scratch, scale=1000, region=area, crs='EPSG:4326', file_per_band=True) 
            print('Download completed')
        else:
            print(f"     - ERA5 Land daily data already exists in {self.working_dir}/era5_land; skipping download.")

    def _download_chirps_prep_data(self):
        data_check = f'{self.working_dir}/chirps/daily_prep.nc'
        if not os.path.exists(data_check):
            ee.Authenticate()
            ee.Initialize()

            chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")

            i_date = str(1981)+'-01-01'
            f_date = str(2021)+'-01-01'

            df = chirps.select('precipitation').filterDate(i_date, f_date)
            area = ee.Geometry.BBox(self.uw.minx, self.uw.miny, self.uw.maxx, self.uw.maxy) 
            geemap.ee_export_image_collection(ee_object=df, out_dir=self.chirps_scratch, scale=1000, region=area, crs='EPSG:4326', file_per_band=True) 
            
            era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            df2 = era5.select('temperature_2m_min', 'temperature_2m_max', 'temperature_2m').filterDate(i_date, f_date)
            geemap.ee_export_image_collection(ee_object=df2, out_dir=self.era5_scratch, scale=1000, region=area, crs='EPSG:4326', file_per_band=True)
            print('Download completed')
        else:
            print(f"     - CHIRPS rainfall and ERA5 Land temperature daily data already exists in {self.working_dir}/era5_land; skipping download.")

    def get_era5_land_meteo_data(self):
        if self.local_data is False:
            self._download_era5_land_data()
            variable_dirs = {
                'pr': f'{self.working_dir}/{self.era5_scratch}/*total_precipitation_sum*.tif',
                'tasmax': f'{self.working_dir}/{self.era5_scratch}/*temperature_2m_max*.tif',
                'tasmin': f'{self.working_dir}/{self.era5_scratch}/*temperature_2m_min*.tif',
                'tmean': f'{self.working_dir}/{self.era5_scratch}/*temperature_2m*.tif'
            }

            output_dir_list = [self.prep_path, self.tasmax_path, self.tasmin_path, self.tmean_path]
            for (var_name, tif_dir), output_dir in zip(variable_dirs.items(), output_dir_list):
                tif_files = sorted(glob.glob(tif_dir))

                # 🕒 Extract timestamps from filenames (e.g., '20210101.tif')
                try:
                    timestamps = [
                        datetime.strptime(os.path.basename(f).split('.')[0], "%Y%m%d")
                        for f in tif_files
                    ]
                except ValueError:
                    raise ValueError(f"Could not parse timestamps from filenames in: {tif_dir}")

                # 📚 Load and stack GeoTIFFs
                data_list = []
                for i, file in enumerate(tif_files):
                    da = rioxarray.open_rasterio(file, masked=True).squeeze()  # (band, y, x) → (y, x)
                    da = da.expand_dims(time=[timestamps[i]])            # Add time dimension
                    data_list.append(da)

                # 📈 Combine into a single xarray DataArray
                data = xr.concat(data_list, dim="time")
                data.name = var_name
                data = data.rename({"y": "lat", "x": "lon"})

                # 💾 Save to NetCDF
                
                data.to_netcdf(f'{output_dir}/{var_name}.nc')
                print(f"✅ Saved NetCDF for '{var_name}': {output_dir}")

            tasmax_nc = xr.open_dataset(f'{output_dir}/tasmax.nc')
            tasmin_nc = xr.open_dataset(f'{output_dir}/tasmin.nc')
            tmean_nc = xr.open_dataset(f'{output_dir}/tmean.nc')
            prep_nc = xr.open_dataset(f'{output_dir}/pr.nc')

        return prep_nc, tasmax_nc, tasmin_nc, tmean_nc 

            

    def get_chirps_prep_meteo_data(self):
        if self.local_data is False:
            self._download_chirps_prep_data()
            variable_dirs = {
                'pr': f'{self.working_dir}/{self.chirps_scratch}/*precipitation*.tif',
                'tasmax': f'{self.working_dir}/{self.era5_scratch}/*temperature_2m_max*.tif',
                'tasmin': f'{self.working_dir}/{self.era5_scratch}/*temperature_2m_min*.tif',
                'tmean': f'{self.working_dir}/{self.era5_scratch}/*temperature_2m*.tif'
            }

            output_dir_list = [self.prep_path, self.tasmax_path, self.tasmin_path, self.tmean_path]
            for (var_name, tif_dir), output_dir in zip(variable_dirs.items(), output_dir_list):
                tif_files = sorted(glob.glob(tif_dir))

                # 🕒 Extract timestamps from filenames (e.g., '20210101.tif')
                try:
                    timestamps = [
                        datetime.strptime(os.path.basename(f).split('.')[0], "%Y%m%d")
                        for f in tif_files
                    ]
                except ValueError:
                    raise ValueError(f"Could not parse timestamps from filenames in: {tif_dir}")

                # 📚 Load and stack GeoTIFFs
                data_list = []
                for i, file in enumerate(tif_files):
                    da = rioxarray.open_rasterio(file, masked=True).squeeze()  # (band, y, x) → (y, x)
                    da = da.expand_dims(time=[timestamps[i]])            # Add time dimension
                    data_list.append(da)

                # 📈 Combine into a single xarray DataArray
                data = xr.concat(data_list, dim="time")
                data.name = var_name
                data = data.rename({"y": "lat", "x": "lon"})

                # 💾 Save to NetCDF
                
                data.to_netcdf(f'{output_dir}/{var_name}.nc')
                print(f"✅ Saved NetCDF for '{var_name}': {output_dir}")

            tasmax_nc = xr.open_dataset(f'{output_dir}/tasmax.nc')
            tasmin_nc = xr.open_dataset(f'{output_dir}/tasmin.nc')
            tmean_nc = xr.open_dataset(f'{output_dir}/tmean.nc')
            prep_nc = xr.open_dataset(f'{output_dir}/pr.nc')

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

        

            # tasmax_nc = self.uw.align_rasters(tasmax_nc, israster=False)
            # tasmin_nc = self.uw.align_rasters(tasmin_nc, israster=False)
            # tmean_nc = self.uw.align_rasters(tmean_nc, israster=False)
            # prep_nc = self.uw.align_rasters(prep_nc, israster=False)
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
            self.get_chelsa_meteo_data()
        elif self.data_source == 'ERA5':
            self.get_era5_land_meteo_data()
        elif self.data_source == 'CHIRPS':
            self.get_chirps_prep_meteo_data()

        
        
                    


        

        
        
                    

