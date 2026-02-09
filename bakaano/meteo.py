"""Meteorological forcing download and preprocessing.

Role: Retrieve and prepare precipitation and temperature forcings.
"""

import os
import ee
import geemap
import glob
import numpy as np
import rioxarray
from isimip_client.client import ISIMIPClient
from bakaano.utils import Utils
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import xarray as xr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


class Meteo:
    def __init__(self, working_dir, study_area, start_date, end_date, local_data=False, data_source='CHELSA', local_prep_path=None, 
                 local_tasmax_path=None, local_tasmin_path=None, local_tmean_path=None):
        
        """
        Role: Download/prepare meteorological forcing for VegET.

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

            for path in [self.prep_path, self.tasmax_path, self.tasmin_path, self.tmean_path]:
                if not os.path.isfile(path) or not path.endswith(".nc"):
                    raise ValueError(f"File not found or not a NetCDF (.nc) file: {path}")
        else:
            if self.data_source == 'CHIRPS':

                self.tasmax_path = Path(f'{self.working_dir}/{self.data_source}/tasmax/')
                self.tasmin_path = Path(f'{self.working_dir}/{self.data_source}/tasmin/')
                self.tmean_path = Path(f'{self.working_dir}/{self.data_source}/tmean/')
                self.prep_path = Path(f'{self.working_dir}/{self.data_source}/prep/')
                self.era5_scratch = Path(f'{self.working_dir}/era5_scratch/')
                self.chirps_scratch = Path(f'{self.working_dir}/chirps_scratch/')
            else:
                self.tasmax_path = Path(f'{self.working_dir}/{self.data_source}/tasmax/')
                self.tasmin_path = Path(f'{self.working_dir}/{self.data_source}/tasmin/')
                self.tmean_path = Path(f'{self.working_dir}/{self.data_source}/tmean/')
                self.prep_path = Path(f'{self.working_dir}/{self.data_source}/prep/')
                self.era5_scratch = Path(f'{self.working_dir}/era5_scratch/')

    
    def check_missing_dates(self, variables=None, scratch_dir=None):
        """Check for missing daily ERA5-Land GeoTIFFs in scratch folder.

        Args:
            variables (list[str], optional): Variable names to check.
            scratch_dir (str | Path, optional): Directory containing raw daily
                GeoTIFFs for the requested variables.

        Returns:
            list[str]: Missing dates in YYYY-MM-DD format.
        """
        import os
        from collections import defaultdict
        from datetime import datetime, timedelta
    
        # variables = [
        #     'total_precipitation_sum',
        #     'temperature_2m_min',
        #     'temperature_2m_max',
        #     'temperature_2m'
        # ]
    
        # Generate list of all expected dates
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        expected_dates = [(start + timedelta(days=i)).strftime("%Y-%m-%d")
                          for i in range((end - start).days + 1)]
    
        if scratch_dir is None:
            scratch_dir = self.era5_scratch

        files = os.listdir(scratch_dir) if os.path.exists(scratch_dir) else []
        var_dates = defaultdict(set)
    
        for fname in files:
            if not fname.endswith(".tif"):
                continue
            parts = fname.split('.')
            if len(parts) >= 3:
                date_part, var_part = parts[0], parts[1]
                if var_part in variables:
                    try:
                        # Convert YYYYMMDD → YYYY-MM-DD
                        date_obj = datetime.strptime(date_part, "%Y%m%d")
                        date_str = date_obj.strftime("%Y-%m-%d")
                        var_dates[var_part].add(date_str)
                    except Exception as e:
                        print(f"⚠️ Skipping malformed filename: {fname} — {e}")
    
        if not var_dates:
            print("❌ No valid ERA5-Land files detected — check filenames and variable matching.")
            return []
    
        # Choose variable with fewest valid days
        min_var = min(var_dates, key=lambda v: len(var_dates[v]))
        observed_dates = var_dates[min_var]
    
        # Identify missing dates
        missing = [d for d in expected_dates if d not in observed_dates]
    
        print(f"🔍 Checked using variable with shortest record: **{min_var}**")
        print(f"📊 {len(observed_dates)} out of {len(expected_dates)} dates present for {min_var}")
        if missing:
            print(f"⚠️ {len(missing)} missing dates (e.g.): {missing[:5]}...")
        else:
            print("✅ All expected dates are present.")
    
        return missing

    def _download_chelsa_data(self, climate_variable, output_folder):
        """Download CHELSA daily climate data using ISIMIP client.

        Args:
            climate_variable (str): Variable id for ISIMIP (e.g., "tasmax", "pr").
            output_folder (str): Output folder name under working_dir.

        Returns:
            None. Downloads and extracts NetCDF files to disk.
        """

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
        """Download ERA5-Land daily data via Earth Engine and store as GeoTIFFs.

        Returns:
            None. Writes GeoTIFFs to the scratch folder.
        """
        ee.Authenticate()
        ee.Initialize()
       
        # Parse start and end dates
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        end_exclusive = (end + timedelta(days=1)).strftime("%Y-%m-%d")

        area = ee.Geometry.BBox(self.uw.minx, self.uw.miny, self.uw.maxx, self.uw.maxy)
    
        era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        existing_era5_tifs = glob.glob(os.path.join(self.era5_scratch, "*.tif"))
        if existing_era5_tifs:
            print("Existing ERA5 scratch files detected; skipping bulk download and checking for missing dates only.")
        else:
            # Step 1: Attempt bulk download by year using image collection
            start_year = start.year
            end_year = end.year
            for year in range(start_year, end_year + 1):
                i_date = self.start_date if year == start_year else f"{year}-01-01"
                f_date = f"{year + 1}-01-01" if year < end_year else end_exclusive
        
                df = era5.select(
                    'total_precipitation_sum',
                    'temperature_2m_min',
                    'temperature_2m_max',
                    'temperature_2m'
                ).filterDate(i_date, f_date)
        
                geemap.ee_export_image_collection(
                    ee_object=df,
                    out_dir=self.era5_scratch,
                    scale=10000,
                    region=area,
                    crs='EPSG:4326',
                    file_per_band=True
                )
            print("Bulk download attempt completed. Verifying files...")
    

        variables = [
            'total_precipitation_sum',
            'temperature_2m_min',
            'temperature_2m_max',
            'temperature_2m'
        ]
        missing_dates = self.check_missing_dates(variables, scratch_dir=self.era5_scratch)
    
        print(f"{len(missing_dates)} missing dates detected. Re-downloading...")
    
        # Step 3: Re-download only missing dates individually
        for date_str in missing_dates:
            try:
                next_day = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                img = era5.filterDate(date_str, next_day).select(
                    ['temperature_2m_min', 'temperature_2m_max', 'temperature_2m', 'total_precipitation_sum']
                ).first()
        
                if img:
                    date_raw = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")
        
                    # Loop over each band and export separately
                    band_names = ['temperature_2m_min', 'temperature_2m_max', 'temperature_2m', 'total_precipitation_sum']
                    for band in band_names:
                        single_band_img = img.select(band)
                        filename = os.path.join(self.era5_scratch, f"{date_raw}.{band}.tif")
        
                        geemap.ee_export_image(
                            ee_object=single_band_img,
                            filename=filename,
                            scale=10000,
                            region=area,
                            crs='EPSG:4326'
                        )
                        print(f"Downloaded {band} for {date_str}")
            except Exception as e:
                print(f"Failed to download {date_str}: {e}")
    
        print("ERA5 download process completed.")

    

    def _download_chirps_prep_data(self):
        """Download CHIRPS precipitation + ERA5 temperatures via Earth Engine.

        Returns:
            None. Writes GeoTIFFs to scratch folders.
        """
        ee.Authenticate()
        ee.Initialize()
    
        chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        area = ee.Geometry.BBox(self.uw.minx, self.uw.miny, self.uw.maxx, self.uw.maxy)
    
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        end_exclusive = (end + timedelta(days=1)).strftime("%Y-%m-%d")
        expected_dates = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end - start).days + 1)]
    
        # Step 1: Bulk download only when scratch folders are empty.
        start_year = start.year
        end_year = end.year
        need_bulk_chirps = not glob.glob(os.path.join(self.chirps_scratch, "*.tif"))
        need_bulk_era5 = not glob.glob(os.path.join(self.era5_scratch, "*.tif"))

        if need_bulk_chirps or need_bulk_era5:
            for year in range(start_year, end_year + 1):
                i_date = self.start_date if year == start_year else f"{year}-01-01"
                f_date = f"{year + 1}-01-01" if year < end_year else end_exclusive

                if need_bulk_chirps:
                    df_chirps = chirps.select('precipitation').filterDate(i_date, f_date)
                    geemap.ee_export_image_collection(
                        ee_object=df_chirps,
                        out_dir=self.chirps_scratch,
                        scale=5000,
                        region=area,
                        crs='EPSG:4326',
                        file_per_band=True
                    )

                if need_bulk_era5:
                    df_era5 = era5.select(
                        'temperature_2m_min',
                        'temperature_2m_max',
                        'temperature_2m'
                    ).filterDate(i_date, f_date)
                    geemap.ee_export_image_collection(
                        ee_object=df_era5,
                        out_dir=self.era5_scratch,
                        scale=10000,
                        region=area,
                        crs='EPSG:4326',
                        file_per_band=True
                    )
            print("Bulk CHIRPS/ERA5 download complete. Checking for missing files...")
        else:
            print("Existing CHIRPS/ERA5 scratch files detected; skipping bulk download and checking for missing dates only.")
    
        
        variables = [
            'precipitation'
        ]
        missing_chirps = self.check_missing_dates(variables, scratch_dir=self.chirps_scratch)
        print(f"Missing CHIRPS dates: {len(missing_chirps)}")
    
        for date_str in missing_chirps:
            try:
                next_day = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                img = chirps.filterDate(date_str, next_day).select('precipitation').first()
                if img:
                    date_raw = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")
                    filename = os.path.join(self.chirps_scratch, f"{date_raw}.precipitation.tif")
                    geemap.ee_export_image(
                        ee_object=img,
                        filename=filename,
                        scale=5000,
                        region=area,
                        crs='EPSG:4326'
                    )
                    print(f"Downloaded CHIRPS for {date_str}")
            except Exception as e:
                print(f"Failed CHIRPS download for {date_str}: {e}")
    
        variables = [
            'temperature_2m_min',
            'temperature_2m_max',
            'temperature_2m'
        ]
        missing_era5 = self.check_missing_dates(variables, scratch_dir=self.era5_scratch)
        print(f"Missing ERA5 dates: {len(missing_era5)}")
    
        for date_str in missing_era5:
            try:
                next_day = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                img = era5.filterDate(date_str, next_day).select(
                    ['temperature_2m_min', 'temperature_2m_max', 'temperature_2m']
                ).first()
        
                if img:
                    date_raw = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")
        
                    # Loop over each band and export separately
                    band_names = ['temperature_2m_min', 'temperature_2m_max', 'temperature_2m']
                    for band in band_names:
                        single_band_img = img.select(band)
                        filename = os.path.join(self.era5_scratch, f"{date_raw}.{band}.tif")
        
                        geemap.ee_export_image(
                            ee_object=single_band_img,
                            filename=filename,
                            scale=10000,
                            region=area,
                            crs='EPSG:4326'
                        )
                        print(f"Downloaded {band} for {date_str}")
            except Exception as e:
                print(f"Failed ERA5 download for {date_str}: {e}")
    
        print("CHIRPS and ERA5 download (with checks) completed.")

    def _build_nc_from_scratch_tifs(self, variable_dirs, output_dir_map):
        """Create missing NetCDF outputs from existing scratch GeoTIFFs.

        Args:
            variable_dirs (dict[str, str]): Mapping variable name -> GeoTIFF glob.
            output_dir_map (dict[str, str | Path]): Mapping variable name -> output dir.

        Returns:
            list[str]: Variable names still missing required GeoTIFFs.
        """
        missing_vars = []
        for var_name, tif_pattern in variable_dirs.items():
            output_dir = output_dir_map[var_name]
            os.makedirs(output_dir, exist_ok=True)
            nc_path = os.path.join(output_dir, f"{var_name}.nc")

            # Skip variables that already have a NetCDF output.
            if os.path.exists(nc_path):
                continue

            tif_files = sorted(glob.glob(tif_pattern))
            if not tif_files:
                missing_vars.append(var_name)
                continue

            missing_dates = self._missing_dates_for_tif_pattern(tif_files)
            if missing_dates:
                print(
                    f"     - Scratch files for '{var_name}' are incomplete "
                    f"({len(missing_dates)} missing days); download required."
                )
                missing_vars.append(var_name)
                continue

            try:
                timestamps = [
                    datetime.strptime(os.path.basename(f).split('.')[0], "%Y%m%d")
                    for f in tif_files
                ]
            except ValueError:
                raise ValueError(f"Could not parse timestamps from filenames in: {tif_pattern}")

            data_list = []
            for i, file in enumerate(tif_files):
                da = rioxarray.open_rasterio(file, masked=True).squeeze()
                da = da.expand_dims(dim={"time": [timestamps[i]]})
                da = da.assign_coords(time=("time", [timestamps[i]]))
                data_list.append(da)

            data = xr.concat(data_list, dim="time")
            data.name = var_name
            data = data.rename({"y": "lat", "x": "lon"})
            data.to_netcdf(nc_path)
            print(f"✅ Saved NetCDF for '{var_name}': {nc_path}")

        return missing_vars

    def _missing_dates_for_tif_pattern(self, tif_files):
        """Return missing dates (YYYY-MM-DD) versus requested daily range."""
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        expected_dates = {
            (start + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range((end - start).days + 1)
        }

        observed_dates = set()
        for file in tif_files:
            try:
                date_str = os.path.basename(file).split('.')[0]
                date_obj = datetime.strptime(date_str, "%Y%m%d")
                observed_dates.add(date_obj.strftime("%Y-%m-%d"))
            except ValueError:
                continue

        return sorted(expected_dates - observed_dates)
        
    def get_era5_land_meteo_data(self):
        """Download/process ERA5-Land daily data and return datasets.

        Returns:
            tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
                prep_nc, tasmax_nc, tasmin_nc, tmean_nc.
        """
        if self.local_data is False:
            variable_dirs = {
                'pr': os.path.join(self.era5_scratch, '*total_precipitation_sum*.tif'),
                'tasmax': os.path.join(self.era5_scratch, '*temperature_2m_max*.tif'),
                'tasmin': os.path.join(self.era5_scratch, '*temperature_2m_min*.tif'),
                'tas': os.path.join(self.era5_scratch, '*temperature_2m.tif')
            }

            output_dir_map = {
                'pr': self.prep_path,
                'tasmax': self.tasmax_path,
                'tasmin': self.tasmin_path,
                'tas': self.tmean_path
            }

            missing_vars = self._build_nc_from_scratch_tifs(variable_dirs, output_dir_map)
            if missing_vars:
                print(
                    "     - Missing scratch GeoTIFFs for variables "
                    f"{missing_vars}; downloading required climate data."
                )
                self._download_era5_land_data()
                missing_vars = self._build_nc_from_scratch_tifs(variable_dirs, output_dir_map)
                if missing_vars:
                    raise FileNotFoundError(
                        "Could not build ERA5 NetCDF outputs. Missing scratch GeoTIFFs for: "
                        f"{missing_vars}"
                    )
            else:
                print("     - ERA5 NetCDF outputs already present (or rebuilt from existing scratch files).")


            # 🔄 Load datasets for return
            prep_nc = xr.open_dataset(os.path.join(self.prep_path, "pr.nc"))
            tasmax_nc = xr.open_dataset(os.path.join(self.tasmax_path, "tasmax.nc"))
            tasmin_nc = xr.open_dataset(os.path.join(self.tasmin_path, "tasmin.nc"))
            tmean_nc = xr.open_dataset(os.path.join(self.tmean_path, "tas.nc"))

        return prep_nc, tasmax_nc, tasmin_nc, tmean_nc

            

    def get_chirps_prep_meteo_data(self):
        """Download/process CHIRPS precipitation and ERA5 temperatures.

        Returns:
            tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
                prep_nc, tasmax_nc, tasmin_nc, tmean_nc.
        """
        if self.local_data is False:
            variable_dirs = {
                'pr': os.path.join(self.chirps_scratch, '*precipitation*.tif'),
                'tasmax': os.path.join(self.era5_scratch, '*temperature_2m_max*.tif'),
                'tasmin': os.path.join(self.era5_scratch, '*temperature_2m_min*.tif'),
                'tas': os.path.join(self.era5_scratch, '*temperature_2m.tif')
            }

            output_dir_map = {
                'pr': self.prep_path,
                'tasmax': self.tasmax_path,
                'tasmin': self.tasmin_path,
                'tas': self.tmean_path
            }

            missing_vars = self._build_nc_from_scratch_tifs(variable_dirs, output_dir_map)
            if missing_vars:
                print(
                    "     - Missing scratch GeoTIFFs for variables "
                    f"{missing_vars}; downloading required climate data."
                )
                self._download_chirps_prep_data()
                missing_vars = self._build_nc_from_scratch_tifs(variable_dirs, output_dir_map)
                if missing_vars:
                    raise FileNotFoundError(
                        "Could not build CHIRPS/ERA5 NetCDF outputs. Missing scratch GeoTIFFs for: "
                        f"{missing_vars}"
                    )
            else:
                print("     - CHIRPS/ERA5 NetCDF outputs already present (or rebuilt from existing scratch files).")

            # 🔄 Load datasets for return
            prep_nc = xr.open_dataset(os.path.join(self.prep_path, "pr.nc"))
            tasmax_nc = xr.open_dataset(os.path.join(self.tasmax_path, "tasmax.nc"))
            tasmin_nc = xr.open_dataset(os.path.join(self.tasmin_path, "tasmin.nc"))
            tmean_nc = xr.open_dataset(os.path.join(self.tmean_path, "tas.nc"))

        return prep_nc, tasmax_nc, tasmin_nc, tmean_nc

    def get_chelsa_meteo_data(self):
        """Download/process CHELSA daily data (or load local NetCDFs).

        Returns:
            tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
                prep_nc, tasmax_nc, tasmin_nc, tmean_nc.
        """
        if self.local_data is False:
            climate_variables = {
                'tasmax': 'CHELSA/tasmax',
                'tasmin': 'CHELSA/tasmin',
                'tas': 'CHELSA/tmean',
                'pr': 'CHELSA/prep'
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
        """Export CHELSA download URLs to text files in the working directory."""
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
        """Get meteo data for the selected source.

        Returns:
            tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
                prep_nc, tasmax_nc, tasmin_nc, tmean_nc.
        """
        if self.data_source == 'CHELSA':
            prep_nc, tasmax_nc, tasmin_nc, tmean_nc = self.get_chelsa_meteo_data()
        elif self.data_source == 'ERA5':
            prep_nc, tasmax_nc, tasmin_nc, tmean_nc = self.get_era5_land_meteo_data()
        elif self.data_source == 'CHIRPS':
            prep_nc, tasmax_nc, tasmin_nc, tmean_nc= self.get_chirps_prep_meteo_data()
        return prep_nc, tasmax_nc, tasmin_nc, tmean_nc
    
    def plot_meteo(self, variable, date):
        """Plot a meteorological field for a given date.

        Args:
            variable (str): One of ``"precip"``, ``"tasmax"``, ``"tasmin"``, ``"tmean"``.
            date (str or datetime): Date to plot (nearest available).

        Returns:
            None. Displays a matplotlib plot.
        """
        prep_nc, tasmax_nc, tasmin_nc, tmean_nc = self.get_meteo_data()
        
        if variable=='precip':
            data = prep_nc['pr'].sel(time=date, method='nearest')
            plt.title(f'Precipitation on {date}')
            plt.imshow(data, interpolation='gaussian')
            plt.colorbar()
        elif variable=='tasmax':
            data = tasmax_nc['tasmax'].sel(time=date, method='nearest') - 273.15
            plt.title(f'Maximum temperature on {date}')
            plt.imshow(data, interpolation='gaussian', vmin=0)
            plt.colorbar()
        elif variable=='tasmin':
            data = tasmin_nc['tasmin'].sel(time=date, method='nearest') - 273.15
            plt.title(f'Minimum temperature on {date}')
            plt.imshow(data, interpolation='gaussian', vmin=0)
            plt.colorbar()
        elif variable=='tmean':
            data = tmean_nc['tas'].sel(time=date, method='nearest') - 273.15
            plt.title(f'Mean temperature on {date}')
            plt.imshow(data, interpolation='gaussian', vmin=0)
            plt.colorbar()
        else:
            raise ValueError("Invalid variable. Select valid variable")


        
        
                    


        

        
        
                    
