
import os
from isimip_client.client import ISIMIPClient
from deepstrmm.utils import Utils
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class Meteo:
    def __init__(self, working_dir, study_area, local_data=False, local_prep_path=None, 
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

        if local_data is True:
            self.prep_path = local_prep_path
            self.tasmax_path = local_tasmax_path
            self.tasmin_path = local_tasmin_path
            self.tmean_path = local_tmean_path
        else:
            self.tasmax_path = Path(f'{self.working_dir}/tasmax/')
            self.tasmin_path = Path(f'{self.working_dir}/tasmin/')
            self.tmean_path = Path(f'{self.working_dir}/tmean/')
            self.prep_path = Path(f'{self.working_dir}/prep/')

    def _download_data(self, climate_variable, output_folder):
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

            tasmax_nc = self.uw.concat_nc(self.tasmax_path, '*tasmax*.nc')
            tasmin_nc = self.uw.concat_nc(self.tasmin_path, '*tasmin*.nc')   
            tmean_nc = self.uw.concat_nc(self.tmean_path, '*tas_*.nc')
            prep_nc = self.uw.concat_nc(self.prep_path, '*pr_*.nc')

            tasmax_nc = self.uw.align_rasters(tasmax_nc, israster=False)
            tasmin_nc = self.uw.align_rasters(tasmin_nc, israster=False)
            tmean_nc = self.uw.align_rasters(tmean_nc, israster=False)
            prep_nc = self.uw.align_rasters(prep_nc, israster=False)
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

            tasmax_nc = self.uw.align_rasters(self.tasmax_path, israster=False)
            tasmin_nc = self.uw.align_rasters(self.tasmin_path, israster=False)
            tmean_nc = self.uw.align_rasters(self.tmean_path, israster=False)
            prep_nc = self.uw.align_rasters(self.prep_path, israster=False)

        return prep_nc, tasmax_nc, tasmin_nc, tmean_nc

        
        
                    


        

        
        
                    

