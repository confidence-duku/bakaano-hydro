import ee
import geemap
import os
import glob
import numpy as np
import rasterio
import rioxarray
import xarray as xr
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
from deepstrmm.utils import Utils
from rasterio.enums import Resampling
from scipy.interpolate import interp1d

class NDVI:
    def __init__(self, working_dir, study_area):
        self.study_area = study_area
        self.working_dir = working_dir
        os.makedirs(f'{self.working_dir}/ndvi', exist_ok=True)
        self.uw = Utils(self.working_dir, self.study_area)
        self.uw.get_bbox('EPSG:4326')
        self.ndvi_folder = f'{self.working_dir}/ndvi'

    def download_ndvi(self):

        #ee.Authenticate()
        ee.Initialize()

        ndvi = ee.ImageCollection("MODIS/061/MOD13A2")

        i_date = str(2001)+'-01-01'
        f_date = str(2021)+'-01-01'
        df = ndvi.select('NDVI').filterDate(i_date, f_date)

        area = ee.Geometry.BBox(self.uw.minx, self.uw.miny, self.uw.maxx, self.uw.maxy) 
        out_path = f'{self.working_dir}/ndvi'
        geemap.ee_export_image_collection(ee_object=df, out_dir=out_path, scale=1000, region=area, crs='EPSG:4326', file_per_band=True) 
        print('Download completed')

    def generate_intervals(self, year):
        """
        Generate 16-day intervals for a given year.
        
        :param year: Year to generate intervals for.
        :return: List of 16-day interval start dates.
        """
        intervals = []
        start_date = datetime(year, 1, 1)
        while start_date.year == year:
            intervals.append(start_date)
            start_date += timedelta(days=16)
        return intervals

    def group_files_by_intervals(self):
        """
        Group NDVI files by their 16-day intervals.
        
        :return: Dictionary of interval start dates and associated file lists.
        """
        ndvi_files = glob.glob(os.path.join(self.ndvi_folder, '*NDVI.tif'))
        base_year = 2000  # A leap year to handle February 29
        base_intervals = self.generate_intervals(base_year)  # Generate intervals for one year
        
        groups = defaultdict(list)
        for file in ndvi_files:
            filename = os.path.basename(file)
            date_str = filename.split('.')[0].replace('_', '-')  # Convert underscores to dashes
            file_date = datetime.strptime(date_str, '%Y-%m-%d')

            # Normalize the date to the base year
            normalized_date = datetime(base_year, file_date.month, file_date.day)

            # Match the normalized date to an interval
            for interval in base_intervals:
                if interval <= normalized_date < interval + timedelta(days=16):
                    interval_key = interval.strftime('%m-%d')  # Use MM-DD as the key
                    groups[interval_key].append(file)
                    break

        return groups

    def calculate_median_raster(self, file_list, output_path):
        """
        Calculate the median raster from a list of NDVI files and save as a TIF.
        
        :param file_list: List of file paths to NDVI rasters.
        :param output_path: Path to save the output TIF.
        """
        # Open the first file to get metadata
        with rasterio.open(file_list[0]) as src:
            meta = src.meta
            meta.update(dtype=rasterio.float32, count=1)

        # Read all rasters into a numpy array
        rasters = []
        for file in file_list:
            with rasterio.open(file) as src:
                rasters.append(src.read(1))  # Read the first band

        # Stack and calculate the median
        rasters_stack = np.stack(rasters)
        median_raster = np.mean(rasters_stack, axis=0)

        # Save the median raster to a new file
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(median_raster.astype(rasterio.float32), 1)

    def interpolate_daily_ndvi(self, medians, interval_dates):
        """
        Linearly interpolate daily NDVI from 16-day median NDVI values.
        :param medians: List of 16-day median NDVI arrays.
        :param interval_dates: List of corresponding interval start dates as day-of-year.
        :return: Dictionary of daily NDVI arrays for each day of the year.
        """
        daily_ndvi = {}
        date_range = [datetime(2000, 1, 1) + timedelta(days=i) for i in range(366)]  # Leap year for 366 days
        daily_doy = np.array([d.timetuple().tm_yday for d in date_range])  # Day of year for each date
    
        # Convert medians to a single 3D NumPy array (time, rows, cols)
        medians_array = np.stack(medians, axis=0)  # Shape: (num_intervals, rows, cols)
    
        # Reshape medians for interpolation
        num_intervals, rows, cols = medians_array.shape
        medians_flat = medians_array.reshape(num_intervals, -1)  # Shape: (num_intervals, pixels)
        
        # Perform interpolation for all pixels simultaneously
        interpolator = interp1d(interval_dates, medians_flat, kind='linear', bounds_error=False, fill_value="extrapolate", axis=0)
        interpolated_values = interpolator(daily_doy)  # Shape: (num_days, pixels)
    
        # Reshape back to (days, rows, cols)
        daily_ndvi_array = interpolated_values.reshape(len(daily_doy), rows, cols)
    
        # Create dictionary with day-of-year keys
        for d, doy in enumerate(daily_doy):
            daily_ndvi[doy] = daily_ndvi_array[d]
    
        return daily_ndvi
    
    def save_daily_ndvi(self, daily_ndvi, template_file):
        """
        Save daily NDVI arrays as GeoTIFF files.
        :param daily_ndvi: Dictionary of daily NDVI arrays.
        :param template_file: A template file to copy spatial metadata from.
        """
        with rasterio.open(template_file) as src:
            meta = src.meta.copy()

        for doy, ndvi_array in daily_ndvi.items():
            output_path = os.path.join(self.output_folder, f"day_{doy:03d}_ndvi.tif")
            meta.update({"dtype": "float32", "count": 1})
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(ndvi_array.astype("float32"), 1)

    def preprocess_ndvi(self):
        """
        Main process to compute the daily NDVI climatology.
        """
        groups = self.group_files_by_intervals()
        interval_dates = [datetime.strptime(k, '%m-%d').timetuple().tm_yday for k in groups.keys()]

        for interval_start, file_list in groups.items():
            print(f'Processing {interval_start} with {len(file_list)} files...')
            output_file = os.path.join(self.ndvi_folder, f'{interval_start}_median_ndvi.tif')
            self.calculate_median_raster(file_list, output_file)

        medians = []
        medians_list = glob.glob(f'{self.working_dir}/ndvi/*median*.tif')
        for file in medians_list:
            median_da = rioxarray.open_rasterio(file)[0]  # Extract the first band as DataArray
            medians.append(median_da)

        print("Interpolating daily NDVI...")
        daily_ndvi = self.interpolate_daily_ndvi(medians, interval_dates)
        for doy, arr in daily_ndvi.items():
            daily_ndvi[doy] = xr.DataArray(
                arr,
                dims=("lat", "lon"),  # Assuming the interpolated array has y and x dimensions
                coords={"lat": medians[0].y, "lon": medians[0].x},  # Use coordinates from a median DataArray
                attrs={"day_of_year": doy},
            )


        pickle_file_path = f'{self.working_dir}/ndvi/daily_ndvi_climatology.pkl'
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(daily_ndvi, f)
        print("Process complete!")
        return daily_ndvi

    