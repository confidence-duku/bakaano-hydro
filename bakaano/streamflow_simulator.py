
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
import tensorflow_probability as tfp
from keras.utils import register_keras_serializable
import glob
from tcn import TCN
import pysheds.grid
import rasterio
import rioxarray
from rasterio.transform import rowcol
from keras.models import load_model # type: ignore
import pickle
import warnings
import geopandas as gpd
from scipy.spatial.distance import cdist
from shapely.geometry import Point
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

tfd = tfp.distributions  # TensorFlow Probability distributions
#=====================================================================================================================================


class PredictDataPreprocessor:
    def __init__(self, working_dir,  study_area, start_date, end_date, grdc_streamflow_nc_file=None):
        """
        Initialize directories, dates and relevant variables
        
        Parameters:
        -----------
        data_dir : Working directory 
            Data directory that contains sub-folders and data
    
        """
        self.study_area = study_area
        self.start_date = start_date
        self.end_date = end_date
        self.working_dir = working_dir
        self.times = pd.date_range(start_date, end_date)
        self.grdc_subset = self.load_observed_streamflow(grdc_streamflow_nc_file)
        self.station_ids = np.unique(self.grdc_subset.to_dataframe().index.get_level_values('id'))
        self.data_list = []
        self.catchment = []  
        
    def _extract_station_rowcol(self, lat, lon):
        """
        Extract the row and column indices for a given latitude and longitude
        from given raster file.

        Parameters
        ----------
        lat : float
            The latitude of the station.
        lon : float
            The longitude of the station.

        Returns
        -------
        row : int
            The row index corresponding to the given latitude and longitude.
        col : int
            The column index corresponding to the given latitude and longitude.

        """
        with rasterio.open(f'{self.working_dir}/elevation/dem_clipped.tif') as src:
            transform = src.transform
            row, col = rowcol(transform, lon, lat)
            return row, col
        
    def _snap_coordinates(self, lat, lon):
        """
        Snap the given latitude and longitude to the nearest river segment based on a river grid.

        Parameters
        ----------
        lat : float
            The latitude to be snapped.
        lon : float
            The longitude to be snapped.

        Returns
        -------
        snapped_lat : float
            The latitude of the nearest river segment.
        snapped_lon : float
            The longitude of the nearest river segment.
        """
        coordinate_to_snap=(lon, lat)
        with rasterio.open(f'{self.working_dir}/elevation/dem_clipped.tif') as src:
            transform = src.transform

            river_coords = []
            for py in range(self.river_grid.shape[0]):
                for px in range(self.river_grid.shape[1]):
                    if self.river_grid[py, px] == 1:
                        river_coords.append(transform * (px + 0.5, py + 0.5))  # Center of the grid cell with river segment

            # Convert river_coords to numpy array for distance calculation
            river_coords = np.array(river_coords)

            # Compute distances from coordinate_to_snap to each river cell
            distances = cdist([coordinate_to_snap], river_coords)

            # Find the index of the nearest river cell
            nearest_index = np.argmin(distances)

            # Get the coordinates of the nearest river cell
            snap_point = river_coords[nearest_index]
            return snap_point[1], snap_point[0]
        
        
    def load_observed_streamflow(self,  grdc_streamflow_nc_file):
        grdc = xr.open_dataset(grdc_streamflow_nc_file)
        stations_df = pd.DataFrame({
            'station_name': grdc['station_name'].values,
            'geo_x': grdc['geo_x'].values,
            'geo_y': grdc['geo_y'].values
        })
        stations_gdf = gpd.GeoDataFrame(
            stations_df, 
            geometry=gpd.points_from_xy(stations_df['geo_x'], stations_df['geo_y']),
            crs="EPSG:4326"  # Assuming WGS84 latitude/longitude; adjust if needed
        )

        # Load the shapefile to check overlap (assuming it has a geometry column)
        region_shape = gpd.read_file(self.study_area)

        # Perform spatial join to find stations within the region shape
        stations_in_region = gpd.sjoin(stations_gdf, region_shape, how='inner', predicate='intersects')

        # Extract the station names that overlap
        overlapping_station_names = stations_in_region['station_name'].unique()

        # Filter the GRDC dataset based on time and station names
        filtered_grdc = grdc.where(
            (grdc['time'] >= pd.to_datetime(self.start_date)) &
            (grdc['time'] <= pd.to_datetime(self.end_date)) &
            (grdc['station_name'].isin(overlapping_station_names)),
            drop=True
        )
        self.sim_station_names = np.unique(filtered_grdc['station_name'].values)
        return filtered_grdc
    
    def encode_lat_lon(self, latitude, longitude):
        # Encode latitude
        sin_latitude = np.sin(np.radians(latitude))
        cos_latitude = np.cos(np.radians(latitude))
        
        # Encode longitude
        sin_longitude = np.sin(np.radians(longitude))
        cos_longitude = np.cos(np.radians(longitude))
        
        return sin_latitude, cos_latitude, sin_longitude, cos_longitude
                          
    def get_data(self):

        count = 1
        
        slope = f'{self.working_dir}/elevation/slope_clipped.tif'
        dem_filepath = f'{self.working_dir}/elevation/dem_clipped.tif'
        tree_cover = f'{self.working_dir}/vcf/mean_tree_cover.tif'
        herb_cover = f'./{self.working_dir}/vcf/mean_herb_cover.tif'
        awc = f'{self.working_dir}/soil/clipped_AWCh3_M_sl6_1km_ll.tif'
        sat_pt = f'{self.working_dir}/soil/clipped_AWCtS_M_sl6_1km_ll.tif'
        
        latlng_ras = rioxarray.open_rasterio(dem_filepath)
        latlng_ras = latlng_ras.rio.write_crs(4326)
        lat = latlng_ras['y'].values
        lon = latlng_ras['x'].values
        
        grid = pysheds.grid.Grid.from_raster(dem_filepath)
        dem = grid.read_raster(dem_filepath)
        
        flooded_dem = grid.fill_depressions(dem)
        inflated_dem = grid.resolve_flats(flooded_dem)
        fdir = grid.flowdir(inflated_dem, routing='mfd')
        acc = grid.accumulation(fdir=fdir, routing='mfd')
        
        facc_thresh = np.nanmax(acc) * 0.0001
        self.river_grid = np.where(acc < facc_thresh, 0, 1)
        river_ras = xr.DataArray(data=self.river_grid, coords=[('lat', lat), ('lon', lon)])
        
        with rasterio.open(dem_filepath) as src:
            ref_meta = src.meta.copy()  # Copy the metadata exactly as is

        with rasterio.open(f'{self.working_dir}/catchment/river_grid.tif', 'w', **ref_meta) as dst:
            dst.write(river_ras.values, 1)  # Write data to the first band
        
        weight2 = grid.read_raster(slope)
        cum_slp = grid.accumulation(fdir=fdir, weights=weight2, routing='mfd')

        weight3 = grid.read_raster(tree_cover)
        cum_tree_cover = grid.accumulation(fdir=fdir, weights=weight3, routing='mfd')
        
        weight4 = grid.read_raster(herb_cover)
        cum_herb_cover = grid.accumulation(fdir=fdir, weights=weight4, routing='mfd')

        weight5 = grid.read_raster(awc)
        cum_awc = grid.accumulation(fdir=fdir, weights=weight5, routing='mfd')
        
        weight6 = grid.read_raster(sat_pt)
        cum_satpt = grid.accumulation(fdir=fdir, weights=weight6, routing='mfd')
        
        acc = xr.DataArray(data=acc, coords=[('lat', lat), ('lon', lon)])
        cum_slp = xr.DataArray(data=cum_slp, coords=[('lat', lat), ('lon', lon)])
        cum_tree_cover = xr.DataArray(data=cum_tree_cover, coords=[('lat', lat), ('lon', lon)])
        cum_herb_cover = xr.DataArray(data=cum_herb_cover, coords=[('lat', lat), ('lon', lon)])
        cum_awc = xr.DataArray(data=cum_awc, coords=[('lat', lat), ('lon', lon)])
        cum_satpt = xr.DataArray(data=cum_satpt, coords=[('lat', lat), ('lon', lon)])
 
        time_index = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        #combine or all yearly output from the runoff and routing module into a single list
        all_years_wfa = sorted(glob.glob(f'{self.working_dir}/runoff_output/*.pkl'))
        wfa_list = []
        for year in all_years_wfa:
            with open(year, 'rb') as f:
                this_arr = pickle.load(f)
            wfa_list = wfa_list + this_arr
        
        all_wfa = []
        for k in self.station_ids:
            station_discharge = self.grdc_subset['runoff_mean'].sel(id=k).to_dataframe(name='station_discharge')
                          
            station_x = np.nanmax(self.grdc_subset['geo_x'].sel(id=k).values)
            station_y = np.nanmax(self.grdc_subset['geo_y'].sel(id=k).values)
            snapped_y, snapped_x = self._snap_coordinates(station_y, station_x)
            self.x = snapped_x
            self.y = snapped_y
            
            acc_data = acc.sel(lat=snapped_y, lon=snapped_x, method='nearest')
            slp_data = cum_slp.sel(lat=snapped_y, lon=snapped_x, method='nearest')
            tree_cover_data = cum_tree_cover.sel(lat=snapped_y, lon=snapped_x, method='nearest').values
            herb_cover_data = cum_herb_cover.sel(lat=snapped_y, lon=snapped_x, method='nearest').values
            awc_data = cum_awc.sel(lat=snapped_y, lon=snapped_x, method='nearest').values
            satpt_data = cum_satpt.sel(lat=snapped_y, lon=snapped_x, method='nearest').values
            acc_data = acc_data.values
            slp_data = slp_data.values
                
            row, col = self._extract_station_rowcol(snapped_y, snapped_x)
        
            station_wfa = []
            for arr in wfa_list:
                arr = arr.tocsr()
                station_wfa.append(arr[int(row), int(col)])
            full_wfa_data = pd.DataFrame(station_wfa, columns=['mfd_wfa'])
            full_wfa_data.set_index(time_index, inplace=True)
            full_wfa_data.index.name = 'time'  # Rename the index to 'time'
            
            wfa_data1 = full_wfa_data[self.sim_start: self.sim_end]
            wfa_data2 = wfa_data1 * ((24 * 60 * 60 * 1000) / (self.acc_data * 1e6))
            wfa_data2.rename(columns={'mfd_wfa': 'scaled_acc'}, inplace=True)
            wfa_data3 = wfa_data1  * ((24 * 60 * 60 * 1000) / (slp_data * 1e6))
            wfa_data3.rename(columns={'mfd_wfa': 'scaled_slp'}, inplace=True)
            wfa_data4 = wfa_data1.join([wfa_data2])
            wfa_data = wfa_data4.join([wfa_data3])
            all_wfa.append(wfa_data)

            station_discharge = self.grdc_subset['runoff_mean'].sel(id=k).to_dataframe(name='station_discharge')

            predictors = wfa_data.copy()
            predictors.replace([np.inf, -np.inf], np.nan, inplace=True)
            response = station_discharge.drop(['id'], axis=1)

            sin_lat, cos_lat, sin_lon, cos_lon = self.encode_lat_lon(snapped_y, snapped_x)
            catch_list = [acc_data, slp_data, sin_lat, cos_lat, sin_lon, cos_lon, tree_cover_data, herb_cover_data, 
                          awc_data, satpt_data]

            self.data_list.append((predictors, response))
            catch_tup = tuple(catch_list)
            self.catchment.append(catch_tup)  
            count = count + 1
            
        return [self.data_list, self.catchment]
    
    def get_data_latlng(self, olat, olon):

        count = 1
        
        count = 1
        
        slope = f'{self.working_dir}/elevation/slope_clipped.tif'
        dem_filepath = f'{self.working_dir}/elevation/dem_clipped.tif'
        tree_cover = f'{self.working_dir}/vcf/mean_tree_cover.tif'
        herb_cover = f'./{self.working_dir}/vcf/mean_herb_cover.tif'
        awc = f'{self.working_dir}/soil/clipped_AWCh3_M_sl6_1km_ll.tif'
        sat_pt = f'{self.working_dir}/soil/clipped_AWCtS_M_sl6_1km_ll.tif'
        
        latlng_ras = rioxarray.open_rasterio(dem_filepath)
        latlng_ras = latlng_ras.rio.write_crs(4326)
        lat = latlng_ras['y'].values
        lon = latlng_ras['x'].values
        
        grid = pysheds.grid.Grid.from_raster(dem_filepath)
        dem = grid.read_raster(dem_filepath)
        
        flooded_dem = grid.fill_depressions(dem)
        inflated_dem = grid.resolve_flats(flooded_dem)
        fdir = grid.flowdir(inflated_dem, routing='mfd')
        acc = grid.accumulation(fdir=fdir, routing='mfd')
        
        facc_thresh = np.nanmax(acc) * 0.0001
        self.river_grid = np.where(acc < facc_thresh, 0, 1)
        river_ras = xr.DataArray(data=self.river_grid, coords=[('lat', lat), ('lon', lon)])
        
        with rasterio.open(dem_filepath) as src:
            ref_meta = src.meta.copy()  # Copy the metadata exactly as is

        with rasterio.open(f'{self.working_dir}/catchment/river_grid.tif', 'w', **ref_meta) as dst:
            dst.write(river_ras.values, 1)  # Write data to the first band
        
        weight2 = grid.read_raster(slope)
        cum_slp = grid.accumulation(fdir=fdir, weights=weight2, routing='mfd')

        weight3 = grid.read_raster(tree_cover)
        cum_tree_cover = grid.accumulation(fdir=fdir, weights=weight3, routing='mfd')
        
        weight4 = grid.read_raster(herb_cover)
        cum_herb_cover = grid.accumulation(fdir=fdir, weights=weight4, routing='mfd')

        weight5 = grid.read_raster(awc)
        cum_awc = grid.accumulation(fdir=fdir, weights=weight5, routing='mfd')
        
        weight6 = grid.read_raster(sat_pt)
        cum_satpt = grid.accumulation(fdir=fdir, weights=weight6, routing='mfd')
        
        acc = xr.DataArray(data=acc, coords=[('lat', lat), ('lon', lon)])
        cum_slp = xr.DataArray(data=cum_slp, coords=[('lat', lat), ('lon', lon)])
        cum_tree_cover = xr.DataArray(data=cum_tree_cover, coords=[('lat', lat), ('lon', lon)])
        cum_herb_cover = xr.DataArray(data=cum_herb_cover, coords=[('lat', lat), ('lon', lon)])
        cum_awc = xr.DataArray(data=cum_awc, coords=[('lat', lat), ('lon', lon)])
        cum_satpt = xr.DataArray(data=cum_satpt, coords=[('lat', lat), ('lon', lon)])
 
        time_index = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        #combine or all yearly output from the runoff and routing module into a single list
        all_years_wfa = sorted(glob.glob(f'{self.working_dir}/output_data/*.pkl'))
        wfa_list = []
        for year in all_years_wfa:
            with open(year, 'rb') as f:
                this_arr = pickle.load(f)
            wfa_list = wfa_list + this_arr
                

        snapped_y, snapped_x = self._snap_coordinates(olat, olon)
        acc_data = acc.sel(lat=snapped_y, lon=snapped_x, method='nearest')
        slp_data = cum_slp.sel(lat=snapped_y, lon=snapped_x, method='nearest')
        tree_cover_data = cum_tree_cover.sel(lat=snapped_y, lon=snapped_x, method='nearest').values
        herb_cover_data = cum_herb_cover.sel(lat=snapped_y, lon=snapped_x, method='nearest').values
        awc_data = cum_awc.sel(lat=snapped_y, lon=snapped_x, method='nearest').values
        satpt_data = cum_satpt.sel(lat=snapped_y, lon=snapped_x, method='nearest').values
        self.acc_data = acc_data.values
        slp_data = slp_data.values
   
        row, col = self._extract_station_rowcol(snapped_y, snapped_x)

        station_wfa = []
        for arr in wfa_list:
            arr = arr.tocsr()
            station_wfa.append(arr[int(row), int(col)])
        full_wfa_data = pd.DataFrame(station_wfa, columns=['mfd_wfa'])
        full_wfa_data.set_index(time_index, inplace=True)
        full_wfa_data.index.name = 'time'  # Rename the index to 'time'

        #extract wfa data based on defined training period
        wfa_data1 = full_wfa_data[self.sim_start: self.sim_end]
        wfa_data2 = wfa_data1 * ((24 * 60 * 60 * 1000) / (self.acc_data * 1e6))
        wfa_data2.rename(columns={'mfd_wfa': 'scaled_acc'}, inplace=True)
        wfa_data3 = wfa_data1  * ((24 * 60 * 60 * 1000) / (slp_data * 1e6))
        wfa_data3.rename(columns={'mfd_wfa': 'scaled_slp'}, inplace=True)
        wfa_data4 = wfa_data1.join([wfa_data2])
        wfa_data = wfa_data4.join([wfa_data3])

        predictors = wfa_data.copy()
        predictors.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        sin_lat, cos_lat, sin_lon, cos_lon = self.encode_lat_lon(snapped_y, snapped_x)
        catch_list = [self.acc_data, slp_data,sin_lat, cos_lat, sin_lon, cos_lon, tree_cover_data, 
                      herb_cover_data, awc_data, satpt_data]
        predictors2 = predictors
        self.data_list.append((predictors2))
        catch_tup = tuple(catch_list)
        self.catchment.append(catch_tup)

        count = count + 1
            
        return [self.data_list, self.catchment]

    
#=====================================================================================================================================
@register_keras_serializable(package="Custom", name="laplacian_nll")
def laplacian_nll(y_true, y_pred):

    mu = y_pred[:, 0]  # Mean prediction (original space)
    b = tf.nn.softplus(y_pred[:, 1]) + 1e-3  # Scale parameter (uncertainty), ensuring positivity

    # Define Laplace distribution in original space
    laplace_dist = tfd.Laplace(loc=mu, scale=b)

    nll = -tf.reduce_mean(laplace_dist.log_prob(y_true[:, 0]))

    # ✅ Regularization to penalize large sigma values
    reg_term = 0.0001 * tf.reduce_mean(tf.square(b))  # Adjust weight if needed

    # Compute Negative Log-Likelihood (NLL)
    return nll + reg_term

class PredictStreamflow:
    """
    A class for preprocessing flow accumulation data and/or observed streamflow data for prediction and comparison. Preprocessing involves data augmentation, remove missing data and breaking them into sequences of specified length.

    """
    def __init__(self, working_dir):
        self.regional_model = None
        self.train_data_list = []
        self.timesteps = 360
        self.num_epochs = 10
        self.batch_size = 64
        self.train_predictors = None
        self.train_response = None
        self.num_dynamic_features = 3
        self.num_static_features = 10
        self.scaled_trained_catchment = None
        self.working_dir = working_dir
    
    def load_global_cdfs_pkl(self):
        """Load the saved empirical CDFs for multiple variables from a pickle file."""
        with open(f'./{self.working_dir}/models/{self.working_dir}_global_cdfs.pkl', "rb") as f:
            global_cdfs = pickle.load(f)
        return global_cdfs

    def compute_global_cdfs_pkl(self, df, variables):
        """
        Compute and save the empirical CDF for each variable separately as a pickle file.
    
        Args:
            df (pd.DataFrame): DataFrame containing multiple variables.
            variables (list): List of column names to apply quantile scaling.
            filename (str): File to save the computed CDFs.
        """
        global_cdfs = {}
    
        for var in variables:
            sorted_values = np.sort(df[var].dropna().values)  # Remove NaNs and sort
            quantiles = np.linspace(0, 1, len(sorted_values))  # Generate percentiles
            global_cdfs[var] = (sorted_values, quantiles)  # Store CDF mapping

        return global_cdfs

    def quantile_transform(self, df, variables, global_cdfs):
        """
        Apply quantile scaling to multiple variables using precomputed global CDFs.
    
        Args:
            df (pd.DataFrame): DataFrame to transform.
            variables (list): List of column names to scale.
            global_cdfs (dict): Dictionary of saved CDF mappings.
        
        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        transformed_df = df.copy()
    
        for var in variables:
            sorted_values, quantiles = global_cdfs[var]
            
            # Apply interpolation to transform values to the percentile space
            transformed_df[var] = np.interp(df[var], sorted_values, quantiles)
    
            # Handle out-of-range values
            transformed_df[var][df[var] < sorted_values[0]] = 0.001  # Assign near 0 for very low values
            transformed_df[var][df[var] > sorted_values[-1]] = 0.999  # Assign near 1 for very high values
    
        return transformed_df
    
    def prepare_data(self, data_list):
        
        """
        Prepare flow accumulation and streamflow data extracted from GRDC database for input in the model. Preparation involves dividing time-series data into desired short sequences based on specified timesteps and reshaping into desired tensor shape.
        
        Parameters:
        -----------
        data_list : Numpy array data 
            The extracted flow accumulation and observed streamflow data i.e. the output of get_grdc_data() functions.

        """

        predictors = list(map(lambda xy: xy[0], data_list[0]))
        train_catchment = np.array(data_list[1])
     
        full_train_predictors = []
        train_catchment_list = []

        with open(f'{self.working_dir}/models/catchment_size_scaler_coarse.pkl', 'rb') as file:
            catchment_scaler = pickle.load(file)

        if len(train_catchment) <= 0:
            return
        
        train_catchment = train_catchment.reshape(-1, self.num_static_features)
        scaled_trained_catchment = catchment_scaler.transform(train_catchment)
        
        variables = ['mfd_wfa', 'scaled_acc', 'scaled_slp']  # Adjust as needed
        global_cdfs = self.load_global_cdfs_pkl()

        for x, z in zip(predictors, scaled_trained_catchment):
            scaled_train_predictor = self.quantile_transform(x, variables, global_cdfs)
            scaled_train_predictor = scaled_train_predictor.values

            num_samples = scaled_train_predictor.shape[0] - self.timesteps
            predictor_samples = []
            catchment_samples = []
            
            for i in range(num_samples):
                predictor_batch = scaled_train_predictor[i:i+self.timesteps, :]
                predictor_batch = predictor_batch.reshape(self.timesteps, self.num_dynamic_features)
                predictor_samples.append(predictor_batch)
                catchment_samples.append(z)
            
            timesteps_to_keep = []
            for i in range(num_samples):
                if not np.isnan(predictor_samples[i]).any():
                    timesteps_to_keep.append(i)

            timesteps_to_keep = np.array(timesteps_to_keep, dtype=np.int64)
            scaled_train_predictor_filtered = np.array(predictor_samples)
            scaled_train_catchment_filtered = np.array(catchment_samples)
            
            full_train_predictors.append(scaled_train_predictor_filtered)
            train_catchment_list.append(scaled_train_catchment_filtered)
            
        self.predictors = np.concatenate(full_train_predictors, axis=0)
        self.catchment_size = np.concatenate(train_catchment_list, axis=0).reshape(-1,self.num_static_features) 
    
    def prepare_data_latlng(self, data_list):
        
        """
        Prepare flow accumulation and streamflow data extracted from GRDC database for input in the model. Preparation involves dividing time-series data into desired short sequences based on specified timesteps and reshaping into desired tensor shape.
        
        Parameters:
        -----------
        data_list : Numpy array data 
            The extracted flow accumulation and observed streamflow data i.e. the output of get_grdc_data() functions.

        """

        predictors = list(map(lambda xy: xy, data_list[0]))
        train_catchment = np.array(data_list[1])  
                
        full_train_predictors = []
        train_catchment_list = []
        
        with open(f'{self.working_dir}/models/catchment_size_scaler_coarse.pkl', 'rb') as file:
            catchment_scaler = pickle.load(file)

        if len(train_catchment) <= 0:
            return
        
        train_catchment = train_catchment.reshape(-1, self.num_static_features)
        scaled_trained_catchment = catchment_scaler.transform(train_catchment)
        
        variables = ['mfd_wfa', 'scaled_acc', 'scaled_slp']  # Adjust as needed
        global_cdfs = self.load_global_cdfs_pkl()

        for x, z in zip(predictors, scaled_trained_catchment):
            scaled_train_predictor = self.quantile_transform(x, variables, global_cdfs)
            scaled_train_predictor = scaled_train_predictor.values
              
            num_samples = scaled_train_predictor.shape[0] - self.timesteps
            predictor_samples = []
            catchment_samples = []
            
            # Iterate over each batch
            for i in range(num_samples):
                # Slice the numpy array using the rolling window
                predictor_batch = scaled_train_predictor[i:i+self.timesteps, :]
                predictor_batch = predictor_batch.reshape(self.timesteps, self.num_dynamic_features)
                
                # Append the batch to the list
                predictor_samples.append(predictor_batch)
                catchment_samples.append(z)
            
            timesteps_to_keep = []
            for i in range(num_samples):
                if not np.isnan(predictor_samples[i]).any():
                    timesteps_to_keep.append(i)

            timesteps_to_keep = np.array(timesteps_to_keep, dtype=np.int64)
            scaled_train_predictor_filtered = np.array(predictor_samples)
            scaled_train_catchment_filtered = np.array(catchment_samples)
            
            full_train_predictors.append(scaled_train_predictor_filtered)
            train_catchment_list.append(scaled_train_catchment_filtered)
            
        self.predictors = np.concatenate(full_train_predictors, axis=0)
        self.catchment_size = np.concatenate(train_catchment_list, axis=0).reshape(-1,self.num_static_features)
        
            
    def load_model(self, path):
        """
        Load saved LSTM model. 
        
        Parameters:
        -----------
        Path : H5 file
            Path to the saved neural network LSTM model

        """
        from tcn import TCN  # Make sure to import TCN
        from tensorflow.keras.utils import custom_object_scope

        custom_objects = {"TCN": TCN, "laplacian_nll": laplacian_nll}

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            with custom_object_scope(custom_objects):  
                self.model = load_model(path, custom_objects=custom_objects, safe_mode=False)
        
    def summary(self):
        self.model.summary()