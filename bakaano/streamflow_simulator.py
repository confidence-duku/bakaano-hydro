
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
from sklearn.preprocessing import MinMaxScaler
from rasterio.transform import rowcol
from keras.models import load_model # type: ignore
import pickle
import warnings
import geopandas as gpd
from scipy.spatial.distance import cdist
from datetime import datetime
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

tfd = tfp.distributions  # TensorFlow Probability distributions
#=====================================================================================================================================


class PredictDataPreprocessor:
    def __init__(self, working_dir,  study_area,  sim_start, sim_end, routing_method, 
                 grdc_streamflow_nc_file=None, catchment_size_threshold=None):
        """
        Initialize the PredictDataPreprocessor object.
        
        Args:
            working_dir (str): The parent working directory where files and outputs will be stored.
            study_area (str): The path to the shapefile of the river basin or watershed.
            start_date (str): The start date for the simulation period in 'YYYY-MM-DD' format.
            end_date (str): The end date for the simulation period in 'YYYY-MM-DD' format.
            grdc_streamflow_nc_file (str): The path to the GRDC streamflow NetCDF file.

        Methods
        -------
        _extract_station_rowcol(lat, lon): Extract the row and column indices for a given latitude and longitude from given raster file.
        _snap_coordinates(lat, lon): Snap the given latitude and longitude to the nearest river segment based on a river grid.
        load_observed_streamflow(grdc_streamflow_nc_file): Load observed streamflow data from GRDC NetCDF file.
        encode_lat_lon(latitude, longitude): Encode latitude and longitude into sine and cosine components.
        get_data(): Extract and process data for each station in the GRDC dataset.
        get_data_latlng(latlist, lonlist): Extract and process data for specified latitude and longitude coordinates.
    
        """
        self.study_area = study_area
        self.working_dir = working_dir
        self.routing_method = routing_method
        
        self.data_list = []
        self.catchment = []  
        self.sim_start = sim_start
        self.sim_end = sim_end
        self.sim_station_names= []
        self.catchment_size_threshold = catchment_size_threshold
        if grdc_streamflow_nc_file is not None:
            self.grdc_subset = self.load_observed_streamflow(grdc_streamflow_nc_file)
            self.station_ids = np.unique(self.grdc_subset.to_dataframe().index.get_level_values('id'))
        
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
            (grdc['time'] >= pd.to_datetime(self.sim_start)) &
            (grdc['time'] <= pd.to_datetime(self.sim_end)) &
            (grdc['station_name'].isin(overlapping_station_names)),
            drop=True
        )
        self.sim_station_names = np.unique(filtered_grdc['station_name'].values)
        return filtered_grdc
                          
    def get_data(self):

        count = 1
        
        #slope = f'{self.working_dir}/elevation/slope_clipped.tif'
        dem_filepath = f'{self.working_dir}/elevation/dem_clipped.tif'
        
        latlng_ras = rioxarray.open_rasterio(dem_filepath)
        latlng_ras = latlng_ras.rio.write_crs(4326)
        lat = latlng_ras['y'].values
        lon = latlng_ras['x'].values
        
        grid = pysheds.grid.Grid.from_raster(dem_filepath)
        dem = grid.read_raster(dem_filepath)
        
        flooded_dem = grid.fill_depressions(dem)
        inflated_dem = grid.resolve_flats(flooded_dem)
        fdir = grid.flowdir(inflated_dem, routing=self.routing_method)
        acc = grid.accumulation(fdir=fdir, routing=self.routing_method)
        
        facc_thresh = np.nanmax(acc) * 0.0001
        self.river_grid = np.where(acc < facc_thresh, 0, 1)
        river_ras = xr.DataArray(data=self.river_grid, coords=[('lat', lat), ('lon', lon)])
        
        with rasterio.open(dem_filepath) as src:
            ref_meta = src.meta.copy()  # Copy the metadata exactly as is

        with rasterio.open(f'{self.working_dir}/catchment/river_grid.tif', 'w', **ref_meta) as dst:
            dst.write(river_ras.values, 1)  # Write data to the first band

        alpha_earth_bands = sorted(glob.glob(f'{self.working_dir}/alpha_earth/band*.tif'))
        alpha_earth_list = []

        for band in alpha_earth_bands:
            weight2 = grid.read_raster(band) + 1
            cum_band = grid.accumulation(fdir=fdir, weights=weight2, routing=self.routing_method)
            cum_band = xr.DataArray(data=cum_band, coords=[('lat', lat), ('lon', lon)])
            alpha_earth_list.append(cum_band)
        
        acc = xr.DataArray(data=acc, coords=[('lat', lat), ('lon', lon)])

        start_dt = datetime.strptime(self.sim_start, "%Y-%m-%d")
        end_dt = datetime.strptime(self.sim_end, "%Y-%m-%d")
        
        #combine or all yearly output from the runoff and routing module into a single list
        all_years_wfa = sorted(glob.glob(f'{self.working_dir}/runoff_output/*.pkl'))
        wfa_list = []
        for year in all_years_wfa:
            with open(year, 'rb') as f:
                this_arr = pickle.load(f)
            wfa_list = wfa_list + this_arr

        # Filter based on time range
        wfa_list = [
            entry for entry in wfa_list
            if start_dt <= datetime.strptime(entry["time"], "%Y-%m-%d") <= end_dt
        ]
        
        time_index = pd.date_range(start=self.sim_start, end=self.sim_end, freq='D')

        all_wfa = []
        for k in self.station_ids:
            station_discharge = self.grdc_subset['runoff_mean'].sel(id=k).to_dataframe(name='station_discharge')

            # if self.catchment_size_threshold is not None:
            #     catchment_size = self.grdc_subset['area'].sel(id=k, method='nearest').values
            #     if catchment_size < self.catchment_size_threshold:
            #         continue

            # self.sim_station_names.append(list(self.grdc_subset['station_name'].sel(id=k).values)[0])
                          
            station_x = np.nanmax(self.grdc_subset['geo_x'].sel(id=k).values)
            station_y = np.nanmax(self.grdc_subset['geo_y'].sel(id=k).values)
            snapped_y, snapped_x = self._snap_coordinates(station_y, station_x)
            self.x = snapped_x
            self.y = snapped_y
            
            acc_data = acc.sel(lat=snapped_y, lon=snapped_x, method='nearest').values
            alpha_earth_stations = []
            for band in alpha_earth_list:
                pixel_data = band.sel(lat=snapped_y, lon=snapped_x, method='nearest').values
                alpha_earth_stations.append(pixel_data/acc_data)
                
            row, col = self._extract_station_rowcol(snapped_y, snapped_x)
        
            station_wfa = []
            for arr in wfa_list:
                arr = arr['matrix'].tocsr()
                station_wfa.append(arr[int(row), int(col)])
            full_wfa_data = pd.DataFrame(station_wfa, columns=['mfd_wfa'])
            full_wfa_data.set_index(time_index, inplace=True)
            full_wfa_data.index.name = 'time'  # Rename the index to 'time'
        
            #wfa_data = full_wfa_data
            wfa_data = full_wfa_data[self.sim_start: self.sim_end]
            all_wfa.append(wfa_data)

            station_discharge = self.grdc_subset['runoff_mean'].sel(id=k).to_dataframe(name='station_discharge')

            predictors = wfa_data.copy()
            predictors.replace([np.inf, -np.inf], np.nan, inplace=True)
            response = station_discharge.drop(['id'], axis=1)

            log_acc = np.log1p(acc_data)
            catch_list = [log_acc] + alpha_earth_stations
            catch_list = [float(x) for x in catch_list]
            predictors2 = predictors
            catch_tup = tuple(catch_list)
            self.catchment.append(catch_tup)  
            self.data_list.append((predictors2, response, catch_tup))
            count = count + 1
            
        return self.data_list
    
    def get_data_latlng(self, latlist, lonlist):

        count = 1
        
        dem_filepath = f'{self.working_dir}/elevation/dem_clipped.tif'
        
        
        latlng_ras = rioxarray.open_rasterio(dem_filepath)
        latlng_ras = latlng_ras.rio.write_crs(4326)
        lat = latlng_ras['y'].values
        lon = latlng_ras['x'].values
        
        grid = pysheds.grid.Grid.from_raster(dem_filepath)
        dem = grid.read_raster(dem_filepath)
        
        flooded_dem = grid.fill_depressions(dem)
        inflated_dem = grid.resolve_flats(flooded_dem)
        fdir = grid.flowdir(inflated_dem, routing=self.routing_method)
        acc = grid.accumulation(fdir=fdir, routing=self.routing_method)
        
        facc_thresh = np.nanmax(acc) * 0.0001
        self.river_grid = np.where(acc < facc_thresh, 0, 1)
        river_ras = xr.DataArray(data=self.river_grid, coords=[('lat', lat), ('lon', lon)])
        
        with rasterio.open(dem_filepath) as src:
            ref_meta = src.meta.copy()  # Copy the metadata exactly as is

        with rasterio.open(f'{self.working_dir}/catchment/river_grid.tif', 'w', **ref_meta) as dst:
            dst.write(river_ras.values, 1)  # Write data to the first band

        alpha_earth_bands = sorted(glob.glob(f'{self.working_dir}/alpha_earth/band*.tif'))
        alpha_earth_list = []

        for band in alpha_earth_bands:
            weight2 = grid.read_raster(band) + 1
            cum_band = grid.accumulation(fdir=fdir, weights=weight2, routing=self.routing_method)
            cum_band = xr.DataArray(data=cum_band, coords=[('lat', lat), ('lon', lon)])
            alpha_earth_list.append(cum_band)
        
        acc = xr.DataArray(data=acc, coords=[('lat', lat), ('lon', lon)])
        

        start_dt = datetime.strptime(self.sim_start, "%Y-%m-%d")
        end_dt = datetime.strptime(self.sim_end, "%Y-%m-%d")
 
        time_index = pd.date_range(start=self.sim_start, end=self.sim_end, freq='D')
        #combine or all yearly output from the runoff and routing module into a single list
        all_years_wfa = sorted(glob.glob(f'{self.working_dir}/runoff_output/*.pkl'))
        wfa_list = []
        for year in all_years_wfa:
            with open(year, 'rb') as f:
                this_arr = pickle.load(f)
            wfa_list = wfa_list + this_arr

        # Filter based on time range
        wfa_list = [
            entry for entry in wfa_list
            if start_dt <= datetime.strptime(entry["time"], "%Y-%m-%d") <= end_dt
        ]

        
        for olat, olon in zip(latlist, lonlist):
            snapped_y, snapped_x = self._snap_coordinates(olat, olon)
            acc_data = acc.sel(lat=snapped_y, lon=snapped_x, method='nearest').values
            alpha_earth_stations = []
            for band in alpha_earth_list:
                pixel_data = band.sel(lat=snapped_y, lon=snapped_x, method='nearest').values
                alpha_earth_stations.append(pixel_data/acc_data)
            
            self.acc_data = acc_data
            
    
            row, col = self._extract_station_rowcol(snapped_y, snapped_x)

            station_wfa = []
            for arr in wfa_list:
                arr = arr['matrix'].tocsr()
                station_wfa.append(arr[int(row), int(col)])
            full_wfa_data = pd.DataFrame(station_wfa, columns=['mfd_wfa'])
            full_wfa_data.set_index(time_index, inplace=True)
            full_wfa_data.index.name = 'time'  # Rename the index to 'time'

            #extract wfa data based on defined training period
            wfa_data = full_wfa_data

            predictors = wfa_data.copy()
            predictors.replace([np.inf, -np.inf], np.nan, inplace=True)
            log_acc = np.log1p(self.acc_data)
            catch_list = [log_acc] + alpha_earth_stations
            catch_list = [float(x) for x in catch_list]
            
            predictors2 = predictors
            catch_tup = tuple(catch_list)
            self.catchment.append(catch_tup)
            self.data_list.append((predictors2, catch_tup))

            count = count + 1
            
        return [self.data_list, self.catchment, latlist, lonlist]

    
#=====================================================================================================================================
@register_keras_serializable(package="Custom", name="laplacian_nll")
def laplacian_nll(y_true, y_pred):

    mu = y_pred[:, 0]  # Log-space mean prediction
    b = tf.nn.softplus(y_pred[:, 1]) + 1e-6  # Scale parameter (uncertainty)

    # ✅ Log-transform the observed streamflow
    log_y_true = tf.math.log(y_true[:, 0] + 1)  # Avoid log(0) issues

    # Define Laplacian distribution in log-space
    laplace_dist = tfd.Laplace(loc=mu, scale=b)

    # Compute NLL in log-space
    return -tf.reduce_mean(laplace_dist.log_prob(log_y_true))

@register_keras_serializable(package="Bakaano", name='mdn_laplacian_nll')
def mdn_laplace_nll_factory(K):
    """
    Serializable MDN Laplace negative log-likelihood loss.
    K = number of mixture components
    """

    def mdn_laplace_nll(y_true, y_pred):
        # Split parameters
        logits = y_pred[:, :K]
        mus    = y_pred[:, K:2*K]
        sigmas = y_pred[:, 2*K:]

        sigmas = tf.nn.softplus(sigmas) + 1e-6

        mixture = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=logits),
            components_distribution=tfd.Laplace(
                loc=mus,
                scale=sigmas
            )
        )

        return -tf.reduce_mean(
            mixture.log_prob(tf.squeeze(y_true))
        )

    return mdn_laplace_nll

class PredictStreamflow:
    def __init__(self, working_dir, lookback):
        """
        Initializes the PredictStreamflow class for streamflow prediction using a temporal convolutional network (TCN).

        Args:
            working_dir (str): The working directory where the model and data are stored.
            lookback (int): The number of timesteps to look back for prediction.

        Methods
        -------
        load_global_cdfs_pkl(): Load the saved empirical CDFs for multiple variables from a pickle file.
        compute_global_cdfs_pkl(df, variables): Compute and save the empirical CDF for each variable separately as a pickle file.
        quantile_transform(df, variables, global_cdfs): Apply quantile scaling to multiple variables using precomputed global CDFs.
        compute_local_cdf(df, variables): Compute and save the empirical CDF for each variable separately as a pickle file.
        prepare_data(data_list): Prepare flow accumulation and streamflow data extracted from GRDC database for input in the model.
        prepare_data_latlng(data_list): Prepare flow accumulation and streamflow data extracted from GRDC database for input in the model.
        load_model(): Load the trained regional model from a file.

        """
        self.regional_model = None
        self.train_data_list = []
        self.timesteps = lookback
        self.num_epochs = 10
        
        self.train_predictors = None
        self.train_response = None
        self.num_dynamic_features = 1
        self.num_static_features = 65
        self.scaled_trained_catchment = None
        self.working_dir = working_dir
    
    def load_global_cdfs_pkl(self):
        """Load the saved empirical CDFs for multiple variables from a pickle file."""
        with open(f'{self.working_dir}/models/global_cdfs.pkl', "rb") as f:
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

        predictors = list(map(lambda xy: xy[0], data_list))
        catchment = list(map(lambda xy: xy[2], data_list))
        catchment_arr = np.array(catchment)

        area = catchment_arr[:, 0:1]      # shape (N, 1)
        alphaearth = catchment_arr[:, 1:] # shape (N, D)
     
        full_train_predictors = []
        full_alphaearth = []
        full_catchsize = []

        with open(f'{self.working_dir}/models/alpha_earth_scaler.pkl', 'rb') as file:
            alphaearth_scaler = pickle.load(file)

        if len(catchment) <= 0:
            return

        alphaearth = alphaearth.reshape(-1,64)
        scaled_alphaearth = alphaearth_scaler.transform(alphaearth) 

        
        variables = ['mfd_wfa']  # Adjust as needed
        global_cdfs = self.load_global_cdfs_pkl()

        for x, z, j in zip(predictors, scaled_alphaearth, area):
            scaled_train_predictor = self.quantile_transform(x, variables, global_cdfs)
            scaled_train_predictor = scaled_train_predictor.values

            num_samples = scaled_train_predictor.shape[0] - self.timesteps
            predictor_samples = []
            area_samples = []
            alphaearth_samples = []
            
            for i in range(num_samples):
                predictor_batch = scaled_train_predictor[i:i+self.timesteps, :]
                predictor_batch = predictor_batch.reshape(self.timesteps, self.num_dynamic_features)

                predictor_samples.append(predictor_batch)
                alphaearth_samples.append(z)
                area_samples.append(j)
            
            timesteps_to_keep = []
            for i in range(num_samples):
                if not np.isnan(predictor_samples[i]).any():
                    timesteps_to_keep.append(i)

            timesteps_to_keep = np.array(timesteps_to_keep, dtype=np.int64)
            scaled_train_predictor_filtered = np.array(predictor_samples)
            scaled_alphaearth_filtered = np.array(alphaearth_samples)
            area_filtered = np.array(area_samples)
            
            full_train_predictors.append(scaled_train_predictor_filtered)
            full_alphaearth.append(scaled_alphaearth_filtered)
            full_catchsize.append(area_filtered)
            
        self.predictors = np.concatenate(full_train_predictors, axis=0)
        self.sim_alphaearth = np.concatenate(full_alphaearth, axis=0).reshape(-1, 64)  
        self.sim_catchsize = np.concatenate(full_catchsize, axis=0).reshape(-1, 1) 
    
    def prepare_data_latlng(self, data_list):
        
        """
        Prepare flow accumulation and streamflow data extracted from GRDC database for input in the model. Preparation involves dividing time-series data into desired short sequences based on specified timesteps and reshaping into desired tensor shape.
        
        Parameters:
        -----------
        data_list : Numpy array data 
            The extracted flow accumulation and observed streamflow data i.e. the output of get_grdc_data() functions.

        """

        predictors = list(map(lambda xy: xy[0], data_list[0]))
        catchment = list(map(lambda xy: xy[1], data_list[0]))
        catchment_arr = np.array(catchment)

        area = catchment_arr[:, 0:1]      # shape (N, 1)
        alphaearth = catchment_arr[:, 1:] # shape (N, D)
     
        full_train_predictors = []
        full_alphaearth = []
        full_catchsize = []
        
        self.latlist = data_list[2]
        self.lonlist = data_list[3] 
                
        
        with open(f'{self.working_dir}/models/alpha_earth_scaler.pkl', 'rb') as file:
            alphaearth_scaler = pickle.load(file)

        if len(catchment) <= 0:
            return
        
        alphaearth = alphaearth.reshape(-1,64)
        scaled_alphaearth = alphaearth_scaler.transform(alphaearth) 
        
        variables = ['mfd_wfa']  # Adjust as needed
        global_cdfs = self.load_global_cdfs_pkl()

        for x, z, j in zip(predictors, scaled_alphaearth, area):
            scaled_train_predictor = self.quantile_transform(x, variables, global_cdfs)
            scaled_train_predictor = scaled_train_predictor.values

            num_samples = scaled_train_predictor.shape[0] - self.timesteps
            predictor_samples = []
            area_samples = []
            alphaearth_samples = []
            
            for i in range(num_samples):
                predictor_batch = scaled_train_predictor[i:i+self.timesteps, :]
                predictor_batch = predictor_batch.reshape(self.timesteps, self.num_dynamic_features)

                predictor_samples.append(predictor_batch)
                alphaearth_samples.append(z)
                area_samples.append(j)
            
            timesteps_to_keep = []
            for i in range(num_samples):
                if not np.isnan(predictor_samples[i]).any():
                    timesteps_to_keep.append(i)

            timesteps_to_keep = np.array(timesteps_to_keep, dtype=np.int64)
            scaled_train_predictor_filtered = np.array(predictor_samples)
            scaled_alphaearth_filtered = np.array(alphaearth_samples)
            area_filtered = np.array(area_samples)
            
            full_train_predictors.append(scaled_train_predictor_filtered)
            full_alphaearth.append(scaled_alphaearth_filtered)
            full_catchsize.append(area_filtered)
            
        self.predictors = np.concatenate(full_train_predictors, axis=0)
        self.sim_alphaearth = np.concatenate(full_alphaearth, axis=0).reshape(-1, 64)  
        self.sim_catchsize = np.concatenate(full_catchsize, axis=0).reshape(-1, 1)
        
            
    def load_model(self, path, loss_fn):
        """
        Load saved LSTM model. 
        
        Parameters:
        -----------
        Path : H5 file
            Path to the saved neural network LSTM model

        """
        from tcn import TCN  # Make sure to import TCN
        from tensorflow.keras.utils import custom_object_scope

        if loss_fn == 'laplacian_nll':
            custom_objects = {"TCN": TCN, "laplacian_nll": laplacian_nll}
        else:
            custom_objects = {"TCN": TCN}

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            with custom_object_scope(custom_objects):  
                self.model = load_model(path, custom_objects=custom_objects)
        
    def summary(self):
        self.model.summary()