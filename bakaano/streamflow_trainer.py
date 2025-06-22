
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Concatenate, Input, LeakyReLU, Multiply, Add
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
from tensorflow.keras.utils import register_keras_serializable
from sklearn.preprocessing import MinMaxScaler
import glob
import pysheds.grid
import rasterio
import rioxarray
from rasterio.transform import rowcol
from tcn import TCN
from keras.models import load_model # type: ignore
import pickle
import warnings
import pandas as pd
import geopandas as gpd
from scipy.spatial.distance import cdist
from datetime import datetime
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

tfd = tfp.distributions  # TensorFlow Probability distributions
#=====================================================================================================================================

class DataPreprocessor:
    def __init__(self,  working_dir, study_area, grdc_streamflow_nc_file, train_start, 
                 train_end, routing_method, catchment_size_threshold):
        """
        Initialize the DataPreprocessor with project details and dates.
        
        Args:
            working_dir (str): The parent working directory where files and outputs will be stored.
            study_area (str): The path to the shapefile defining the study area.
            grdc_streamflow_nc_file (str): The path to the GRDC streamflow NetCDF file.
            start_date (str): The start date for the simulation (training + validation) period in 'YYYY-MM-DD' format.
            end_date (str): The end date for the simulation (training + validation) period in 'YYYY-MM-DD' format.
            train_start (str): The start date for the training period in 'YYYY-MM-DD' format.
            train_end (str): The end date for the training period in 'YYYY-MM-DD' format.

        Methods
        -------
        __init__(working_dir, study_area, grdc_streamflow_nc_file, start_date, end_date):
            Initializes the DataPreprocessor with project details and dates.
        load_observed_streamflow(grdc_streamflow_nc_file):
            Loads and filters observed streamflow data based on the study area and simulation period.
        encode_lat_lon(latitude, longitude):
            Encodes latitude and longitude into sine and cosine components.
        get_data():
            Extracts and preprocesses predictor and response variables for each station based on its coordinates.

        """
        
        self.study_area = study_area
        self.working_dir = working_dir
        #self.times = pd.date_range(start_date, end_date)
        
        self.data_list = []
        self.catchment = []    
        self.sim_station_names= []
        self.train_start = train_start
        self.train_end = train_end
        self.grdc_subset = self.load_observed_streamflow(grdc_streamflow_nc_file)
        self.station_ids = np.unique(self.grdc_subset.to_dataframe().index.get_level_values('id'))
        self.catchment_size_threshold = catchment_size_threshold
        self.routing_method = routing_method
    
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
            #data = src.read(1)
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

    def load_observed_streamflow(self, grdc_streamflow_nc_file):
        """
        Load and filter observed streamflow data based on the study area and simulation period.

        Returns
        -------
        filtered_grdc : xarray.Dataset
            The filtered GRDC dataset containing streamflow data for the specified
            simulation period and study area.
        """
        grdc = xr.open_dataset(grdc_streamflow_nc_file)

        # Create a GeoDataFrame from the GRDC dataset using geo_x and geo_y attributes
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
            (grdc['time'] >= pd.to_datetime(self.train_start)) &
            (grdc['time'] <= pd.to_datetime(self.train_end)) &
            (grdc['station_name'].isin(overlapping_station_names)),
            drop=True
        )
        
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
        """
        Extract and preprocess predictor and response variables for each station based on its coordinates.

        Returns
        -------
        list
            A list containing two elements:
            - self.data_list: A list of tuples, each containing predictors (DataFrame) and response (DataFrame).
            - self.catchment: A list of tuples, each containing catchment data (accumulation and slope values).
        """
        count = 1
        
        slope = f'{self.working_dir}/elevation/slope_clipped.tif'
        dem_filepath = f'{self.working_dir}/elevation/dem_clipped.tif'
        tree_cover = f'{self.working_dir}/vcf/mean_tree_cover.tif'
        herb_cover = f'{self.working_dir}/vcf/mean_herb_cover.tif'
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
        fdir = grid.flowdir(inflated_dem, routing=self.routing_method)
        acc = grid.accumulation(fdir=fdir, routing=self.routing_method)
        
        facc_thresh = np.nanmax(acc) * 0.0001
        self.river_grid = np.where(acc < facc_thresh, 0, 1)
        river_ras = xr.DataArray(data=self.river_grid, coords=[('lat', lat), ('lon', lon)])
        
        with rasterio.open(dem_filepath) as src:
            ref_meta = src.meta.copy()  # Copy the metadata exactly as is

        with rasterio.open(f'{self.working_dir}/catchment/river_grid.tif', 'w', **ref_meta) as dst:
            dst.write(river_ras.values, 1)  # Write data to the first band
        
        weight2 = grid.read_raster(slope)
        cum_slp = grid.accumulation(fdir=fdir, weights=weight2, routing=self.routing_method)

        weight3 = grid.read_raster(tree_cover)
        cum_tree_cover = grid.accumulation(fdir=fdir, weights=weight3, routing=self.routing_method)
        
        weight4 = grid.read_raster(herb_cover)
        cum_herb_cover = grid.accumulation(fdir=fdir, weights=weight4, routing=self.routing_method)

        weight5 = grid.read_raster(awc)
        cum_awc = grid.accumulation(fdir=fdir, weights=weight5, routing=self.routing_method)
        
        weight6 = grid.read_raster(sat_pt)
        cum_satpt = grid.accumulation(fdir=fdir, weights=weight6, routing=self.routing_method)
        
        acc = xr.DataArray(data=acc, coords=[('lat', lat), ('lon', lon)])
        cum_slp = xr.DataArray(data=cum_slp, coords=[('lat', lat), ('lon', lon)])
        cum_tree_cover = xr.DataArray(data=cum_tree_cover, coords=[('lat', lat), ('lon', lon)])
        cum_herb_cover = xr.DataArray(data=cum_herb_cover, coords=[('lat', lat), ('lon', lon)])
        cum_awc = xr.DataArray(data=cum_awc, coords=[('lat', lat), ('lon', lon)])
        cum_satpt = xr.DataArray(data=cum_satpt, coords=[('lat', lat), ('lon', lon)])
 
        time_index = pd.date_range(start=self.train_start, end=self.train_end, freq='D')
        
        #combine or all yearly output from the runoff and routing module into a single list
        start_dt = datetime.strptime(self.train_start, "%Y-%m-%d")
        end_dt = datetime.strptime(self.train_end, "%Y-%m-%d")

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
        
        #extract station predictor and response variables based on station coordinates
        for k in self.station_ids:
            station_discharge = self.grdc_subset['runoff_mean'].sel(id=k).to_dataframe(name='station_discharge')
                          
            station_x = np.nanmax(self.grdc_subset['geo_x'].sel(id=k).values)
            station_y = np.nanmax(self.grdc_subset['geo_y'].sel(id=k).values)
            snapped_y, snapped_x = self._snap_coordinates(station_y, station_x)
            
            acc_data = acc.sel(lat=snapped_y, lon=snapped_x, method='nearest')
            slp_data = cum_slp.sel(lat=snapped_y, lon=snapped_x, method='nearest')
            tree_cover_data = cum_tree_cover.sel(lat=snapped_y, lon=snapped_x, method='nearest').values
            herb_cover_data = cum_herb_cover.sel(lat=snapped_y, lon=snapped_x, method='nearest').values
            awc_data = cum_awc.sel(lat=snapped_y, lon=snapped_x, method='nearest').values
            satpt_data = cum_satpt.sel(lat=snapped_y, lon=snapped_x, method='nearest').values
            acc_data = acc_data.values
            slp_data = slp_data.values

            if acc_data < self.catchment_size_threshold:
                continue

            self.sim_station_names.append(list(self.grdc_subset['station_name'].sel(id=k).values)[0])
        
            row, col = self._extract_station_rowcol(snapped_y, snapped_x)
            
            station_wfa = []
            for arr in wfa_list:
                arr = arr['matrix'].tocsr()
                station_wfa.append(arr[int(row), int(col)])
            full_wfa_data = pd.DataFrame(station_wfa, columns=['mfd_wfa'])
            full_wfa_data.set_index(time_index, inplace=True)
            full_wfa_data.index.name = 'time'  # Rename the index to 'time'
    
            #extract wfa data based on defined training period
            wfa_data1 = full_wfa_data
            wfa_data2 = wfa_data1 * ((24 * 60 * 60 * 1000) / (acc_data * 1e6))
            wfa_data2.rename(columns={'mfd_wfa': 'scaled_acc'}, inplace=True)
            wfa_data3 = wfa_data1  * ((24 * 60 * 60 * 1000) / (slp_data * 1e6))
            wfa_data3.rename(columns={'mfd_wfa': 'scaled_slp'}, inplace=True)
            wfa_data4 = wfa_data1.join([wfa_data2])
            wfa_data = wfa_data4.join([wfa_data3])
            
            station_discharge = self.grdc_subset['runoff_mean'].sel(id=k).to_dataframe(name='station_discharge')

            predictors = wfa_data.copy()
            predictors.replace([np.inf, -np.inf], np.nan, inplace=True)
            response = station_discharge.drop(['id'], axis=1)

            sin_lat, cos_lat, sin_lon, cos_lon = self.encode_lat_lon(snapped_y, snapped_x)
            catch_list = [acc_data, slp_data, sin_lat, cos_lat, sin_lon, cos_lon, tree_cover_data, herb_cover_data, 
                          awc_data, satpt_data]
            predictors2 = predictors
            catch_tup = tuple(catch_list)
            self.catchment.append(catch_tup)
            self.data_list.append((predictors2, response, catch_tup))
            
            count = count + 1

        with open(f'{self.working_dir}/models/predictor_response_data.pkl', 'wb') as file:
                pickle.dump(self.data_list, file)
            
        return self.data_list
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
class StreamflowModel:
    
    def __init__(self, working_dir, lookback, batch_size, num_epochs):
        """
        Initialize the StreamflowModel with project details.

        Args:
            working_dir (str): The parent working directory where files and outputs will be stored.
            lookback (int): The number of timesteps to look back for the model.
            batch_size (int): The batch size for training the model.
            num_epochs (int): The number of epochs for training the model.

        Methods
        -------
        __init__(working_dir, lookback, batch_size, num_epochs):
            Initializes the StreamflowModel with project details.
        compute_global_cdfs_pkl(df, variables):
            Computes and saves the empirical CDF for each variable separately as a pickle file.
        compute_local_cdf(df, variables):
            Computes the empirical CDF for each variable separately.
        load_global_cdfs_pkl():
            Loads the saved empirical CDFs for multiple variables from a pickle file.
        quantile_transform(df, variables, global_cdfs):
            Applies quantile scaling to multiple variables using precomputed global CDFs.
        prepare_data(data_list):
            Prepares the data for training the streamflow prediction model.
        build_model_3_input_branches(loss_fn):
            Builds and compiles the streamflow prediction model using TCN and dense layers with three input branches.
        build_model_2_input_branches(loss_fn):
            Builds and compiles the streamflow prediction model using TCN and dense layers with two input branches.
        train_model():
            Trains the streamflow prediction model using the prepared data.
        load_regional_model():
            Loads a pre-trained regional model from a specified directory.
        """
        self.regional_model = None
        self.train_data_list = []
        self.timesteps = lookback
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.train_predictors = None
        self.train_response = None
        self.num_dynamic_features = 3
        self.num_static_features = 10
        self.scaled_trained_catchment = None
        self.working_dir = working_dir
        

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
        
        # Save as a pickle file
        with open(f'{self.working_dir}/models/global_cdfs.pkl', "wb") as f:
            pickle.dump(global_cdfs, f)

    def compute_local_cdf(self, df, variables):
        """
        Compute and save the empirical CDF for each variable separately as a pickle file.
    
        Args:
            df (pd.DataFrame): DataFrame containing multiple variables.
            variables (list): List of column names to apply quantile scaling.
            filename (str): File to save the computed CDFs.
        """
        transformed_df = pd.DataFrame(index=df.index)
    
        for var in variables:
            sorted_values = np.sort(df[var].dropna().values)  # Remove NaNs and sort
            quantiles = np.linspace(0, 1, len(sorted_values))  # Generate percentiles
            transformed_df[var] = np.interp(df[var], sorted_values, quantiles)
        return transformed_df

    def load_global_cdfs_pkl(self):
        """Load the saved empirical CDFs for multiple variables from a pickle file."""
        with open(f'{self.working_dir}/models/global_cdfs.pkl', "rb") as f:
            global_cdfs = pickle.load(f)
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
        Prepare the data for training the streamflow prediction model.

        Parameters
        ----------
        data_list : list
            A list containing tuples of predictors and responses, and an array of catchment data.

        Returns
        -------
        None
        """
        train_predictors = list(map(lambda xy: xy[0], data_list))
        train_response = list(map(lambda xy: xy[1], data_list))
        catchment = list(map(lambda xy: xy[2], data_list))
        train_catchment = np.array(catchment)
                
        full_train_predictors = []
        full_train_response = []
        train_catchment_list = []
        full_local_predictors = []
        
        catchment_scaler = MinMaxScaler()
        #train_catchment = train_catchment.reshape(-1,1)
        trained_catchment_scaler = catchment_scaler.fit(train_catchment)
        with open(f'{self.working_dir}/models/catchment_size_scaler_coarse.pkl', 'wb') as file:
            pickle.dump(trained_catchment_scaler, file)

        concatenated_predictors = pd.concat(train_predictors, axis=0)
        variables = ['mfd_wfa', 'scaled_acc', 'scaled_slp']  # Adjust as needed
        self.compute_global_cdfs_pkl(concatenated_predictors, variables)
        global_cdfs = self.load_global_cdfs_pkl()

        for x, y,z in zip(train_predictors, train_response, train_catchment):
            scaled_train_predictor = self.quantile_transform(x, variables, global_cdfs)
            scaled_train_predictor = scaled_train_predictor.values

            local_var = ['scaled_acc']
            local_predictor = self.compute_local_cdf(x, local_var)
            local_predictor = local_predictor.values

            scaled_train_response = y.values/1

            z2 = z.reshape(-1,self.num_static_features)
            scaled_train_catchment = trained_catchment_scaler.transform(z2)   
            
            # Calculate the 
            num_samples = scaled_train_predictor.shape[0] - self.timesteps - 1
            predictor_samples = []
            response_samples = []
            catchment_samples = []
            local_predictor_samples = []
            
            # Iterate over each batch
            for i in range(num_samples):
                # Slice the numpy array using the rolling window
                predictor_batch = scaled_train_predictor[i:i+self.timesteps, :]
                predictor_batch = predictor_batch.reshape(self.timesteps, self.num_dynamic_features)

                local_predictor_batch = local_predictor[i:i+self.timesteps, :]
                local_predictor_batch = local_predictor_batch.reshape(self.timesteps, 1)
                
                response_batch = scaled_train_response[i+self.timesteps]
                response_batch = response_batch.reshape(1)
                
                # Append the batch to the list
                predictor_samples.append(predictor_batch)
                local_predictor_samples.append(local_predictor_batch)
                response_samples.append(response_batch)

                catchment_samples.append(scaled_train_catchment)
            
            timesteps_to_keep = []
            for i in range(num_samples):
                if not np.isnan(predictor_samples[i]).any() and not np.isnan(response_samples[i]).any():
                    timesteps_to_keep.append(i)

            timesteps_to_keep = np.array(timesteps_to_keep, dtype=np.int64)
            scaled_train_predictor_filtered = np.array(predictor_samples)[timesteps_to_keep]
            scaled_train_response_filtered = np.array(response_samples)[timesteps_to_keep]
            scaled_train_catchment_filtered = np.array(catchment_samples)[timesteps_to_keep]
            train_local_predictor_filtered = np.array(local_predictor_samples)[timesteps_to_keep]
            
            full_train_predictors.append(scaled_train_predictor_filtered)
            full_train_response.append(scaled_train_response_filtered)
            train_catchment_list.append(scaled_train_catchment_filtered)
            full_local_predictors.append(train_local_predictor_filtered)
            
        self.train_predictors = np.concatenate(full_train_predictors, axis=0)
        self.train_response = np.concatenate(full_train_response, axis=0)
        self.train_local_predictors = np.concatenate(full_local_predictors, axis=0)
        self.train_catchment_size = np.concatenate(train_catchment_list, axis=0).reshape(-1, self.num_static_features)  
    
              
    def build_model_3_input_branches(self, loss_fn):
        """
        Build and compile the streamflow prediction model using TCN and dense layers.

        The model uses a TCN for the dynamic input and a dense network for the static input,
        then concatenates their outputs and passes them through additional dense layers.

        Returns
        -------
        None
        """
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            global_input = Input(shape=(self.timesteps, self.num_dynamic_features), name='global_input')
            local_input = Input(shape=(self.timesteps, 1), name='local_input')
            static_input = Input(shape=(self.num_static_features,), name="static_input")
    
            tcn_output = TCN(nb_filters = 128, kernel_size=3, dilations=(1,2,4,8,16,32, 64, 128, 256),
                             return_sequences=False)(global_input)
            tcn_output = BatchNormalization()(tcn_output)
            tcn_output = Dropout(0.4)(tcn_output)


            tcn_output1 = TCN(nb_filters = 128, kernel_size=3, dilations=(1,2,4,8,16,32, 64, 128, 256),
                             return_sequences=False)(local_input)
            tcn_output1 = BatchNormalization()(tcn_output1)
            tcn_output1 = Dropout(0.4)(tcn_output1)
        
            static_dense = Dense(32, activation="relu")(static_input)
            static_dense = BatchNormalization()(static_dense)
            static_dense = Dropout(0.4)(static_dense)
        
            # --- Enhanced FiLM Conditioning ---
            def enhanced_film_layer(static_input, feature_dim, hidden_dim=64):
                """
                Enhanced FiLM conditioning using a deeper MLP for gamma and beta modulation.
                
                Parameters:
                    static_input: Tensor (Static catchment descriptors)
                    feature_dim: int (Size of TCN output features)
                    hidden_dim: int (Size of hidden layers in FiLM MLP)
                
                Returns:
                    gamma, beta: Scaling and shifting parameters for FiLM conditioning
                """
                x = Dense(hidden_dim, activation="relu")(static_input)
                x = BatchNormalization()(x)
                x = Dense(hidden_dim, activation="relu")(x)
                x = BatchNormalization()(x)
        
                gamma = Dense(feature_dim, activation="sigmoid")(x)  # Scaling
                beta = Dense(feature_dim, activation="tanh")(x)  # Shifting
                return gamma, beta
        
            # Apply FiLM at multiple levels
            gamma1, beta1 = enhanced_film_layer(static_dense, feature_dim=128)
            tcn_output = Multiply()([tcn_output, gamma1])  # Scaling
            tcn_output = Add()([tcn_output, beta1])  # Shifting
        
            gamma2, beta2 = enhanced_film_layer(static_dense, feature_dim=128)
            tcn_output = Multiply()([tcn_output, gamma2])  # Additional FiLM layer
            tcn_output = Add()([tcn_output, beta2])
    
    
            # --- Attention Mechanism ---
            def attention_block(inputs):
                attention_scores = Dense(1, activation="softmax")(inputs)
                weighted_output = Multiply()([inputs, attention_scores])
                return weighted_output
    
            tcn_output = attention_block(tcn_output)
            tcn_output1 = attention_block(tcn_output1)
    
            # Merge all outputs
            merged_output = Concatenate()([tcn_output, tcn_output1, static_dense])
            merged_output = BatchNormalization()(merged_output)
    
            # Fully connected layers
            output3 = Dense(64, activation='relu')(merged_output)
            output3 = BatchNormalization()(output3)
            output3 = Dropout(0.4)(output3)
    
            output2 = Dense(32, activation='relu')(output3)
            output2 = BatchNormalization()(output2)
            output2 = Dropout(0.4)(output2)
    
            output1 = Dense(16)(output2)
            output1 = LeakyReLU(alpha=0.01)(output1) # LeakyReLU with alpha=0.01
    
            # Create the model
            
            if loss_fn == 'laplacian_nll':
                mu = Dense(1, name="mu")(output1)  # Mean prediction
                sigma = Dense(1, activation="softplus", name="sigma")(output1)  # Std dev (must be positive)
                output = Concatenate(name="streamflow_distribution")([mu, sigma])
                self.regional_model = Model(inputs=[global_input, local_input, static_input], outputs=output)
                self.regional_model.compile(optimizer='adam', loss=laplacian_nll)
            else:
                output = Dense(1)(output1)
                self.regional_model = Model(inputs=[global_input, local_input, static_input], outputs=output)
                self.regional_model.compile(optimizer='adam', loss='mean_squared_logarithmic_error')
            return self.regional_model


    def build_model_2_input_branches(self, loss_fn):
        """
        Build and compile the streamflow prediction model using TCN and dense layers.

        The model uses a TCN for the dynamic input and a dense network for the static input,
        then concatenates their outputs and passes them through additional dense layers.

        Returns
        -------
        None
        """
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            global_input = Input(shape=(self.timesteps, self.num_dynamic_features), name='global_input')
            static_input = Input(shape=(self.num_static_features,), name="static_input")
    
            tcn_output = TCN(nb_filters = 128, kernel_size=3, dilations=(1,2,4,8,16,32, 64, 128, 256),
                             return_sequences=False)(global_input)
            tcn_output = BatchNormalization()(tcn_output)
            tcn_output = Dropout(0.4)(tcn_output)

        
            static_dense = Dense(32, activation="relu")(static_input)
            static_dense = BatchNormalization()(static_dense)
            static_dense = Dropout(0.4)(static_dense)
        
            # --- Enhanced FiLM Conditioning ---
            def enhanced_film_layer(static_input, feature_dim, hidden_dim=64):
                """
                Enhanced FiLM conditioning using a deeper MLP for gamma and beta modulation.
                
                Parameters:
                    static_input: Tensor (Static catchment descriptors)
                    feature_dim: int (Size of TCN output features)
                    hidden_dim: int (Size of hidden layers in FiLM MLP)
                
                Returns:
                    gamma, beta: Scaling and shifting parameters for FiLM conditioning
                """
                x = Dense(hidden_dim, activation="relu")(static_input)
                x = BatchNormalization()(x)
                x = Dense(hidden_dim, activation="relu")(x)
                x = BatchNormalization()(x)
        
                gamma = Dense(feature_dim, activation="sigmoid")(x)  # Scaling
                beta = Dense(feature_dim, activation="tanh")(x)  # Shifting
                return gamma, beta
        
            # Apply FiLM at multiple levels
            gamma1, beta1 = enhanced_film_layer(static_dense, feature_dim=128)
            tcn_output = Multiply()([tcn_output, gamma1])  # Scaling
            tcn_output = Add()([tcn_output, beta1])  # Shifting
        
            gamma2, beta2 = enhanced_film_layer(static_dense, feature_dim=128)
            tcn_output = Multiply()([tcn_output, gamma2])  # Additional FiLM layer
            tcn_output = Add()([tcn_output, beta2])
    
    
            # --- Attention Mechanism ---
            def attention_block(inputs):
                attention_scores = Dense(1, activation="softmax")(inputs)
                weighted_output = Multiply()([inputs, attention_scores])
                return weighted_output
    
            tcn_output = attention_block(tcn_output)
            #tcn_output1 = attention_block(tcn_output1)
    
            # Merge all outputs
            merged_output = Concatenate()([tcn_output, static_dense])
            merged_output = BatchNormalization()(merged_output)
    
            # Fully connected layers
            output3 = Dense(64, activation='relu')(merged_output)
            output3 = BatchNormalization()(output3)
            output3 = Dropout(0.4)(output3)
    
            output2 = Dense(32, activation='relu')(output3)
            output2 = BatchNormalization()(output2)
            output2 = Dropout(0.4)(output2)
    
            output1 = Dense(16)(output2)
            output1 = LeakyReLU(alpha=0.01)(output1) # LeakyReLU with alpha=0.01
    
            # Create the model
            
            if loss_fn == 'laplacian_nll':
                mu = Dense(1, name="mu")(output1)  # Mean prediction
                sigma = Dense(1, activation="softplus", name="sigma")(output1)  # Std dev (must be positive)
                output = Concatenate(name="streamflow_distribution")([mu, sigma])
                self.regional_model = Model(inputs=[global_input, static_input], outputs=output)
                self.regional_model.compile(optimizer='adam', loss=laplacian_nll)
            else:
                output = Dense(1)(output1)
                self.regional_model = Model(inputs=[global_input,  static_input], outputs=output)
                self.regional_model.compile(optimizer='adam', loss='mean_squared_logarithmic_error')
            return self.regional_model


    
    def train_model(self, loss_fn, num_input_branch): 
        """
        Train the streamflow prediction model using the prepared training data.

        This method defines a checkpoint callback to save the best model based on the training loss,
        and then trains the model using the training predictors and responses.

        Returns
        -------
        None
        """
        # Define the checkpoint callback
        checkpoint_callback = ModelCheckpoint(filepath=f'{self.working_dir}/models/bakaano_model_{loss_fn}_{num_input_branch}_branches.keras', 
                                              save_best_only=True, monitor='loss', mode='min')

        if num_input_branch == 3:
            self.regional_model.fit(x=[self.train_predictors, self.train_local_predictors, self.train_catchment_size], y=self.train_response, 
                                    batch_size=self.batch_size, epochs=self.num_epochs, verbose=2, callbacks=[checkpoint_callback])
        else:
            self.regional_model.fit(x=[self.train_predictors, self.train_catchment_size], y=self.train_response, 
                                    batch_size=self.batch_size, epochs=self.num_epochs, verbose=2, callbacks=[checkpoint_callback])
        
    def load_regional_model(self, path, loss_fn):
        """
        Load a pre-trained regional model from the specified path.

        Parameters
        ----------
        path : str
            The path to the saved model file.

        Returns
        -------
        None
        """
        #self.regional_model = load_model(path)
       
        from tcn import TCN  # Make sure to import TCN
        from tensorflow.keras.utils import custom_object_scope # type: ignore

        if loss_fn == 'laplacian_nll':
            custom_objects = {"TCN": TCN, "laplacian_nll": laplacian_nll}
        else:
            custom_objects = {"TCN": TCN}
        
        strategy = tf.distribute.MirroredStrategy()
        
        with strategy.scope():
            with custom_object_scope(custom_objects):  
                self.regional_model = load_model(path, custom_objects=custom_objects)
        
    def regional_summary(self):
        """
        Print a summary of the regional model's architecture.
        
        This method prints out the layer names, output shapes, and number of parameters
        of the loaded regional model.
        
        Returns
        -------
        None
        """
        self.regional_model.summary()
        
