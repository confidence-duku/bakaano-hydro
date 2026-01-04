
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Concatenate, Input, LeakyReLU, Multiply, Add, Reshape, Activation
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
from tensorflow.keras.utils import register_keras_serializable
from sklearn.preprocessing import StandardScaler
import glob
import pysheds.grid
import rasterio
import rioxarray
from rasterio.transform import rowcol
from tcn import TCN
from keras.models import load_model # type: ignore
import pickle
import warnings
from itertools import chain
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
        #self.sim_station_names= []
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
        self.sim_station_names = list(overlapping_station_names)
        return filtered_grdc
    
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
            catchment_size = self.grdc_subset['area'].sel(id=k, method='nearest').values

            # if catchment_size < self.catchment_size_threshold:
            #     continue
            
            # if station_discharge['station_discharge'].notna().sum() < 1095:
            #     continue
                          
            station_x = np.nanmax(self.grdc_subset['geo_x'].sel(id=k).values)
            station_y = np.nanmax(self.grdc_subset['geo_y'].sel(id=k).values)
            snapped_y, snapped_x = self._snap_coordinates(station_y, station_x)
            
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
    
            wfa_data = full_wfa_data
            
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

        basin_name = os.path.split(self.study_area)[1][:-4]
        with open(f'{self.working_dir}/models/{basin_name}_predictor_response_data.pkl', 'wb') as file:
                pickle.dump(self.data_list, file)
            
        return self.data_list
#=====================================================================================================================================                          
class StreamflowModel:
    
    def __init__(self, working_dir, batch_size, num_epochs, train_start, train_end):
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
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.train_predictors = None
        self.train_response = None
        self.scaled_trained_catchment = None
        self.working_dir = working_dir
        self.train_start = train_start
        self.train_end = train_end

    
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
        catchment_arr = np.array(catchment, dtype=np.float32)

        area = catchment_arr[:, 0:1]      # shape (N, 1)
        alphaearth = catchment_arr[:, 1:] # shape (N, D)

        train_response = [
            df.loc[self.train_start:self.train_end]
            for df in train_response
        ]

        train_predictors = [
            df.loc[self.train_start:self.train_end]
            for df in train_predictors
        ]
                
        full_train_30d, full_train_90d = [], []
        full_train_180d, full_train_365d = [], []
        full_train_response = []
        full_alphaearth = []
        full_catchsize = []
        
        scaler = StandardScaler()
        alphaearth_scaler = scaler.fit(alphaearth)
        with open(f'{self.working_dir}/models/alpha_earth_scaler.pkl', 'wb') as file:
            pickle.dump(alphaearth_scaler, file)

        for x, y,z,j in zip(train_predictors, train_response, alphaearth, area):
            this_area = np.expm1(j)
            scaled_train_predictor = np.log1p((x.values/this_area) + 0.001)
            scaled_train_response = np.log1p((y.values * 86400 * 1000/this_area) + 0.001)

            z2 = z.reshape(-1,64)
            scaled_alphaearth = alphaearth_scaler.transform(z2)   
            
            # Calculate the 
            num_samples = scaled_train_predictor.shape[0] - 365 - 1
            
            # Temporary lists for this specific basin
            p30_samples, p90_samples, p180_samples, p365_samples = [], [], [], []
            response_samples = []
            area_samples = []
            alphaearth_samples = []
            
            for i in range(num_samples):
                # Slice the 365-day MASTER window
                full_window = scaled_train_predictor[i : i + 365, :]
                
                # --- MULTI-SCALE SLICING ---
                # We slice from the END of the full_window so all branches share 'Day 365'
                p30_samples.append(full_window[-30:, :])
                p90_samples.append(full_window[-90:, :])
                p180_samples.append(full_window[-180:, :])
                p365_samples.append(full_window) # This is the full 365
                
                # Target value is the day immediately following the window
                response_batch = scaled_train_response[i + 365].reshape(1)
                response_samples.append(response_batch)
        
                alphaearth_samples.append(scaled_alphaearth)
                area_samples.append(j)
            
            # --- FILER NAANS ---
            timesteps_to_keep = []
            for i in range(num_samples):
                # Checking the 365d window automatically validates the 30/90/180 subsets
                if not np.isnan(p365_samples[i]).any() and not np.isnan(response_samples[i]).any():
                    timesteps_to_keep.append(i)
        
            timesteps_to_keep = np.array(timesteps_to_keep, dtype=np.int64)
        
            if len(timesteps_to_keep) > 0:
                # Filter and append this basin's data to the global lists
                full_train_30d.append(np.array(p30_samples)[timesteps_to_keep])
                full_train_90d.append(np.array(p90_samples)[timesteps_to_keep])
                full_train_180d.append(np.array(p180_samples)[timesteps_to_keep])
                full_train_365d.append(np.array(p365_samples)[timesteps_to_keep])
                
                full_train_response.append(np.array(response_samples)[timesteps_to_keep])
                full_alphaearth.append(np.array(alphaearth_samples)[timesteps_to_keep])
                full_catchsize.append(np.array(area_samples)[timesteps_to_keep])
        
        # --- FINAL CONCATENATION ---
        # These are the variables your model.fit() will use
        self.train_30d = np.concatenate(full_train_30d, axis=0)
        self.train_90d = np.concatenate(full_train_90d, axis=0)
        self.train_180d = np.concatenate(full_train_180d, axis=0)
        self.train_365d = np.concatenate(full_train_365d, axis=0)
        
        self.train_response = np.concatenate(full_train_response, axis=0)
        self.train_alphaearth = np.concatenate(full_alphaearth, axis=0).reshape(-1, 64)  
        self.train_catchsize = np.concatenate(full_catchsize, axis=0).reshape(-1, 1)
    

    def build_model(self):
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
    
            # --------------------------------------------------
            # 1. Temporal inputs
            # --------------------------------------------------
            input_30d  = Input((30,  self.num_dynamic_features), name="input_30d")
            input_90d  = Input((90,  self.num_dynamic_features), name="input_90d")
            input_180d = Input((180, self.num_dynamic_features), name="input_180d")
            input_365d = Input((365, self.num_dynamic_features), name="input_365d")
    
            alphaearth_input = Input((64,), name="alphaearth")
            area_input       = Input((1,),  name="catchment_area")  # log(area)
    
            # --------------------------------------------------
            # 2. Multi-timescale TCNs
            # --------------------------------------------------
            def tcn_block(x, filters, kernel, dilations, name):
                x = TCN(
                    nb_filters=filters,
                    kernel_size=kernel,
                    dilations=dilations,
                    return_sequences=False,
                    name=name
                )(x)
                return BatchNormalization()(x)
    
            t30  = tcn_block(input_30d,  32, 3, (1,2,4,8),        "tcn_30")
            t90  = tcn_block(input_90d,  32, 3, (1,2,4,8,16),     "tcn_90")
            t180 = tcn_block(input_180d, 32, 5, (1,2,4,8,16),     "tcn_180")
            t365 = tcn_block(input_365d, 64, 7, (1,2,4,8,16),     "tcn_365")
    
            merged_temporal = Concatenate()([t30, t90, t180, t365])
            merged_temporal = BatchNormalization()(merged_temporal)
            merged_temporal = Dropout(0.4)(merged_temporal)
    
            temporal_dim = 160
    
            # --------------------------------------------------
            # 3. AlphaEarth encoder (NEW)
            # --------------------------------------------------
            alpha_latent = Dense(64, activation="relu")(alphaearth_input)
            alpha_latent = BatchNormalization()(alpha_latent)
    
            # --------------------------------------------------
            # 4. FiLM conditioning (single, controlled)
            # --------------------------------------------------
            def film(alpha, dim, gamma_scale=0.1):
                x = Dense(64, activation="relu")(alpha)
                x = Dense(64, activation="relu")(x)
                gamma = Dense(dim)(x)
                beta  = Dense(dim)(x)
                gamma = 1.0 + gamma_scale * gamma
                return gamma, beta
    
            gamma, beta = film(alpha_latent, temporal_dim)
            h = gamma * merged_temporal + beta
    
            # --------------------------------------------------
            # 5. Base predictor (shape)
            # --------------------------------------------------
            h = Dense(64, activation="relu")(h)
            h = Dense(32, activation="relu")(h)
            y_base = Dense(1, activation=None)(h)
    
            # --------------------------------------------------
            # 6. Amplitude scaling (area only)
            # --------------------------------------------------
            scale = Dense(1, activation=None)(area_input)
            scale = Activation("exponential")(scale)
    
            y_hat = Multiply()([y_base, scale])
    
            # --------------------------------------------------
            # 7. Model
            # --------------------------------------------------
            model = Model(
                inputs=[
                    input_30d, input_90d, input_180d, input_365d,
                    alphaearth_input, area_input
                ],
                outputs=y_hat
            )
    
            model.compile(
                optimizer="adam",
                loss="msle"
            )
    
            self.regional_model = model
            return model

    
    
    def train_model(self): 
        """
        Train the streamflow prediction model using the prepared training data.

        This method defines a checkpoint callback to save the best model based on the training loss,
        and then trains the model using the training predictors and responses.

        Returns
        -------
        None
        """
        # Define the checkpoint callback
        checkpoint_callback = ModelCheckpoint(filepath=f'{self.working_dir}/models/bakaano_model.keras', 
                                              save_best_only=True, monitor='loss', mode='min')

        self.regional_model.fit(x=[self.train_30d, self.train_90d, self.train_180d, self.train_365d, self.train_alphaearth, 
                                   self.train_catchsize], y=self.train_response, 
                                batch_size=self.batch_size, epochs=self.num_epochs, verbose=2, callbacks=[checkpoint_callback])
        
    def load_regional_model(self, path):
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
        
