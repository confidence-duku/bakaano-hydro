
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Concatenate, Input, LeakyReLU # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
#=====================================================================================================================================

class DataPreprocessor:
    """
    A class used to preprocess data for training DeepSTRMM hydrological model.

    Attributes
    ----------
    study_area : str
        The file path to a shapefile of the study area.
    start_date : str
        The start date of the data period in 'YYYY-MM-DD' format.
    end_date : str
        The end date of the data period in 'YYYY-MM-DD' format.
    sim_start : str
        The start date of the training period in 'YYYY-MM-DD' format.
    sim_end : str
        The end date of the training period in 'YYYY-MM-DD' format.
    project_name : str
        The name of the project.
    times : pd.DatetimeIndex
        A range of dates from start_date to end_date.
    grdc_subset : pd.DataFrame
        A DataFrame containing the observed streamflow data.
    station_ids : np.ndarray
        An array of unique station IDs from the grdc_subset DataFrame.
    data_list : list
        A list to store preprocessed data.
    catchment : list
        A list to store catchment information.

    Methods
    -------
    __init__(project_name, study_area, start_date, end_date, sim_start, sim_end):
        Initializes the DataPreprocessor with project details and dates.
    load_observed_streamflow():
        Load observed streamflow data from GRDC Data. 
    _extract_station_rowcol(lat, lon):
        Transforms coordinates into row and column numbers.
    _snap_coordinates(lat, lon):
        Snaps given coordinates to the neares river network grid
    get_data():
        Extracts predictors for multiple stations
    """
    def __init__(self,  working_dir, study_area, start_date, end_date):
        """
        Initialize the DataPreprocessor with project details and dates.
        
        Parameters
        ----------
        project_name : str
            The name of the project.
        study_area : str
            The geographical area of the study.
        start_date : str
            The start date of the data period in 'YYYY-MM-DD' format.
        end_date : str
            The end date of the data period in 'YYYY-MM-DD' format.
        sim_start : str
            The start date of the simulation period in 'YYYY-MM-DD' format.
        sim_end : str
            The end date of the simulation period in 'YYYY-MM-DD' format.
        """
        
        self.study_area = study_area
        self.start_date = start_date
        self.end_date = end_date
        self.working_dir = working_dir
        #self.times = pd.date_range(start_date, end_date)
        self.grdc_subset = self.load_observed_streamflow()
        self.station_ids = np.unique(self.grdc_subset.to_dataframe().index.get_level_values('id'))
        self.data_list = []
        self.catchment = []    
        self.sim_station_names= []
    
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
            data = src.read(1)
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
        grdc = xr.open_dataset(grdc_streamflow_nc_file) #africa
        #grdc = xr.open_dataset('/lustre/backup/WUR/ESG/duku002/NBAT/hydro/input_data/GRDC-Daily-aus.nc')

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
            (grdc['time'] >= pd.to_datetime(self.start_date)) &
            (grdc['time'] <= pd.to_datetime(self.end_date)) &
            (grdc['station_name'].isin(overlapping_station_names)),
            drop=True
        )
        
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
        
        slope = f'{self.working_dir}/elevation/slope_clipped.tif'
        dem_filepath = f'{self.working_dir}/elevation/dem_clipped.tif'
        
        
        grid = pysheds.grid.Grid.from_raster(dem_filepath)
        dem = grid.read_raster(dem_filepath)
        
        flooded_dem = grid.fill_depressions(dem)
        inflated_dem = grid.resolve_flats(flooded_dem)
        fdir = grid.flowdir(inflated_dem, routing='mfd')
        acc = grid.accumulation(fdir=fdir, routing='mfd')
        
        facc_thresh = np.nanmax(acc) * 0.0001
        self.river_grid = np.where(acc < facc_thresh, 0, 1)
        
        weight2 = grid.read_raster(slope)
        cum_slp = grid.accumulation(fdir=fdir, weights=weight2, routing='mfd')
        
        latlng_ras = rioxarray.open_rasterio(dem_filepath)
        latlng_ras = latlng_ras.rio.write_crs(4326)
        lat = latlng_ras['y'].values
        lon = latlng_ras['x'].values
        
        acc = xr.DataArray(data=acc, coords=[('lat', lat), ('lon', lon)])
        cum_slp = xr.DataArray(data=cum_slp, coords=[('lat', lat), ('lon', lon)])
                
        #create time index for the station_wfa data extracted per station
        time_index = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        #combine or all yearly output from the runoff and routing module into a single list
        all_years_wfa = sorted(glob.glob(f'{self.working_dir}/runoff_output/*.pkl'))
        wfa_list = []
        for year in all_years_wfa:
            with open(year, 'rb') as f:
                this_arr = pickle.load(f)
            wfa_list = wfa_list + this_arr
        
        #extract station predictor and response variables based on station coordinates
        acc_list = []
        id_list = []
        for k in self.station_ids:
            station_discharge = self.grdc_subset['runoff_mean'].sel(id=k).to_dataframe(name='station_discharge')
            
            if station_discharge['station_discharge'].notna().sum() < 1825:
                continue
                          
            station_x = np.nanmax(self.grdc_subset['geo_x'].sel(id=k).values)
            station_y = np.nanmax(self.grdc_subset['geo_y'].sel(id=k).values)
            snapped_y, snapped_x = self._snap_coordinates(station_y, station_x)
            
            acc_data = acc.sel(lat=snapped_y, lon=snapped_x, method='nearest')
            slp_data = cum_slp.sel(lat=snapped_y, lon=snapped_x, method='nearest')
            acc_data = acc_data.values
            slp_data = slp_data.values
            
            # if acc_data > 5000:
            #     continue
                
            id_list.append(k)

            self.sim_station_names.append(list(self.grdc_subset['station_name'].sel(id=k).values)[0])
        
    
            row, col = self._extract_station_rowcol(snapped_y, snapped_x)
            
            col_name = f'mfd_wfa_{k}'
            station_wfa = []
            for arr in wfa_list:
                arr = arr.tocsr()
                #station_wfa.append(arr[row, col])
                station_wfa.append(arr[int(row), int(col)])
            full_wfa_data = pd.DataFrame(station_wfa, columns=['mfd_wfa'])
            full_wfa_data.set_index(time_index, inplace=True)
            full_wfa_data.index.name = 'time'  # Rename the index to 'time'
            
            wfa_data1 = full_wfa_data[self.start_date: self.end_date]
            wfa_data2 = wfa_data1/acc_data
            wfa_data2.rename(columns={'mfd_wfa': 'scaled_with_acc'}, inplace=True)
            wfa_data3 = wfa_data1 / slp_data
            wfa_data3.rename(columns={'mfd_wfa': 'scaled_with_slp'}, inplace=True)
            wfa_data4 = wfa_data1.join([wfa_data2])
            wfa_data = wfa_data4.join([wfa_data3])
            
            station_discharge = self.grdc_subset['runoff_mean'].sel(id=k).to_dataframe(name='station_discharge')

            predictors = wfa_data.copy()
            predictors.replace([np.inf, -np.inf], np.nan, inplace=True)
            response = station_discharge.drop(['id'], axis=1)

            self.data_list.append((predictors, response))
            catch_list = [acc_data, slp_data]
            catch_tup = tuple(catch_list)
            self.catchment.append(catch_tup)
            acc_list.append(acc_data)
            
            count = count + 1
            
        return [self.data_list, self.catchment, acc_list, id_list]
#=====================================================================================================================================                          

class StreamflowModel:
    """
    A class used to create and train a streamflow prediction model.

    Attributes
    ----------
    project_name : str
        The name of the project.
    regional_model : object, optional
        The regional model used for streamflow prediction.
    train_data_list : list
        A list to store the training data.
    timesteps : int
        The number of timesteps for the model input.
    num_epochs : int
        The number of epochs for training the model.
    batch_size : int
        The batch size for training the model.
    train_predictors : DataFrame, optional
        The predictors used for training the model.
    train_response : DataFrame, optional
        The response variables used for training the model.
    num_dynamic_features : int
        The number of dynamic features in the model.
    num_static_features : int
        The number of static features in the model.
    scaled_trained_catchment : object, optional
        The scaled catchment data used for training the model.
    """
    def __init__(self, working_dir):
        """
        Initialize the StreamflowModel with project details.

        Parameters
        ----------
        project_name : str
            The name of the project.
        """
        self.regional_model = None
        self.train_data_list = []
        self.timesteps = 360
        self.num_epochs = 500
        self.batch_size = 128
        self.train_predictors = None
        self.train_response = None
        self.num_dynamic_features = 3
        self.num_static_features = 2
        self.scaled_trained_catchment = None
        self.working_dir = working_dir
    
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
        predictors = list(map(lambda xy: xy[0], data_list[0]))
        response = list(map(lambda xy: xy[1], data_list[0]))

        train_predictors = predictors
        train_response = response
        train_catchment = np.array(data_list[1])
        weight_list = 1/np.array(data_list[2])
        weight_list = weight_list/np.sum(weight_list)
        id_list = np.array(data_list[3])
                
        full_train_predictors = []
        full_train_response = []
        train_catchment_list = []
        full_weights = []
        
        catchment_scaler = MinMaxScaler()
        rscaler = StandardScaler()
        
        trained_catchment_scaler = catchment_scaler.fit(train_catchment)
        with open(f'{self.working_dir}/models/catchment_size_scaler_coarse.pkl', 'wb') as file:
            pickle.dump(trained_catchment_scaler, file)

        concatenated_predictors = pd.concat(train_predictors, axis=0)
        
        scaler1 = rscaler.fit(concatenated_predictors)
        with open(f'{self.working_dir}/models/global_predictor_scaler.pkl', 'wb') as file:
            pickle.dump(scaler1, file)

        for x, y, z, w, i in zip(predictors, train_response, train_catchment, weight_list, id_list):
            
            scaled_train_predictor = pd.DataFrame(scaler1.transform(x), columns=['mfd_wfa', 'scaled_acc', 'scaled_slp']).values 
            
            scaled_train_response = y.values
           
            
            z2 = z.reshape(-1,self.num_static_features)
            scaled_train_catchment = trained_catchment_scaler.transform(z2)
            
            
            # Calculate the 
            num_samples = scaled_train_predictor.shape[0] - self.timesteps - 1
            predictor_samples = []
            response_samples = []
            catchment_samples = []
            weight_samples = []
            
            # Iterate over each batch
            for i in range(num_samples):
                # Slice the numpy array using the rolling window
                predictor_batch = scaled_train_predictor[i:i+self.timesteps, :]
                predictor_batch = predictor_batch.reshape(self.timesteps, self.num_dynamic_features)
                
                response_batch = scaled_train_response[i+self.timesteps]
                response_batch = response_batch.reshape(1)
                
                # Append the batch to the list
                predictor_samples.append(predictor_batch)
                response_samples.append(response_batch)
                
                catchment_samples.append(scaled_train_catchment)
                weight_samples.append(w)
            
            timesteps_to_keep = []
            for i in range(num_samples):
                if not np.isnan(predictor_samples[i]).any() and not np.isnan(response_samples[i]).any():
                    timesteps_to_keep.append(i)

            timesteps_to_keep = np.array(timesteps_to_keep, dtype=np.int64)
            scaled_train_predictor_filtered = np.array(predictor_samples)[timesteps_to_keep]
            scaled_train_response_filtered = np.array(response_samples)[timesteps_to_keep]
            scaled_train_catchment_filtered = np.array(catchment_samples)[timesteps_to_keep]
            filtered_weights = np.array(weight_samples)[timesteps_to_keep]
            
            full_train_predictors.append(scaled_train_predictor_filtered)
            full_train_response.append(scaled_train_response_filtered)
            train_catchment_list.append(scaled_train_catchment_filtered)
            full_weights.append(filtered_weights)
            #count = count + 1
            
        self.train_predictors = np.concatenate(full_train_predictors, axis=0)
        self.train_response = np.concatenate(full_train_response, axis=0)
        self.train_catchment_size = np.concatenate(train_catchment_list, axis=0).reshape(-1, self.num_static_features)   
        self.loss_weights = np.concatenate(full_weights, axis=0)
    
              
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
            dynamic_input = Input(shape=(self.timesteps, self.num_dynamic_features), name='dynamic_input')
            static_input = Input(shape=(self.num_static_features,), name='static_input')

            tcn_output = TCN(nb_filters = 128, kernel_size=3, dilations=(1,2,4,8,16,32, 64, 128, 256),
                             return_sequences=False)(dynamic_input)
            tcn_output = BatchNormalization()(tcn_output)
            tcn_output = Dropout(0.4)(tcn_output)

            dense_output = Dense(16, activation='relu')(static_input)
            dense_output = BatchNormalization()(dense_output)
            dense_output = Dropout(0.4)(dense_output)

            # Concatenate the LSTM and Dense outputs
            merged_output = Concatenate()([tcn_output,  dense_output])

            output1 = Dense(16)(merged_output)
            output1 = LeakyReLU(alpha=0.01)(output1)

            # Final output layer
            output = Dense(1)(output1)

            # Create the model
            self.regional_model = Model(inputs=[dynamic_input, static_input], outputs=output)

            # Compile the model
            #optimizer = Adam(learning_rate=0.00001)
            #self.regional_model.compile(optimizer='adam', loss=weighted_mse(self.loss_weights))
            self.regional_model.compile(optimizer='adam', loss='mean_squared_logarithmic_error')

    
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
        checkpoint_callback = ModelCheckpoint(filepath=f'{self.working_dir}/models/deepstrmm_model_tcn360.keras', 
                                              save_best_only=True, monitor='loss', mode='min')

        self.regional_model.fit(x=[self.train_predictors, self.train_catchment_size], y=self.train_response, batch_size=self.batch_size, 
                       epochs=self.num_epochs, verbose=2, callbacks=[checkpoint_callback])
        
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
        
        strategy = tf.distribute.MirroredStrategy()
        
        with strategy.scope():
            with custom_object_scope({'TCN': TCN}):
                self.regional_model = load_model(path)
        
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
        
