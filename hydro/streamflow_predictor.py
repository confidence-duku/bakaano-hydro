### Use only weighted flow accumulation of runoff, regular flow accumulation and data augmentation variables as inputs
### the model architecture comprises two separate inputs, one for dynamic inputs and the other for static inputs

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout, Reshape, Concatenate, Input, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import math
import glob
import pysheds.grid
import rasterio
import rioxarray
from rasterio.transform import rowcol
from tcn import TCN
from keras.models import load_model
import pickle
import warnings
from shapely.geometry import Point
import geopandas as gpd
from scipy.spatial.distance import cdist
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
#=====================================================================================================================================


class PredictDataPreprocessor:
    def __init__(self, project_name,  study_area):
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
        self.sim_start = sim_start
        self.sim_end = sim_end
        self.project_name = project_name
        #self.project_name = project_name
        self.times = pd.date_range(start_date, end_date)
        self.grdc_subset = self.load_observed_streamflow()
        self.station_ids = np.unique(self.grdc_subset.to_dataframe().index.get_level_values('id'))
        self.data_list = []
        self.catchment = []  
        
    def get_data(self):

        all_predictors = []
        all_responses = []
        count = 1
        
        slope = f'./{self.project_name}/elevation/slope_{self.project_name}.tif'
        dem_filepath = f'./{self.project_name}/elevation/dem_{self.project_name}.tif'
        
        grid = pysheds.grid.Grid.from_raster(dem_filepath)
        dem = grid.read_raster(dem_filepath)
        
        flooded_dem = grid.fill_depressions(dem)
        # Resolve flats
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
        all_years_wfa = sorted(glob.glob(f'/lustre/backup/WUR/ESG/duku002/NBAT/hydro/{self.project_name}/output_data/*.pkl'))
        wfa_list = []
        for year in all_years_wfa:
            with open(year, 'rb') as f:
                this_arr = pickle.load(f)
            wfa_list = wfa_list + this_arr
        
        
        for k in self.station_ids:
            station_discharge = self.grdc_subset['runoff_mean'].sel(id=k).to_dataframe(name='station_discharge')
            
            if station_discharge['station_discharge'].notna().sum() < 10:
                continue
                          
            station_x = np.nanmax(self.grdc_subset['geo_x'].sel(id=k).values)
            station_y = np.nanmax(self.grdc_subset['geo_y'].sel(id=k).values)
            
            #wfa_data = wfa.sel(lat=station_y, lon=station_x, method='nearest').to_dataframe(name='d8_weighted_flowacc')
            acc_data = acc.sel(lat=station_y, lon=station_x, method='nearest')
            slp_data = cum_slp.sel(lat=station_y, lon=station_x, method='nearest')
            #wfa_data = wfa_data.drop(['lat', 'lon'], axis=1)
            acc_data = acc_data.values
            slp_data = slp_data.values
                          
#             row, col = self._extract_station_wfa_data(station_lat, station_lon)
#             wfa_data = pd.DataFrame(wfa[:, row, col], columns=['mfd_wfa'])
#             acc_data = acc[row, col]
#             slp_data = cum_slp[row, col]
            #twi_data = w_twi[row, col]
    
            snapped_y, snapped_x = self._snap_coordinates(station_y, station_x)    
            row, col = self._extract_station_rowcol(snapped_y, snapped_x)
        
            station_wfa = []
            #col_name = f'mfd_wfa_{k}'
            for arr in wfa_list:
                arr = arr.tocsr()
                station_wfa.append(arr[row, col])
            full_wfa_data = pd.DataFrame(station_wfa, columns=['mfd_wfa'])
            full_wfa_data.set_index(time_index, inplace=True)
            full_wfa_data.index.name = 'time'  # Rename the index to 'time'
            
            #extract wfa data based on defined training period
            wfa_data = full_wfa_data[self.sim_start: self.sim_end]

            station_discharge = self.grdc_subset['runoff_mean'].sel(id=k).to_dataframe(name='station_discharge')

            predictors = wfa_data.copy()
            predictors.replace([np.inf, -np.inf], np.nan, inplace=True)
            response = station_discharge.drop(['id'], axis=1)

            self.data_list.append((predictors, response))
            #catch_descriptors = []
            self.catchment.append((acc_data, slp_data))
            #catch_descriptors = np.array(catch_descriptors)
            #self.catchment.append(catch_descriptors)   
            count = count + 1
            
        return [self.data_list, self.catchment]
#==========================================================================================================    
    def get_data_latlng(self, outlet_row_index, outlet_col_index):

        all_predictors = []
        all_responses = []
        count = 1
        
        slope = f'./projects/{self.project_name}/elevation/slope_{self.project_name}.tif'
        dem_filepath = f'./projects/{self.project_name}/elevation/dem_{self.project_name}.tif'
        
        grid = pysheds.grid.Grid.from_raster(dem_filepath)
        dem = grid.read_raster(dem_filepath)
        
        flooded_dem = grid.fill_depressions(dem)
        # Resolve flats
        inflated_dem = grid.resolve_flats(flooded_dem)
        fdir = grid.flowdir(inflated_dem, routing='mfd')
        acc = grid.accumulation(fdir=fdir, routing='mfd')
        
        facc_thresh = np.nanmax(acc) * 0.001
        self.river_grid = np.where(acc < facc_thresh, 0, 1)
        
        weight2 = grid.read_raster(slope)
        cum_slp = grid.accumulation(fdir=fdir, weights=weight2, routing='mfd')
        
        #create time index for the station_wfa data extracted per station
        #time_index = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        #wfa_data = wfa.sel(lat=station_y, lon=station_x, method='nearest').to_dataframe(name='d8_weighted_flowacc')
        acc_data = acc[outlet_row_index, outlet_col_index]
        slp_data = cum_slp[outlet_row_index, outlet_col_index]


        all_years_wfa = sorted(glob.glob(f'./projects/{self.project_name}/output_data/*.pkl'))
        wfa_list = []
        for year in all_years_wfa:
            with open(year, 'rb') as f:
                this_arr = pickle.load(f)
            wfa_list = wfa_list + this_arr

        station_wfa = []
        for arr in wfa_list:
            arr = arr.tocsr()
            station_wfa.append(arr[outlet_row_index, outlet_col_index])
        wfa_data = pd.DataFrame(station_wfa, columns=['mfd_wfa'])
        
        predictors = wfa_data.copy()
        predictors.replace([np.inf, -np.inf], np.nan, inplace=True)
        #response = station_discharge.drop(['id'], axis=1)

        self.data_list.append(predictors)
        self.catchment.append((acc_data, slp_data))   
        count = count + 1
            
        return [self.data_list, self.catchment]
    
#=====================================================================================================================================
                                    
class PredictStreamflow:
    """
    A class for preprocessing flow accumulation data and/or observed streamflow data for prediction and comparison. Preprocessing involves data augmentation, remove missing data and breaking them into sequences of specified length.

    """
    def __init__(self, project_name):
        self.regional_model = None
        self.train_data_list = []
        self.timesteps = 360
        self.num_epochs = 500
        self.batch_size = 64
        self.train_predictors = None
        self.train_response = None
        self.num_dynamic_features = 1
        self.num_static_features = 2
        self.scaled_trained_catchment = None
        self.project_name = project_name
        #self.project_name = project_name
    
    
    def prepare_data(self, data_list):
        
        """
        Prepare flow accumulation and streamflow data extracted from GRDC database for input in the model. Preparation involves dividing time-series data into desired short sequences based on specified timesteps and reshaping into desired tensor shape.
        
        Parameters:
        -----------
        data_list : Numpy array data 
            The extracted flow accumulation and observed streamflow data i.e. the output of get_grdc_data() functions.

        """

        predictors = list(map(lambda xy: xy[0], data_list[0]))
        response = list(map(lambda xy: xy[1], data_list[0]))

        train_predictors = predictors
        train_response = response
        train_catchment = np.array(data_list[1])
                
        full_train_predictors = []
        full_train_response = []
        full_val_predictors = []
        full_val_response = []
        train_catchment_list = []
        val_catchment_list = []
        
        #catchment_scaler = MinMaxScaler()
        #pscaler = StandardScaler()
        
        #catchment_scaler = MinMaxScaler()
        with open(f'./projects/{self.project_name}/models/catchment_size_scaler_coarse.pkl', 'rb') as file:
            catchment_scaler = pickle.load(file)
        
        #catchment_num = catchment_num.reshape(-1,self.num_static_features)
        
        #trained_catchment_scaler = catchment_scaler.fit(catchment_num)
        train_catchment = train_catchment.reshape(-1, self.num_static_features)
        scaled_trained_catchment = catchment_scaler.transform(train_catchment)
        
        with open(f'./projects/{self.project_name}/models/global_predictor_scaler.pkl', 'rb') as file:
            scaler1 = pickle.load(file)
            #pickle.dump(scaler1, file)

        for x, y, z in zip(predictors, response, scaled_trained_catchment):
            
            scaled_train_predictor = pd.DataFrame(scaler1.transform(x), columns=['global']).values
            
            # Calculate the 
            num_samples = scaled_train_predictor.shape[0] - self.timesteps
            predictor_samples = []
            #response_samples = []
            catchment_samples = []
            
            # Iterate over each batch
            for i in range(num_samples):
                # Slice the numpy array using the rolling window
                predictor_batch = scaled_train_predictor[i:i+self.timesteps, :]
                predictor_batch = predictor_batch.reshape(self.timesteps, self.num_dynamic_features)
                
                #response_batch = scaled_train_response[i+self.timesteps]
                #response_batch = response_batch.reshape(1)
                
                # Append the batch to the list
                predictor_samples.append(predictor_batch)
                #response_samples.append(response_batch)
                
                catchment_samples.append(z)
            
            timesteps_to_keep = []
            for i in range(num_samples):
                if not np.isnan(predictor_samples[i]).any():
                    timesteps_to_keep.append(i)

            timesteps_to_keep = np.array(timesteps_to_keep, dtype=np.int64)
            scaled_train_predictor_filtered = np.array(predictor_samples)
            #scaled_train_response_filtered = np.array(response_samples)
            scaled_train_catchment_filtered = np.array(catchment_samples)
            
            full_train_predictors.append(scaled_train_predictor_filtered)
            #full_train_response.append(scaled_train_response_filtered)
            train_catchment_list.append(scaled_train_catchment_filtered)
            
        self.predictors = np.concatenate(full_train_predictors, axis=0)
        #self.response = np.concatenate(full_train_response, axis=0)
        self.catchment_size = np.concatenate(train_catchment_list, axis=0).reshape(-1,self.num_static_features)
        
#=========================================================================================================    
    def prepare_data_latlng(self, data_list):
        
        """
        Prepare flow accumulation and streamflow data extracted from GRDC database for input in the model. Preparation involves dividing time-series data into desired short sequences based on specified timesteps and reshaping into desired tensor shape.
        
        Parameters:
        -----------
        data_list : Numpy array data 
            The extracted flow accumulation and observed streamflow data i.e. the output of get_grdc_data() functions.

        """

        predictors = data_list[0]
        #response = list(map(lambda xy: xy[1], data_list[0]))

        train_predictors = predictors
        train_catchment = np.array(data_list[1])
                
        full_train_predictors = []
        train_catchment_list = []
        val_catchment_list = []
        
        with open(f'./projects/{self.project_name}/models/catchment_size_scaler_coarse.pkl', 'rb') as file:
            catchment_scaler = pickle.load(file)
        
        train_catchment = train_catchment.reshape(-1, self.num_static_features)
        scaled_trained_catchment = catchment_scaler.transform(train_catchment)
        
        with open(f'./projects/{self.project_name}/models/global_predictor_scaler.pkl', 'rb') as file:
            scaler1 = pickle.load(file)

        for x, z in zip(predictors, scaled_trained_catchment):
            
            scaled_train_predictor = pd.DataFrame(scaler1.transform(x), columns=['global']).values
            
            # Calculate the 
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
                #response_samples.append(response_batch)
                
                catchment_samples.append(z)
            
            timesteps_to_keep = []
            for i in range(num_samples):
                if not np.isnan(predictor_samples[i]).any():
                    timesteps_to_keep.append(i)

            timesteps_to_keep = np.array(timesteps_to_keep, dtype=np.int64)
            scaled_train_predictor_filtered = np.array(predictor_samples)
            #scaled_train_response_filtered = np.array(response_samples)
            scaled_train_catchment_filtered = np.array(catchment_samples)
            
            full_train_predictors.append(scaled_train_predictor_filtered)
            #full_train_response.append(scaled_train_response_filtered)
            train_catchment_list.append(scaled_train_catchment_filtered)
            
        self.predictors = np.concatenate(full_train_predictors, axis=0)
        self.catchment_size = np.concatenate(train_catchment_list, axis=0).reshape(-1,self.num_static_features)
        
#===============================================================================================================            
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

        strategy = tf.distribute.MirroredStrategy()
        
        with strategy.scope():
            with custom_object_scope({'TCN': TCN}):
                self.model = load_model(path)
        
        #self.model = load_model(path)
        
    def summary(self):
        self.model.summary()