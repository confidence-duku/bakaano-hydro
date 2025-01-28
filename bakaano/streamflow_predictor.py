
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import glob
import pysheds.grid
import rasterio
import rioxarray
from rasterio.transform import rowcol
from keras.models import load_model # type: ignore
import pickle
import warnings
import geopandas as gpd
from scipy.spatial.distance import cdist
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
#=====================================================================================================================================


class PredictDataPreprocessor:
    def __init__(self, working_dir,  study_area, start_date, end_date):
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
        self.grdc_subset = self.load_observed_streamflow()
        self.station_ids = np.unique(self.grdc_subset.to_dataframe().index.get_level_values('id'))
        self.data_list = []
        self.catchment = []  
        
    def _extract_station_rowcol(self, lat, lon):
        with rasterio.open(f'{self.working_dir}/elevation/dem_{self.working_dir}.tif') as src:
            data = src.read(1)
            transform = src.transform
            row, col = rowcol(transform, lon, lat)
            return row, col
        
    def _snap_coordinates(self, lat, lon):
        coordinate_to_snap=(lon, lat)
        with rasterio.open(f'{self.working_dir}/elevation/dem_{self.working_dir}.tif') as src:
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
        #grdc = xr.open_dataset('/lustre/backup/WUR/ESG/duku002/NBAT/hydro/input_data/GRDC-Daily-EU.nc')
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
            (grdc['time'] >= pd.to_datetime(self.start_date)) &
            (grdc['time'] <= pd.to_datetime(self.end_date)) &
            (grdc['station_name'].isin(overlapping_station_names)),
            drop=True
        )
        self.sim_station_names = np.unique(filtered_grdc['station_name'].values)
        return filtered_grdc
                          
    def get_data(self):

        count = 1
        
        slope = f'{self.working_dir}/elevation/slope_clipped.tif'
        dem_filepath = f'{self.working_dir}/elevation/dem_clipped.tif'
        
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
        all_years_wfa = sorted(glob.glob(f'{self.working_dir}/runoff_output/*.pkl'))
        wfa_list = []
        for year in all_years_wfa:
            with open(year, 'rb') as f:
                this_arr = pickle.load(f)
            wfa_list = wfa_list + this_arr
        
        all_wfa = []
        id_list = []
        for k in self.station_ids:
            station_discharge = self.grdc_subset['runoff_mean'].sel(id=k).to_dataframe(name='station_discharge')
            
            if station_discharge['station_discharge'].notna().sum() < 1825:
                continue
                          
            station_x = np.nanmax(self.grdc_subset['geo_x'].sel(id=k).values)
            station_y = np.nanmax(self.grdc_subset['geo_y'].sel(id=k).values)
            snapped_y, snapped_x = self._snap_coordinates(station_y, station_x)
            self.x = snapped_x
            self.y = snapped_y

            id_list.append(k)
            
            acc_data = acc.sel(lat=snapped_y, lon=snapped_x, method='nearest')
            slp_data = cum_slp.sel(lat=snapped_y, lon=snapped_x, method='nearest')
            acc_data = acc_data.values
            slp_data = slp_data.values
            
                
            row, col = self._extract_station_rowcol(snapped_y, snapped_x)
        
            station_wfa = []
            #col_name = f'mfd_wfa_{k}'
            for arr in wfa_list:
                arr = arr.tocsr()
                #station_wfa.append(arr[row, col])
                station_wfa.append(arr[int(row), int(col)])
            full_wfa_data = pd.DataFrame(station_wfa, columns=['mfd_wfa'])
            full_wfa_data.set_index(time_index, inplace=True)
            full_wfa_data.index.name = 'time'  # Rename the index to 'time'
            
            #extract wfa data based on defined training period
            wfa_data1 = full_wfa_data[self.start_date: self.end_date]
            wfa_data2 = wfa_data1/acc_data
            wfa_data2.rename(columns={'mfd_wfa': 'scaled_with_acc'}, inplace=True)
            wfa_data3 = wfa_data1 / slp_data
            wfa_data3.rename(columns={'mfd_wfa': 'scaled_with_slp'}, inplace=True)
            wfa_data4 = wfa_data1.join([wfa_data2])
            wfa_data = wfa_data4.join([wfa_data3])
            #wfa_data = wfa_data / acc_data
            all_wfa.append(wfa_data)

            station_discharge = self.grdc_subset['runoff_mean'].sel(id=k).to_dataframe(name='station_discharge')

            predictors = wfa_data.copy()
            predictors.replace([np.inf, -np.inf], np.nan, inplace=True)
            response = station_discharge.drop(['id'], axis=1)

            self.data_list.append((predictors, response))
            catch_list = [acc_data, slp_data]
            catch_tup = tuple(catch_list)
            self.catchment.append(catch_tup)  
            count = count + 1
            
        return [self.data_list, self.catchment, all_wfa, id_list]
    
    def get_data_latlng(self, olat, olon):

        count = 1
        
        slope = f'{self.working_dir}/elevation/slope_{self.working_dir}.tif'
        dem_filepath = f'{self.working_dir}/elevation/dem_{self.working_dir}.tif'
        
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
        all_years_wfa = sorted(glob.glob(f'{self.working_dir}/output_data/*.pkl'))
        wfa_list = []
        for year in all_years_wfa:
            with open(year, 'rb') as f:
                this_arr = pickle.load(f)
            wfa_list = wfa_list + this_arr
                

        snapped_y, snapped_x = self._snap_coordinates(olat, olon)
        #wfa_data = wfa.sel(lat=station_y, lon=station_x, method='nearest').to_dataframe(name='d8_weighted_flowacc')
        acc_data = acc.sel(lat=snapped_y, lon=snapped_x, method='nearest')
        slp_data = cum_slp.sel(lat=snapped_y, lon=snapped_x, method='nearest')
        acc_data = acc_data.values
        slp_data = slp_data.values
   
        row, col = self._extract_station_rowcol(snapped_y, snapped_x)

        station_wfa = []
        #col_name = f'mfd_wfa_{k}'
        for arr in wfa_list:
            arr = arr.tocsr()
            #station_wfa.append(arr[row, col])
            station_wfa.append(arr[int(row), int(col)])
        full_wfa_data = pd.DataFrame(station_wfa, columns=['mfd_wfa'])
        full_wfa_data.set_index(time_index, inplace=True)
        full_wfa_data.index.name = 'time'  # Rename the index to 'time'

        #extract wfa data based on defined training period
        wfa_data1 = full_wfa_data[self.start_date: self.end_date]
        wfa_data2 = wfa_data1/acc_data
        wfa_data2.rename(columns={'mfd_wfa': 'scaled_with_acc'}, inplace=True)
        wfa_data3 = wfa_data1 / slp_data
        wfa_data3.rename(columns={'mfd_wfa': 'scaled_with_slp'}, inplace=True)
        wfa_data4 = wfa_data1.join([wfa_data2])
        wfa_data = wfa_data4.join([wfa_data3])

        predictors = wfa_data.copy()
        predictors.replace([np.inf, -np.inf], np.nan, inplace=True)

        self.data_list.append(predictors)
        catch_list = [acc_data, slp_data]
        catch_tup = tuple(catch_list)
        self.catchment.append(catch_tup)
        count = count + 1
            
        return [self.data_list, self.catchment]

    
#=====================================================================================================================================
    
class PredictStreamflow:
    """
    A class for preprocessing flow accumulation data and/or observed streamflow data for prediction and comparison. Preprocessing involves data augmentation, remove missing data and breaking them into sequences of specified length.

    """
    def __init__(self, working_dir):
        self.regional_model = None
        self.train_data_list = []
        self.timesteps = 360
        self.num_epochs = 500
        self.batch_size = 64
        self.train_predictors = None
        self.train_response = None
        self.num_dynamic_features = 3
        self.num_static_features = 2
        self.scaled_trained_catchment = None
        self.working_dir = working_dir
    
    
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

        train_catchment = np.array(data_list[1])
        all_wfa = data_list[2]
                
        full_train_predictors = []
        train_catchment_list = []
        
        with open(f'{self.working_dir}/models/catchment_size_scaler_coarse.pkl', 'rb') as file:
            catchment_scaler = pickle.load(file)
        
        
        train_catchment = train_catchment.reshape(-1, self.num_static_features)
        scaled_trained_catchment = catchment_scaler.transform(train_catchment)
        
        with open(f'{self.working_dir}/models/global_predictor_scaler.pkl', 'rb') as file:
            scaler1 = pickle.load(file)


        for x, y, z, g in zip(predictors, response, scaled_trained_catchment, all_wfa):
            scaled_train_predictor = pd.DataFrame(scaler1.transform(x), columns=['mfd_wfa', 'scaled_acc', 'scaled_slp']).values 
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
        self.this_wfa = g.values
        self.catchment_size = np.concatenate(train_catchment_list, axis=0).reshape(-1,self.num_static_features)
        
    
    def prepare_data_latlng(self, data_list):
        
        """
        Prepare flow accumulation and streamflow data extracted from GRDC database for input in the model. Preparation involves dividing time-series data into desired short sequences based on specified timesteps and reshaping into desired tensor shape.
        
        Parameters:
        -----------
        data_list : Numpy array data 
            The extracted flow accumulation and observed streamflow data i.e. the output of get_grdc_data() functions.

        """

        predictors = data_list[0]

        train_predictors = predictors
        train_catchment = np.array(data_list[1])
        
                
        full_train_predictors = []
        train_catchment_list = []
        val_catchment_list = []
        
        with open(f'{self.working_dir}/models/catchment_size_scaler_coarse.pkl', 'rb') as file:
            catchment_scaler = pickle.load(file)
        
        train_catchment = train_catchment.reshape(-1, self.num_static_features)
        scaled_trained_catchment = catchment_scaler.transform(train_catchment)
        
        with open(f'{self.working_dir}/models/global_predictor_scaler.pkl', 'rb') as file:
            scaler1 = pickle.load(file)

        for x, z in zip(predictors, scaled_trained_catchment):
            pscaler = StandardScaler()
            scaled_train_predictor1 = pd.DataFrame(scaler1.transform(x), columns=['global'])
            scaled_train_predictor3 = pd.DataFrame(pscaler.fit_transform(x), columns=['independent'])
            scaled_train_predictor = scaled_train_predictor1.join([scaled_train_predictor3]).values
            
            # Calculate the 
            num_samples = scaled_train_predictor.shape[0] - self.timesteps
            predictor_samples = []
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
        
            
    def load_model(self, path):
        """
        Load saved LSTM model. 
        
        Parameters:
        -----------
        Path : H5 file
            Path to the saved neural network LSTM model

        """
        from tcn import TCN  # Make sure to import TCN
        from tf.keras.utils import custom_object_scope # type: ignore

        strategy = tf.distribute.MirroredStrategy()
        
        with strategy.scope():
            with custom_object_scope({'TCN': TCN}):
                #self.model = load_model(path, custom_objects={'weighted_loss': weighted_mse})
                self.model = load_model(path)
        
        #self.model = load_model(path)
        
    def summary(self):
        self.model.summary()