
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
from shapely.geometry import Point
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
        
    def _check_point_in_region(self, olat, olon):
        """
        Check whether a single (olat, olon) point lies within a study-area shapefile.
    
        - If NOT inside: raise SystemExit with a formatted, user-facing message
        - If inside: print confirmation and do nothing
        """
    
        # Load study-area shapefile
        try:
            region_gdf = gpd.read_file(self.study_area)
        except Exception as e:
            raise SystemExit(f"""
    ERROR: Failed to load study-area shapefile
    
    The study-area shapefile could not be read.
    
    File:
      {self.study_area}
    
    Original error:
      {str(e)}
    
    Please verify that the shapefile exists and is readable.
    """.strip())
    
        # Create point geometry
        point = gpd.GeoSeries(
            [Point(olon, olat)],
            crs="EPSG:4326"
        )
    
        # Ensure CRS match
        if region_gdf.crs != point.crs:
            region_gdf = region_gdf.to_crs(point.crs)
    
        # Spatial check
        inside = region_gdf.contains(point.iloc[0]).any()
    
        if not inside:
            raise SystemExit(f"""
    ERROR: Point outside study area
    
    The provided coordinates do not intersect the study area.
    
    Point location:
      latitude:  {olat}
      longitude: {olon}
    
    Study-area shapefile:
      {self.study_area}
    
    Please verify:
      - the input coordinates (EPSG:4326)
      - the spatial extent of the study area
      - that the point is not outside or on the boundary
    """.strip())
    
        # Confirmation message
        print(f"""
    INFO: Point accepted
    
    The point at:
      latitude:  {olat}
      longitude: {olon}
    
    lies within the study area.
    """.strip())

        
    
    def load_observed_streamflow(self, grdc_streamflow_nc_file):
        """
        Load and filter observed GRDC streamflow data in a schema-robust way.
        Works for single- and multi-station NetCDFs.
        """
    
        try:
            grdc = xr.open_dataset(grdc_streamflow_nc_file)
    
            # ---- 1. Sanity checks ----
            required_vars = ['runoff_mean', 'geo_x', 'geo_y', 'station_name']
            missing_vars = [v for v in required_vars if v not in grdc]
    
            if missing_vars:
                raise SystemExit(f"""
                    ERROR: Invalid GRDC NetCDF file
                    
                    The GRDC file is missing one or more required variables:
                    {", ".join(missing_vars)}
                    
                    Required variables are:
                    - runoff_mean
                    - geo_x
                    - geo_y
                    - station_name
                    
                    Please verify that the provided NetCDF file is a valid
                    GRDC daily discharge dataset.
                    """.strip())
    
            if 'id' not in grdc.dims:
                raise SystemExit(f"""
                    ERROR: Unsupported GRDC NetCDF format
                    
                    The GRDC dataset does not contain an 'id' dimension.
                    
                    This usually indicates a single-station GRDC file or a
                    non-standard export format.
                    
                    Please ensure the GRDC file is formatted with dimensions:
                    - time
                    - id
                    or preprocess the file to include an explicit station dimension.
                    """.strip())
    
            # ---- 2. Build station GeoDataFrame ----
            stations_df = pd.DataFrame({
                'id': grdc['id'].values,
                'station_name': grdc['station_name'].values,
                'geo_x': grdc['geo_x'].values,
                'geo_y': grdc['geo_y'].values,
            })
    
            stations_gdf = gpd.GeoDataFrame(
                stations_df,
                geometry=gpd.points_from_xy(stations_df['geo_x'], stations_df['geo_y']),
                crs="EPSG:4326"
            )
    
            # ---- 3. Spatial filtering ----
            region_shape = gpd.read_file(self.study_area)
    
            stations_in_region = gpd.sjoin(
                stations_gdf,
                region_shape,
                how='inner',
                predicate='intersects'
            )
    
            if stations_in_region.empty:
                raise SystemExit(f"""
                    ERROR: No GRDC stations found in study area
                    
                    None of the GRDC stations intersect the provided study area.
                    
                    Please check:
                    - the spatial extent of the study area shapefile
                    - the coordinate reference system (CRS)
                    - whether the GRDC stations fall within the selected region
                    """.strip())
    
            overlapping_ids = stations_in_region['id'].unique()
    
            # ---- 4. Dataset filtering ----
            filtered_grdc = grdc.sel(
                id=overlapping_ids,
                time=slice(self.sim_start, self.sim_end)
            )
    
            # ---- 5. Store metadata ----
            self.sim_station_names = filtered_grdc['station_name'].values.tolist()
            self.station_ids = filtered_grdc['id'].values.tolist()
    
            return filtered_grdc
    
        except SystemExit:
            # User-facing errors: re-raise cleanly
            raise
    
        except Exception as e:
            # Unexpected failure: add context, suppress traceback
            raise SystemExit(f"""
                ERROR: Failed to load GRDC streamflow data
                
                An unexpected error occurred while loading or filtering
                the GRDC streamflow dataset.
                
                This may indicate:
                - corrupted or unreadable NetCDF files
                - inconsistent dimensions or indexing
                - unexpected CRS or geometry issues
                
                Original error:
                {str(e)}
                
                Please verify the input data and try again.
                """.strip())

                          
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
        
        
        #combine or all yearly output from the runoff and routing module into a single list
        start_dt = datetime.strptime(self.sim_start, "%Y-%m-%d")
        end_dt = datetime.strptime(self.sim_end, "%Y-%m-%d")

        all_years_wfa = sorted(glob.glob(f'{self.working_dir}/runoff_output/*.pkl'))
        wfa_list = []
        for year in all_years_wfa:
            with open(year, 'rb') as f:
                this_arr = pickle.load(f)
            wfa_list = wfa_list + this_arr

        try:
            # --- Safety checks ---
            if not wfa_list:
                raise SystemExit(f"""
        ERROR: No routed runoff data found
        
        The routed runoff list is empty.
        
        This usually indicates that the runoff and routing modules
        have not been run yet, or that the expected output files
        could not be found.
        
        Please check the runoff_output directory and ensure that
        the runoff and routing steps completed successfully.
        """.strip())
        
            # --- Parse available times ---
            try:
                wfa_times = [
                    datetime.strptime(entry["time"], "%Y-%m-%d")
                    for entry in wfa_list
                ]
            except Exception:
                raise SystemExit(f"""
        ERROR: Invalid routed runoff metadata
        
        The routed runoff entries do not contain valid time information.
        
        Each entry is expected to include a 'time' field formatted as:
          YYYY-MM-DD
        
        Please verify the routed runoff output files and metadata.
        """.strip())
        
            available_start = min(wfa_times)
            available_end   = max(wfa_times)
        
            # --- Coverage check ---
            if start_dt < available_start or end_dt > available_end:
                raise SystemExit(f"""
        ERROR: Requested simulation period outside routed runoff coverage
        
        Requested simulation period:
          start: {start_dt.date()}
          end:   {end_dt.date()}
        
        Available routed runoff data:
          from:  {available_start.date()}
          to:    {available_end.date()}
        
        Please re-run the runoff and routing modules and ensure that
        the simulation period covers the intended training, validation,
        and other simulation periods.
        """.strip())
        
        except SystemExit:
            # User-facing errors: re-raise cleanly
            raise
        
        except Exception as e:
            # Unexpected failure
            raise SystemExit(f"""
        ERROR: Failed during routed runoff availability checks
        
        An unexpected error occurred while validating the routed
        runoff data against the requested simulation period.
        
        Original error:
          {str(e)}
        
        Please verify the runoff outputs and try again.
        """.strip())

        

        # Filter based on time range
        wfa_list = [
            entry for entry in wfa_list
            if start_dt <= datetime.strptime(entry["time"], "%Y-%m-%d") <= end_dt
        ]
        time_index = pd.date_range(start=self.sim_start, end=self.sim_end, freq='D')
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

        # basin_name = os.path.split(self.study_area)[1][:-4]
        # with open(f'{self.working_dir}/models/{basin_name}_predictor_response_data.pkl', 'wb') as file:
        #         pickle.dump(self.data_list, file)
            
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
        time_index = pd.date_range(start=self.sim_start, end=self.sim_end, freq='D')
        
        #combine or all yearly output from the runoff and routing module into a single list
        start_dt = datetime.strptime(self.sim_start, "%Y-%m-%d")
        end_dt = datetime.strptime(self.sim_end, "%Y-%m-%d")

        all_years_wfa = sorted(glob.glob(f'{self.working_dir}/runoff_output/*.pkl'))
        wfa_list = []
        for year in all_years_wfa:
            with open(year, 'rb') as f:
                this_arr = pickle.load(f)
            wfa_list = wfa_list + this_arr

        try:
            # --- Safety checks ---
            if not wfa_list:
                raise SystemExit(f"""
        ERROR: No routed runoff data found
        
        The routed runoff list is empty.
        
        This usually indicates that the runoff and routing modules
        have not been run yet, or that the expected output files
        could not be found.
        
        Please check the runoff_output directory and ensure that
        the runoff and routing steps completed successfully.
        """.strip())
        
            # --- Parse available times ---
            try:
                wfa_times = [
                    datetime.strptime(entry["time"], "%Y-%m-%d")
                    for entry in wfa_list
                ]
            except Exception:
                raise SystemExit(f"""
        ERROR: Invalid routed runoff metadata
        
        The routed runoff entries do not contain valid time information.
        
        Each entry is expected to include a 'time' field formatted as:
          YYYY-MM-DD
        
        Please verify the routed runoff output files and metadata.
        """.strip())
        
            available_start = min(wfa_times)
            available_end   = max(wfa_times)
        
            # --- Coverage check ---
            if start_dt < available_start or end_dt > available_end:
                raise SystemExit(f"""
        ERROR: Requested simulation period outside routed runoff coverage
        
        Requested simulation period:
          start: {start_dt.date()}
          end:   {end_dt.date()}
        
        Available routed runoff data:
          from:  {available_start.date()}
          to:    {available_end.date()}
        
        Please re-run the runoff and routing modules and ensure that
        the simulation period covers the intended training, validation,
        and other simulation periods.
        """.strip())
        
        except SystemExit:
            # User-facing errors: re-raise cleanly
            raise
        
        except Exception as e:
            # Unexpected failure
            raise SystemExit(f"""
        ERROR: Failed during routed runoff availability checks
        
        An unexpected error occurred while validating the routed
        runoff data against the requested simulation period.
        
        Original error:
          {str(e)}
        
        Please verify the runoff outputs and try again.
        """.strip()) 

        # Filter based on time range
        wfa_list = [
            entry for entry in wfa_list
            if start_dt <= datetime.strptime(entry["time"], "%Y-%m-%d") <= end_dt
        ]
        
        for olat, olon in zip(latlist, lonlist):
            self._check_point_in_region(olat, olon)
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


class PredictStreamflow:
    def __init__(self, working_dir):
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

        predictors = list(map(lambda xy: xy[0], data_list))
        catchment = list(map(lambda xy: xy[2], data_list))
        catchment_arr = np.array(catchment)

        area = catchment_arr[:, 0:1]      # shape (N, 1)
        alphaearth = catchment_arr[:, 1:] # shape (N, D)
     
        full_train_30d, full_train_90d = [], []
        full_train_180d, full_train_365d = [], []
        full_alphaearth = []
        full_catchsize = []

        with open(f'{self.working_dir}/models/alpha_earth_scaler.pkl', 'rb') as file:
            alphaearth_scaler = pickle.load(file)

        if len(catchment) <= 0:
            return

        alphaearth = alphaearth.reshape(-1,64)
        scaled_alphaearth = alphaearth_scaler.transform(alphaearth) 

        self.catch_area_list = []
        for x, z, j in zip(predictors, scaled_alphaearth, area):
            this_area = np.expm1(j)
            self.catch_area_list.append(this_area)
            scaled_train_predictor = np.log1p((x.values /this_area) + 0.001)

            num_samples = scaled_train_predictor.shape[0] - 365
            p30_samples, p90_samples, p180_samples, p365_samples = [], [], [], []
            area_samples = []
            alphaearth_samples = []

            self.catch_area = np.expm1(j)
            
            for i in range(num_samples):
                full_window = scaled_train_predictor[i : i + 365, :]
                
                # --- MULTI-SCALE SLICING ---
                # We slice from the END of the full_window so all branches share 'Day 365'
                p30_samples.append(full_window[-30:, :])
                p90_samples.append(full_window[-90:, :])
                p180_samples.append(full_window[-180:, :])
                p365_samples.append(full_window) # This is the full 365
        
                alphaearth_samples.append(z)
                area_samples.append(j)
            
            # --- FILER NAANS ---
            timesteps_to_keep = []
            for i in range(num_samples):
                if not np.isnan(p365_samples[i]).any():
                    timesteps_to_keep.append(i)

            timesteps_to_keep = np.array(timesteps_to_keep, dtype=np.int64)
            full_train_30d.append(np.array(p30_samples))
            full_train_90d.append(np.array(p90_samples))
            full_train_180d.append(np.array(p180_samples))
            full_train_365d.append(np.array(p365_samples))
            
            full_alphaearth.append(np.array(alphaearth_samples))
            full_catchsize.append(np.array(area_samples))
            
        self.sim_30d = np.concatenate(full_train_30d, axis=0)
        self.sim_90d = np.concatenate(full_train_90d, axis=0)
        self.sim_180d = np.concatenate(full_train_180d, axis=0)
        self.sim_365d = np.concatenate(full_train_365d, axis=0)
        
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
     
        full_train_30d, full_train_90d = [], []
        full_train_180d, full_train_365d = [], []
        full_alphaearth = []
        full_catchsize = [] 
                
        
        with open(f'{self.working_dir}/models/alpha_earth_scaler.pkl', 'rb') as file:
            alphaearth_scaler = pickle.load(file)

        if len(catchment) <= 0:
            return
        
        alphaearth = alphaearth.reshape(-1,64)
        scaled_alphaearth = alphaearth_scaler.transform(alphaearth) 

        self.catch_area_list = []
        for x, z, j in zip(predictors, scaled_alphaearth, area):
            this_area = np.expm1(j)
            self.catch_area_list.append(this_area)
            scaled_train_predictor = np.log1p((x.values /this_area) + 0.001)

            num_samples = scaled_train_predictor.shape[0] - 365
            p30_samples, p90_samples, p180_samples, p365_samples = [], [], [], []
            area_samples = []
            alphaearth_samples = []

            #self.catch_area = np.expm1(j)
            
            for i in range(num_samples):
                full_window = scaled_train_predictor[i : i + 365, :]
                
                # --- MULTI-SCALE SLICING ---
                # We slice from the END of the full_window so all branches share 'Day 365'
                p30_samples.append(full_window[-30:, :])
                p90_samples.append(full_window[-90:, :])
                p180_samples.append(full_window[-180:, :])
                p365_samples.append(full_window) # This is the full 365
        
                alphaearth_samples.append(z)
                area_samples.append(j)
            
            # --- FILER NAANS ---
            timesteps_to_keep = []
            for i in range(num_samples):
                if not np.isnan(p365_samples[i]).any():
                    timesteps_to_keep.append(i)

            timesteps_to_keep = np.array(timesteps_to_keep, dtype=np.int64)
            full_train_30d.append(np.array(p30_samples))
            full_train_90d.append(np.array(p90_samples))
            full_train_180d.append(np.array(p180_samples))
            full_train_365d.append(np.array(p365_samples))
            
            full_alphaearth.append(np.array(alphaearth_samples))
            full_catchsize.append(np.array(area_samples))
            
        self.sim_30d = np.concatenate(full_train_30d, axis=0)
        self.sim_90d = np.concatenate(full_train_90d, axis=0)
        self.sim_180d = np.concatenate(full_train_180d, axis=0)
        self.sim_365d = np.concatenate(full_train_365d, axis=0)
        
        self.sim_alphaearth = np.concatenate(full_alphaearth, axis=0).reshape(-1, 64)  
        self.sim_catchsize = np.concatenate(full_catchsize, axis=0).reshape(-1, 1)
    
            
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

        custom_objects = {"TCN": TCN}
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            with custom_object_scope(custom_objects):  
                self.model = load_model(path, custom_objects=custom_objects)
        
    def summary(self):
        self.model.summary()