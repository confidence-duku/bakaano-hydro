"""Training pipeline for regional streamflow models.

Role: Build training datasets and train the TCN-based streamflow model.
"""

import os
import math
import glob
import pickle
import warnings
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import geopandas as gpd
import numpy as np
import pandas as pd
import pysheds.grid
import rasterio
import rioxarray
import tensorflow as tf
import xarray as xr
from keras.models import load_model  # type: ignore
from rasterio.transform import rowcol
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from tcn import TCN
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Concatenate,
    Dense,
    Dropout,
    Input,
    LeakyReLU,
    Multiply,
    Reshape,
)
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.utils import custom_object_scope, register_keras_serializable

mixed_precision.set_global_policy('mixed_bfloat16')
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

#=====================================================================================================================================


@tf.keras.utils.register_keras_serializable()
def asym_laplace_plus_mse_sqrt(
    y_true,
    params,
    scale_min=1e-4,
    mse_weight=0.05
):
    import tensorflow as tf

    y_true = tf.cast(y_true, tf.float32)
    params = tf.cast(params, tf.float32)

    mu = params[:, 0:1]
    log_b_plus = params[:, 1:2]
    log_b_minus = params[:, 2:3]

    b_plus = tf.maximum(tf.nn.softplus(log_b_plus), scale_min)
    b_minus = tf.maximum(tf.nn.softplus(log_b_minus), scale_min)

    r = y_true - mu

    ald = tf.where(
        r >= 0.0,
        tf.math.log(b_plus) + r / b_plus,
        tf.math.log(b_minus) - r / b_minus,
    )

    ald = tf.reduce_mean(ald)

    mse = tf.reduce_mean(tf.square(y_true - mu))

    total = ald + mse_weight * mse

    total = tf.where(
        tf.math.is_finite(total),
        total,
        tf.constant(1e6, dtype=tf.float32)
    )

    return total



@tf.keras.utils.register_keras_serializable()
def asym_laplace_nll(
    y_true,
    params,
    r_clip=5.0,
    scale_clip=(1e-3, 5.0),
    peak_weight=0.3
):
    import tensorflow as tf

    # Ensure float32 for stability
    y_true = tf.cast(y_true, tf.float32)
    params = tf.cast(params, tf.float32)

    mu = params[:, 0:1]
    log_b_plus = params[:, 1:2]
    log_b_minus = params[:, 2:3]

    # Positive scale
    b_plus = tf.nn.softplus(log_b_plus)
    b_minus = tf.nn.softplus(log_b_minus)

    # Clip scale in standardized space
    b_plus = tf.clip_by_value(b_plus, scale_clip[0], scale_clip[1])
    b_minus = tf.clip_by_value(b_minus, scale_clip[0], scale_clip[1])

    # Residual (standardized)
    r = y_true - mu
    r = tf.clip_by_value(r, -r_clip, r_clip)

    # Asymmetric Laplace NLL
    nll = tf.where(
        r >= 0.0,
        tf.math.log(b_plus) + r / b_plus,
        tf.math.log(b_minus) - r / b_minus,
    )

    # Peak emphasis in standardized space
    # Only weight positive extremes
    weights = 1.0 + peak_weight * tf.nn.relu(y_true)

    return tf.reduce_mean(weights * nll)


class DataPreprocessor:
    def __init__(self,  working_dir, study_area, grdc_streamflow_nc_file, train_start, 
                 train_end, routing_method, catchment_size_threshold):
        """
        Role: Build station-level predictors/responses for training.

        Initialize the DataPreprocessor with project details and dates.
        
        Args:
            working_dir (str): The parent working directory where files and outputs will be stored.
            study_area (str): The path to the shapefile defining the study area.
            grdc_streamflow_nc_file (str, optional): Path to GRDC streamflow NetCDF file.
            train_start (str): Training start date (YYYY-MM-DD).
            train_end (str): Training end date (YYYY-MM-DD).
            routing_method (str): Routing method ("mfd", "d8", "dinf").
            catchment_size_threshold (float): Minimum catchment size for stations.

        Methods
        -------
        __init__(working_dir, study_area, grdc_streamflow_nc_file, train_start, train_end, routing_method, catchment_size_threshold):
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
        self.sim_station_names = []
        self.train_start = train_start
        self.train_end = train_end
        self.grdc_subset = None
        self.station_ids = []
        if grdc_streamflow_nc_file is not None:
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
        Load and filter observed GRDC streamflow data in a schema-robust way.
        Works for single- and multi-station NetCDFs.

        Args:
            grdc_streamflow_nc_file (str): Path to GRDC NetCDF file.

        Returns:
            xarray.Dataset: Filtered GRDC subset for the study area.
        """
    
        try:
            grdc = self._open_grdc_dataset(grdc_streamflow_nc_file)
    
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
                time=slice(self.train_start, self.train_end)
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

    def _open_grdc_dataset(self, grdc_streamflow_nc_file):
        """Open GRDC NetCDF with backend fallback for Colab/Drive compatibility."""
        open_errors = []

        for engine in (None, "h5netcdf"):
            try:
                if engine is None:
                    return xr.open_dataset(grdc_streamflow_nc_file)
                return xr.open_dataset(grdc_streamflow_nc_file, engine=engine)
            except Exception as e:
                engine_name = "netcdf4(default)" if engine is None else engine
                open_errors.append(f"{engine_name}: {str(e)}")

        raise OSError(
            "Unable to open GRDC NetCDF with available backends. "
            "Install/enable a compatible backend (e.g., h5netcdf) or verify the file.\n"
            + "\n".join(open_errors)
        )

    
    def load_observed_streamflow_from_csv_dir(
        self,
        csv_dir,
        lookup_csv,
        id_col="id",
        lat_col="latitude",
        lon_col="longitude",
        date_col="date",
        discharge_col="discharge",
        file_pattern="{id}.csv",
    ):
        """
        Load observed streamflow from per-station CSV files using a lookup table.

        The lookup table must include station identifiers and coordinates. The method
        filters stations to the study area, then loads per-station CSVs by ID.

        Args:
            csv_dir (str): Directory containing per-station CSV files.
            lookup_csv (str): CSV file with station ids and coordinates.
            id_col (str): Station id column in lookup CSV.
            lat_col (str): Latitude column in lookup CSV.
            lon_col (str): Longitude column in lookup CSV.
            date_col (str): Date column in station CSVs.
            discharge_col (str): Discharge column in station CSVs.
            file_pattern (str): Pattern for station CSV filenames (e.g., ``"{id}.csv"``).

        Returns:
            dict: Mapping of station_id to observed discharge DataFrame.
        """
        lookup = pd.read_csv(lookup_csv)
        required_cols = [id_col, lat_col, lon_col]
        missing_cols = [c for c in required_cols if c not in lookup.columns]
        if missing_cols:
            raise SystemExit(
                "Lookup CSV is missing required columns: "
                + ", ".join(missing_cols)
            )

        stations_gdf = gpd.GeoDataFrame(
            lookup,
            geometry=gpd.points_from_xy(lookup[lon_col], lookup[lat_col]),
            crs="EPSG:4326",
        )
        region_shape = gpd.read_file(self.study_area)
        stations_in_region = gpd.sjoin(
            stations_gdf,
            region_shape,
            how="inner",
            predicate="intersects",
        )
        if stations_in_region.empty:
            raise SystemExit(
                "No stations from the lookup table intersect the study area."
            )

        station_ids = stations_in_region[id_col].astype(str).unique().tolist()
        self.station_ids = station_ids
        self.sim_station_names = station_ids
        self.station_meta = stations_in_region[[id_col, lat_col, lon_col]].copy()
        self.station_meta_cols = {"id": id_col, "lat": lat_col, "lon": lon_col}

        observed = {}
        missing_files = []
        for station_id in station_ids:
            pattern = file_pattern.format(id=station_id)
            matches = sorted(glob.glob(os.path.join(csv_dir, pattern)))
            if not matches:
                missing_files.append(station_id)
                continue
            df = pd.read_csv(matches[0])
            if date_col not in df.columns or discharge_col not in df.columns:
                raise SystemExit(
                    f"Missing columns in station CSV for id={station_id}. "
                    f"Required: {date_col}, {discharge_col}"
                )
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col).sort_index()
            df = df.loc[self.train_start:self.train_end]
            observed[station_id] = df[[discharge_col]].rename(
                columns={discharge_col: "station_discharge"}
            )

        if missing_files:
            raise SystemExit(
                "Missing observed CSV files for station ids: "
                + ", ".join(missing_files)
            )

        self.observed_streamflow_csv = observed
        return observed

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
        use_csv_obs = hasattr(self, "observed_streamflow_csv") and self.observed_streamflow_csv
        use_grdc = hasattr(self, "grdc_subset") and self.grdc_subset is not None
        
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
            station_discharge = None
            catchment_size = None

            if use_csv_obs:
                station_discharge = self.observed_streamflow_csv.get(str(k))
            elif use_grdc:
                station_discharge = (
                    self.grdc_subset['runoff_mean']
                    .sel(id=k)
                    .to_dataframe(name='station_discharge')
                )

            if station_discharge is None:
                continue

            # if catchment_size < self.catchment_size_threshold:
            #     continue
            
            # if station_discharge['station_discharge'].notna().sum() < 1095:
            #     continue
                          
            if use_csv_obs:
                meta = self.station_meta
                cols = self.station_meta_cols
                row = meta.loc[meta[cols["id"]].astype(str) == str(k)]
                if row.empty:
                    continue
                station_y = np.nanmax(row[cols["lat"]].values)
                station_x = np.nanmax(row[cols["lon"]].values)
            else:
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
            
            predictors = wfa_data.copy()
            predictors.replace([np.inf, -np.inf], np.nan, inplace=True)
            response = station_discharge
            if use_grdc and 'id' in response.columns:
                response = response.drop(['id'], axis=1)
            this_id = tuple([k])

            log_acc = np.log1p(acc_data)
            catch_list = [log_acc] + alpha_earth_stations
            catch_list = [float(x) for x in catch_list]
            predictors2 = predictors
            catch_tup = tuple(catch_list)
            self.catchment.append(catch_tup)
            self.data_list.append((predictors2, response, catch_tup, this_id))
            
            count = count + 1

        #basin_name = os.path.split(self.study_area)[1][:-4]
        with open(f'{self.working_dir}/models/predictor_response_data.pkl', 'wb') as file:
                pickle.dump(self.data_list, file)
            
        return self.data_list
#=====================================================================================================================================                          
#=====================================================================================================================================
class StreamflowModel:
    """
    Role: Define and train the multi-scale TCN streamflow model.

    Full-materialization training variant of the regional streamflow model.

    Key characteristics (actual behavior):
    - Prepares per-station scaled series using area normalization (optional).
    - Materializes all valid 365-day sliding windows in memory.
    - Trains directly with in-memory NumPy arrays.
    - Enables XLA globally via tf.config.optimizer.set_jit(True).
    """

    def __init__(self, working_dir, batch_size, num_epochs,
                 learning_rate=1e-4, loss_function="huber", train_start=None, train_end=None, seed=100,
                 area_normalize=True, lr_schedule=None, warmup_epochs=3, min_learning_rate=1e-5):
        """
        Initialize the full-materialization training model configuration.

        Parameters
        ----------
        working_dir : str
            Base directory for model artifacts.
        batch_size : int
            Batch size for training.
        num_epochs : int
            Number of training epochs.
        learning_rate : float
            Optimizer learning rate.
        loss_function : str or callable
            Loss used for model compilation.
        train_start : str
            Training start date (YYYY-MM-DD).
        train_end : str
            Training end date (YYYY-MM-DD).
        seed : int or None
            Random seed for reproducible sampling. If None, sampling is random.
        area_normalize : bool
            Whether to area-normalize predictors/response before model fitting.
        lr_schedule : str or None
            Learning-rate schedule ("cosine", "exp_decay", or None).
        warmup_epochs : int
            Number of warmup epochs before scheduling.
        min_learning_rate : float
            Minimum learning rate for schedules.
        """
        self.working_dir = working_dir
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.train_start = train_start
        self.train_end = train_end
        self.regional_model = None

        # training arrays
        self.train_45d = None
        self.train_90d = None
        self.train_180d = None
        self.train_365d = None
        self.train_response = None
        self.train_alphaearth = None
        self.train_area = None

        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.seed = seed
        self.area_normalize = area_normalize
        self.lr_schedule = lr_schedule
        self.warmup_epochs = int(warmup_epochs or 0)
        self.min_learning_rate = float(min_learning_rate)

        try:
            tf.config.optimizer.set_jit(True)
        except Exception:
            pass

    def _build_lr_callback(self):
        """Create a learning-rate schedule callback with optional warmup."""
        if not self.lr_schedule:
            return None

        base_lr = float(self.learning_rate)
        min_lr = float(self.min_learning_rate)
        warmup_epochs = max(0, int(self.warmup_epochs))
        schedule = str(self.lr_schedule).lower()

        def _lr_fn(epoch, lr):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return min_lr + (base_lr - min_lr) * (epoch + 1) / warmup_epochs
            t = epoch - warmup_epochs
            if schedule == "cosine":
                if self.num_epochs <= warmup_epochs:
                    return base_lr
                total = max(1, self.num_epochs - warmup_epochs)
                cos_inner = math.pi * min(t, total) / total
                return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(cos_inner))
            if schedule == "exp_decay":
                decay_rate = 0.95
                return max(min_lr, base_lr * (decay_rate ** t))
            return base_lr

        return tf.keras.callbacks.LearningRateScheduler(_lr_fn, verbose=0)

    # --------------------------------------------------
    # DATA PREPARATION (FULL MATERIALIZATION)
    # --------------------------------------------------
    def prepare_data(self, data_list):
        """
        Prepare the data for training the streamflow prediction model.

        This materializes all sliding windows (365),
        filters NaNs once, and concatenates across stations.
        """
        train_predictors = list(map(lambda xy: xy[0], data_list))
        train_response = list(map(lambda xy: xy[1], data_list))
        catchment = list(map(lambda xy: xy[2], data_list))
        catchment_arr = np.array(catchment, dtype=np.float32)

        area = catchment_arr[:, 0:1]
        alphaearth = catchment_arr[:, 1:]

        train_response = [
            df.loc[self.train_start:self.train_end]
            for df in train_response
        ]

        train_predictors = [
            df.loc[self.train_start:self.train_end]
            for df in train_predictors
        ]

        full_train_45d = []
        full_train_90d = []
        full_train_180d = []
        full_train_365d = []
        full_train_response = []
        full_alphaearth = []
        full_area = []

        scaler = StandardScaler()
        alphaearth_scaler = scaler.fit(alphaearth)
        with open(f"{self.working_dir}/models/alpha_earth_scaler.pkl", "wb") as file:
            pickle.dump(alphaearth_scaler, file)

        for x, y, z, j in zip(train_predictors, train_response, alphaearth, area):
            this_area = np.expm1(j)
            area_m2 = this_area * 1000000.0

            if self.area_normalize:
                scaled_train_predictor = x.values / this_area
            else:
                scaled_train_predictor = x.values

            if self.area_normalize:
                scaled_train_response = (y.values * 86400 * 1000) / area_m2
            else:
                scaled_train_response = y.values

            z2 = z.reshape(-1, 64)
            scaled_alphaearth = alphaearth_scaler.transform(z2)

            num_samples = scaled_train_predictor.shape[0] - 365 - 1
            if num_samples <= 0:
                continue

            p45_samples = []
            p90_samples = []
            p180_samples = []
            p365_samples = []
            response_samples = []
            alphaearth_samples = []
            area_samples = []

            for i in range(num_samples):
                full_window = scaled_train_predictor[i:i + 365, :]

                p45_samples.append(full_window[-45:, :])
                p90_samples.append(full_window[-90:, :])
                p180_samples.append(full_window[-180:, :])
                p365_samples.append(full_window)

                response_batch = scaled_train_response[i + 365].reshape(1)
                response_samples.append(response_batch)

                alphaearth_samples.append(scaled_alphaearth)
                area_samples.append(j.reshape(1))

            timesteps_to_keep = []
            for i in range(num_samples):
                if (
                    not np.isnan(p45_samples[i]).any()
                    and not np.isnan(p90_samples[i]).any()
                    and not np.isnan(p180_samples[i]).any()
                    and not np.isnan(p365_samples[i]).any()
                    and not np.isnan(response_samples[i]).any()
                ):
                    timesteps_to_keep.append(i)

            timesteps_to_keep = np.array(timesteps_to_keep, dtype=np.int64)

            if len(timesteps_to_keep) > 0:
                full_train_45d.append(np.array(p45_samples)[timesteps_to_keep])
                full_train_90d.append(np.array(p90_samples)[timesteps_to_keep])
                full_train_180d.append(np.array(p180_samples)[timesteps_to_keep])
                full_train_365d.append(np.array(p365_samples)[timesteps_to_keep])

                full_train_response.append(np.array(response_samples)[timesteps_to_keep])
                full_alphaearth.append(np.array(alphaearth_samples)[timesteps_to_keep])
                full_area.append(np.array(area_samples)[timesteps_to_keep])

        if not full_train_45d:
            raise ValueError("No valid training windows were created. Check date range and NaN coverage.")

        self.train_45d = np.concatenate(full_train_45d, axis=0).astype("float32")
        self.train_90d = np.concatenate(full_train_90d, axis=0).astype("float32")
        self.train_180d = np.concatenate(full_train_180d, axis=0).astype("float32")
        self.train_365d = np.concatenate(full_train_365d, axis=0).astype("float32")

        self.train_response = np.concatenate(full_train_response, axis=0).astype("float32")
        self.train_alphaearth = np.concatenate(full_alphaearth, axis=0).reshape(-1, 64).astype("float32")
        self.train_area = np.concatenate(full_area, axis=0).reshape(-1, 1).astype("float32")


    # --------------------------------------------------
    # MODEL DEFINITION
    # --------------------------------------------------
    def build_model(self):
        strategy = tf.distribute.MirroredStrategy()
        print(f"GPUs in sync: {strategy.num_replicas_in_sync}")

        with strategy.scope():
            custom_loss_names = {"asym_laplace_nll", "asym_laplace_plus_mse_sqrt"}
            loss_name = (
                self.loss_function.lower()
                if isinstance(self.loss_function, str)
                else getattr(self.loss_function, "__name__", "").lower()
            )
            uses_asym_laplace = (
                self.loss_function in {asym_laplace_nll, asym_laplace_plus_mse_sqrt}
                or loss_name in custom_loss_names
            )

            # -------------------------------------------------------
            # Inputs
            # -------------------------------------------------------
            in45 = Input((45, 1), name="input_45d")
            in90 = Input((90, 1), name="input_90d")
            in180 = Input((180, 1), name="input_180d")
            in365 = Input((365, 1), name="input_365d")
    
            in_alpha = Input((64,), name="alphaearth")
            in_area = Input((1,), name="area")
    
            # -------------------------------------------------------
            # Static conditioning branch
            # -------------------------------------------------------
            alpha_latent = Dense(64, activation="relu")(in_alpha)
            alpha_latent = BatchNormalization()(alpha_latent)
    
            area_latent = Dense(8, activation="relu")(in_area)
            area_latent = BatchNormalization()(area_latent)
    
            cond = Concatenate()([alpha_latent, area_latent])
    
            # -------------------------------------------------------
            # FiLM generator
            # -------------------------------------------------------
            def film(alpha, dim, gamma_scale=0.1, beta_scale=0.1):
                x = Dense(64, activation="relu")(alpha)
                x = Dense(64, activation="relu")(x)
    
                gamma = Dense(dim)(x)
                beta = Dense(dim)(x)
    
                gamma = 1.0 + gamma_scale * gamma
                beta = beta_scale * beta
    
                return gamma, beta
    
            # -------------------------------------------------------
            # TCN block
            # -------------------------------------------------------
            def tcn_block(x, filters, kernel, dilations, name):
    
                x = TCN(
                    nb_filters=filters,
                    kernel_size=kernel,
                    dilations=dilations,
                    return_sequences=False,
                    kernel_initializer=tf.keras.initializers.HeNormal(),
                    name=name,
                )(x)
    
                x = BatchNormalization()(x)
    
                return x
    
            # -------------------------------------------------------
            # Temporal branches
            # -------------------------------------------------------
            b1 = tcn_block(in45, 32, 3, (1,2,4,8,16), "tcn_45")
            b2 = tcn_block(in90, 32, 3, (1,2,4,8,16), "tcn_90")
            b3 = tcn_block(in180, 32, 5, (1,2,4,8,16,32), "tcn_180")
            b4 = tcn_block(in365, 64, 7, (1,2,4,8,16,32,64), "tcn_365")
    
            # -------------------------------------------------------
            # FiLM modulation per branch
            # -------------------------------------------------------
            g1, b_1 = film(cond, b1.shape[-1])
            g2, b_2 = film(cond, b2.shape[-1])
            g3, b_3 = film(cond, b3.shape[-1])
            g4, b_4 = film(cond, b4.shape[-1])
    
            b1 = g1 * b1 + b_1
            b2 = g2 * b2 + b_2
            b3 = g3 * b3 + b_3
            b4 = g4 * b4 + b_4
    
            b1 = BatchNormalization()(b1)
            b2 = BatchNormalization()(b2)
            b3 = BatchNormalization()(b3)
            b4 = BatchNormalization()(b4)
    
            # -------------------------------------------------------
            # Combine temporal features
            # -------------------------------------------------------
            temporal = Concatenate()([b1, b2, b3, b4])
            temporal = BatchNormalization()(temporal)
            temporal = Dropout(0.2)(temporal)
    
            # -------------------------------------------------------
            # Prediction head
            # -------------------------------------------------------
            h = Dense(128, activation="relu")(temporal)
            h = BatchNormalization()(h)
            h = Dropout(0.2)(h)
    
            h = Dense(64, activation="relu")(h)
            h = BatchNormalization()(h)
            h = Dropout(0.2)(h)
    
            h = Dense(32)(h)
            h = LeakyReLU(alpha=0.01)(h)

            out_dim = 3 if uses_asym_laplace else 1
            out = Dense(out_dim, name="streamflow")(h)

            # -------------------------------------------------------
            # Model
            # -------------------------------------------------------
            self.regional_model = Model(
                inputs=[in45, in90, in180, in365, in_alpha, in_area],
                outputs=out,
            )

            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate
            )

            self.regional_model.compile(
                optimizer=optimizer,
                loss=self.loss_function,
                jit_compile=False,
            )

            return self.regional_model

    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------
    def train_model(self):
        if self.regional_model is None:
            raise ValueError("Call build_model() before train_model().")

        if self.train_response is None:
            raise ValueError("Call prepare_data() before train_model().")

        checkpoint = ModelCheckpoint(
            filepath=f"{self.working_dir}/models/bakaano_model.keras",
            monitor="loss",
            save_best_only=True,
            mode="min",
        )

        early_stop = EarlyStopping(
            monitor="loss",
            patience=40,
            restore_best_weights=True,
        )

        callbacks = [checkpoint]
        lr_callback = self._build_lr_callback()
        if lr_callback:
            callbacks.append(lr_callback)
        callbacks.append(early_stop)

        self.regional_model.fit(
            x=[
                self.train_45d,
                self.train_90d,
                self.train_180d,
                self.train_365d,
                self.train_alphaearth,
                self.train_area,
            ],
            y=self.train_response,
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            callbacks=callbacks,
            verbose=2,
            shuffle=True,
        )

    def load_regional_model(self, path):
        """
        Load a previously saved regional model from disk.

        Parameters
        ----------
        path : str
            Path to the saved model file.

        Returns:
            tensorflow.keras.Model: Loaded model instance.
        """
        custom_objects = {
            "TCN": TCN,
            "asym_laplace_nll": asym_laplace_nll,
            "asym_laplace_plus_mse_sqrt": asym_laplace_plus_mse_sqrt,
        }
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            with custom_object_scope(custom_objects):
                self.regional_model = load_model(path, custom_objects=custom_objects)

    def regional_summary(self):
        """
        Print the Keras model summary.
        """
        if self.regional_model is None:
            raise ValueError("No model loaded/built yet.")
        self.regional_model.summary()
