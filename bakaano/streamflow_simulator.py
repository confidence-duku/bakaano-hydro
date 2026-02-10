"""Simulation and inference utilities for streamflow prediction.

Role: Prepare simulation inputs and run trained model inference.
"""

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

def _open_dataset_with_fallback(nc_path):
    """Open NetCDF with backend fallback for Colab/Drive compatibility."""
    open_errors = []
    for engine in (None, "h5netcdf"):
        try:
            if engine is None:
                return xr.open_dataset(nc_path)
            return xr.open_dataset(nc_path, engine=engine)
        except Exception as e:
            name = "netcdf4(default)" if engine is None else engine
            open_errors.append(f"{name}: {str(e)}")

    raise OSError(
        "Unable to open NetCDF with available backends.\n" + "\n".join(open_errors)
    )


class PredictDataPreprocessor:
    def __init__(self, working_dir,  study_area,  sim_start, sim_end, routing_method, 
                 grdc_streamflow_nc_file=None, catchment_size_threshold=None):
        """
        Role: Build predictors for simulation/inference.

        Initialize the PredictDataPreprocessor object.
        
        Args:
            working_dir (str): The parent working directory where files and outputs will be stored.
            study_area (str): The path to the shapefile of the river basin or watershed.
            sim_start (str): Simulation start date (YYYY-MM-DD).
            sim_end (str): Simulation end date (YYYY-MM-DD).
            routing_method (str): Routing method ("mfd", "d8", "dinf").
            grdc_streamflow_nc_file (str, optional): GRDC NetCDF path.
            catchment_size_threshold (float, optional): Minimum catchment size for stations.

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

        Args:
            grdc_streamflow_nc_file (str): Path to GRDC NetCDF file.

        Returns:
            xarray.Dataset: Filtered GRDC subset for the study area.
        """
    
        try:
            grdc = _open_dataset_with_fallback(grdc_streamflow_nc_file)
    
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
            df = df.loc[self.sim_start:self.sim_end]
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
        
        river_grid_path = f'{self.working_dir}/catchment/river_grid.tif'
        if os.path.exists(river_grid_path):
            with rasterio.open(river_grid_path) as src:
                self.river_grid = src.read(1)
        else:
            facc_thresh = np.nanmax(acc) * 0.0001
            self.river_grid = np.where(acc < facc_thresh, 0, 1)
            river_ras = xr.DataArray(data=self.river_grid, coords=[('lat', lat), ('lon', lon)])
            with rasterio.open(dem_filepath) as src:
                ref_meta = src.meta.copy()  # Copy the metadata exactly as is
            with rasterio.open(river_grid_path, 'w', **ref_meta) as dst:
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
        """Prepare predictors for arbitrary latitude/longitude points.

        Args:
            latlist (list[float]): Latitudes to simulate.
            lonlist (list[float]): Longitudes to simulate.

        Returns:
            list: [data_list, catchment, latlist, lonlist].
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
        
        river_grid_path = f'{self.working_dir}/catchment/river_grid.tif'
        if os.path.exists(river_grid_path):
            with rasterio.open(river_grid_path) as src:
                self.river_grid = src.read(1)
        else:
            facc_thresh = np.nanmax(acc) * 0.0001
            self.river_grid = np.where(acc < facc_thresh, 0, 1)
            river_ras = xr.DataArray(data=self.river_grid, coords=[('lat', lat), ('lon', lon)])
            with rasterio.open(dem_filepath) as src:
                ref_meta = src.meta.copy()  # Copy the metadata exactly as is
            with rasterio.open(river_grid_path, 'w', **ref_meta) as dst:
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
    def __init__(self, working_dir, area_normalize=True):
        """
        Role: Prepare model inputs and run inference.

        Initializes the PredictStreamflow class for streamflow prediction using a temporal convolutional network (TCN).

        Args:
            working_dir (str): The working directory where the model and data are stored.
            area_normalize (bool): Whether to area-normalize predictors/response.

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
        self.area_normalize = area_normalize

    def prepare_data(self, data_list):
        
        """
        Prepare flow accumulation and streamflow data extracted from GRDC database for input in the model. Preparation involves dividing time-series data into desired short sequences based on specified timesteps and reshaping into desired tensor shape.
        
        Parameters:
        -----------
        data_list : Numpy array data 
            The extracted flow accumulation and observed streamflow data i.e. the output of get_grdc_data() functions.

        Returns:
            None. Populates model input arrays on the instance.
        """

        predictors = list(map(lambda xy: xy[0], data_list))
        catchment = list(map(lambda xy: xy[2], data_list))
        catchment_arr = np.array(catchment)

        area = catchment_arr[:, 0:1]      # shape (N, 1)
        alphaearth = catchment_arr[:, 1:] # shape (N, D)
     
        full_train_45d = []
        full_train_90d = []
        full_train_180d = []
        full_train_365d = []
        full_alphaearth = []
        full_area = []

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
            if self.area_normalize:
                scaled_train_predictor = x.values / this_area
            else:
                scaled_train_predictor = x.values
            scaled_train_predictor = np.log1p(scaled_train_predictor)

            num_samples = scaled_train_predictor.shape[0] - 365
            p45_samples = []
            p90_samples = []
            p180_samples = []
            p365_samples = []
            alphaearth_samples = []
            area_samples = []

            self.catch_area = np.expm1(j)
            
            for i in range(num_samples):
                full_window = scaled_train_predictor[i : i + 365, :]
                
                p45_samples.append(full_window[-45:, :])
                p90_samples.append(full_window[-90:, :])
                p180_samples.append(full_window[-180:, :])
                p365_samples.append(full_window)
        
                alphaearth_samples.append(z)
                area_samples.append(j.reshape(1))
            
            # --- FILER NAANS ---
            timesteps_to_keep = []
            for i in range(num_samples):
                if (
                    not np.isnan(p45_samples[i]).any()
                    and not np.isnan(p90_samples[i]).any()
                    and not np.isnan(p180_samples[i]).any()
                    and not np.isnan(p365_samples[i]).any()
                ):
                    timesteps_to_keep.append(i)

            timesteps_to_keep = np.array(timesteps_to_keep, dtype=np.int64)
            if len(timesteps_to_keep) > 0:
                full_train_45d.append(np.array(p45_samples)[timesteps_to_keep])
                full_train_90d.append(np.array(p90_samples)[timesteps_to_keep])
                full_train_180d.append(np.array(p180_samples)[timesteps_to_keep])
                full_train_365d.append(np.array(p365_samples)[timesteps_to_keep])
                full_alphaearth.append(np.array(alphaearth_samples)[timesteps_to_keep])
                full_area.append(np.array(area_samples)[timesteps_to_keep])
            
        self.sim_45d = np.concatenate(full_train_45d, axis=0)
        self.sim_90d = np.concatenate(full_train_90d, axis=0)
        self.sim_180d = np.concatenate(full_train_180d, axis=0)
        self.sim_365d = np.concatenate(full_train_365d, axis=0)
        self.sim_alphaearth = np.concatenate(full_alphaearth, axis=0).reshape(-1, 64)  
        self.sim_area = np.concatenate(full_area, axis=0).reshape(-1, 1)
    
    def prepare_data_latlng(self, data_list):
        
        """
        Prepare model inputs for user-defined latitude/longitude points.

        This uses routed runoff time series at specified lat/lon points (not GRDC
        stations), slices multi-scale windows, and reshapes tensors for inference.
        
        Parameters:
        -----------
        data_list : Numpy array data 
            Output of get_data_latlng(), containing predictors and catchment info.

        Returns:
            None. Populates model input arrays on the instance.
        """

        predictors = list(map(lambda xy: xy[0], data_list[0]))
        catchment = list(map(lambda xy: xy[1], data_list[0]))
        catchment_arr = np.array(catchment)

        area = catchment_arr[:, 0:1]      # shape (N, 1)
        alphaearth = catchment_arr[:, 1:] # shape (N, D)
     
        full_train_45d = []
        full_train_90d = []
        full_train_180d = []
        full_train_365d = []
        full_alphaearth = []
        full_area = []
                
        
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
            if self.area_normalize:
                scaled_train_predictor = x.values / this_area
            else:
                scaled_train_predictor = x.values
            if scaled_train_predictor.ndim == 1:
                scaled_train_predictor = scaled_train_predictor.reshape(-1, 1)
            scaled_train_predictor = np.log1p(scaled_train_predictor)

            num_samples = scaled_train_predictor.shape[0] - 365
            p45_samples = []
            p90_samples = []
            p180_samples = []
            p365_samples = []
            alphaearth_samples = []
            area_samples = []

            #self.catch_area = np.expm1(j)
            
            for i in range(num_samples):
                full_window = scaled_train_predictor[i : i + 365, :]
                
                p45_samples.append(full_window[-45:, :])
                p90_samples.append(full_window[-90:, :])
                p180_samples.append(full_window[-180:, :])
                p365_samples.append(full_window)
        
                alphaearth_samples.append(z)
                area_samples.append(j.reshape(1))
            
            # --- FILER NAANS ---
            timesteps_to_keep = []
            for i in range(num_samples):
                if (
                    not np.isnan(p45_samples[i]).any()
                    and not np.isnan(p90_samples[i]).any()
                    and not np.isnan(p180_samples[i]).any()
                    and not np.isnan(p365_samples[i]).any()
                ):
                    timesteps_to_keep.append(i)

            timesteps_to_keep = np.array(timesteps_to_keep, dtype=np.int64)
            if len(timesteps_to_keep) > 0:
                full_train_45d.append(np.array(p45_samples)[timesteps_to_keep])
                full_train_90d.append(np.array(p90_samples)[timesteps_to_keep])
                full_train_180d.append(np.array(p180_samples)[timesteps_to_keep])
                full_train_365d.append(np.array(p365_samples)[timesteps_to_keep])
                full_alphaearth.append(np.array(alphaearth_samples)[timesteps_to_keep])
                full_area.append(np.array(area_samples)[timesteps_to_keep])
            
        self.sim_45d = np.concatenate(full_train_45d, axis=0)
        self.sim_90d = np.concatenate(full_train_90d, axis=0)
        self.sim_180d = np.concatenate(full_train_180d, axis=0)
        self.sim_365d = np.concatenate(full_train_365d, axis=0)
        self.sim_alphaearth = np.concatenate(full_alphaearth, axis=0).reshape(-1, 64)  
        self.sim_area = np.concatenate(full_area, axis=0).reshape(-1, 1)
    
            
    def load_model(self, path):
        """
        Load a trained regional model from disk.

        Args:
            path (str): Path to the saved Keras model.

        Returns:
            tensorflow.keras.Model: Loaded model instance.
        """
        from tcn import TCN  # Make sure to import TCN
        from tensorflow.keras.utils import custom_object_scope

        from bakaano.streamflow_trainer import asym_laplace_nll
        custom_objects = {"TCN": TCN, "asym_laplace_nll": asym_laplace_nll}
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            with custom_object_scope(custom_objects):  
                self.model = load_model(path, custom_objects=custom_objects)
        
    def summary(self):
        """Print a summary of the loaded model."""
        self.model.summary()
