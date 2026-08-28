"""Simulation and inference utilities for streamflow prediction.

Role: Prepare simulation inputs and run trained model inference.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import json
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
import tensorflow_probability as tfp
from keras.utils import register_keras_serializable
import glob
from tcn import TCN
import rasterio
import rioxarray
from rasterio.transform import rowcol
from keras.models import load_model # type: ignore
import pickle
import warnings
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial.distance import cdist
from collections.abc import Iterable
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

tfd = tfp.distributions  # TensorFlow Probability distributions


def _load_pysheds_grid():
    """Import pysheds lazily to avoid import-time backend failures."""
    import pysheds.grid

    return pysheds.grid


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


def split_predictions_by_station(flat_predictions, station_window_counts):
    """Split flat model outputs into station-aligned chunks."""
    total = int(np.sum(station_window_counts))
    if total != len(flat_predictions):
        raise ValueError(
            "Prediction/window mismatch: "
            f"predictions={len(flat_predictions)} vs expected={total}"
        )

    chunks = []
    start = 0
    for count in station_window_counts:
        end = start + int(count)
        chunks.append(flat_predictions[start:end])
        start = end
    return chunks


def _preview_items(items, limit=8):
    """Return a compact preview string for notebook progress messages."""
    values = list(items or [])
    if len(values) <= limit:
        return ", ".join(map(str, values))
    shown = ", ".join(map(str, values[:limit]))
    return f"{shown}, ... ({len(values) - limit} more)"


def coerce_prediction_array(predicted_streamflow):
    """Convert model outputs to a NumPy float array safe for NumPy ops."""
    return np.asarray(predicted_streamflow, dtype=np.float32)


def clip_negative_predictions(predicted_streamflow):
    """Clamp negative predictions to zero after normalizing dtype."""
    predicted_streamflow = coerce_prediction_array(predicted_streamflow)
    return np.maximum(predicted_streamflow, 0.0)


def convert_area_normalized_flow(predicted_streamflow, catch_area):
    """Convert area-normalized depth (mm/day) back to discharge (m3/s)."""
    return (predicted_streamflow * catch_area * 1_000_000.0) / (86400 * 1000)


def inverse_log1p_predictions(predicted_streamflow):
    """Convert model outputs from log1p target space back to linear values."""
    return np.expm1(coerce_prediction_array(predicted_streamflow))


def build_prediction_frame(predicted_streamflow, sim_start):
    """Return a dated prediction frame after the 365-day model lookback."""
    adjusted_start_date = pd.to_datetime(sim_start) + pd.DateOffset(days=365)
    period = pd.date_range(adjusted_start_date, periods=len(predicted_streamflow), freq="D")
    return pd.DataFrame({
        "time": period,
        "streamflow (m3/s)": predicted_streamflow.reshape(-1),
    })


def _plot_grdc_streamflow(observed_streamflow, predicted_streamflow, val_start):
    """Plot observed vs predicted streamflow for one interactive evaluation run."""
    from matplotlib import pyplot as plt

    adjusted_start_date = pd.to_datetime(val_start) + pd.DateOffset(days=365)
    num_days = len(predicted_streamflow)
    period = pd.date_range(adjusted_start_date, periods=num_days, freq="D")
    
    # Extract observed data with proper time-period alignment
    obs_df = observed_streamflow[0].copy()
    
    # Ensure time index is datetime
    if 'time' in obs_df.columns:
        obs_df['time'] = pd.to_datetime(obs_df['time'])
        obs_df.set_index('time', inplace=True)
    elif not isinstance(obs_df.index, pd.DatetimeIndex):
        obs_df.index = pd.to_datetime(obs_df.index)
    
    # Filter observed data to match the prediction period
    obs_in_period = obs_df.loc[adjusted_start_date:period[-1], 'station_discharge']
    obs_values = obs_in_period.values[:num_days] if len(obs_in_period) >= num_days else obs_in_period.values
    
    plt.figure(figsize=(15, 5))
    plt.plot(period[:len(obs_values)], obs_values, color="blue", label="Observed streamflow")
    plt.plot(period, predicted_streamflow, color="red", label="Bakaano predicted streamflow")
    plt.title("Observed vs Bakaano predicted streamflow")
    plt.xlabel("Time")
    plt.ylabel("Streamflow")
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()


def evaluate_streamflow_model_interactively(
    working_dir,
    study_area,
    model_path,
    val_start,
    val_end,
    grdc_netcdf=None,
    routing_method="mfd",
    catchment_size_threshold=1000,
    area_normalize=True,
    log_transform=True,
    csv_dir=None,
    lookup_csv=None,
    id_col="id",
    lat_col="latitude",
    lon_col="longitude",
    date_col="date",
    discharge_col="discharge",
    file_pattern="{id}.csv",
    runoff_output_dir=None,
):
    """Interactively evaluate a trained model against one station."""
    csv_mode = bool(csv_dir and lookup_csv)
    grdc_mode = grdc_netcdf is not None
    if csv_mode == grdc_mode:
        raise ValueError(
            "Provide exactly one observed-data source for evaluation: either grdc_netcdf or csv_dir+lookup_csv."
        )
    if grdc_mode and not os.path.isfile(os.fspath(grdc_netcdf)):
        raise FileNotFoundError(f"GRDC NetCDF file was not found: {grdc_netcdf}")
    if csv_mode and not os.path.isdir(os.fspath(csv_dir)):
        raise FileNotFoundError(f"Observed streamflow CSV directory was not found: {csv_dir}")
    if csv_mode and not os.path.isfile(os.fspath(lookup_csv)):
        raise FileNotFoundError(f"Observed streamflow lookup CSV was not found: {lookup_csv}")

    vdp = PredictDataPreprocessor(
        working_dir,
        study_area,
        val_start,
        val_end,
        routing_method,
        grdc_netcdf,
        catchment_size_threshold,
        runoff_output_dir=runoff_output_dir,
    )

    if csv_mode:
        vdp.load_observed_streamflow_from_csv_dir(
            csv_dir=csv_dir,
            lookup_csv=lookup_csv,
            id_col=id_col,
            lat_col=lat_col,
            lon_col=lon_col,
            date_col=date_col,
            discharge_col=discharge_col,
            file_pattern=file_pattern,
        )
        print("Available station ids:")
        print(_preview_items(vdp.station_ids))
        station_id = input("\n Please enter the station id: ")
        vdp.station_ids = np.unique([str(station_id)])
        station = vdp.observed_streamflow_csv.get(str(station_id))
        if station is None:
            raise SystemExit("Station id not found in observed CSV directory.")
    else:
        fulldata = vdp.load_observed_streamflow(grdc_netcdf)
        print("Available station names:")
        print(_preview_items(vdp.sim_station_names))
        station_name = input("\n Please enter the station name: ")
        extracted_data = fulldata.where(fulldata.station_name.astype(str) == station_name, drop=True)
        full_ids = list(extracted_data.id.values)
        station = extracted_data["runoff_mean"].where(
            extracted_data["station_name"] == station_name,
            drop=True,
        ).to_dataframe(name="station_discharge").reset_index()
        station_id = station["id"][0]
        station_index = full_ids.index(station_id)
        vdp.station_ids = np.unique([full_ids[station_index]])

    rawdata = vdp.get_data()
    observed_streamflow = list(map(lambda xy: xy[1], rawdata))

    vmodel = PredictStreamflow(
        working_dir,
        area_normalize=area_normalize,
        log_transform=log_transform,
    )
    vmodel.load_model_config(model_path)
    vmodel.prepare_data(rawdata)
    vmodel.print_prediction_summary(point_count=1)
    vmodel.load_model(model_path)
    predicted_streamflow = vmodel.predict_station_series()[0]
    _plot_grdc_streamflow(observed_streamflow, predicted_streamflow, val_start)
    return predicted_streamflow


def simulate_streamflow(
    working_dir,
    study_area,
    model_path,
    sim_start,
    sim_end,
    latlist,
    lonlist,
    routing_method="mfd",
    area_normalize=True,
    log_transform=True,
    runoff_output_dir=None,
):
    """Simulate streamflow at arbitrary latitude/longitude points."""
    if not isinstance(latlist, Iterable) or isinstance(latlist, (str, bytes)):
        raise ValueError("latlist must be an iterable of latitude values.")
    if not isinstance(lonlist, Iterable) or isinstance(lonlist, (str, bytes)):
        raise ValueError("lonlist must be an iterable of longitude values.")
    latlist = list(latlist)
    lonlist = list(lonlist)
    if not latlist or not lonlist:
        raise ValueError("latlist and lonlist must not be empty.")
    if len(latlist) != len(lonlist):
        raise ValueError(
            f"latlist and lonlist must have the same length. Received {len(latlist)} and {len(lonlist)}."
        )

    print(" 1. Loading runoff data and other predictors")
    vdp = PredictDataPreprocessor(
        working_dir,
        study_area,
        sim_start,
        sim_end,
        routing_method,
        runoff_output_dir=runoff_output_dir,
    )
    rawdata = vdp.get_data_latlng(latlist, lonlist)

    vmodel = PredictStreamflow(
        working_dir,
        area_normalize=area_normalize,
        log_transform=log_transform,
    )
    vmodel.load_model_config(model_path)
    vmodel.prepare_data_latlng(rawdata)
    vmodel.print_prediction_summary(point_count=len(latlist))
    batch_size = len(latlist)
    vmodel.load_model(model_path)
    print(" 2. Batch prediction")
    predicted_streamflow_list = vmodel.predict_station_series(
        batch_size=batch_size,
        area_normalize=vmodel.area_normalize,
    )
    print(" 3. Generating csv file for each coordinate")
    valid_pairs = [(latlist[i], lonlist[i]) for i in vmodel.valid_entry_indices]
    output_paths = []
    for predicted_streamflow, (lat, lon) in zip(predicted_streamflow_list, valid_pairs):
        df = build_prediction_frame(predicted_streamflow, sim_start)
        output_path = os.path.join(
            working_dir,
            f"predicted_streamflow_data/predicted_streamflow_lat{lat}_lon{lon}.csv",
        )
        df.to_csv(output_path, index=False)
        output_paths.append(output_path)
    out_folder = os.path.join(working_dir, "predicted_streamflow_data")
    print(f"Completed. Predicted streamflow CSV files are available at {out_folder}")
    return output_paths


def simulate_grdc_csv_stations(
    working_dir,
    study_area,
    model_path,
    sim_start,
    sim_end,
    grdc_netcdf=None,
    routing_method="mfd",
    csv_dir=None,
    lookup_csv=None,
    id_col="id",
    lat_col="latitude",
    lon_col="longitude",
    date_col="date",
    discharge_col="discharge",
    file_pattern="{id}.csv",
    area_normalize=True,
    log_transform=True,
    runoff_output_dir=None,
):
    """Simulate streamflow for GRDC or CSV-defined station sets."""
    csv_mode = bool(csv_dir and lookup_csv)
    grdc_mode = grdc_netcdf is not None
    if csv_mode == grdc_mode:
        raise ValueError(
            "Provide exactly one station source for simulation: either grdc_netcdf or csv_dir+lookup_csv."
        )
    if grdc_mode and not os.path.isfile(os.fspath(grdc_netcdf)):
        raise FileNotFoundError(f"GRDC NetCDF file was not found: {grdc_netcdf}")
    if csv_mode and not os.path.isdir(os.fspath(csv_dir)):
        raise FileNotFoundError(f"Observed streamflow CSV directory was not found: {csv_dir}")
    if csv_mode and not os.path.isfile(os.fspath(lookup_csv)):
        raise FileNotFoundError(f"Observed streamflow lookup CSV was not found: {lookup_csv}")

    print(" 1. Loading runoff data and other predictors")
    vdp = PredictDataPreprocessor(
        working_dir,
        study_area,
        sim_start,
        sim_end,
        routing_method,
        grdc_netcdf,
        runoff_output_dir=runoff_output_dir,
    )
    if csv_mode:
        vdp.load_observed_streamflow_from_csv_dir(
            csv_dir=csv_dir,
            lookup_csv=lookup_csv,
            id_col=id_col,
            lat_col=lat_col,
            lon_col=lon_col,
            date_col=date_col,
            discharge_col=discharge_col,
            file_pattern=file_pattern,
        )
        print("Available station ids:")
        print(_preview_items(vdp.station_ids))
    else:
        print("Available station names:")
        print(_preview_items(vdp.sim_station_names))

    rawdata = vdp.get_data()
    vmodel = PredictStreamflow(
        working_dir,
        area_normalize=area_normalize,
        log_transform=log_transform,
    )
    vmodel.load_model_config(model_path)
    vmodel.prepare_data(rawdata)
    vmodel.print_prediction_summary(point_count=len(vdp.station_ids))
    batch_size = len(vdp.station_ids)
    vmodel.load_model(model_path)
    print(" 2. Batch prediction")
    predicted_streamflow_list = vmodel.predict_station_series(
        batch_size=batch_size,
        area_normalize=vmodel.area_normalize,
    )
    print(" 3. Generating csv file for each coordinate")
    valid_station_ids = [vdp.station_ids[i] for i in vmodel.valid_entry_indices]
    output_paths = []
    for predicted_streamflow, station_id in zip(predicted_streamflow_list, valid_station_ids):
        df = build_prediction_frame(predicted_streamflow, sim_start)
        output_path = os.path.join(working_dir, f"predicted_streamflow_data/bakaano_{station_id}.csv")
        df.to_csv(output_path, index=False)
        output_paths.append(output_path)
    out_folder = os.path.join(working_dir, "predicted_streamflow_data")
    print(f"Completed. Predicted streamflow CSV files are available at {out_folder}")
    return output_paths


class PredictDataPreprocessor:
    def __init__(
        self,
        working_dir,
        study_area,
        sim_start,
        sim_end,
        routing_method,
        grdc_streamflow_nc_file=None,
        catchment_size_threshold=None,
        runoff_output_dir=None,
    ):
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
        self.runoff_output_dir = runoff_output_dir or f"{self.working_dir}/runoff_output"
        
        self.data_list = []
        self.catchment = []  
        self.sim_start = sim_start
        self.sim_end = sim_end
        self.sim_station_names= []
        self.catchment_size_threshold = catchment_size_threshold
        if grdc_streamflow_nc_file is not None:
            self.grdc_subset = self.load_observed_streamflow(grdc_streamflow_nc_file)
            self.station_ids = np.unique(self.grdc_subset.to_dataframe().index.get_level_values('id'))

    def _load_runoff_entries_for_period(self):
        """Load routed runoff entries and require exact daily coverage for simulation."""
        preferred_file = os.path.join(self.runoff_output_dir, 'wacc_sparse_arrays.pkl')
        candidate_files = [preferred_file] if os.path.exists(preferred_file) else sorted(glob.glob(f'{self.runoff_output_dir}/*.pkl'))

        if not candidate_files:
            raise SystemExit(f"""
ERROR: No routed runoff output found

The simulation pipeline could not find any routed runoff pickle files in:
  {self.runoff_output_dir}

Please run VegET runoff routing first for the period needed by simulation.
""".strip())

        requested_dates = pd.date_range(start=self.sim_start, end=self.sim_end, freq='D')
        requested_labels = [dt.strftime("%Y-%m-%d") for dt in requested_dates]
        runoff_by_date = {}

        for path in candidate_files:
            try:
                with open(path, 'rb') as f:
                    loaded = pickle.load(f)
            except Exception as exc:
                raise SystemExit(f"""
ERROR: Failed to read routed runoff output

Bakaano could not read:
  {path}

Original error:
  {str(exc)}
""".strip()) from exc

            if not isinstance(loaded, list):
                continue

            for entry in loaded:
                if not isinstance(entry, dict):
                    continue
                date_str = entry.get("time")
                matrix = entry.get("matrix")
                if date_str is None or matrix is None:
                    continue
                runoff_by_date[str(date_str)] = entry

        if not runoff_by_date:
            raise SystemExit(f"""
ERROR: No routed runoff matrices found

Files were found in:
  {self.runoff_output_dir}

but none contained daily routed runoff entries with both 'time' and 'matrix' fields.
Please verify that VegET runoff routing completed successfully.
""".strip())

        available_dates = sorted(runoff_by_date.keys())
        missing_dates = [date_str for date_str in requested_labels if date_str not in runoff_by_date]
        if missing_dates:
            preview = ", ".join(missing_dates[:5])
            suffix = " ..." if len(missing_dates) > 5 else ""
            raise SystemExit(f"""
ERROR: Requested simulation period is not covered by routed runoff output

Requested simulation period:
  start: {self.sim_start}
  end:   {self.sim_end}

Available routed runoff dates:
  from:  {available_dates[0]}
  to:    {available_dates[-1]}

Missing requested dates:
  {preview}{suffix}

This usually means VegET runoff was computed for a different date range
than the one now requested for simulation.
Please rerun VegET for the required period or adjust sim_start/sim_end.
""".strip())

        return [runoff_by_date[date_str] for date_str in requested_labels]

    def _load_optional_rainfall_entries_for_period(self):
        """Load routed rainfall entries if available and fully aligned."""
        rainfall_file = os.path.join(self.runoff_output_dir, "rainfall_sparse_arrays.pkl")
        if not os.path.exists(rainfall_file):
            print("     Routed rainfall not found; using routed runoff only.")
            return None

        requested_dates = pd.date_range(start=self.sim_start, end=self.sim_end, freq="D")
        requested_labels = [dt.strftime("%Y-%m-%d") for dt in requested_dates]

        try:
            with open(rainfall_file, "rb") as f:
                loaded = pickle.load(f)
        except Exception as exc:
            raise SystemExit(
                f"Failed to read routed rainfall output from {rainfall_file}: {str(exc)}"
            ) from exc

        rainfall_by_date = {}
        if isinstance(loaded, list):
            for entry in loaded:
                if not isinstance(entry, dict):
                    continue
                date_str = entry.get("time")
                matrix = entry.get("matrix")
                if date_str is None or matrix is None:
                    continue
                rainfall_by_date[str(date_str)] = entry

        missing_dates = [date_str for date_str in requested_labels if date_str not in rainfall_by_date]
        if missing_dates:
            preview = ", ".join(missing_dates[:5])
            suffix = " ..." if len(missing_dates) > 5 else ""
            raise SystemExit(
                "Routed rainfall output is present but does not cover the requested simulation period. "
                f"Missing dates: {preview}{suffix}. "
                "Regenerate routed rainfall for the full period, or remove rainfall_sparse_arrays.pkl "
                "to simulate with runoff-only predictors."
            )

        print("     Routed rainfall found; using routed runoff + routed rainfall predictors.")
        return [rainfall_by_date[date_str] for date_str in requested_labels]
        
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
    
        # Accept points inside the polygon or lying exactly on its boundary.
        inside = region_gdf.geometry.covers(point.iloc[0]).any()
    
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
      - that the point is not outside the study area
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
        
        pysheds_grid = _load_pysheds_grid()
        grid = pysheds_grid.Grid.from_raster(dem_filepath)
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
        
        
        wfa_list = self._load_runoff_entries_for_period()
        rainfall_list = self._load_optional_rainfall_entries_for_period()
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
    
            predictors = full_wfa_data.copy()
            if rainfall_list is not None:
                station_rainfall = []
                for arr in rainfall_list:
                    arr = arr["matrix"].tocsr()
                    station_rainfall.append(arr[int(row), int(col)])
                rainfall_data = pd.DataFrame(station_rainfall, columns=["routed_rainfall"])
                rainfall_data.set_index(time_index, inplace=True)
                predictors = predictors.join(rainfall_data)

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
        
        pysheds_grid = _load_pysheds_grid()
        grid = pysheds_grid.Grid.from_raster(dem_filepath)
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
        
        wfa_list = self._load_runoff_entries_for_period()
        rainfall_list = self._load_optional_rainfall_entries_for_period()
        
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

            predictors = full_wfa_data.copy()
            if rainfall_list is not None:
                station_rainfall = []
                for arr in rainfall_list:
                    arr = arr["matrix"].tocsr()
                    station_rainfall.append(arr[int(row), int(col)])
                rainfall_data = pd.DataFrame(station_rainfall, columns=["routed_rainfall"])
                rainfall_data.set_index(time_index, inplace=True)
                predictors = predictors.join(rainfall_data)

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
    def __init__(self, working_dir, area_normalize=True, log_transform=True):
        """
        Role: Prepare model inputs and run inference.

        Initializes the PredictStreamflow class for streamflow prediction using a temporal convolutional network (TCN).

        Args:
            working_dir (str): The working directory where the model and data are stored.
            area_normalize (bool): Whether to area-normalize predictors/response.
            log_transform (bool): Whether model inputs/outputs use log1p target space.

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
        self.log_transform = bool(log_transform)
        self.sim_p14 = None
        self.sim_p180 = None
        self.sim_p365 = None
        self.sim_alphaearth = None
        self.sim_area = None
        self.catch_area = None
        self.catch_area_list = []
        self.station_window_counts = []
        self.valid_entry_indices = []
        self.predictor_columns = []
        self.load_default_model_config()

    def predict(self, batch_size=None):
        """Run model inference using the prepared predictor tensors."""
        predicted_streamflows = self.model.predict(
            [
                self.sim_p14,
                self.sim_p180,
                self.sim_p365,
                self.sim_alphaearth,
                self.sim_area,
            ],
            batch_size=batch_size,
        )
        if predicted_streamflows.ndim == 2 and predicted_streamflows.shape[1] == 3:
            predicted_streamflows = predicted_streamflows[:, 0:1]
        return coerce_prediction_array(predicted_streamflows)

    def predict_station_series(self, batch_size=None, area_normalize=None):
        """Run inference and return one prediction array per valid station/point."""
        if area_normalize is None:
            area_normalize = self.area_normalize
        predicted_streamflows = self.predict(batch_size=batch_size)
        station_preds = split_predictions_by_station(
            predicted_streamflows,
            self.station_window_counts,
        )
        predicted_streamflow_list = []
        for predicted_streamflow, catch_area in zip(station_preds, self.catch_area_list):
            if self.log_transform:
                predicted_streamflow = inverse_log1p_predictions(predicted_streamflow)
            if area_normalize:
                predicted_streamflow = convert_area_normalized_flow(predicted_streamflow, catch_area)
            predicted_streamflow = clip_negative_predictions(predicted_streamflow)
            predicted_streamflow_list.append(predicted_streamflow)
        return predicted_streamflow_list

    def load_model_config(self, model_path):
        """Load saved model-scale options before preparing inference tensors."""
        from bakaano.neuralnet.train import model_config_paths

        for config_path in model_config_paths(model_path):
            config = self._load_config_file(config_path)
            if config is not None:
                return config
        return None

    def load_default_model_config(self):
        """Load default working-dir model config when available."""
        config_path = os.path.join(
            self.working_dir,
            "models",
            "bakaano_model_config.json",
        )
        return self._load_config_file(config_path)

    def _load_config_file(self, config_path):
        if not os.path.exists(config_path):
            return None
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        if "area_normalize" in config:
            self.area_normalize = bool(config["area_normalize"])
        if "log_transform" in config:
            self.log_transform = bool(config["log_transform"])
        return config

    def prepare_data(self, data_list):
        from numpy.lib.stride_tricks import sliding_window_view

        """
        Prepare flow accumulation and streamflow data extracted from GRDC database for input in the model. Preparation involves dividing time-series data into desired short sequences based on specified timesteps and reshaping into desired tensor shape.
        
        Parameters:
        -----------
        data_list : Numpy array data 
            The extracted flow accumulation and observed streamflow data i.e. the output of get_grdc_data() functions.

        Returns:
            None. Populates model input arrays on the instance.
        """

        predictors = [xy[0] for xy in data_list]
        self.predictor_columns = list(predictors[0].columns) if predictors else []
        catchment = [xy[2] for xy in data_list]
        catchment_arr = np.array(catchment, dtype=np.float32)

        area = catchment_arr[:, 0:1]
        alphaearth = catchment_arr[:, 1:]

        with open(f'{self.working_dir}/models/alpha_earth_scaler.pkl', 'rb') as file:
            alphaearth_scaler = pickle.load(file)

        if len(catchment) <= 0:
            raise ValueError("No catchment data available for prediction.")

        alphaearth = alphaearth.reshape(-1, 64)
        scaled_alphaearth = alphaearth_scaler.transform(alphaearth)

        all_p14 = []
        all_p180 = []
        all_p365 = []
        all_alpha = []
        all_area = []
        self.catch_area_list = []
        self.station_window_counts = []
        self.valid_entry_indices = []
        for idx, (x, z, j) in enumerate(zip(predictors, scaled_alphaearth, area)):
            this_area = np.expm1(j)

            if self.area_normalize:
                scaled_train_predictor = x.values / this_area
            else:
                scaled_train_predictor = x.values
            if self.log_transform:
                scaled_train_predictor = np.log1p(scaled_train_predictor)

            if scaled_train_predictor.ndim == 1:
                scaled_train_predictor = scaled_train_predictor.reshape(-1, 1)
                

            num_samples = scaled_train_predictor.shape[0] - 365 - 1
            if num_samples <= 0:
                continue

            windows_raw = sliding_window_view(
                scaled_train_predictor,
                (365, scaled_train_predictor.shape[1]),
            )[:, 0]
            full_windows = windows_raw[:num_samples]
            mask = ~np.isnan(full_windows).any(axis=(1, 2))
            if not np.any(mask):
                continue

            n_valid = int(np.sum(mask))
            valid_windows = full_windows[mask]
            all_p14.append(valid_windows[:, -14:, :])
            all_p180.append(valid_windows[:, -180:, :])
            all_p365.append(valid_windows)
            all_alpha.append(np.tile(z.reshape(1, -1), (n_valid, 1)))
            all_area.append(np.tile(j.reshape(1, -1), (n_valid, 1)))
            self.catch_area_list.append(this_area)
            self.station_window_counts.append(n_valid)
            self.valid_entry_indices.append(idx)

        if not all_p365:
            raise ValueError("No valid simulation windows were created.")

        self.sim_p14 = np.concatenate(all_p14, axis=0).astype("float32")
        self.sim_p180 = np.concatenate(all_p180, axis=0).astype("float32")
        self.sim_p365 = np.concatenate(all_p365, axis=0).astype("float32")
        self.sim_alphaearth = np.concatenate(all_alpha, axis=0).astype("float32")
        self.sim_area = np.concatenate(all_area, axis=0).astype("float32")
        self.catch_area = self.catch_area_list[0] if self.catch_area_list else None
    
    def prepare_data_latlng(self, data_list):
        from numpy.lib.stride_tricks import sliding_window_view

        """
        Prepare model inputs for user-defined latitude/longitude points.

        This uses routed runoff time series at specified lat/lon points (not GRDC
        stations), builds 365-day windows, and reshapes tensors for inference.
        
        Parameters:
        -----------
        data_list : Numpy array data 
            Output of get_data_latlng(), containing predictors and catchment info.

        Returns:
            None. Populates model input arrays on the instance.
        """

        predictors = [xy[0] for xy in data_list[0]]
        self.predictor_columns = list(predictors[0].columns) if predictors else []
        catchment = [xy[1] for xy in data_list[0]]
        catchment_arr = np.array(catchment, dtype=np.float32)

        area = catchment_arr[:, 0:1]
        alphaearth = catchment_arr[:, 1:]

        with open(f'{self.working_dir}/models/alpha_earth_scaler.pkl', 'rb') as file:
            alphaearth_scaler = pickle.load(file)

        if len(catchment) <= 0:
            raise ValueError("No catchment data available for prediction.")

        alphaearth = alphaearth.reshape(-1, 64)
        scaled_alphaearth = alphaearth_scaler.transform(alphaearth)

        all_p14 = []
        all_p180 = []
        all_p365 = []
        all_alpha = []
        all_area = []
        self.catch_area_list = []
        self.station_window_counts = []
        self.valid_entry_indices = []
        for idx, (x, z, j) in enumerate(zip(predictors, scaled_alphaearth, area)):
            this_area = np.expm1(j)

            if self.area_normalize:
                scaled_train_predictor = x.values / this_area
            else:
                scaled_train_predictor = x.values
            if self.log_transform:
                scaled_train_predictor = np.log1p(scaled_train_predictor)
            if scaled_train_predictor.ndim == 1:
                scaled_train_predictor = scaled_train_predictor.reshape(-1, 1)

            num_samples = scaled_train_predictor.shape[0] - 365 - 1
            if num_samples <= 0:
                continue

            windows_raw = sliding_window_view(
                scaled_train_predictor,
                (365, scaled_train_predictor.shape[1]),
            )[:, 0]
            full_windows = windows_raw[:num_samples]
            mask = ~np.isnan(full_windows).any(axis=(1, 2))
            if not np.any(mask):
                continue

            n_valid = int(np.sum(mask))
            valid_windows = full_windows[mask]
            all_p14.append(valid_windows[:, -14:, :])
            all_p180.append(valid_windows[:, -180:, :])
            all_p365.append(valid_windows)
            all_alpha.append(np.tile(z.reshape(1, -1), (n_valid, 1)))
            all_area.append(np.tile(j.reshape(1, -1), (n_valid, 1)))
            self.catch_area_list.append(this_area)
            self.station_window_counts.append(n_valid)
            self.valid_entry_indices.append(idx)

        if not all_p365:
            raise ValueError("No valid simulation windows were created.")

        self.sim_p14 = np.concatenate(all_p14, axis=0).astype("float32")
        self.sim_p180 = np.concatenate(all_p180, axis=0).astype("float32")
        self.sim_p365 = np.concatenate(all_p365, axis=0).astype("float32")
        self.sim_alphaearth = np.concatenate(all_alpha, axis=0).astype("float32")
        self.sim_area = np.concatenate(all_area, axis=0).astype("float32")
        self.catch_area = self.catch_area_list[0] if self.catch_area_list else None

    def print_prediction_summary(self, point_count=None):
        """Print a compact summary of prepared simulation tensors."""
        if self.sim_p14 is None:
            return

        point_text = "unknown" if point_count is None else str(point_count)
        predictors = ", ".join(self.predictor_columns) if self.predictor_columns else "unknown"
        print("     Simulation data prepared:")
        print(f"       points/stations: {point_text}")
        print(f"       valid windows: {self.sim_p14.shape[0]}")
        print(
            "       temporal inputs: "
            f"p14={tuple(self.sim_p14.shape[1:])}, "
            f"p180={tuple(self.sim_p180.shape[1:])}, "
            f"p365={tuple(self.sim_p365.shape[1:])}"
        )
        print(f"       predictors: {predictors}")
        print(f"       area_normalize: {self.area_normalize}")
        print(f"       log_transform: {self.log_transform}")
    
            
    def load_model(self, path):
        """
        Load a trained regional model from disk.

        Args:
            path (str): Path to the saved Keras model.

        Returns:
            tensorflow.keras.Model: Loaded model instance.
        """
        self.load_model_config(path)
        from tensorflow.keras.utils import custom_object_scope

        from bakaano.neuralnet.train import (
            asym_laplace_nll_linear,
            EntropyRegulariser,
            ExpandDims,
            OneMinus,
            ReduceSum,
            ScaleShift,
            Slice1D,
            SliceTimestep,
            asym_laplace_nll,
            validate_keras_archive,
        )
        custom_objects = {
            "TCN": TCN,
            "asym_laplace_nll": asym_laplace_nll,
            "asym_laplace_nll_linear": asym_laplace_nll_linear,
            "ExpandDims": ExpandDims,
            "ReduceSum": ReduceSum,
            "ScaleShift": ScaleShift,
            "Slice1D": Slice1D,
            "SliceTimestep": SliceTimestep,
            "OneMinus": OneMinus,
            "EntropyRegulariser": EntropyRegulariser,
        }
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            with custom_object_scope(custom_objects):  
                validate_keras_archive(path)
                self.model = load_model(path, custom_objects=custom_objects)
        self._align_temporal_feature_count_to_model()

    def _align_temporal_feature_count_to_model(self):
        """Match prepared temporal feature channels to the loaded model."""
        expected_lengths = [14, 180, 365]
        actual_lengths = [int(inp.shape[1]) for inp in self.model.inputs[:3]]
        if len(self.model.inputs) != 5 or actual_lengths != expected_lengths:
            raise ValueError(
                "Loaded model does not use the supported p14/p180/p365 input layout. "
                "Retrain the model with the current training module before simulation."
            )

        expected = int(self.model.inputs[0].shape[-1])
        actual = int(self.sim_p14.shape[-1])
        if actual == expected:
            return
        if actual > expected:
            self.sim_p14 = self.sim_p14[:, :, :expected]
            self.sim_p180 = self.sim_p180[:, :, :expected]
            self.sim_p365 = self.sim_p365[:, :, :expected]
            return
        if actual < expected:
            raise ValueError(
                "Prepared predictor feature count does not match the trained model. "
                f"Model expects {expected} temporal feature(s), but the prepared inputs contain {actual}. "
                "If the model was trained with routed rainfall enabled, make sure rainfall_sparse_arrays.pkl "
                "is available for the requested simulation period."
            )
        raise ValueError(
            "Prepared predictor feature count does not match the trained model. "
            f"Model expects {expected} temporal feature(s), but the prepared inputs contain {actual}. "
            "If the model was trained with routed rainfall enabled, make sure rainfall_sparse_arrays.pkl is available "
            "for the requested simulation period."
        )
        
    def summary(self):
        """Print a summary of the loaded model."""
        self.model.summary()
