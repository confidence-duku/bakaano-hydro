"""Training pipeline for regional streamflow models.

Role: Build training datasets and train the TCN-based streamflow model.
"""

import os
import math
import glob
import json
import pickle
import warnings
import zipfile
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import geopandas as gpd
import numpy as np
import pandas as pd
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
    Layer,
    LayerNormalization,
    LeakyReLU,
    Multiply,
    Reshape,
    Softmax,
)
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.utils import custom_object_scope, register_keras_serializable

mixed_precision.set_global_policy('mixed_bfloat16')

MODEL_CONFIG_VERSION = 1


def model_config_paths(model_path):
    """Return supported sidecar config paths for a saved Keras model."""
    model_path = os.fspath(model_path)
    return [
        f"{model_path}.config.json",
        os.path.join(os.path.dirname(model_path), "bakaano_model_config.json"),
    ]
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def validate_keras_archive(model_path):
    """Raise a clear error if a .keras archive is missing model weights."""
    model_path = os.fspath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file was not found: {model_path}")
    if not zipfile.is_zipfile(model_path):
        raise ValueError(
            f"Model file is not a valid .keras archive: {model_path}. "
            "Delete it and retrain the model."
        )
    with zipfile.ZipFile(model_path) as zf:
        names = set(zf.namelist())
    if "model.weights.h5" not in names and "model.weights.npz" not in names:
        raise ValueError(
            f"Model file is incomplete or corrupted: {model_path}. "
            "Expected model.weights.h5 or model.weights.npz inside the .keras archive. "
            "Delete this file and retrain the model."
        )


@register_keras_serializable(package="bakaano")
class ExpandDims(Layer):
    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        return tf.expand_dims(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape.insert(self.axis, 1)
        return tuple(shape)

    def get_config(self):
        return {**super().get_config(), "axis": self.axis}


@register_keras_serializable(package="bakaano")
class ReduceSum(Layer):
    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        return tf.reduce_sum(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape.pop(self.axis)
        return tuple(shape)

    def get_config(self):
        return {**super().get_config(), "axis": self.axis}


@register_keras_serializable(package="bakaano")
class ScaleShift(Layer):
    def __init__(self, scale, shift, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.shift = shift

    def call(self, x):
        return self.scale * x + self.shift

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "scale": self.scale, "shift": self.shift}


@register_keras_serializable(package="bakaano")
class Slice1D(Layer):
    def __init__(self, start, stop, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.stop = stop

    def call(self, x):
        return x[:, self.start:self.stop]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.stop - self.start)

    def get_config(self):
        return {**super().get_config(), "start": self.start, "stop": self.stop}


@register_keras_serializable(package="bakaano")
class SliceTimestep(Layer):
    def __init__(self, timestep, ch_start, ch_stop, **kwargs):
        super().__init__(**kwargs)
        self.timestep = timestep
        self.ch_start = ch_start
        self.ch_stop = ch_stop

    def call(self, x):
        return x[:, self.timestep, self.ch_start:self.ch_stop]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.ch_stop - self.ch_start)

    def get_config(self):
        return {
            **super().get_config(),
            "timestep": self.timestep,
            "ch_start": self.ch_start,
            "ch_stop": self.ch_stop,
        }


@register_keras_serializable(package="bakaano")
class OneMinus(Layer):
    def call(self, x):
        return 1.0 - x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config()}


@register_keras_serializable(package="bakaano")
class EntropyRegulariser(Layer):
    def __init__(self, strength=0.01, **kwargs):
        super().__init__(**kwargs)
        self.strength = strength

    def call(self, w):
        entropy = -tf.reduce_sum(w * tf.math.log(w + 1e-8), axis=-1)
        self.add_loss(-self.strength * tf.reduce_mean(entropy))
        return w

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "strength": self.strength}


def _load_pysheds_grid():
    """Import pysheds lazily to avoid import-time backend failures."""
    import pysheds.grid

    return pysheds.grid


def _normalize_station_ids(station_ids):
    """Return station ids as a comparable set across scalar/array-like inputs."""
    if isinstance(station_ids, (str, bytes)):
        return {station_ids}
    if np.isscalar(station_ids):
        return {station_ids.item() if isinstance(station_ids, np.generic) else station_ids}
    if hasattr(station_ids, "tolist"):
        values = station_ids.tolist()
        if isinstance(values, list):
            return set(values)
        return {values}
    if isinstance(station_ids, (set, tuple, list)):
        return set(station_ids)
    return set(station_ids)


def _predictor_cache_path(working_dir):
    """Return the canonical predictor-response cache path."""
    return os.path.join(working_dir, "models", "predictor_response_data.pkl")


def _preview_items(items, limit=8):
    """Return a compact preview string for notebook progress messages."""
    values = list(items or [])
    if len(values) <= limit:
        return ", ".join(map(str, values))
    shown = ", ".join(map(str, values[:limit]))
    return f"{shown}, ... ({len(values) - limit} more)"


def _build_predictor_cache_payload(data, train_start, train_end):
    """Wrap predictor-response data with period metadata for safe cache reuse."""
    return {
        "metadata": {
            "train_start": str(train_start),
            "train_end": str(train_end),
        },
        "data": data,
    }


def _load_predictor_cache_if_compatible(cache_path, train_start, train_end):
    """Load cached predictors only when the stored period matches exactly."""
    if not os.path.exists(cache_path):
        return None

    with open(cache_path, "rb") as f:
        loaded = pickle.load(f)

    if not isinstance(loaded, dict) or "metadata" not in loaded or "data" not in loaded:
        return None

    metadata = loaded["metadata"]
    if (
        str(metadata.get("train_start")) != str(train_start)
        or str(metadata.get("train_end")) != str(train_end)
    ):
        return None

    return loaded["data"]


def filter_training_data_by_station_ids(rawdata, station_ids):
    """Keep only training entries whose embedded station id matches the target set."""
    target_ids = _normalize_station_ids(station_ids)
    filtered = [
        item for item in rawdata
        if len(item) >= 4
        and isinstance(item[3], tuple)
        and len(item[3]) == 1
        and item[3][0] in target_ids
    ]
    if not filtered:
        raise SystemExit(f"""
ERROR: Station ID not found in raw data

Requested station ID(s):
  {sorted(target_ids)}

No matching station entries were found.

Please verify that the station ID(s) exist in the dataset.
""".strip())
    return filtered


def validate_training_runoff_window(rawdata, train_start, train_end):
    """Require exact routed-runoff coverage for the requested training dates."""
    try:
        start_dt = datetime.strptime(train_start, "%Y-%m-%d")
        end_dt = datetime.strptime(train_end, "%Y-%m-%d")

        if not rawdata:
            raise SystemExit(
                "No runoff data loaded. Check the runoff_output directory and pickle files."
            )

        df_runoff = rawdata[0][0]
        if not isinstance(df_runoff.index, pd.DatetimeIndex):
            df_runoff.index = pd.to_datetime(df_runoff.index)

        available_start = df_runoff.index.min()
        available_end = df_runoff.index.max()

        missing = []
        if start_dt not in df_runoff.index:
            missing.append(f"start date ({start_dt.date()})")
        if end_dt not in df_runoff.index:
            missing.append(f"end date ({end_dt.date()})")

        if missing:
            raise SystemExit(f"""
ERROR: Invalid simulation period

Requested period:
  start: {start_dt.date()}
  end:   {end_dt.date()}

Available routed runoff data:
  from:  {available_start.date()}
  to:    {available_end.date()}

Please re-run the runoff and routing modules and ensure the simulation
period covers the intended training, validation, and inference periods.
""".strip())
    except ValueError:
        raise
    except Exception as exc:
        raise SystemExit(f"""
ERROR: Simulation period validation failed

The model failed while validating the simulation period against the
available routed runoff data.

This may indicate one of the following:
  - corrupted or incomplete runoff files
  - an unexpected runoff data format
  - inconsistent or non-datetime time indexing

Please verify the runoff outputs and ensure they were generated
correctly before running training or evaluation again.
""".strip()) from exc


def train_streamflow_model(
    working_dir,
    study_area,
    train_start,
    train_end,
    grdc_netcdf=None,
    batch_size=32,
    num_epochs=300,
    learning_rate=0.0005,
    loss_function="asym_laplace_nll",
    seed=100,
    routing_method="mfd",
    catchment_size_threshold=1,
    area_normalize=True,
    log_transform=True,
    lr_schedule="cosine",
    warmup_epochs=1,
    min_learning_rate=5e-5,
    csv_dir=None,
    lookup_csv=None,
    id_col="id",
    lat_col="latitude",
    lon_col="longitude",
    date_col="date",
    discharge_col="discharge",
    file_pattern="{id}.csv",
    model_overwrite=True,
):
    """Train the Bakaano streamflow model directly from the neuralnet module."""
    cache_path = _predictor_cache_path(working_dir)

    print("\nTRAINING BAKAANO-HYDRO DEEP LEARNING STREAMFLOW PREDICTION MODEL")
    print(" 1. Loading observed streamflow")

    csv_mode = bool(csv_dir and lookup_csv)
    grdc_mode = grdc_netcdf is not None
    if csv_mode == grdc_mode:
        raise SystemExit(
            "Provide exactly one observed-data source: either grdc_netcdf or csv_dir+lookup_csv."
        )
    if grdc_mode and not os.path.isfile(os.fspath(grdc_netcdf)):
        raise FileNotFoundError(f"GRDC NetCDF file was not found: {grdc_netcdf}")
    if csv_mode and not os.path.isdir(os.fspath(csv_dir)):
        raise FileNotFoundError(f"Observed streamflow CSV directory was not found: {csv_dir}")
    if csv_mode and not os.path.isfile(os.fspath(lookup_csv)):
        raise FileNotFoundError(f"Observed streamflow lookup CSV was not found: {lookup_csv}")

    sdp = DataPreprocessor(
        working_dir,
        study_area,
        grdc_netcdf if grdc_mode else None,
        train_start,
        train_end,
        routing_method,
        catchment_size_threshold,
    )

    if csv_mode:
        sdp.load_observed_streamflow_from_csv_dir(
            csv_dir=csv_dir,
            lookup_csv=lookup_csv,
            id_col=id_col,
            lat_col=lat_col,
            lon_col=lon_col,
            date_col=date_col,
            discharge_col=discharge_col,
            file_pattern=file_pattern,
        )

    print(" 2. Loading runoff data and other predictors")
    rawdata = _load_predictor_cache_if_compatible(cache_path, train_start, train_end)
    if rawdata is None:
        rawdata = sdp.get_data()

    rawdata = filter_training_data_by_station_ids(rawdata, sdp.station_ids)
    validate_training_runoff_window(rawdata, train_start, train_end)

    station_count = len(sdp.sim_station_names)
    print(f"     Stations selected for training: {station_count}")
    if station_count:
        print(f"     Station preview: {_preview_items(sdp.sim_station_names)}")

    smodel = StreamflowModel(
        working_dir=working_dir,
        batch_size=batch_size,
        num_epochs=num_epochs,
        loss_function=loss_function,
        learning_rate=learning_rate,
        train_start=train_start,
        train_end=train_end,
        seed=seed,
        area_normalize=area_normalize,
        log_transform=log_transform,
        lr_schedule=lr_schedule,
        warmup_epochs=warmup_epochs,
        min_learning_rate=min_learning_rate,
    )
    smodel.prepare_data(rawdata)
    smodel.print_training_summary(station_count=station_count)
    model_path = f"{working_dir}/models/bakaano_model.keras"
    if (not model_overwrite) and os.path.exists(model_path):
        print(f" 3. Loading existing model for continued training: {model_path}")
        smodel.load_regional_model(model_path)
    else:
        if not model_overwrite and not os.path.exists(model_path):
            print(" 3. No existing model found; starting fresh training run.")
        print(" 3. Building neural network model")
        smodel.build_model()
    print(" 4. Training neural network model")
    smodel.train_model()
    print(f"     Completed! Trained model saved at {model_path}")
    return model_path

#=====================================================================================================================================


@tf.keras.utils.register_keras_serializable()
def asym_laplace_nll(
    y_true,
    params,
    r_clip=10.0,                   # raised: 5.0 was clipping large-basin peaks
    scale_clip=(1e-3, 10.0),       # raised upper: 5.0 prevented high-uncertainty expression
    peak_weight=0.3
):
    import tensorflow as tf

    y_true = tf.cast(y_true, tf.float32)
    params = tf.cast(params, tf.float32)

    mu          = params[:, 0:1]
    log_b_plus  = params[:, 1:2]
    log_b_minus = params[:, 2:3]

    b_plus  = tf.nn.softplus(log_b_plus)
    b_minus = tf.nn.softplus(log_b_minus)

    b_plus  = tf.clip_by_value(b_plus,  scale_clip[0], scale_clip[1])
    b_minus = tf.clip_by_value(b_minus, scale_clip[0], scale_clip[1])

    r_raw = y_true - mu
    r     = tf.clip_by_value(r_raw, -r_clip, r_clip)

    # ── diagnostic: log clipping fraction ────────────────────────────────────
    frac_clipped = tf.reduce_mean(tf.cast(tf.abs(r_raw) > r_clip, tf.float32))
    tf.summary.scalar("ald/frac_clipped_residuals", frac_clipped)

    # ── asymmetric Laplace NLL ────────────────────────────────────────────────
    nll = tf.where(
        r >= 0.0,
        tf.math.log(b_plus)  + r / b_plus,
        tf.math.log(b_minus) - r / b_minus,
    )

    # ── peak weight: amplify only when model actually misses the peak ─────────
    # Old: weights = 1.0 + peak_weight * relu(y_true)
    #   → upweights all high-flow timesteps regardless of prediction quality
    # New: additional penalty only fires when |r| > 1σ at a high-flow timestep
    peak_w = 1.0 + peak_weight * tf.nn.relu(y_true) * tf.nn.relu(tf.abs(r) - 1.0)

    # ── low-flow penalty: stop arid stations predicting zero ──────────────────
    # Soft penalty when model over-predicts dryness (r < 0 near zero flow)
    low_flow_mask = tf.cast(y_true < 0.5, tf.float32)   # near-zero in log1p space
    low_flow_w    = 1.0 + 0.2 * low_flow_mask * tf.nn.relu(-r)

    weights = peak_w * low_flow_w

    # ── KGE bias correction in raw domain ────────────────────────────────────
    # Penalises systematic mean bias (the FHV blowout in Cfb/Cwa zones)
    # computed via expm1 to approximate raw-space means from log1p targets
    mu_raw    = tf.math.expm1(tf.nn.relu(mu))
    ytrue_raw = tf.math.expm1(tf.nn.relu(y_true))
    beta      = tf.reduce_mean(mu_raw) / (tf.reduce_mean(ytrue_raw) + 1e-6)
    bias_penalty = 0.1 * tf.square(beta - 1.0)

    return tf.reduce_mean(weights * nll) + bias_penalty


@tf.keras.utils.register_keras_serializable()
def asym_laplace_nll_linear(
    y_true,
    params,
    r_clip=10.0,
    scale_clip=(1e-3, 10.0),
    peak_weight=0.3
):
    """Asymmetric Laplace NLL for models trained on linear targets."""
    import tensorflow as tf

    y_true = tf.cast(y_true, tf.float32)
    params = tf.cast(params, tf.float32)

    mu = params[:, 0:1]
    log_b_plus = params[:, 1:2]
    log_b_minus = params[:, 2:3]

    b_plus = tf.clip_by_value(tf.nn.softplus(log_b_plus), scale_clip[0], scale_clip[1])
    b_minus = tf.clip_by_value(tf.nn.softplus(log_b_minus), scale_clip[0], scale_clip[1])

    r_raw = y_true - mu
    r = tf.clip_by_value(r_raw, -r_clip, r_clip)

    frac_clipped = tf.reduce_mean(tf.cast(tf.abs(r_raw) > r_clip, tf.float32))
    tf.summary.scalar("ald/frac_clipped_residuals", frac_clipped)

    nll = tf.where(
        r >= 0.0,
        tf.math.log(b_plus) + r / b_plus,
        tf.math.log(b_minus) - r / b_minus,
    )

    y_scale = tf.reduce_mean(tf.nn.relu(y_true)) + 1e-6
    peak_signal = tf.nn.relu(y_true) / y_scale
    peak_w = 1.0 + peak_weight * peak_signal * tf.nn.relu(tf.abs(r) - y_scale)

    low_flow_threshold = 0.05 * y_scale
    low_flow_mask = tf.cast(y_true < low_flow_threshold, tf.float32)
    low_flow_w = 1.0 + 0.2 * low_flow_mask * tf.nn.relu(-r / y_scale)

    weights = peak_w * low_flow_w

    mu_raw = tf.nn.relu(mu)
    ytrue_raw = tf.nn.relu(y_true)
    beta = tf.reduce_mean(mu_raw) / (tf.reduce_mean(ytrue_raw) + 1e-6)
    bias_penalty = 0.1 * tf.square(beta - 1.0)

    return tf.reduce_mean(weights * nll) + bias_penalty


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

    def _load_runoff_entries_for_period(self):
        """Load routed runoff entries and require exact daily coverage for training."""
        runoff_dir = f'{self.working_dir}/runoff_output'
        preferred_file = os.path.join(runoff_dir, 'wacc_sparse_arrays.pkl')
        candidate_files = [preferred_file] if os.path.exists(preferred_file) else sorted(glob.glob(f'{runoff_dir}/*.pkl'))

        if not candidate_files:
            raise SystemExit(f"""
ERROR: No routed runoff output found

The training pipeline could not find any routed runoff pickle files in:
  {runoff_dir}

Please run VegET runoff routing first for the period needed by training.
""".strip())

        requested_dates = pd.date_range(start=self.train_start, end=self.train_end, freq='D')
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
  {runoff_dir}

but none contained daily routed runoff entries with both 'time' and 'matrix' fields.
Please verify that VegET runoff routing completed successfully.
""".strip())

        available_dates = sorted(runoff_by_date.keys())
        missing_dates = [date_str for date_str in requested_labels if date_str not in runoff_by_date]
        if missing_dates:
            preview = ", ".join(missing_dates[:5])
            suffix = " ..." if len(missing_dates) > 5 else ""
            raise SystemExit(f"""
ERROR: Requested training period is not covered by routed runoff output

Requested training period:
  start: {self.train_start}
  end:   {self.train_end}

Available routed runoff dates:
  from:  {available_dates[0]}
  to:    {available_dates[-1]}

Missing requested dates:
  {preview}{suffix}

This usually means VegET runoff was computed for a different date range
than the one now requested for training.
Please rerun VegET for the required period or adjust train_start/train_end.
""".strip())

        return [runoff_by_date[date_str] for date_str in requested_labels]

    def _load_optional_rainfall_entries_for_period(self):
        """Load routed rainfall entries if available and fully aligned."""
        rainfall_file = os.path.join(self.working_dir, "runoff_output", "rainfall_sparse_arrays.pkl")
        if not os.path.exists(rainfall_file):
            print("     Routed rainfall not found; using routed runoff only.")
            return None

        requested_dates = pd.date_range(start=self.train_start, end=self.train_end, freq="D")
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
                "Routed rainfall output is present but does not cover the requested training period. "
                f"Missing dates: {preview}{suffix}. "
                "Regenerate routed rainfall for the full period, or remove rainfall_sparse_arrays.pkl "
                "to train a runoff-only model."
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
        
        pysheds_grid = _load_pysheds_grid()
        grid = pysheds_grid.Grid.from_raster(dem_filepath)
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
        wfa_list = self._load_runoff_entries_for_period()
        rainfall_list = self._load_optional_rainfall_entries_for_period()
        
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
        with open(_predictor_cache_path(self.working_dir), 'wb') as file:
                pickle.dump(
                    _build_predictor_cache_payload(
                        self.data_list,
                        self.train_start,
                        self.train_end,
                    ),
                    file,
                )
            
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

    def __init__(self, working_dir, batch_size, num_epochs, loss_function,
                 learning_rate=1e-4,  train_start=None, train_end=None, seed=100,
                 area_normalize=True, log_transform=True, lr_schedule=None,
                 warmup_epochs=3, min_learning_rate=1e-5):
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
        log_transform : bool
            Whether to apply log1p to temporal predictors and response targets.
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
        self.train_p14 = None
        self.train_p180 = None
        self.train_p365 = None
        self.train_response = None
        self.train_alphaearth = None
        self.train_area = None
        self.predictor_columns = []

        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.seed = seed
        self.area_normalize = area_normalize
        self.log_transform = bool(log_transform)
        self.lr_schedule = lr_schedule
        self.warmup_epochs = int(warmup_epochs or 0)
        self.min_learning_rate = float(min_learning_rate)

        try:
            tf.config.optimizer.set_jit(True)
        except Exception:
            pass

    def _model_config(self):
        """Return inference-critical options that must match this checkpoint."""
        loss_name = (
            self.loss_function
            if isinstance(self.loss_function, str)
            else getattr(self.loss_function, "__name__", str(self.loss_function))
        )
        return {
            "config_version": MODEL_CONFIG_VERSION,
            "area_normalize": bool(self.area_normalize),
            "log_transform": bool(self.log_transform),
            "loss_function": loss_name,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "predictor_columns": list(self.predictor_columns),
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }

    def save_model_config(self, model_path):
        """Save inference-critical checkpoint settings next to the model."""
        config = self._model_config()
        for path in model_config_paths(model_path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as file:
                json.dump(config, file, indent=2, sort_keys=True)
        return config

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
        self.predictor_columns = list(train_predictors[0].columns) if train_predictors else []

        full_train_p14 = []
        full_train_p180 = []
        full_train_p365 = []
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
            if self.log_transform:
                scaled_train_predictor = np.log1p(scaled_train_predictor)

            if self.area_normalize:
                scaled_train_response = (y.values * 86400 * 1000) / area_m2
            else:
                scaled_train_response = y.values
            if self.log_transform:
                scaled_train_response = np.log1p(scaled_train_response)

            z2 = z.reshape(-1, 64)
            scaled_alphaearth = alphaearth_scaler.transform(z2)

            num_samples = scaled_train_predictor.shape[0] - 365 - 1
            if num_samples <= 0:
                continue

            p14_samples = []
            p180_samples = []
            p365_samples = []
            response_samples = []
            alphaearth_samples = []
            area_samples = []

            for i in range(num_samples):
                full_window = scaled_train_predictor[i:i + 365, :]

                p14_samples.append(full_window[-14:, :])
                p180_samples.append(full_window[-180:, :])
                p365_samples.append(full_window)

                response_batch = scaled_train_response[i + 365].reshape(1)
                response_samples.append(response_batch)

                alphaearth_samples.append(scaled_alphaearth)
                area_samples.append(j.reshape(1))

            timesteps_to_keep = []
            for i in range(num_samples):
                if (
                    not np.isnan(p14_samples[i]).any()
                    and not np.isnan(p180_samples[i]).any()
                    and not np.isnan(p365_samples[i]).any()
                    and not np.isnan(response_samples[i]).any()
                ):
                    timesteps_to_keep.append(i)

            timesteps_to_keep = np.array(timesteps_to_keep, dtype=np.int64)

            if len(timesteps_to_keep) > 0:
                full_train_p14.append(np.array(p14_samples)[timesteps_to_keep])
                full_train_p180.append(np.array(p180_samples)[timesteps_to_keep])
                full_train_p365.append(np.array(p365_samples)[timesteps_to_keep])

                full_train_response.append(np.array(response_samples)[timesteps_to_keep])
                full_alphaearth.append(np.array(alphaearth_samples)[timesteps_to_keep])
                full_area.append(np.array(area_samples)[timesteps_to_keep])

        if not full_train_p14:
            raise ValueError("No valid training windows were created. Check date range and NaN coverage.")

        self.train_p14 = np.concatenate(full_train_p14, axis=0).astype("float32")
        self.train_p180 = np.concatenate(full_train_p180, axis=0).astype("float32")
        self.train_p365 = np.concatenate(full_train_p365, axis=0).astype("float32")
        if self.train_p14.shape[-1] > 2:
            self.train_p14 = self.train_p14[:, :, :2]
            self.train_p180 = self.train_p180[:, :, :2]
            self.train_p365 = self.train_p365[:, :, :2]
            self.predictor_columns = self.predictor_columns[:2]

        self.train_response = np.concatenate(full_train_response, axis=0).astype("float32")
        self.train_alphaearth = np.concatenate(full_alphaearth, axis=0).reshape(-1, 64).astype("float32")
        self.train_area = np.concatenate(full_area, axis=0).reshape(-1, 1).astype("float32")

    def print_training_summary(self, station_count=None):
        """Print a compact summary of prepared training tensors."""
        if self.train_p14 is None:
            return

        station_text = "unknown" if station_count is None else str(station_count)
        predictors = ", ".join(self.predictor_columns) if self.predictor_columns else "unknown"
        print("     Training data prepared:")
        print(f"       stations: {station_text}")
        print(f"       samples: {self.train_p14.shape[0]}")
        print(
            "       temporal inputs: "
            f"p14={tuple(self.train_p14.shape[1:])}, "
            f"p180={tuple(self.train_p180.shape[1:])}, "
            f"p365={tuple(self.train_p365.shape[1:])}"
        )
        print(f"       predictors: {predictors}")
        print(f"       loss: {self.loss_function}")
        print(f"       area_normalize: {self.area_normalize}")
        print(f"       log_transform: {self.log_transform}")
        print(f"       batch_size: {self.batch_size}, epochs: {self.num_epochs}")

    # --------------------------------------------------
    # MODEL DEFINITION
    # --------------------------------------------------
    def build_model(self):
        """
        Multi-scale TCN with sequence FiLM, adaptive timescale weighting,
        attention pooling, rainfall-runoff gating, and persistence correction.
        """
        strategy = tf.distribute.MirroredStrategy()
        print(f"GPUs in sync: {strategy.num_replicas_in_sync}")

        with strategy.scope():
            if self.train_p14 is None:
                raise ValueError("Call prepare_data() before build_model().")

            if (
                isinstance(self.loss_function, str)
                and self.loss_function.lower() == "asym_laplace_nll"
                and not self.log_transform
            ):
                self.loss_function = asym_laplace_nll_linear
            elif self.loss_function is asym_laplace_nll and not self.log_transform:
                self.loss_function = asym_laplace_nll_linear

            loss_name = (
                self.loss_function.lower()
                if isinstance(self.loss_function, str)
                else getattr(self.loss_function, "__name__", "").lower()
            )
            uses_asym_laplace = (
                self.loss_function in {asym_laplace_nll, asym_laplace_nll_linear}
                or loss_name in {"asym_laplace_nll", "asym_laplace_nll_linear"}
            )

            n_features = int(self.train_p14.shape[-1])
            if n_features > 2:
                n_features = 2

            in14 = Input((14, n_features), name="input_14d")
            in180 = Input((180, n_features), name="input_180d")
            in365 = Input((365, n_features), name="input_365d")
            in_alpha = Input((64,), name="alphaearth")
            in_area = Input((1,), name="area")

            alpha_latent = Dense(64, activation="relu")(in_alpha)
            alpha_latent = LayerNormalization()(alpha_latent)

            area_latent = Dense(8, activation="relu")(in_area)
            area_latent = LayerNormalization()(area_latent)

            cond = Concatenate(name="static_cond")([alpha_latent, area_latent])

            def film(cond_input, dim, name):
                x = Dense(64, activation="relu")(cond_input)
                x = Dense(64, activation="relu")(x)
                gamma = Dense(dim)(x)
                beta = Dense(dim)(x)
                gamma = ScaleShift(0.4, 1.0, name=f"{name}_gamma")(gamma)
                beta = ScaleShift(0.4, 0.0, name=f"{name}_beta")(beta)
                return gamma, beta

            def apply_film_sequence(seq, gamma, beta, name):
                gamma_exp = ExpandDims(axis=1, name=f"{name}_gamma_expand")(gamma)
                beta_exp = ExpandDims(axis=1, name=f"{name}_beta_expand")(beta)
                return Add(name=f"{name}_film")([
                    Multiply(name=f"{name}_film_mul")([seq, gamma_exp]),
                    beta_exp,
                ])

            def tcn_block(x, filters, kernel, dilations, name):
                y = TCN(
                    nb_filters=filters,
                    kernel_size=kernel,
                    dilations=dilations,
                    return_sequences=True,
                    name=name,
                )(x)
                return LayerNormalization(name=f"{name}_ln")(y)

            def attention_pool(seq, name):
                scores = Dense(16, activation="relu", name=f"{name}_attn_hidden")(seq)
                scores = Dense(1, name=f"{name}_attn_score")(scores)
                weights = Softmax(axis=1, name=f"{name}_attn_weights")(scores)
                weighted = Multiply(name=f"{name}_attn_apply")([seq, weights])
                return ReduceSum(axis=1, name=f"{name}_attn_pool")(weighted)

            b14_seq = tcn_block(in14, 32, 3, (1, 2, 4), "tcn_14")
            b180_seq = tcn_block(in180, 32, 5, (1, 2, 4, 8, 16, 32), "tcn_180")
            b365_seq = tcn_block(in365, 32, 7, (1, 2, 4, 8, 16, 32), "tcn_365")

            g14, beta14 = film(cond, 32, "b14")
            g180, beta180 = film(cond, 32, "b180")
            g365, beta365 = film(cond, 32, "b365")

            b14_seq = apply_film_sequence(b14_seq, g14, beta14, "b14")
            b180_seq = apply_film_sequence(b180_seq, g180, beta180, "b180")
            b365_seq = apply_film_sequence(b365_seq, g365, beta365, "b365")

            b14 = attention_pool(b14_seq, "b14")
            b180 = attention_pool(b180_seq, "b180")
            b365 = attention_pool(b365_seq, "b365")

            weight_context = Concatenate(name="weight_context")([cond, b14, b180, b365])
            w = Dense(3, activation="softmax", name="timescale_weights")(weight_context)
            w14 = Slice1D(0, 1, name="w_14")(w)
            w180 = Slice1D(1, 2, name="w_180")(w)
            w365 = Slice1D(2, 3, name="w_365")(w)

            temporal = Add(name="temporal_fusion")([
                Multiply(name="weighted_b14")([b14, w14]),
                Multiply(name="weighted_b180")([b180, w180]),
                Multiply(name="weighted_b365")([b365, w365]),
            ])

            h = Dense(128, name="head_dense_1")(temporal)
            h = LeakyReLU(negative_slope=0.01, name="head_act_1")(h)
            h = LayerNormalization(name="head_ln_1")(h)
            h = Dropout(0.1, name="head_drop")(h)

            h = Dense(64, name="head_dense_2")(h)
            h = LeakyReLU(negative_slope=0.01, name="head_act_2")(h)
            h = LayerNormalization(name="head_ln_2")(h)

            tcn_pred = Dense(1, name="tcn_pred")(h)

            last_runoff = SliceTimestep(-1, 0, 1, name="last_runoff")(in14)
            if n_features >= 2:
                last_rain = SliceTimestep(-1, 1, 2, name="last_rain")(in14)
                pers_features = [last_runoff, last_rain]
            else:
                last_rain = None
                pers_features = [last_runoff]

            pers_input = (
                Concatenate(name="pers_input")(pers_features)
                if len(pers_features) > 1
                else last_runoff
            )
            pers_pred = Dense(16, activation="relu", name="pers_dense")(pers_input)
            pers_pred = Dense(1, name="pers_out")(pers_pred)

            gate_features = [h, last_runoff, cond]
            if last_rain is not None:
                gate_features.insert(2, last_rain)
            gate_input = Concatenate(name="gate_input")(gate_features)
            gate = Dense(1, activation="sigmoid", name="rain_runoff_gate")(gate_input)
            one_minus_gate = OneMinus(name="one_minus_gate")(gate)

            mu = Add(name="mu")([
                Multiply(name="gate_tcn")([gate, tcn_pred]),
                Multiply(name="gate_pers")([one_minus_gate, pers_pred]),
            ])

            scale_params = Dense(2, name="scale_params")(h)
            streamflow = Concatenate(name="streamflow")([mu, scale_params])
            out = streamflow if uses_asym_laplace else mu

            self.regional_model = Model(
                inputs=[in14, in180, in365, in_alpha, in_area],
                outputs=out,
                name="bakaano_hydro",
            )

            self.regional_model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.learning_rate,
                    clipnorm=1.0,
                ),
                loss=self.loss_function,
                steps_per_execution=8,
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

        model_path = f"{self.working_dir}/models/bakaano_model.keras"
        self.save_model_config(model_path)

        checkpoint = ModelCheckpoint(
            filepath=model_path,
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
                self.train_p14,
                self.train_p180,
                self.train_p365,
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
        self.regional_model.save(model_path)
        validate_keras_archive(model_path)
        self.save_model_config(model_path)

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
                self.regional_model = load_model(path, custom_objects=custom_objects)

    def regional_summary(self):
        """
        Print the Keras model summary.
        """
        if self.regional_model is None:
            raise ValueError("No model loaded/built yet.")
        self.regional_model.summary()
