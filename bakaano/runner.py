"""High-level orchestration for Bakaano-Hydro workflows.

Role: Provide a user-facing API to train, evaluate, and simulate streamflow.
"""

import numpy as np
import pandas as pd
import os
from bakaano.utils import Utils
import importlib.util
from pathlib import Path
from bakaano.router import RunoffRouter
import hydroeval
import matplotlib.pyplot as plt
import xarray as xr
import rasterio
import glob
import pickle
import pandas as pd
import geopandas as gpd
from datetime import datetime
from leafmap.foliumap import Map

# Load "copy" modules by filename to avoid import issues with spaces in filenames.
_here = Path(__file__).resolve().parent
_trainer_mod_path = _here / "streamflow_trainer copy.py"
_sim_mod_path = _here / "streamflow_simulator copy.py"

def _load_module(mod_name, path):
    """Load a Python module from an explicit file path.

    Args:
        mod_name (str): Name to assign the loaded module.
        path (pathlib.Path): Path to the module file.

    Returns:
        module: Imported module object.
    """
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module

if _trainer_mod_path.exists():
    _trainer_mod = _load_module("streamflow_trainer_copy", _trainer_mod_path)
else:
    from bakaano import streamflow_trainer as _trainer_mod

if _sim_mod_path.exists():
    _sim_mod = _load_module("streamflow_simulator_copy", _sim_mod_path)
else:
    from bakaano import streamflow_simulator as _sim_mod

DataPreprocessor = _trainer_mod.DataPreprocessor
StreamflowModel = _trainer_mod.StreamflowModel
PredictDataPreprocessor = _sim_mod.PredictDataPreprocessor
PredictStreamflow = _sim_mod.PredictStreamflow

#========================================================================================================================  
class BakaanoHydro:
    """Main user-facing interface for Bakaano-Hydro workflows."""
    def __init__(self, working_dir, study_area, climate_data_source):
        """Initialize the BakaanoHydro object with project details.

        Args:
            working_dir (str): The parent working directory where files and outputs will be stored.
            study_area (str): The path to the shapefile of the river basin or watershed.
            climate_data_source (str): The source of climate data: 'CHELSA', 'ERA5', or 'CHIRPS'.

        Methods
        -------
        __init__(working_dir, study_area_path, start_date, end_date, climate_data_source):
            Initializes the BakaanoHydro object with project details.
        train_streamflow_model(grdc_netcdf, prep_nc, tasmax_nc, tasmin_nc, tmean_nc, loss_fn, num_input_branch, lookback, batch_size, num_epochs):
            Train the deep learning streamflow prediction model.
        evaluate_streamflow_model(model_path, grdc_netcdf, prep_nc, tasmax_nc, tasmin_nc, tmean_nc, loss_fn, num_input_branch, lookback, batch_size):
            Evaluate the streamflow prediction model.
        simulate_streamflow(model_path, latlist, lonlist, prep_nc, tasmax_nc, tasmin_nc, tmean_nc, loss_fn, num_input_branch, lookback, batch_size):
            Simulate streamflow using the trained model.
        simulate_streamflow_batch(model_path, latlist, lonlist, prep_nc, tasmax_nc, tasmin_nc, tmean_nc, loss_fn, num_input_branch, lookback):
            Simulate streamflow in batch mode using the trained model.
        plot_grdc_streamflow(observed_streamflow, predicted_streamflow, loss_fn):
            Plot the observed and predicted streamflow data.
        compute_metrics(observed_streamflow, predicted_streamflow, loss_fn):
            Compute performance metrics for the model.

        """
         # Initialize the project name
        self.working_dir = working_dir
        self.climate_data_source = climate_data_source
        
        # Initialize the study area
        self.study_area = study_area
        
        # Initialize utility class with project name and study area.
        self.uw = Utils(self.working_dir, self.study_area)
        
        # Set the start and end dates for the project

        # Create necessary directories for the project structure   
        os.makedirs(f'{self.working_dir}/models', exist_ok=True)
        os.makedirs(f'{self.working_dir}/runoff_output', exist_ok=True)
        os.makedirs(f'{self.working_dir}/scratch', exist_ok=True)
        os.makedirs(f'{self.working_dir}/shapes', exist_ok=True)
        os.makedirs(f'{self.working_dir}/catchment', exist_ok=True)
        os.makedirs(f'{self.working_dir}/predicted_streamflow_data', exist_ok=True)
      
        self.clipped_dem = f'{self.working_dir}/elevation/dem_clipped.tif'

#=========================================================================================================================================
    def train_streamflow_model(
        self,
        train_start,
        train_end,
        grdc_netcdf,
        batch_size,
        num_epochs,
        learning_rate=0.0001,
        loss_function="mse",
        seed=100,
        routing_method="mfd",
        catchment_size_threshold=1,
        csv_dir=None,
        lookup_csv=None,
        id_col="id",
        lat_col="latitude",
        lon_col="longitude",
        date_col="date",
        discharge_col="discharge",
        file_pattern="{id}.csv",
    ):
        """Train the deep learning streamflow prediction model.

        Args:
            train_start (str): Training start date (YYYY-MM-DD).
            train_end (str): Training end date (YYYY-MM-DD).
            grdc_netcdf (str): GRDC NetCDF path (if using GRDC data).
            batch_size (int): Training batch size.
            num_epochs (int): Number of training epochs.
            learning_rate (float): Optimizer learning rate.
            loss_function (str): Loss name (e.g., "mse").
            seed (int): Random seed for sampling.
            routing_method (str): Routing method ("mfd", "d8", "dinf").
            catchment_size_threshold (float): Minimum catchment size for stations.
            csv_dir (str, optional): Directory of per-station CSVs.
            lookup_csv (str, optional): CSV lookup file with station coords.
            id_col (str): Station id column in lookup CSV.
            lat_col (str): Latitude column in lookup CSV.
            lon_col (str): Longitude column in lookup CSV.
            date_col (str): Date column in station CSVs.
            discharge_col (str): Discharge column in station CSVs.
            file_pattern (str): Filename pattern for station CSVs.
        """

        rawdata = glob.glob(f'{self.working_dir}/models/*_predictor_response*.pkl')
    
        print('\nTRAINING BAKAANO-HYDRO DEEP LEARNING STREAMFLOW PREDICTION MODEL')
        
            
        sdp = DataPreprocessor(self.working_dir, self.study_area, grdc_netcdf, train_start, train_end, 
                               routing_method, catchment_size_threshold)
        print(' 1. Loading observed streamflow')
        if csv_dir and lookup_csv:
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
        else:
            sdp.load_observed_streamflow(grdc_netcdf)
        print(' 2. Loading runoff data and other predictors')
        if len(rawdata) > 0:
            with open(rawdata[0], "rb") as f:
                self.rawdata = pickle.load(f)
        else:
            self.rawdata = sdp.get_data()
        sn = str(len(sdp.sim_station_names))

        # Normalize station_ids to a set (supports single int or iterable)
        if isinstance(sdp.station_ids, (list, tuple, set)):
            target_ids = set(sdp.station_ids)
        else:
            target_ids = {sdp.station_ids}
        
        filtered = [
            item for item in self.rawdata
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
        
        self.rawdata = filtered

        try:
            # --- Parse dates ---
            start_dt = datetime.strptime(train_start, "%Y-%m-%d")
            end_dt   = datetime.strptime(train_end, "%Y-%m-%d")
        
            # --- Sanity check on runoff data ---
            if not self.rawdata:
                raise SystemExit(
                    "No runoff data loaded. "
                    "Check the runoff_output directory and pickle files."
                )
        
            # Use the runoff dataframe of the first entry as reference
            df_runoff = self.rawdata[0][0]   # first element of tuple
        
            # --- Ensure datetime index ---
            if not isinstance(df_runoff.index, pd.DatetimeIndex):
                df_runoff.index = pd.to_datetime(df_runoff.index)
        
            # --- Available date range ---
            available_start = df_runoff.index.min()
            available_end   = df_runoff.index.max()
        
            # --- Explicit presence check ---
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
            # Re-raise ValueErrors unchanged (user-facing, informative)
            raise
        
        except Exception as e:
            # Catch-all for unexpected issues
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
                """.strip()
            ) from e
        
        print(f'     Training deepstrmm model based on {sn} stations in the GRDC database')
        print(sdp.sim_station_names)
        
        print(' 3. Building neural network model')
        smodel = StreamflowModel(
            self.working_dir,
            batch_size,
            num_epochs,
            learning_rate,
            loss_function,
            train_start,
            train_end,
            seed=seed,
        )
        smodel.prepare_data(self.rawdata)
        smodel.build_model()
        print(' 4. Training neural network model')
        smodel.train_model()
        print(f'     Completed! Trained model saved at {self.working_dir}/models/bakaano_model.keras')
#========================================================================================================================  
                
    def evaluate_streamflow_model_interactively(
        self,
        model_path,
        val_start,
        val_end,
        grdc_netcdf,
        routing_method="mfd",
        catchment_size_threshold=1000,
        csv_dir=None,
        lookup_csv=None,
        id_col="id",
        lat_col="latitude",
        lon_col="longitude",
        date_col="date",
        discharge_col="discharge",
        file_pattern="{id}.csv",
    ):
        """Interactively evaluate a trained streamflow model.

        Args:
            model_path (str): Path to a trained Keras model.
            val_start (str): Validation start date (YYYY-MM-DD).
            val_end (str): Validation end date (YYYY-MM-DD).
            grdc_netcdf (str): GRDC NetCDF path (if using GRDC data).
            routing_method (str): Routing method ("mfd", "d8", "dinf").
            catchment_size_threshold (float): Minimum catchment size for stations.
            csv_dir (str, optional): Directory of per-station CSVs.
            lookup_csv (str, optional): CSV lookup file with station coords.
            id_col (str): Station id column in lookup CSV.
            lat_col (str): Latitude column in lookup CSV.
            lon_col (str): Longitude column in lookup CSV.
            date_col (str): Date column in station CSVs.
            discharge_col (str): Discharge column in station CSVs.
            file_pattern (str): Filename pattern for station CSVs.
        """

        vdp = PredictDataPreprocessor(
            self.working_dir,
            self.study_area,
            val_start,
            val_end,
            routing_method,
            grdc_netcdf,
            catchment_size_threshold,
        )

        if csv_dir and lookup_csv:
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
            print(vdp.station_ids)
            station_id = input("\n Please enter the station id: ")
            vdp.station_ids = np.unique([str(station_id)])
            self.station = vdp.observed_streamflow_csv.get(str(station_id))
            if self.station is None:
                raise SystemExit("Station id not found in observed CSV directory.")
        else:
            fulldata = vdp.load_observed_streamflow(grdc_netcdf)
            self.stat_names = vdp.sim_station_names
            print("Available station names:")
            print(self.stat_names)

            station_name = input("\n Please enter the station name: ")
            
            extracted_data = fulldata.where(fulldata.station_name.astype(str) == station_name, drop=True)
            full_ids = list(extracted_data.id.values)
            
            self.station = extracted_data['runoff_mean'].where(
                extracted_data['station_name'] == station_name,
                drop=True,
            ).to_dataframe(name='station_discharge').reset_index()

            station_id = self.station['id'][0]
            station_index = full_ids.index(station_id)

            vdp.station_ids = np.unique([full_ids[station_index]])
        
        rawdata = vdp.get_data()
        
        observed_streamflow = list(map(lambda xy: xy[1], rawdata))

        self.vmodel = PredictStreamflow(self.working_dir)
        self.vmodel.prepare_data(rawdata)

        self.vmodel.load_model(model_path)

        predicted_streamflow = self.vmodel.model.predict(
            [
                self.vmodel.sim_45d,
                self.vmodel.sim_90d,
                self.vmodel.sim_180d,
                self.vmodel.sim_365d,
                self.vmodel.sim_alphaearth,
                self.vmodel.sim_area,
            ]
        )
        if predicted_streamflow.ndim == 2 and predicted_streamflow.shape[1] == 3:
            predicted_streamflow = predicted_streamflow[:, 0:1]

        predicted_streamflow = np.expm1(predicted_streamflow)
        predicted_streamflow = (predicted_streamflow * self.vmodel.catch_area * 1_000_000.0) / (86400 * 1000)
        predicted_streamflow = np.where(predicted_streamflow < 0, 0, predicted_streamflow) 

        self._plot_grdc_streamflow(observed_streamflow, predicted_streamflow,  val_start)
        
#==============================================================================================================================
    def simulate_streamflow(self, model_path, sim_start, sim_end, latlist, lonlist, 
                            routing_method='mfd'):
        """Simulate streamflow for given coordinates using a trained model.

        Args:
            model_path (str): Path to a trained Keras model.
            sim_start (str): Simulation start date (YYYY-MM-DD).
            sim_end (str): Simulation end date (YYYY-MM-DD).
            latlist (list[float]): List of latitudes.
            lonlist (list[float]): List of longitudes.
            routing_method (str): Routing method ("mfd", "d8", "dinf").
        """
        print(' 1. Loading runoff data and other predictors')
        vdp = PredictDataPreprocessor(self.working_dir, self.study_area, sim_start, sim_end, routing_method)
        rawdata = vdp.get_data_latlng(latlist, lonlist)

        self.vmodel = PredictStreamflow(self.working_dir)
        self.vmodel.prepare_data_latlng(rawdata)
        batch_size = len(latlist)
        self.vmodel.load_model(model_path)
        print(' 2. Batch prediction')
        predicted_streamflows = self.vmodel.model.predict(
            [
                self.vmodel.sim_45d,
                self.vmodel.sim_90d,
                self.vmodel.sim_180d,
                self.vmodel.sim_365d,
                self.vmodel.sim_alphaearth,
                self.vmodel.sim_area,
            ],
            batch_size=batch_size,
        )
        if predicted_streamflows.ndim == 2 and predicted_streamflows.shape[1] == 3:
            predicted_streamflows = predicted_streamflows[:, 0:1]

        predicted_streamflows = np.expm1(predicted_streamflows)
        seq = int(len(predicted_streamflows)/batch_size)
        predicted_streamflows = predicted_streamflows.reshape(batch_size, seq, 1)

        predicted_streamflow_list = []
        for predicted_streamflow, catch_area in zip(predicted_streamflows, self.vmodel.catch_area_list):
            predicted_streamflow = (predicted_streamflow * catch_area * 1_000_000.0) / (86400 * 1000)
            predicted_streamflow = np.where(predicted_streamflow < 0, 0, predicted_streamflow)
            
            predicted_streamflow_list.append(predicted_streamflow)
        print(' 3. Generating csv file for each coordinate')
        for predicted_streamflow, lat, lon in zip(predicted_streamflow_list, latlist, lonlist):
            predicted_streamflow = predicted_streamflow.reshape(-1)

            adjusted_start_date = pd.to_datetime(sim_start) + pd.DateOffset(days=365)
            period = pd.date_range(adjusted_start_date, periods=len(predicted_streamflow), freq='D')  # Match time length with mu
            df = pd.DataFrame({
                'time': period,  # Adjusted time column
                'streamflow (m3/s)': predicted_streamflow
            })
            output_path = os.path.join(self.working_dir, f"predicted_streamflow_data/predicted_streamflow_lat{lat}_lon{lon}.csv")
            df.to_csv(output_path, index=False)
        out_folder = os.path.join(self.working_dir, 'predicted_streamflow_data')
        print(f' COMPLETED! csv files available at {out_folder}')

#==============================================================================================================================
    def simulate_grdc_csv_stations(
        self,
        model_path,
        sim_start,
        sim_end,
        grdc_netcdf,
        routing_method="mfd",
        csv_dir=None,
        lookup_csv=None,
        id_col="id",
        lat_col="latitude",
        lon_col="longitude",
        date_col="date",
        discharge_col="discharge",
        file_pattern="{id}.csv",
    ):
        """Simulate streamflow for GRDC or CSV stations in batch.

        Args:
            model_path (str): Path to a trained Keras model.
            sim_start (str): Simulation start date (YYYY-MM-DD).
            sim_end (str): Simulation end date (YYYY-MM-DD).
            grdc_netcdf (str): GRDC NetCDF path (if using GRDC data).
            routing_method (str): Routing method ("mfd", "d8", "dinf").
            csv_dir (str, optional): Directory of per-station CSVs.
            lookup_csv (str, optional): CSV lookup file with station coords.
            id_col (str): Station id column in lookup CSV.
            lat_col (str): Latitude column in lookup CSV.
            lon_col (str): Longitude column in lookup CSV.
            date_col (str): Date column in station CSVs.
            discharge_col (str): Discharge column in station CSVs.
            file_pattern (str): Filename pattern for station CSVs.
        """
        print(' 1. Loading runoff data and other predictors')
        vdp = PredictDataPreprocessor(
            self.working_dir,
            self.study_area,
            sim_start,
            sim_end,
            routing_method,
            grdc_netcdf,
        )
    
        if csv_dir and lookup_csv:
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
            print(vdp.station_ids)
        else:
            self.stat_names = vdp.sim_station_names
            print("Available station names:")
            print(self.stat_names)

        rawdata = vdp.get_data()

        self.vmodel = PredictStreamflow(self.working_dir)
        self.vmodel.prepare_data(rawdata)
        batch_size = len(vdp.station_ids)
        self.vmodel.load_model(model_path)
        print(' 2. Batch prediction')
        predicted_streamflows = self.vmodel.model.predict(
            [
                self.vmodel.sim_45d,
                self.vmodel.sim_90d,
                self.vmodel.sim_180d,
                self.vmodel.sim_365d,
                self.vmodel.sim_alphaearth,
                self.vmodel.sim_area,
            ],
            batch_size=batch_size,
        )
        if predicted_streamflows.ndim == 2 and predicted_streamflows.shape[1] == 3:
            predicted_streamflows = predicted_streamflows[:, 0:1]

        predicted_streamflows = np.expm1(predicted_streamflows)
        seq = int(len(predicted_streamflows)/batch_size)
        predicted_streamflows = predicted_streamflows.reshape(batch_size, seq, 1)

        predicted_streamflow_list = []
        for predicted_streamflow, catch_area in zip(predicted_streamflows, self.vmodel.catch_area_list):
            predicted_streamflow = (predicted_streamflow * catch_area * 1000000.0) / (86400 * 1000)
            predicted_streamflow = np.where(predicted_streamflow < 0, 0, predicted_streamflow)
            
            predicted_streamflow_list.append(predicted_streamflow)
        print(' 3. Generating csv file for each coordinate')
        for predicted_streamflow, snames, sids in zip(predicted_streamflow_list, vdp.sim_station_names, vdp.station_ids):
            predicted_streamflow = predicted_streamflow.reshape(-1)

            adjusted_start_date = pd.to_datetime(sim_start) + pd.DateOffset(days=365)
            period = pd.date_range(adjusted_start_date, periods=len(predicted_streamflow), freq='D')  # Match time length with mu
            df = pd.DataFrame({
                'time': period,  # Adjusted time column
                'streamflow (m3/s)': predicted_streamflow
            })
            output_path = os.path.join(self.working_dir, f"predicted_streamflow_data/bakaano_{sids}.csv")
            df.to_csv(output_path, index=False)
        out_folder = os.path.join(self.working_dir, 'predicted_streamflow_data')
        print(f' COMPLETED! csv files available at {out_folder}')

#========================================================================================================================  
            
    def _plot_grdc_streamflow(self, observed_streamflow, predicted_streamflow, val_start):
        """Plot the observed and predicted streamflow data.

        Args:
            observed_streamflow (list[pd.DataFrame]): Observed discharge data.
            predicted_streamflow (np.ndarray): Predicted discharge array.
            val_start (str): Validation start date (YYYY-MM-DD).
        """
        nse, kge = self._compute_metrics(observed_streamflow, predicted_streamflow)
        kge1 = kge[0][0]
        R = kge[1][0]
        Beta = kge[2][0]
        Alpha = kge[3][0]

        start_date = pd.to_datetime(val_start) + pd.Timedelta(days=365)
        num_days = len(predicted_streamflow)
        date_range = pd.date_range(start=start_date, periods=num_days, freq='D')

        print(f"Nash-Sutcliffe Efficiency (NSE): {nse}")
        print(f"Kling-Gupta Efficiency (KGE): {kge1}")
        plt.plot(date_range, predicted_streamflow[:], color='blue', label='Predicted Streamflow')
        plt.plot(date_range, observed_streamflow[0]['station_discharge'][365:].values[:], color='red', label='Observed Streamflow')
        plt.title('Comparison of observed and simulated streamflow')  # Add a title
        plt.xlabel('Date')  # Label the x-axis
        plt.ylabel('River Discharge (m³/s)')
        plt.legend()  # Add a legend to label the lines
        plt.show()

#========================================================================================================================  
        
    def _compute_metrics(self, observed_streamflow, predicted_streamflow):
        """Compute performance metrics for the model.

        Args:
            observed_streamflow (list[pd.DataFrame]): Observed discharge data.
            predicted_streamflow (np.ndarray): Predicted discharge array.

        Returns:
            tuple: (nse, kge) metric values from hydroeval.
        """
        observed = observed_streamflow[0]['station_discharge'][365:].values
        predicted = predicted_streamflow[:, 0].flatten()
        nan_indices = np.isnan(observed) | np.isnan(predicted)
        observed = observed[~nan_indices]
        predicted = predicted[~nan_indices]
        nse = hydroeval.nse(predicted, observed)
        kge = hydroeval.kge(predicted, observed)
        return nse, kge
  
#===========================================================================================================================
    def explore_data_interactively(self, start_date, end_date, grdc_netcdf=None):
        """Launch an interactive map to explore inputs and stations.

        Args:
            start_date (str): Start date (YYYY-MM-DD) for GRDC filtering.
            end_date (str): End date (YYYY-MM-DD) for GRDC filtering.
            grdc_netcdf (str, optional): GRDC NetCDF path for stations overlay.

        Returns:
            leafmap.foliumap.Map: Interactive map object.
        """
        m = Map()
        rout = RunoffRouter(self.working_dir, self.clipped_dem, 'mfd')
        fdir, acc = rout.compute_flow_dir()

        with rasterio.open(self.clipped_dem) as dm:
            dem_profile = dm.profile

        dem_profile.update(dtype=rasterio.float32, count=1)
        acc_name = f'{self.working_dir}/scratch/river_network.tif'
        with rasterio.open(acc_name, 'w', **dem_profile) as dst:
                dst.write(acc, 1)
    

        try:
            tree_cover = f'{self.working_dir}/vcf/mean_tree_cover.tif'
            dem = f'{self.working_dir}/elevation/dem_clipped.tif'
            slope = f'{self.working_dir}/elevation/slope_clipped.tif'
            awc = f'{self.working_dir}/soil/clipped_AWCh3_M_sl6_1km_ll.tif'
        

            m.add_raster(dem, layer_name="DEM", colormap='gist_ncar', zoom_to_layer=True, opacity=0.6)
            vmx = np.nanmax(np.array(acc))*0.025
            for path, name, cmap, vmax, opacity in [
                (awc, "Available water content", "terrain", 10, 0.75),
                (tree_cover, "Tree cover", "viridis_r", 70, 0.75),
                (slope, "Slope", "gist_ncar", 20, 0.75),
                (acc_name, "River network", "viridis", vmx, 0.9),
            ]:
                try:
                    m.add_raster(path, layer_name=name, colormap=cmap, zoom_to_layer=True, opacity=opacity, 
                                 vmax=vmax, visible=False)
                except Exception as e:
                    print(f"⚠️ Failed to load raster '{name}': {e}")

        except Exception as e:
            print(f"❌ Raster setup failed: {e}")

        # Process GRDC data if provided
        if grdc_netcdf is not None:
            try:
                grdc = xr.open_dataset(grdc_netcdf)

                stations_df = pd.DataFrame({
                    'station_name': grdc['station_name'].values,
                    'geo_x': grdc['geo_x'].values,
                    'geo_y': grdc['geo_y'].values
                })

                stations_gdf = gpd.GeoDataFrame(
                    stations_df,
                    geometry=gpd.points_from_xy(stations_df['geo_x'], stations_df['geo_y']),
                    crs="EPSG:4326"
                )

                # Load region shapefile
                try:
                    region_shape = gpd.read_file(self.study_area)
                except Exception as e:
                    print(f"❌ Could not read study area shapefile: {e}")
                    return m

                # Spatial join
                stations_in_region = gpd.sjoin(stations_gdf, region_shape, how='inner', predicate='intersects')
                overlapping_station_names = stations_in_region['station_name'].unique()

                # Filter by time and station
                grdc_time_filtered = grdc.sel(time=slice(start_date, end_date))

                filtered_grdc = grdc_time_filtered.where(
                    grdc_time_filtered['station_name'].isin(overlapping_station_names), drop=True
                )

                x = filtered_grdc["geo_x"].values.flatten()
                y = filtered_grdc["geo_y"].values.flatten()
                name = filtered_grdc['station_name'].values.flatten()
                runoff = filtered_grdc["runoff_mean"]

                percent_missing = (
                    runoff.isnull().sum(dim="time") / runoff.sizes["time"] * 100
                ).round(1).values.flatten()

                df = pd.DataFrame({
                    'Station_name': name,
                    'Percent_missing': percent_missing,
                    "Longitude": x,
                    "Latitude": y
                }).dropna()

                m.add_points_from_xy(
                    data=df, x="Longitude", y="Latitude", color="brown",
                    layer_name="Stations", radius=3
                )

            except Exception as e:
                print(f"❌ Failed to process GRDC NetCDF: {e}")

        return m
