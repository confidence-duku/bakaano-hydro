import numpy as np
import pandas as pd
import os
from bakaano.utils import Utils
from bakaano.streamflow_trainer import DataPreprocessor, StreamflowModel
from bakaano.streamflow_simulator import PredictDataPreprocessor, PredictStreamflow
from bakaano.router import RunoffRouter
import hydroeval
import matplotlib.pyplot as plt
import xarray as xr
import rasterio
import pandas as pd
import geopandas as gpd
from leafmap.foliumap import Map

#========================================================================================================================  
class BakaanoHydro:
    """Generate an instance
    """
    def __init__(self, working_dir, study_area, climate_data_source):
        """Initialize the BakaanoHydro object with project details.

        Args:
            working_dir (str): The parent working directory where files and outputs will be stored.
            study_area_path (str)): The path to the shapefile of the river basin or watershed.
            start_date (str): The start date for the project in 'YYYY-MM-DD' format.
            end_date (): The end date for the project in 'YYYY-MM-DD' format.
            climate_data_source (str): The source of climate data, either 'CHELSA', 'ERA5' or 'CHIRPS'.

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
    def train_streamflow_model(self, train_start, train_end, grdc_netcdf, loss_fn, num_input_branch, 
                               lookback, batch_size, num_epochs, routing_method='mfd', catchment_size_threshold=1000):
        """Train the deep learning streamflow prediction model."
        """
    
        print('\nTRAINING BAKAANO-HYDRO DEEP LEARNING STREAMFLOW PREDICTION MODEL')
        sdp = DataPreprocessor(self.working_dir, self.study_area, grdc_netcdf, train_start, train_end, 
                               routing_method, catchment_size_threshold)
        print(' 1. Loading observed streamflow')
        sdp.load_observed_streamflow(grdc_netcdf)
        
        print(' 2. Loading runoff data and other predictors')
        self.rawdata = sdp.get_data()
        sn = str(len(sdp.sim_station_names))
        
        print(f'     Training deepstrmm model based on {sn} stations in the GRDC database')
        print(sdp.sim_station_names)
        
        print(' 3. Building neural network model')
        smodel = StreamflowModel(self.working_dir, lookback, batch_size, num_epochs)
        smodel.prepare_data(self.rawdata)
        if num_input_branch == 3:
            smodel.build_model_3_input_branches(loss_fn)
        elif num_input_branch == 2:
            smodel.build_model_2_input_branches(loss_fn)
        else:
            raise ValueError("Invalid number. Choose either 2 or 3")
        print(' 4. Training neural network model')
        smodel.train_model(loss_fn, num_input_branch)
        print(f'     Completed! Trained model saved at {self.working_dir}/models/bakaano_model_{loss_fn}_{num_input_branch}_branches.keras')
#========================================================================================================================  
                
    def evaluate_streamflow_model_interactively(self, model_path, val_start, val_end, grdc_netcdf, loss_fn, 
                                                num_input_branch, lookback, routing_method='mfd', catchment_size_threshold=1000):
        """Evaluate the streamflow prediction model."
        """

        vdp = PredictDataPreprocessor(self.working_dir, self.study_area, val_start, val_end, routing_method, 
                                      grdc_netcdf, catchment_size_threshold)
        fulldata = vdp.load_observed_streamflow(grdc_netcdf)
        self.stat_names = vdp.sim_station_names
        print("Available station names:")
        print(self.stat_names)

        station_name = input("\n Please enter the station name: ")
        
        extracted_data = fulldata.where(fulldata.station_name.astype(str) == station_name, drop=True)
        full_ids = list(extracted_data.id.values)
        
        self.station = extracted_data['runoff_mean'].where(extracted_data['station_name'] == station_name, 
                                                drop=True).to_dataframe(name='station_discharge').reset_index()

        station_id = self.station['id'][0]
        station_index = full_ids.index(station_id)

        vdp.station_ids = np.unique([full_ids[station_index]])
        
        rawdata = vdp.get_data()
        observed_streamflow = list(map(lambda xy: xy[1], rawdata))

        self.vmodel = PredictStreamflow(self.working_dir, lookback)
        self.vmodel.prepare_data(rawdata)

        self.vmodel.load_model(model_path, loss_fn)
        if num_input_branch == 3:
            predicted_streamflow = self.vmodel.model.predict([self.vmodel.predictors, self.vmodel.local_predictors, self.vmodel.catchment_size])
        elif num_input_branch == 2:
            predicted_streamflow = self.vmodel.model.predict([self.vmodel.predictors, self.vmodel.catchment_size])
        else:
            raise ValueError("Invalid number. Choose either 2 or 3")

        if loss_fn == 'laplacian_nll':
        
            mu_log = predicted_streamflow[:, 0]  # Mean in log-space
            b_log = predicted_streamflow[:, 1]  # Uncertainty in log-space
            
            # ✅ Convert back to original streamflow units
            mu = np.exp(mu_log) - 1  # Mean streamflow prediction
            sigma = (np.exp(mu_log) - 1) * (np.exp(b_log) - 1) # Uncertainty in original scale
            
            predicted_streamflow = mu
        else:
            predicted_streamflow = np.where(predicted_streamflow < 0, 0, predicted_streamflow)

        self._plot_grdc_streamflow(observed_streamflow, predicted_streamflow, loss_fn, val_start, lookback)
        
#==============================================================================================================================
    def simulate_streamflow(self, model_path, sim_start, sim_end, latlist, lonlist, loss_fn, num_input_branch, 
                            lookback, routing_method='mfd'):
        """Simulate streamflow in batch mode using the trained model."
        """
        print(' 1. Loading runoff data and other predictors')
        vdp = PredictDataPreprocessor(self.working_dir, self.study_area, sim_start, sim_end, routing_method)
        rawdata = vdp.get_data_latlng(latlist, lonlist)

        self.vmodel = PredictStreamflow(self.working_dir, lookback)
        self.vmodel.prepare_data_latlng(rawdata)
        batch_size = len(self.vmodel.latlist)
        self.vmodel.load_model(model_path, loss_fn)
        print(' 2. Batch prediction')
        if num_input_branch == 3:
            predicted_streamflows = self.vmodel.model.predict([self.vmodel.predictors, self.vmodel.local_predictors, self.vmodel.catchment_size],
                                                                  batch_size=batch_size)
        elif num_input_branch ==2:
            predicted_streamflows = self.vmodel.model.predict([self.vmodel.predictors, self.vmodel.catchment_size], batch_size=batch_size)
        else:
            raise ValueError("Invalid number. Choose either 2 or 3")

        if loss_fn == 'laplacian_nll':
            seq = int(len(predicted_streamflows)/batch_size)
            predicted_streamflows = predicted_streamflows.reshape(batch_size, seq, 2)

            predicted_streamflow_list = []
        
            for predicted_streamflow in predicted_streamflows:
                mu_log = predicted_streamflow[:, 0]  # Mean in log-space
                b_log = predicted_streamflow[:, 1]  # Uncertainty in log-space
                
                # ✅ Convert back to original streamflow units
                mu = np.exp(mu_log) - 1  # Mean streamflow prediction
                sigma = (np.exp(mu_log) - 1) * (np.exp(b_log) - 1) # Uncertainty in original scale
                
                predicted_streamflow = mu
                predicted_streamflow_list.append(predicted_streamflow)
        else:
            seq = int(len(predicted_streamflows)/batch_size)
            predicted_streamflows = predicted_streamflows.reshape(batch_size, seq, 1)

            predicted_streamflow_list = []
            for predicted_streamflow in predicted_streamflows:
                predicted_streamflow = np.where(predicted_streamflow < 0, 0, predicted_streamflow)
                predicted_streamflow_list.append(predicted_streamflow)
        print(' 3. Generating csv file for each coordinate')
        for predicted_streamflow, lat, lon in zip(predicted_streamflow_list, latlist, lonlist):
            adjusted_start_date = pd.to_datetime(self.start_date) + pd.DateOffset(days=lookback)
            period = pd.date_range(adjusted_start_date, periods=len(predicted_streamflow), freq='D')  # Match time length with mu
            df = pd.DataFrame({
                'time': period,  # Adjusted time column
                'streamflow (m3/s)': predicted_streamflow
            })
            output_path = os.path.join(self.working_dir, f"predicted_streamflow_data/predicted_streamflow_lat{lat}_lon{lon}.csv")
            df.to_csv(output_path, index=False)
        out_folder = os.path.join(self.working_dir, 'predicted_streamflow_data')
        print(f' COMPLETED! csv files available at {out_folder}')
#========================================================================================================================  
            
    def _plot_grdc_streamflow(self, observed_streamflow, predicted_streamflow, loss_fn, val_start, lookback):
        """Plot the observed and predicted streamflow data.
        """
        nse, kge = self._compute_metrics(observed_streamflow, predicted_streamflow, loss_fn)
        kge1 = kge[0][0]
        R = kge[1][0]
        Beta = kge[2][0]
        Alpha = kge[3][0]

        start_date = pd.to_datetime(val_start) + pd.Timedelta(days=lookback)
        num_days = len(predicted_streamflow)
        date_range = pd.date_range(start=start_date, periods=num_days, freq='D')

        print(f"Nash-Sutcliffe Efficiency (NSE): {nse}")
        print(f"Kling-Gupta Efficiency (KGE): {kge1}")
        plt.plot(date_range, predicted_streamflow[:], color='blue', label='Predicted Streamflow')
        plt.plot(date_range, observed_streamflow[0]['station_discharge'][self.vmodel.timesteps:].values[:], color='red', label='Observed Streamflow')
        plt.title('Comparison of observed and simulated streamflow')  # Add a title
        plt.xlabel('Date')  # Label the x-axis
        plt.ylabel('River Discharge (m³/s)')
        plt.legend()  # Add a legend to label the lines
        plt.show()
#========================================================================================================================  
        
    def _compute_metrics(self, observed_streamflow, predicted_streamflow, loss_fn):
        """Compute performance metrics for the model.
        """
        observed = observed_streamflow[0]['station_discharge'][self.vmodel.timesteps:].values
        if loss_fn == 'laplacian_nll':
            predicted = predicted_streamflow[:]
        else: 
            predicted = predicted_streamflow[:, 0].flatten()
        nan_indices = np.isnan(observed) | np.isnan(predicted)
        observed = observed[~nan_indices]
        predicted = predicted[~nan_indices]
        nse = hydroeval.nse(predicted, observed)
        kge = hydroeval.kge(predicted, observed)
        return nse, kge
  
#===========================================================================================================================
    def explore_data_interactively(self, start_date, end_date, grdc_netcdf=None):
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

