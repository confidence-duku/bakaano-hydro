import numpy as np
import pandas as pd
import os
from bakaano.utils import Utils
from bakaano.streamflow_trainer import DataPreprocessor, StreamflowModel
from bakaano.streamflow_simulator import PredictDataPreprocessor, PredictStreamflow
from bakaano.veget import VegET
import hydroeval
import matplotlib.pyplot as plt

#========================================================================================================================  
class BakaanoHydro:
    """Generate an instance
    """
    def __init__(self, working_dir, study_area_path, start_date, end_date, climate_data_source):
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
        self.study_area = study_area_path
        
        # Initialize utility class with project name and study area.
        self.uw = Utils(self.working_dir, self.study_area)
        self.times = pd.date_range(start_date, end_date)
        
        # Set the start and end dates for the project
        self.start_date = start_date
        self.end_date = end_date

        # Create necessary directories for the project structure   
        os.makedirs(f'{self.working_dir}/models', exist_ok=True)
        os.makedirs(f'{self.working_dir}/runoff_output', exist_ok=True)
        os.makedirs(f'{self.working_dir}/scratch', exist_ok=True)
        os.makedirs(f'{self.working_dir}/shapes', exist_ok=True)
        os.makedirs(f'{self.working_dir}/catchment', exist_ok=True)
        os.makedirs(f'{self.working_dir}/predicted_streamflow_data', exist_ok=True)
      
        self.clipped_dem = f'{self.working_dir}/elevation/dem_clipped.tif'

#=========================================================================================================================================
    def train_streamflow_model(self, train_start, train_end, grdc_netcdf,  loss_fn, num_input_branch, lookback, batch_size, num_epochs):
        """Train the deep learning streamflow prediction model."
        """
    
        print('TRAINING DEEP LEARNING STREAMFLOW PREDICTION MODEL')
        sdp = DataPreprocessor(self.working_dir, self.study_area, grdc_netcdf, self.start_date, self.end_date,train_start, train_end)
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
        elif num_input_branch == 3:
            smodel.build_model_2_input_branches(loss_fn)
        else:
            raise ValueError("Invalid number. Choose either 2 or 3")
        print(' 4. Training neural network model')
        smodel.train_model(loss_fn, num_input_branch)
        print(f'     Completed! Trained model saved at {self.working_dir}/models/bakaano_model_{loss_fn}_{num_input_branch}_branches.keras')
#========================================================================================================================  
                
    def evaluate_streamflow_model_interactively(self, model_path, val_start, val_end, grdc_netcdf, loss_fn, num_input_branch, lookback, batch_size):
        """Evaluate the streamflow prediction model."
        """

        vdp = PredictDataPreprocessor(self.working_dir, self.study_area, self.start_date, self.end_date,val_start, val_end, grdc_netcdf)
        fulldata = vdp.load_observed_streamflow(grdc_netcdf)
        self.stat_names = vdp.sim_station_names
        print("Available station names:")
        print(self.stat_names)

        station_name = input("Please enter the station name: ")
        
        extracted_data = fulldata.where(fulldata.station_name.astype(str) == station_name, drop=True)
        full_ids = list(extracted_data.id.values)
        
        self.station = extracted_data['runoff_mean'].where(extracted_data['station_name'] == station_name, 
                                                drop=True).to_dataframe(name='station_discharge').reset_index()

        station_id = self.station['id'][0]
        station_index = full_ids.index(station_id)

        vdp.station_ids = np.unique([full_ids[station_index]])
        
        rawdata = vdp.get_data()
        observed_streamflow = list(map(lambda xy: xy[1], rawdata))

        self.vmodel = PredictStreamflow(self.working_dir, lookback, batch_size)
        self.vmodel.prepare_data(rawdata)

        self.vmodel.load_model(model_path, loss_fn)
        if num_input_branch == 3:
            predicted_streamflow = self.vmodel.model.predict([self.vmodel.predictors, self.vmodel.local_predictors, self.vmodel.catchment_size])
        elif num_input_branch == 3:
            predicted_streamflow = self.vmodel.model.predict([self.vmodel.predictors, self.vmodel.catchment_size])

        if loss_fn == 'laplacian_nll':
        
            mu_log = predicted_streamflow[:, 0]  # Mean in log-space
            b_log = predicted_streamflow[:, 1]  # Uncertainty in log-space
            
            # ✅ Convert back to original streamflow units
            mu = np.exp(mu_log) - 1  # Mean streamflow prediction
            sigma = (np.exp(mu_log) - 1) * (np.exp(b_log) - 1) # Uncertainty in original scale
            
            predicted_streamflow = mu
        else:
            predicted_streamflow = np.where(predicted_streamflow < 0, 0, predicted_streamflow)

        self.plot_grdc_streamflow(observed_streamflow, predicted_streamflow, loss_fn)
        
#==============================================================================================================================
    def simulate_streamflow(self, model_path, sim_start, sim_end, latlist, lonlist, loss_fn, num_input_branch, lookback):
        """Simulate streamflow in batch mode using the trained model."
        """

        vdp = PredictDataPreprocessor(self.working_dir, self.study_area, self.start_date, self.end_date, sim_start, sim_end)
        
        rawdata = vdp.get_data_latlng(latlist, lonlist)

        self.vmodel = PredictStreamflow(self.working_dir, lookback, batch_size)
        self.vmodel.prepare_data_latlng(rawdata)
        batch_size = len(self.vmodel.latlist)
        self.vmodel.load_model(model_path, loss_fn)
        if num_input_branch == 3:
            predicted_streamflows = self.vmodel.model.predict([self.vmodel.predictors, self.vmodel.local_predictors, self.vmodel.catchment_size],
                                                                  batch_size=batch_size)
            
        else:
            predicted_streamflow = self.vmodel.model.predict([self.vmodel.predictors, self.vmodel.catchment_size], batch_size=batch_size)

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

        for predicted_streamflow, lat, lon in zip(predicted_streamflow_list, latlist, lonlist):
            adjusted_start_date = pd.to_datetime(self.start_date) + pd.DateOffset(days=lookback)
            period = pd.date_range(adjusted_start_date, periods=len(predicted_streamflow), freq='D')  # Match time length with mu
            df = pd.DataFrame({
                'time': period,  # Adjusted time column
                'streamflow (m3/s)': predicted_streamflow
            })
            output_path = os.path.join(self.working_dir, f"predicted_streamflow_data/streamflow_{lat}_{lon}.csv")
            df.to_csv(output_path, index=False)

#========================================================================================================================  
            
    def plot_grdc_streamflow(self, observed_streamflow, predicted_streamflow, loss_fn):
        """Plot the observed and predicted streamflow data.
        """
        nse, kge = self.compute_metrics(observed_streamflow, predicted_streamflow, loss_fn)
        kge1 = kge[0][0]
        R = kge[1][0]
        Beta = kge[2][0]
        Alpha = kge[3][0]
        
        print(f"Nash-Sutcliffe Efficiency (NSE): {nse}")
        print(f"Kling-Gupta Efficiency (KGE): {kge1}")
        plt.plot(predicted_streamflow[:], color='blue', label='Predicted Streamflow')
        plt.plot(observed_streamflow[0]['station_discharge'][self.vmodel.timesteps:].values[:], color='red', label='Observed Streamflow')
        plt.title('Comparison of observed and simulated streamflow for River ' + self.working_dir)  # Add a title
        plt.xlabel('Date')  # Label the x-axis
        plt.ylabel('River Discharge (m³/s)')
        plt.legend()  # Add a legend to label the lines
        plt.show()
#========================================================================================================================  
        
    def compute_metrics(self, observed_streamflow, predicted_streamflow, loss_fn):
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
  

    