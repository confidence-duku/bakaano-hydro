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
    def __init__(self, working_dir, study_area_path, start_date, end_date):
        """_summary_

        Args:
            project_name (str): _description_
            study_area (str): _description_
            start_date (str): _description_
            end_date (str): _description_
            rp (_type_, optional): _description_. Defaults to None.
        """
         # Initialize the project name
        self.working_dir = working_dir
        
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
      
        self.clipped_dem = f'{self.working_dir}/elevation/dem_clipped.tif'

#=========================================================================================================================================
    def train_streamflow_model(self, grdc_netcdf, prep_nc, tasmax_nc, tasmin_nc, tmean_nc):
        if not os.path.exists(f'{self.working_dir}/runoff_output/wacc_sparse_arrays.pkl'):
            print('Computing VegET runoff and routing flow to river network')
            vg = VegET(self.working_dir, self.study_area, self.start_date, self.start_date)
            vg.compute_veget_runoff_route_flow(prep_nc, tasmax_nc, tasmin_nc, tmean_nc)

        print('TRAINING DEEP LEARNING STREAMFLOW PREDICTION MODEL')
        sdp = DataPreprocessor(self.working_dir, self.study_area, self.start_date, self.start_date)
        print(' 1. Loading observed streamflow')
        sdp.load_observed_streamflow(grdc_netcdf)
        
        print(' 2. Loading runoff data and other predictors')
        self.rawdata = sdp.get_data()
        sn = str(len(sdp.sim_station_names))
        
        print(f'     Training deepstrmm model based on {sn} stations in the GRDC database')
        print(sdp.sim_station_names)
        
        print(' 3. Building neural network model')
        smodel = StreamflowModel(self.working_dir)
        smodel.prepare_data(self.rawdata)
        smodel.build_model()
        #smodel.load_regional_model(f'{self.working_dir}/models/deepstrmm_model_tcn360.keras')
        print(' 4. Training neural network model')
        smodel.train_model()
#========================================================================================================================  
                
    def evaluate_streamflow_model(self, station_name, model_path, grdc_netcdf, prep_nc, tasmax_nc, tasmin_nc, tmean_nc):
        if not os.path.exists(f'{self.working_dir}/runoff_output/wacc_sparse_arrays.pkl'):
            print('Computing VegET runoff and routing flow to river network')
            vg = VegET(self.working_dir, self.study_area, self.start_date, self.start_date)
            vg.compute_veget_runoff_route_flow(prep_nc, tasmax_nc, tasmin_nc, tmean_nc)

        vdp = PredictDataPreprocessor(self.working_dir, self.study_area, self.start_date, self.start_date)
        fulldata = vdp.load_observed_streamflow(grdc_netcdf)
        self.stat_names = vdp.sim_station_names
        
        extracted_data = fulldata.where(fulldata.station_name.astype(str) == station_name, drop=True)
        full_ids = list(extracted_data.id.values)
        
        self.station = extracted_data['runoff_mean'].where(extracted_data['station_name'] == station_name, 
                                                drop=True).to_dataframe(name='station_discharge').reset_index()

        station_id = self.station['id'][0]
        station_index = full_ids.index(station_id)

        vdp.station_ids = np.unique([full_ids[station_index]])
        
        rawdata = vdp.get_data()
        observed_streamflow = list(map(lambda xy: xy[1], rawdata[0]))

        self.vmodel = PredictStreamflow(self.working_dir)
        self.vmodel.prepare_data(rawdata)

        self.vmodel.load_model(model_path)
        predicted_streamflow = self.vmodel.model.predict([self.vmodel.predictors, self.vmodel.catchment_size])
        mu = predicted_streamflow[:, 0]  # Mean in original space
        sigma = predicted_streamflow[:, 1]  # Standard deviation in original space
        mu = pd.DataFrame(mu.reshape(-1, 1)).rolling(window=30, min_periods=1).mean().values.flatten()

        self.plot_grdc_streamflow(observed_streamflow, mu)
        
#========================================================================================================================  

    def simulate_streamflow_latlng(self, model_path, lat, lon, prep_nc, tasmax_nc, tasmin_nc, tmean_nc):
        if not os.path.exists(f'{self.working_dir}/runoff_output/wacc_sparse_arrays.pkl'):
            print('Computing VegET runoff and routing flow to river network')
            vg = VegET(self.working_dir, self.study_area, self.start_date, self.start_date)
            vg.compute_veget_runoff_route_flow(prep_nc, tasmax_nc, tasmin_nc, tmean_nc)

        vdp = PredictDataPreprocessor(self.working_dir, self.study_area, self.start_date, self.end_date)
        
        rawdata = vdp.get_data_latlng(lat, lon)

        self.vmodel = PredictStreamflow(self.working_dir)
        self.vmodel.prepare_data_latlng(rawdata)

        self.vmodel.load_model(model_path)
        predicted_streamflow = self.vmodel.model.predict([self.vmodel.predictors, self.vmodel.catchment_size])
        mu = predicted_streamflow[:, 0]  # Mean in original space
        sigma = predicted_streamflow[:, 1]  # Standard deviation in original space
        lower_bound = np.maximum(mu - 1.65 * sigma, 0)
        upper_bound = mu + 1.65 * sigma
        mu_smoothed = pd.DataFrame(mu.reshape(-1, 1)).rolling(window=30, min_periods=1).mean().values.flatten()

        adjusted_start_date = pd.to_datetime(self.start_date) + pd.DateOffset(days=365)
        period = pd.date_range(adjusted_start_date, periods=len(mu), freq='D')  # Match time length with mu
        df = pd.DataFrame({
            'time': period,  # Adjusted time column
            'streamflow (m3/s)': mu_smoothed,
            'lower_bound (95% CI)': lower_bound,
            'upper_bound (95% CI)': upper_bound
        })
        output_path = os.path.join(self.working_dir, f"output_data/streamflow_{lat}_{lon}.csv")
        df.to_csv(output_path, index=False)
        
#========================================================================================================================  
            
    def plot_grdc_streamflow(self, observed_streamflow, predicted_streamflow, sigma=None):
        nse, kge = self.compute_metrics(observed_streamflow, predicted_streamflow)
        kge1 = kge[0][0]
        R = kge[1][0]
        Beta = kge[2][0]
        Alpha = kge[3][0]
        
        print(f"Nash-Sutcliffe Efficiency (NSE): {nse}")
        print(f"Kling-Gupta Efficiency (KGE): {kge1}")
        plt.plot(predicted_streamflow[:], color='blue', label='Predicted Streamflow')
        plt.plot(observed_streamflow[0]['station_discharge'][self.vmodel.timesteps:].values[:], color='red', label='Observed Streamflow')
        if sigma is not None:
            lower_bound = np.maximum(predicted_streamflow - 1.65 * sigma, 0)
            upper_bound = predicted_streamflow[:] + 1.65 * sigma
            plt.fill_between(range(len(predicted_streamflow[:])), lower_bound, upper_bound, color='orange', alpha=0.3, label='95% Confidence Interval')
        plt.title('Comparison of observed and simulated streamflow for River ' + self.working_dir)  # Add a title
        plt.xlabel('Date')  # Label the x-axis
        plt.ylabel('River Discharge (m³/s)')
        plt.legend()  # Add a legend to label the lines
        plt.show()
#========================================================================================================================  
        
    def compute_metrics(self, observed_streamflow, predicted_streamflow):
        observed = observed_streamflow[0]['station_discharge'][self.vmodel.timesteps:].values
        predicted = predicted_streamflow[:]
        nan_indices = np.isnan(observed) | np.isnan(predicted)
        observed = observed[~nan_indices]
        predicted = predicted[~nan_indices]
        nse = hydroeval.nse(predicted, observed)
        kge = hydroeval.kge(predicted, observed)
        return nse, kge
  

    