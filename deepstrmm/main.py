import numpy as np
import pandas as pd
import os
import xarray as xr
import rioxarray
import rasterio
from datetime import datetime
from deepstrmm.utils import Utils
from deepstrmm.pet import PotentialEvapotranspiration
from deepstrmm.router import RunoffRouter
from deepstrmm.streamflow_trainer import DataPreprocessor, StreamflowModel
from deepstrmm.streamflow_predictor import PredictDataPreprocessor, PredictStreamflow
import richdem as rd
import warnings
import hydroeval
import matplotlib.pyplot as plt
import pickle
import gc
import scipy as sp
from concurrent.futures import ProcessPoolExecutor
warnings.filterwarnings("ignore", category=RuntimeWarning)


#========================================================================================================================  
class DeepSTRMM:
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
        
        # Set the Curve Number coefficient
        self.cncoef = -1
        
        # File paths for input data
        self.hsg_filename = f'{self.working_dir}/soil/hsg.tif'
        self.hsg_filename_prj = f'{self.working_dir}/soil/hsg_clipped.tif'
        self.clipped_dem = f'{self.working_dir}/elevation/dem_clipped.tif'
        self.clipped_lc = f'{self.working_dir}/land_cover/lc_clipped.tif'

#============================================================================================================================        
    def compute_CN2(self):
         #initialize land cover model and download land_cover data
        #print(' 3. Calculating CN2')
        with rasterio.open(self.clipped_lc) as lcd:
            lc = lcd.read(1)

        with rasterio.open(self.hsg_filename_prj) as hd:
            hsg = hd.read(1)
            hsg = np.where(lc !=50, hsg, 4) #adjusting soil for urban areas

        # Define lookup table for computin cn2
        #land cover is based on esa world cover
        lookup_table = {
            ("10", "1"): 30,("10", "2"): 55,("10", "3"): 70,("10", "4"): 77,
            ("20", "1"): 35,("20", "2"): 56,("20", "3"): 70,("20", "4"): 77,
            ("30", "1"): 49,("30", "2"): 69,("30", "3"): 79,("30", "4"): 84,
            ("40", "1"): 64,("40", "2"): 75,("40", "3"): 82,("40", "4"): 85,
            ("50", "1"): 77,("50", "2"): 85,("50", "3"): 90,("50", "4"): 92,
            ("60", "1"): 77,("60", "2"): 86,("60", "3"): 91,("60", "4"): 94,
            ("70", "1"): 0,("70", "2"): 0,("70", "3"): 0,("70", "4"): 0,
            ("80", "1"): 0,("80", "2"): 0,("80", "3"): 0,("80", "4"): 0,
            ("90", "1"): 0,("90", "2"): 0,("90", "3"): 0,("90", "4"): 0,
            ("95", "1"): 0,("95", "2"): 0,("95", "3"): 0,("95", "4"): 0,
            ("100", "1"): 71,("100", "2"): 71,("100", "3"): 81,("100", "4"): 89
        }
                
        self.cn2 = np.zeros_like(self.lc)
        lc = lc.astype(int)
        for keys, val in lookup_table.items():  
            lc_key, soil_key = keys          
            pp = np.where((lc==int(lc_key)) & (hsg==int(soil_key)), val, 0)
            self.cn2 = np.add(self.cn2, pp)

#========================================================================================================================  

    def adjust_CN2_slp(self):
        #clip slope to study area
        
        slp_name = f'{self.working_dir}/elevation/slope_clipped.tif'
        self.slope = rd.TerrainAttribute(self.rd_dem, attrib='slope_riserun')
        self.uw.save_to_scratch(slp_name, self.slope)
        
        p1 = np.divide(np.subtract(self.cn3, self.cn2), 3)
        p2 = np.subtract(1,np.multiply(np.exp(-13.86 * (self.slope/100)), 2))
        self.cn2_slp = p1 * p2 + self.cn2
#========================================================================================================================  

    def compute_CN1(self, cn2):
        p1 = np.subtract(100, cn2)
        p2 = np.multiply(20, p1)
        p3 = np.add(np.exp(np.subtract(2.533, np.multiply(0.0636, p1))), p1)
        self.cn1 = np.subtract(cn2, np.divide(p2,p3))
        #self.cn1 = np.where(self.cn1 <0, 0, self.cn1)
#========================================================================================================================  

    def compute_CN3(self, cn2):        
        p1 = np.subtract(100, cn2)
        self.cn3 = np.multiply(cn2, np.exp(np.multiply(0.00673, p1)))


#=================================================================================================================
    def compute_runoff_route_flow(self, prep_nc, tasmax_nc, tasmin_nc, tmean_nc):  

        align_dem = rasterio.open(self.clipped_dem).read(1)
        self.rd_dem = rd.rdarray(align_dem, no_data=-9999)
        self.lc = rioxarray.open_rasterio(self.clipped_lc)[0]

        #self._download_input_data()
        #self.compute_HSG()
        self.compute_CN2()
        self.compute_CN3(self.cn2)
        self.adjust_CN2_slp()
        self.compute_CN3(self.cn2_slp)
        self.compute_CN1(self.cn2_slp)

        # Initialize potential evapotranspiration and data preprocessor
        eto = PotentialEvapotranspiration(self.working_dir, self.study_area, self.start_date, self.end_date)

        # Load observed streamflow and climate data
        tasmax_var = list(tasmax_nc.data_vars)[0]
        tasmin_var = list(tasmin_nc.data_vars)[0]
        tmean_var = list(tmean_nc.data_vars)[0]
        prep_var = list(prep_nc.data_vars)[0]

        tasmax_period = tasmax_nc[tasmax_var].sel(time=slice(self.start_date, self.end_date)) - 273.15
        tasmin_period = tasmin_nc[tasmin_var].sel(time=slice(self.start_date, self.end_date)) - 273.15
        tmean_period = tmean_nc[tmean_var].sel(time=slice(self.start_date, self.end_date)) - 273.15
        rf = prep_nc[prep_var].sel(time=slice(self.start_date, self.end_date)) * 86400  # Conversion from kg/m2/s to mm/day
        rf = rf.astype(np.float32).assign_coords(lat=rf['lat'].astype(np.float32), lon=rf['lon'].astype(np.float32))
        
        td = np.sqrt(tasmax_period - tasmin_period)
        pet_params = 0.408 * 0.0023 * (tmean_period + 17.8) * td
        pet_params = pet_params.astype(np.float32)
        self.pet_params = pet_params.assign_coords(
            lat=pet_params['lat'].astype(np.float32),
            lon=pet_params['lon'].astype(np.float32)
        )

        #pet_params = pet_params.values
        # Extract latitude and expand dimensions to match the lon dimension
        latsg = tmean_period[0]['lat']
        latsg = latsg.astype(np.float32)
        self.latgrids = latsg.expand_dims(lon=tmean_period[0]['lon'], axis=[1]).values
        
        # Initial soil moisture condition
        smax = ((1000 / self.cn1) - 10) * 25.4
        s = 0.9 * smax

        # Initialize runoff router and compute flow direction
        rout = RunoffRouter(self.working_dir, self.clipped_dem)
        fdir, acc = rout.compute_flow_dir()
        
        facc_thresh = np.nanmax(acc) * 0.0001
        facc_mask = np.where(acc < facc_thresh, 0, 1)

        # Lists for storing results
        self.wacc_list = []
        self.mam_ro, self.jja_ro, self.son_ro, self.djf_ro = 0, 0, 0, 0
        self.mam_wfa, self.jja_wfa, self.son_wfa, self.djf_wfa = 0, 0, 0, 0

        for count in range(rf.shape[0]):
            if count % 365 == 0:
                year_num = (count // 365) + 1
                print(f'Computing surface runoff and routing flow to river channels in year {year_num}')
            this_rf = rf[count]
            this_rf = self.uw.align_rasters(this_rf, israster=False)

            this_et = eto.compute_PET(self.pet_params[count], self.latgrids, tmean_period[count])
            this_et = self.uw.align_rasters(this_et, israster=False)
            
            p1 = this_rf - (0.2 * s)
            p2 = p1**2
            p3 = np.where(p1<0,0,p2)
            p4 = this_rf + (0.8 * s)
            q_surf = p3 / p4
            q_surf = np.where(q_surf>0, q_surf, 0)

            # Adjust retention parameter at the end of the day
            s1 = np.exp((self.cncoef * s) / smax)
            s = np.add(np.multiply(this_et, s1), s) - this_rf + q_surf

            # Use Pysheds for weighted flow accumulation            
            ro_tiff = rout.convert_runoff_layers(q_surf)
            wacc = rout.compute_weighted_flow_accumulation(ro_tiff)                  
            
            this_month = tmean_period[count]['time'].dt.month.values
            if this_month in [1, 2, 12]:
                self.djf_ro = q_surf + self.djf_ro
                self.djf_wfa = wacc + self.djf_wfa
            elif this_month in [3, 4, 5]:
                self.mam_ro = q_surf + self.mam_ro
                self.mam_wfa = wacc + self.mam_wfa
            elif this_month in [6, 7, 8]:
                self.jja_ro = q_surf + self.jja_ro
                self.jja_wfa = wacc + self.jja_wfa
            elif this_month in [9, 10, 11]:
                self.son_ro = q_surf + self.son_ro
                self.son_wfa = wacc + self.son_wfa
                
            wacc = wacc * facc_mask            
            wacc = sp.sparse.coo_array(wacc)
            self.wacc_list.append(wacc)            
            #gc.collect()
            
       
        # Save seasonal data using rasterio
        with rasterio.open(self.clipped_dem) as out:
            s_meta = out.profile

        out_meta = s_meta.copy()
        out_meta.update({
            "dtype": "float64",
            "compress": "lzw"
        })

        seasons = ['djf', 'mam', 'jja', 'son']
        attributes = ['wfa', 'ro']

        for season in seasons:
            for attr in attributes:
                filename = f'{self.working_dir}/runoff_output/{season}_{attr}.tif'
                data = getattr(self, f'{season}_{attr}')
                with rasterio.open(filename, 'w', **out_meta) as dst:
                    dst.write(data, indexes=1)

        # Save station weighted flow accumulation data
        filename = f'{self.working_dir}/runoff_output/wacc_sparse_arrays.pkl'
        
        with open(filename, 'wb') as f:
            pickle.dump(self.wacc_list, f)
    
#=========================================================================================================================================
    def train_streamflow_model(self, grdc_netcdf):
        print('TRAINING DEEP LEARNING STREAMFLOW PREDICTION MODEL')
        sdp = DataPreprocessor(self.working_dir, self.study_area, self.start_date, self.start_date)
        print(' 1. Loading observed streamflow')
        sdp.load_observed_streamflow(grdc_netcdf)
        #print('There are ')
        
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
                
    def evaluate_streamflow_model(self, station_name, model_path, grdc_netcdf):
        vdp = PredictDataPreprocessor(self.working_dir, self.study_area, self.start_date, self.start_date)
        fulldata = vdp.load_observed_streamflow(grdc_netcdf)
        #print(vdp.sim_station_names)
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
        y2 = self.vmodel.model.predict([self.vmodel.predictors, self.vmodel.catchment_size])
        predicted_streamflow = np.where(y2<0, 0, y2)
        self.plot_grdc_streamflow(observed_streamflow, predicted_streamflow)
       
        
#========================================================================================================================  

    def simulate_streamflow_latlng(self, model_path, lat, lon):
        vdp = PredictDataPreprocessor(self.working_dir, self.study_area, self.start_date, self.end_date)
        
        rawdata = vdp.get_data_latlng(lat, lon)

        self.vmodel = PredictStreamflow(self.working_dir)
        self.vmodel.prepare_data_latlng(rawdata)

        self.vmodel.load_model(model_path)
        y2 = self.vmodel.model.predict([self.vmodel.predictors, self.vmodel.catchment_size])
        predicted_streamflow = np.where(y2<0, 0, y2)
        return predicted_streamflow
        
    
#========================================================================================================================  
            
    def plot_grdc_streamflow(self, observed_streamflow, predicted_streamflow):
        nse, kge = self.compute_metrics(observed_streamflow, predicted_streamflow)
        
        kge1 = kge[0][0]
        R = kge[1][0]
        Beta = kge[2][0]
        Alpha = kge[3][0]
        
        print(f"Nash-Sutcliffe Efficiency (NSE): {nse}")
        print(f"Kling-Gupta Efficiency (KGE): {kge1}")
        plt.plot(predicted_streamflow[:], color='blue', label='Predicted Streamflow')
        plt.plot(observed_streamflow[0]['station_discharge'][self.vmodel.timesteps:].values[:], color='red', label='Observed Streamflow')
        #plt.plot(self.vmodel.this_wfa[self.vmodel.timesteps:], color='green', label='wfa')
        plt.title('Comparison of observed and simulated streamflow for River ' + self.working_dir)  # Add a title
        plt.xlabel('Date')  # Label the x-axis
        plt.ylabel('River Discharge (m³/s)')
        plt.legend()  # Add a legend to label the lines
        plt.show()
#========================================================================================================================  
        
    def compute_metrics(self, observed_streamflow, predicted_streamflow):
        observed = observed_streamflow[0]['station_discharge'][self.vmodel.timesteps:].values
        predicted = predicted_streamflow[:, 0].flatten()
        nan_indices = np.isnan(observed) | np.isnan(predicted)
        observed = observed[~nan_indices]
        predicted = predicted[~nan_indices]
        nse = hydroeval.nse(predicted, observed)
        kge = hydroeval.kge(predicted, observed)
        return nse, kge
        


    

#===========================================================================================================      

    