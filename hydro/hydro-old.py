import numpy as np
import pandas as pd
import os
import xarray as xr
from pathlib import Path
import rioxarray
import rasterio
import glob
from datetime import datetime, timedelta
from deepstrmm.utils import Utils
from deepstrmm.pet import PotentialEvapotranspiration
from deepstrmm.soil import SoilGridsData
from deepstrmm.land_cover import LandCover
from deepstrmm.dem import DEMDownloader
from deepstrmm.climate import ChelsaDataDownloader
from deepstrmm.router import RunoffRouter
from deepstrmm.streamflow_trainer import DataPreprocessor, StreamflowModel
from deepstrmm.streamflow_predictor import PredictDataPreprocessor, PredictStreamflow
import richdem as rd
from rasterio.io import MemoryFile
import warnings
#from joblib import Parallel, delayed
warnings.filterwarnings("ignore", category=RuntimeWarning)
from IPython.display import HTML
import hydroeval
import math
import matplotlib.pyplot as plt
import time
import pickle
import gc
import scipy as sp


#========================================================================================================================  
class DeepSTRMM:
    """
    Initialize an instance of the DeepSTRMM hydrological model.

    Attributes:
    -----------
    project_name : str
        The name of the project. Used for creating directories and organizing project-related files.
    study_area : str
        The path to the shapefile of the study area.
    start_date : str
        The start date of the study period in 'YYYY-MM-DD' format.
    end_date : str
        The end date of the study period in 'YYYY-MM-DD' format.
    cncoef : int
        Coefficient related to Curve Number (CN) used for hydrological calculations.
    s : object or None, optional
        Additional parameter `s` if provided.
    uw : Utils
        Instance of Utils class for utility functions related to the project.
    times : pandas.DatetimeIndex
        Date range between `start_date` and `end_date` inclusive.
    hsg_filename : str
        File path for soil data related to Hydrologic Soil Groups (HSG).
    hsg_filename_prj : str
        File path for projected soil data related to Hydrologic Soil Groups (HSG).

    Methods:
    --------
    __init__(project_name, study_area, start_date, end_date, cncoef, rp=None):
        Initialize the DeepSTRMM model with project details and create necessary directories.

    prepare_data(data_list):
        Method to prepare data for the DeepSTRMM model training.

    """
    def __init__(self, project_name, study_area, start_date, end_date, cncoef, rp=None):
        
         # Initialize the project name
        self.project_name = project_name
        
        # Initialize the study area
        self.study_area = study_area
        
        # Initialize utility class with project name and study area.
        self.uw = Utils(self.project_name, self.study_area)
        self.times = pd.date_range(start_date, end_date)
        
        # Set the start and end dates for the project
        self.start_date = start_date
        self.end_date = end_date

        # Create necessary directories for the project structure
        os.makedirs(f'./{self.project_name}/soil', exist_ok=True)
        os.makedirs(f'./{self.project_name}/elevation', exist_ok=True)
        os.makedirs(f'./{self.project_name}/models', exist_ok=True)
        os.makedirs(f'./{self.project_name}/output_data', exist_ok=True)
        os.makedirs(f'./{self.project_name}/land_cover', exist_ok=True)
        os.makedirs(f'./{self.project_name}/tasmax', exist_ok=True)
        os.makedirs(f'./{self.project_name}/tasmin', exist_ok=True)
        os.makedirs(f'./{self.project_name}/prep', exist_ok=True)
        os.makedirs(f'./{self.project_name}/tmean', exist_ok=True)
        os.makedirs(f'./{self.project_name}/scratch', exist_ok=True)
        os.makedirs(f'./{self.project_name}/shapes', exist_ok=True)
        
        # Set the Curve Number coefficient
        self.cncoef = cncoef
        self.s = rp
        
        # File paths for soil data
        self.hsg_filename = f'./{self.project_name}/soil/hsg.tif'
        self.hsg_filename_prj = f'./{self.project_name}/soil/hsg_{self.project_name}.tif'
#========================================================================================================================  

    def _download_input_data(self):
        #print(' 1. Downloading and preprocessing input data')
        
        #print('     - Downloading and preprocessing climate data')
#         cd = ChelsaDataDownloader(self.project_name, self.study_area, self.start_date, self.end_date)
#         cd.get_chelsa_clim_data()
     
        self.tasmax_nc = self.uw.concat_nc(Path(f'/lustre/backup/WUR/ESG/duku002/NBAT/hydro/{self.project_name}/tasmax/'), '*tasmax*.nc')
        self.tasmin_nc = self.uw.concat_nc(Path(f'/lustre/backup/WUR/ESG/duku002/NBAT/hydro/{self.project_name}/tasmin/'), '*tasmin*.nc')   
        self.tmean_nc = self.uw.concat_nc(Path(f'/lustre/backup/WUR/ESG/duku002/NBAT/hydro/{self.project_name}/tmean/'), '*tas_*.nc')
        self.prep_nc = self.uw.concat_nc(Path(f'/lustre/backup/WUR/ESG/duku002/NBAT/hydro/{self.project_name}/prep/'), '*pr_*.nc')
        self.lats = self.prep_nc.pr.sel()[0]['lat'].values
        self.lons = self.prep_nc.pr.sel()[0]['lon'].values
                
#         print('     - Downloading soil data')
#         filetest = self.uw.process_existing_file(self.hsg_filename)
#         if filetest is False:
#             sgd = SoilGridsData(self.project_name, self.study_area)
#             sgd.get_soil_data()
            
        #print('     - Downloading land cover data')
        ldc = LandCover(self.project_name, self.study_area)
        self.clipped_lc = f'./{self.project_name}/land_cover/lc_{self.project_name}.tif'
        filetest = self.uw.process_existing_file(self.clipped_lc)
        if filetest is False:
            #ldc.download_lc()
            #ldc.mosaic_lc()
            self.lc = self.uw.align_rasters(ldc.out_fp, self.tasmax_nc)
            self.lc.rio.to_raster(self.clipped_lc, dtype='float32')
        else:
            self.lc = rioxarray.open_rasterio(self.clipped_lc)

        
        #print('     - Preprocessing DEM data')
#         dd = DEMDownloader(
#             self.project_name,
#             self.study_area,                 
#             self.username,                 
#             self.password
#          )
#         dem = dd.download_dem()

        #dem = './common_data/hyd_af_dem_15s.tif' #africa
        dem = './common_data/hyd_glo_dem_15s.tif'
        self.clipped_dem = f'./{self.project_name}/elevation/dem_{self.project_name}.tif'
        filetest2 = self.uw.process_existing_file(self.clipped_dem)
        if filetest2 is False:
            align_dem = self.uw.align_rasters(dem, self.tasmax_nc)
            self.rd_dem = rd.rdarray(align_dem, no_data=-9999)
            align_dem.rio.to_raster(self.clipped_dem, dtype='float32')
        else:
            align_dem = rasterio.open(self.clipped_dem).read(1)
            self.rd_dem = rd.rdarray(align_dem, no_data=-9999)
            
#========================================================================================================================  
        
    def compute_HYSOG(self):
        #print(' 2. Calculating hydrologic soil groups')
        filetest = self.uw.process_existing_file(self.hsg_filename_prj)
        if filetest is False:
            hysog = './common_data/HYSOGs250m.tif'
            hsg = self.uw.align_rasters(hysog, self.tasmax_nc)
            hsg.rio.to_raster(self.hsg_filename_prj, dtype='int32') 
        else:
            with rasterio.open(self.hsg_filename_prj) as src:
                hsg = src.read(1)
        hsg = np.where(hsg > 10, (hsg - 10), hsg)
        self.hsg = np.nan_to_num(hsg, 0)       
#========================================================================================================================  

    def compute_HSG(self):
        #initialize soil model and download soil data
        #print(' 2. Calculating hydrologic soil groups')
        
        soil_dir = Path(f'./{self.project_name}/soil')
        clay_list = list(map(str, soil_dir.glob('clay*.tif')))
        sand_list = list(map(str, soil_dir.glob('sand*.tif')))
        hsg_list = []
        for cl, sd in zip(clay_list, sand_list):
            with rasterio.open(cl) as src:
                clay = src.read(1)
                clay = np.divide(clay, 10)

            with rasterio.open(sd) as src1:
                sand = src1.read(1)
                sand = np.divide(sand, 10)
                

            # Conditions for each group
            conditions = [
                (sand > 90) & (clay < 10),  # Group A
                (sand >= 50) & (sand <= 90) & (clay >= 10) & (clay < 20),  # Group B
                (sand < 50) & (clay >= 20) & (clay <= 40),  # Group C
                (sand < 50) & (clay > 40)  # Group D
            ]

            # Corresponding group numbers
            choices = [1, 2, 3, 4]

            # Use np.select to classify the samples
            this_hsg = np.select(conditions, choices, default=0)
            hsg_list.append(this_hsg)

        hsg1 = np.max(np.array(hsg_list), axis=0)

        with rasterio.open(clay_list[0]) as slp:
            slp_meta = slp.profile

        out_meta = slp_meta.copy()
        out_meta.update({            
             "compress": "lzw"
        })

        with rasterio.open(self.hsg_filename, 'w', **out_meta) as dst:
            dst.write(hsg1, indexes=1)

        self.uw.reproject_raster(self.hsg_filename, self.hsg_filename_prj)
        self.hsg = self.uw.align_rasters(self.hsg_filename_prj, self.tasmax_nc)
#========================================================================================================================  
        
    def compute_CN2(self):
         #initialize land cover model and download land_cover data
        #print(' 3. Calculating CN2')

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
                
        self.cn2 = 0
        for keys, val in lookup_table.items():  
            lc_key, soil_key = keys
            #print('Processing for ' + lc_key + ' ' + soil_key)            
            pp = np.where((self.lc==int(lc_key)) & (self.hsg==int(soil_key)), val, 0)
            self.cn2 = np.add(self.cn2, pp)
        self.cn2 = xr.DataArray(data=self.cn2[0], coords=[('lat', self.lats), ('lon', self.lons)])
        chunk_size = {'lat': 1000, 'lon': 1000}  # Adjust chunk sizes based on your data and memory
        self.cn2 = self.cn2.chunk(chunk_size)
        self.cn2 = self.cn2.astype(np.int32)
        self.cn2 = self.cn2.assign_coords(
            lat=self.cn2['lat'].astype(np.float32),
            lon=self.cn2['lon'].astype(np.float32)
        )
        self.cn2 = self.cn2.values
        #print(self.cn2)

#========================================================================================================================  

    def adjust_CN2_slp(self):
        #clip slope to study area
        
        slp_name = f'./{self.project_name}/elevation/slope_{self.project_name}.tif'
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


#========================================================================================================================  
    def compute_runoff(self):        
        self._download_input_data()
        self.compute_HYSOG()
        self.compute_CN2()
        self.compute_CN3(self.cn2)
        self.adjust_CN2_slp()
        self.compute_CN3(self.cn2_slp)
        self.compute_CN1(self.cn2_slp)

        # Initialize potential evapotranspiration and data preprocessor
        eto = PotentialEvapotranspiration(self.project_name, self.study_area, self.start_date, self.end_date)
        sdp = DataPreprocessor(self.project_name, self.study_area, self.start_date, self.end_date, self.start_date, self.end_date)

        # Load observed streamflow and climate data
        roi_stations = sdp.station_ids
        tasmax_period = self.tasmax_nc.tasmax.sel(time=slice(self.start_date, self.end_date)) - 273.15
        tasmin_period = self.tasmin_nc.tasmin.sel(time=slice(self.start_date, self.end_date)) - 273.15
        tmean_period = self.tmean_nc.tas.sel(time=slice(self.start_date, self.end_date)) - 273.15
        rf = self.prep_nc.pr.sel(time=slice(self.start_date, self.end_date)) * 86400  # Conversion from kg/m2/s to mm/day
        rf = rf.astype(np.float32).assign_coords(lat=rf['lat'].astype(np.float32), lon=rf['lon'].astype(np.float32)).values
        
        td = np.sqrt(tasmax_period - tasmin_period)
        pet_params = 0.408 * 0.0023 * (tmean_period + 17.8) * td
        pet_params = pet_params.astype(np.float32)
        self.pet_params = pet_params.assign_coords(
            lat=pet_params['lat'].astype(np.float32),
            lon=pet_params['lon'].astype(np.float32)
        ).values

        #pet_params = pet_params.values
        # Extract latitude and expand dimensions to match the lon dimension
        latsg = tmean_period[0]['lat']
        latsg = latsg.astype(np.float32)
        self.latgrids = latsg.expand_dims(lon=tmean_period[0]['lon'], axis=[1]).values
       
        
        # Initial soil moisture condition
        smax = ((1000 / self.cn1) - 10) * 25.4
        s = 0.9 * smax

        # Initialize runoff router and compute flow direction
        rout = RunoffRouter(self.project_name, self.clipped_dem)
        fdir, acc = rout.compute_flow_dir()
        
        facc_thresh = np.nanmax(acc) * 0.0001
        facc_mask = np.where(acc < facc_thresh, 0, 1)

        # Lists for storing results
        self.wacc_list = []
        #self.qlist =[]
        self.mam_ro, self.jja_ro, self.son_ro, self.djf_ro = 0, 0, 0, 0
        self.mam_wfa, self.jja_wfa, self.son_wfa, self.djf_wfa = 0, 0, 0, 0
        this_date = datetime.strptime(self.start_date, '%Y-%m-%d')
        for count in range(rf.shape[0]):
            #start_time = time.perf_counter()
            this_rf = rf[count]
            this_et = eto.compute_PET(self.pet_params[count], self.latgrids, tmean_period[count])
            #this_et = eto.compute_PET_single(tasmax_period[count], tasmin_period[count], tmean_period[count])
            
            p1 = this_rf - (0.2 * s)
            p2 = p1**2
            p3 = np.where(p1<0,0,p2)
            p4 = this_rf + (0.8 * s)
            q_surf = p3 / p4
            #q_surf = q_surf_sp.toarray()
            q_surf = np.where(q_surf>0, q_surf, 0)
            
            #self.qlist.append(q_surf)
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
        
            #end_time = time.perf_counter()
            #print(f"     Execution time: for day {count} is {end_time - start_time} seconds")            
            gc.collect()
            
       
        # Save seasonal data using rasterio
        year_name = datetime.strptime(self.start_date, '%Y-%m-%d').year
        with rasterio.open(self.clipped_dem) as out:
            s_meta = out.profile

        out_meta = s_meta.copy()
        out_meta.update({"compress": "lzw"})

        seasons = ['djf', 'mam', 'jja', 'son']
        attributes = ['wfa', 'ro']

        for season in seasons:
            for attr in attributes:
                filename = f'./{self.project_name}/output_data/{season}_{attr}_{self.project_name}_{year_name}.tif'
                data = getattr(self, f'{season}_{attr}')
                with rasterio.open(filename, 'w', **out_meta) as dst:
                    dst.write(data, indexes=1)

        # Save station weighted flow accumulation data
        filename = f'/lustre/backup/WUR/ESG/duku002/NBAT/hydro/{self.project_name}/output_data/wacc_sparse_arrays_{self.project_name}_{year_name}.pkl'
        
        with open(filename, 'wb') as f:
            pickle.dump(self.wacc_list, f)
        return s
    
#=========================================================================================================================================
    def train_streamflow_model(self):
        print('TRAINING DEEP LEARNING STREAMFLOW PREDICTION MODEL')
        sdp = DataPreprocessor(self.project_name, self.study_area, self.start_date, self.end_date, '1989-01-01', '2016-12-31')
        print(' 1. Loading observed streamflow')
        sdp.load_observed_streamflow()
        #print('There are ')
        sn = str(len(sdp.sim_station_names))
        
        print(f'     Training deepstrmm model for {self.project_name} based on {sn} number of stations in the GRDC database')
        print(sdp.sim_station_names)
        print(' 2. Loading runoff data and other predictors')
        self.rawdata = sdp.get_data()
        
        print(' 3. Building neural network model')
        smodel = StreamflowModel(self.project_name)
        smodel.prepare_data(self.rawdata)
        smodel.build_model()
        #smodel.load_regional_model(f'./{self.project_name}/models/{self.project_name}_model_tcn360.h5')
        print(' 4. Training neural network model')
        smodel.train_model()
#========================================================================================================================  
                
    def simulate_streamflow(self, station_name, model_path):
        vdp = PredictDataPreprocessor(self.project_name, self.study_area, self.start_date, self.end_date, '1981-01-01', '1988-12-31')
        fulldata = vdp.load_observed_streamflow()
        print(vdp.sim_station_names)
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

        self.vmodel = PredictStreamflow(self.project_name)
        self.vmodel.prepare_data(rawdata)

        self.vmodel.load_model(model_path)
        y1 = self.vmodel.model.predict([self.vmodel.predictors, self.vmodel.catchment_size])
        y2 = y1
        #y2 = y1 * 100
        predicted_streamflow = np.where(y2<0, 0, y2)
        self.ps = predicted_streamflow
        
        lat = vdp.y
        lon = vdp.x
        self.plot_grdc_streamflow(observed_streamflow, predicted_streamflow)
       
        
#========================================================================================================================  
                
    def simulate_streamflow_all(self, model_path):
        vdp = PredictDataPreprocessor(self.project_name, self.study_area, self.start_date, self.end_date, '1981-01-01', '1988-12-31')
        fulldata = vdp.load_observed_streamflow()
        #print(vdp.sim_station_names)
        self.stat_names = vdp.sim_station_names
        
        #extracted_data = fulldata.where(fulldata.station_name.astype(str) == self.stat_names, drop=True)
        full_ids = list(fulldata.id.values)
        
        #self.station = extracted_data['runoff_mean'].where(extracted_data['station_name'] == self.stat_names, 
        #                                        drop=True).to_dataframe(name='station_discharge').reset_index()
        
        #if self.station['station_discharge'].notna().sum() < 1825:

        #station_id = self.station['id'][0]
        #station_index = full_ids.index(station_id)

        vdp.station_ids = np.unique([full_ids])
        rawdata = vdp.get_data()
        #if vdp.acc > 5000:
        observed_streamflow = list(map(lambda xy: xy[1], rawdata[0]))

        self.vmodel = PredictStreamflow(self.project_name)
        self.vmodel.prepare_data(rawdata)

        self.vmodel.load_model(model_path)
        y1 = self.vmodel.model.predict([self.vmodel.predictors, self.vmodel.catchment_size])
        #y2 = y1 * 100
        predicted_streamflow = np.where(y2<0, 0, y2)
        

        lat = vdp.y
        lon = vdp.x
        #self.plot_grdc_streamflow(observed_streamflow, predicted_streamflow)
        nse, kge = self.compute_metrics(observed_streamflow, predicted_streamflow)

        return {

                'station_name': station_name,
                'lat': lat,
                'lon': lon,
                'nse': nse,
                'kge': kge[0][0]

            }
            
        
#========================================================================================================================  

    def simulate_streamflow_latlng(self, model_path, lat, lon):
        vdp = PredictDataPreprocessor(self.project_name, self.study_area, self.start_date, self.end_date, self.start_date, self.end_date)
        
        rawdata = vdp.get_data_latlng(lat, lon)

        self.vmodel = PredictStreamflow(self.project_name)
        self.vmodel.prepare_data_latlng(rawdata)

        self.vmodel.load_model(model_path)
        y1 = self.vmodel.model.predict([self.vmodel.predictors, self.vmodel.catchment_size])
        y2 = y1 * 100
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
        plt.title('Comparison of observed and simulated streamflow for River ' + self.project_name)  # Add a title
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

    