import numpy as np
import pandas as pd
import os
import xarray as xr
from pathlib import Path
import rioxarray
import rasterio
import glob
from datetime import datetime, timedelta
from hydro.utils import Utils
from hydro.router import RunoffRouter
from hydro.streamflow_trainer import DataPreprocessor, StreamflowModel
from hydro.streamflow_predictor import PredictDataPreprocessor, PredictStreamflow
import richdem as rd
from rasterio.io import MemoryFile
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from IPython.display import HTML
import hydroeval
import math
import matplotlib
import time
import pickle
import gc
import scipy as sp
from whitebox import WhiteboxTools


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
    def __init__(self, project_name, study_area):
        
         # Initialize the project name
        
        self.ddir = os.getcwd()
        os.chdir('..')
        
        self.project_name = project_name
        os.makedirs(f'./projects/{self.project_name}', exist_ok=True)
        
        # Initialize the study area
        self.study_area = study_area
        self.start_date = '1981-01-01'
        self.end_date = '2016-12-31'
        # Initialize utility class with project name and study area.
        self.uw = Utils(self.project_name, self.study_area)
        self.times = pd.date_range(self.start_date, self.end_date)
        self.row = None
        self.col = None
        
        # File paths for soil data
        self.global_hsg = './common_data/HYSOGs250m.tif'
        self.global_lc = './common_data/lc_mosaic.tif'
        self.global_dem = './common_data/hyd_glo_dem_15s.tif'
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
#============================================================================================================================
    def find_outlet_location(self):
        #running whitebox tools
        fil = self.ddir + f'/projects/{self.project_name}/scratch/fil.tif'
        wbt.fill_depressions(self.dem, fil)

        fdr = self.ddir + f'/projects/{self.project_name}/scratch/fdr.tif'
        wbt.d8_pointer(fil, fdr)

        facc = self.ddir + f'/projects/{self.project_name}/scratch/facc.tif'
        wbt.d8_flow_accumulation(fil, facc)

        with rasterio.open(facc) as src:
            facc_data = src.read(1)
            facc_data = np.where(facc_data == src.nodata, np.nan, facc_data)
            facc_threshold = np.nanmax(facc_data) * 0.001

        str = self.ddir + f'/projects/{self.project_name}/scratch/str.tif'
        wbt.extract_streams(facc, str, facc_threshold)

        dist_outlet = self.ddir + f'/projects/{self.project_name}/scratch/dist_outlet.tif'
        wbt.distance_to_outlet(fdr, str, dist_outlet)

        with rasterio.open(dist_outlet) as do:
            dist_outlet_data = do.read(1)
            min_index = np.nanargmin(dist_outlet_data)
            self.row, self.col = np.unravel_index(min_index, dist_outlet_data.shape)
            
#=============================================================================================================================
    def get_facc_mask(self):
        # Initialize runoff router and compute flow direction
        rout = RunoffRouter(self.project_name, self.dem)
        fdir, acc = rout.compute_flow_dir()

        facc_thresh = np.nanmax(acc) * 0.001
        facc_mask = np.where(acc < facc_thresh, 0, 1)
        return facc_mask     

#==========================================================================================================================================
    def climate_scenarios(self, temp_change, rf_change):
        self.tasmax_nc = self.uw.clip_nc(f'./projects/{self.project_name}/climate_normals/tasmax_normal.nc')
        self.tasmin_nc = self.uw.clip_nc(f'./projects/{self.project_name}/climate_normals/tasmin_normal.nc')  
        self.tmean_nc = self.uw.clip_nc(f'./projects/{self.project_name}/climate_normals/tas_normal.nc')
        self.prep_nc = self.uw.clip_nc(f'./projects/{self.project_name}/climate_normals/pr_normal.nc')

        self.lats = self.prep_nc.pr.sel()[0]['lat'].values
        self.lons = self.prep_nc.pr.sel()[0]['lon'].values

        # # Initialize potential evapotranspiration and data preprocessor
        # eto = PotentialEvapotranspiration(self.project_name, self.study_area)
        #sdp = DataPreprocessor(self.project_name, self.study_area, self.start_date, self.end_date, self.start_date, self.end_date)

        # Load observed streamflow and climate data
        #roi_stations = sdp.station_ids
        tasmax_period = (self.tasmax_nc.tasmax.sel(time=slice(self.start_date, self.end_date)) - 273.15) + temp_change
        tasmin_period = (self.tasmin_nc.tasmin.sel(time=slice(self.start_date, self.end_date)) - 273.15) + temp_change
        tmean_period = (self.tmean_nc.tas.sel(time=slice(self.start_date, self.end_date)) - 273.15) + temp_change
        rf = (self.prep_nc.pr.sel(time=slice(self.start_date, self.end_date)) * 86400) * (1 - rf_change/100)  # Conversion from kg/m2/s to mm/day
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
        return rf, td, tmean_period

#==================================================================================================================================
    def surface_runoff_loop(self, rf, scenario_name):
        eto = PotentialEvapotranspiration(self.project_name, self.study_area)
        # Initial soil moisture condition
        smax = ((1000 / self.cn1) - 10) * 25.4
        s = 0.9 * smax

        # Initialize runoff router and compute flow direction
        rout = RunoffRouter(self.project_name, self.dem)
        fdir, acc = rout.compute_flow_dir()

        facc_thresh = np.nanmax(acc) * 0.001
        facc_mask = np.where(acc < facc_thresh, 0, 1)

        self.wacc_list = []

        if self.row is None and self.col is None:
            self.find_outlet_location()
            facc_mask = self.get_facc_mask()
            
        for count in range(rf.shape[0]):
            #start_time = time.perf_counter()
            this_rf = rf[count]
            this_et = eto.compute_PET(self.pet_params[count], self.latgrids, tmean_period[count])
            
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
            ro_out_name = self.ddir + f'/projects/{self.project_name}/scratch/runoff_scratch_1km.tif'
            ro_tiff = rout.convert_runoff_layers(q_surf, ro_out_name)
            wacc = rout.compute_weighted_flow_accumulation(ro_tiff)
            #wfa.append(wacc[self.row, self.col])

            wacc = wacc * facc_mask            
            wacc = sp.sparse.coo_array(wacc)
            self.wacc_list.append(wacc)
                  
            gc.collect()

        filename = f'/projects/{self.project_name}/output_data/wacc_sparse_arrays_{self.project_name}_{scenario_name}.pkl'
        
        with open(filename, 'wb') as f:
            pickle.dump(self.wacc_list, f)
        

#===========================================================================================================================
    def soil_lc_scenarios(self, nbs_lc, nbs_dem):  
        wbt = WhiteboxTools()
        wbt.verbose = False

        # #concatenate and clip netcdf files to study area extent
        # self.tasmax_nc = self.uw.concat_nc(f'./projects/{self.project_name}/climate_normals/*tasmax*.nc')
        # self.tasmin_nc = self.uw.concat_nc(f'./projects/{self.project_name}/climate_normals/*tasmin*.nc')  
        # self.tmean_nc = self.uw.concat_nc(f'./projects/{self.project_name}/climate_normals/*tas_*.nc')
        # self.prep_nc = self.uw.concat_nc(f'./projects/{self.project_name}/climate_normals/*pr_*.nc')
        # self.lats = self.prep_nc.pr.sel()[0]['lat'].values
        # self.lons = self.prep_nc.pr.sel()[0]['lon'].values

        #clip and resample lc file
        clipped_lc = f'./projects/{self.project_name}/land_cover/lc_{self.project_name}.tif'
        self.lc = self.uw.align_rasters_1km(self.global_lc, self.tasmax_nc)
        self.lc.rio.to_raster(clipped_lc, dtype='float32')
       
        #clip and resample dem file
        clipped_dem = f'./projects/{self.project_name}/elevation/dem_{self.project_name}.tif'
        self.dem = self.uw.align_rasters_1km(self.global_dem, self.tasmax_nc)
        self.dem.rio.to_raster(clipped_dem, dtype='float32')
        self.rd_dem = rd.rdarray(self.dem, no_data=-9999)

        #clip and resample soil file
        clipped_hsg = f'./projects/{self.project_name}/soil/hsg_{self.project_name}.tif'
        self.hsg = self.uw.align_rasters_1km(self.global_hsg, self.tasmax_nc)
        self.hsg.rio.to_raster(clipped_hsg)
       
        self.compute_CN2()
        self.compute_CN3(self.cn2)
        self.adjust_CN2_slp()
        self.compute_CN3(self.cn2_slp)
        self.compute_CN1(self.cn2_slp)

        

        # # Initialize potential evapotranspiration and data preprocessor
        
        # sdp = DataPreprocessor(self.project_name, self.study_area, self.start_date, self.end_date, self.start_date, self.end_date)

        #  # Load observed streamflow and climate data
        # roi_stations = sdp.station_ids
        # tasmax_period = self.tasmax_nc.tasmax.sel(time=slice(self.start_date, self.end_date)) - 273.15
        # tasmin_period = self.tasmin_nc.tasmin.sel(time=slice(self.start_date, self.end_date)) - 273.15
        # tmean_period = self.tmean_nc.tas.sel(time=slice(self.start_date, self.end_date)) - 273.15
        # rf = self.prep_nc.pr.sel(time=slice(self.start_date, self.end_date)) * 86400  # Conversion from kg/m2/s to mm/day
        # rf = rf.astype(np.float32).assign_coords(lat=rf['lat'].astype(np.float32), lon=rf['lon'].astype(np.float32)).values
        
        # td = np.sqrt(tasmax_period - tasmin_period)
        # pet_params = 0.408 * 0.0023 * (tmean_period + 17.8) * td
        # pet_params = pet_params.astype(np.float32)
        # self.pet_params = pet_params.assign_coords(
        #     lat=pet_params['lat'].astype(np.float32),
        #     lon=pet_params['lon'].astype(np.float32)
        # ).values

        # #pet_params = pet_params.values
        # # Extract latitude and expand dimensions to match the lon dimension
        # latsg = tmean_period[0]['lat']
        # latsg = latsg.astype(np.float32)
        # self.latgrids = latsg.expand_dims(lon=tmean_period[0]['lon'], axis=[1]).values       
        
        # Initial soil moisture condition
        # smax = ((1000 / self.cn1) - 10) * 25.4
        # s = 0.9 * smax

        # # Initialize runoff router and compute flow direction
        # rout = RunoffRouter(self.project_name, self.dem)
        # fdir, acc = rout.compute_flow_dir()

        # facc_thresh = np.nanmax(acc) * 0.001
        # facc_mask = np.where(acc < facc_thresh, 0, 1)

        # self.wacc_list = []

        # if self.row is None and self.col is None:
        #     self.find_outlet_location()
        #     facc_mask = self.get_facc_mask()

        # return s, facc_mask, rf, eto, rout
            
        #self._surface_runoff(s, facc_mask, rf, eto, rout, scenario_name)              
        

        #wfa = []
        # for count in range(rf.shape[0]):
        #     #start_time = time.perf_counter()
        #     this_rf = rf[count]
        #     this_et = eto.compute_PET(self.pet_params[count], self.latgrids, tmean_period[count])
            
        #     p1 = this_rf - (0.2 * s)
        #     p2 = p1**2
        #     p3 = np.where(p1<0,0,p2)
        #     p4 = this_rf + (0.8 * s)
        #     q_surf = p3 / p4
        #     #q_surf = q_surf_sp.toarray()
        #     q_surf = np.where(q_surf>0, q_surf, 0)
            
        #     #self.qlist.append(q_surf)
        #     # Adjust retention parameter at the end of the day
        #     s1 = np.exp((self.cncoef * s) / smax)
        #     s = np.add(np.multiply(this_et, s1), s) - this_rf + q_surf

        #     # Use Pysheds for weighted flow accumulation            
        #     ro_out_name = self.ddir + f'/projects/{self.project_name}/scratch/runoff_scratch_1km.tif'
        #     ro_tiff = rout.convert_runoff_layers(q_surf, ro_out_name)
        #     wacc = rout.compute_weighted_flow_accumulation(ro_tiff)
        #     #wfa.append(wacc[self.row, self.col])

        #     wacc = wacc * facc_mask            
        #     wacc = sp.sparse.coo_array(wacc)
        #     self.wacc_list.append(wacc)
                  
        #     gc.collect()

        # filename = f'/projects/{self.project_name}/output_data/wacc_sparse_arrays_{self.project_name}_{scenario_name}.pkl'
        
        # with open(filename, 'wb') as f:
        #     pickle.dump(self.wacc_list, f)
        
        # df = pd.DataFrame(wfa, columns=['weighted_flow_acc'])
        ## output_csv_path
        ## df.to_csv(self.ddir + f'/projects/{self.project_name}/output_data/wfa_1km.csv', index=False)
        # df.to_csv(output_csv_path, index=False)

# ===========================================================================================================================
#     def train_streamflow_model(self):
#         print('TRAINING DEEP LEARNING STREAMFLOW PREDICTION MODEL')
#         sdp = DataPreprocessor(self.project_name, self.study_area, self.start_date, self.end_date, '1990-01-01', '2016-12-31')
#         print(' 1. Loading observed streamflow')
#         sdp.load_observed_streamflow()
#         #print('There are ')
#         sn = str(len(sdp.sim_station_names))
#         print(f'     Training deepstrmm model for {self.project_name} based on {sn} number of stations in the GRDC database')
#         print(sdp.sim_station_names)
#         print(' 2. Loading runoff data and other predictors')
#         self.rawdata = sdp.get_data()
        
#         print(' 3. Building neural network model')
#         smodel = StreamflowModel(self.project_name)
#         smodel.prepare_data(self.rawdata)
#         #smodel.build_model()
#         smodel.load_regional_model(self.ddir + f'/projects/{self.project_name}/models/{self.project_name}_model_tcn360.h5')
#         print(' 4. Training neural network model')
#         smodel.train_model()

#===========================================================================================================================
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


    