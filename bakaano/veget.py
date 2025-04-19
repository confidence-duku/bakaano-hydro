
import numpy as np
import pandas as pd
import os
from datetime import datetime
from bakaano.utils import Utils
from bakaano.pet import PotentialEvapotranspiration
from bakaano.router import RunoffRouter
import pickle
import scipy as sp
from bakaano.meteo import Meteo
from tqdm import tqdm
from datetime import datetime, timedelta

class VegET:
    """Generate an instance
    """
    def __init__(self, working_dir, study_area, start_date, end_date, climate_data_source, routing_method='mfd'):
        """Initialize a VegET object.

        Args:
            working_dir (str): The parent working directory where files and outputs will be stored.
            study_area_path (str): The path to the shapefile of the river basin or watershed.
            start_date (str): The start date of the simulation period in YYYY-MM-DD format
            end_date (str): The end date of the simulation period in YYYY-MM-DD format
            climate_data_source (str): The source of climate data. Options are 'CHELSA', 'ERA5', or 'CHIRPS'.
            routing_method (str): The method used for routing runoff. Options are 'mfd', 'd8' or 'dinf'. Default is 'mfd'.

        Methods
        -------
        __init__(working_dir, study_area_path, start_date, end_date, climate_data_source):
            Initializes the VegET object with project details.
        compute_veget_runoff_route_flow(prep_nc, tasmax_nc, tasmin_nc, tmean_nc):
            Computes the vegetation evapotranspiration and runoff routing flow.
        """
         # Initialize the project name
        self.working_dir = working_dir
        
        # Initialize the study area
        self.study_area = study_area
        
        # Initialize utility class with project name and study area.
        self.uw = Utils(self.working_dir, self.study_area)
        self.times = pd.date_range(start_date, end_date)
        
        # Set the start and end dates for the project
        self.start_date = start_date
        self.end_date = end_date
        self.routing_method = routing_method

        # Create necessary directories for the project structure   
        os.makedirs(f'{self.working_dir}/models', exist_ok=True)
        os.makedirs(f'{self.working_dir}/runoff_output', exist_ok=True)
        os.makedirs(f'{self.working_dir}/scratch', exist_ok=True)
        os.makedirs(f'{self.working_dir}/shapes', exist_ok=True)
        os.makedirs(f'{self.working_dir}/catchment', exist_ok=True)

        self.clipped_dem = f'{self.working_dir}/elevation/dem_clipped.tif'
        self.climate_data_source = climate_data_source

    def compute_veget_runoff_route_flow(self):  
        if not os.path.exists(f'{self.working_dir}/runoff_output/wacc_sparse_arrays.pkl'):
            # Initialize potential evapotranspiration and data preprocessor
            print('Computing VegET runoff and routing flow to river network')
            # Initialize potential evapotranspiration and data preprocessor
            eto = PotentialEvapotranspiration(self.working_dir, self.study_area, self.start_date, self.end_date)

            cd = Meteo(self.working_dir, self.study_area, start_date=self.start_date, end_date=self.end_date, 
                       local_data=False, data_source=self.climate_data_source, local_prep_path=None, local_tasmax_path=None, 
                       local_tasmin_path=None, local_tmean_path=None)
            prep_nc, tasmax_nc, tasmin_nc, tmean_nc = cd.get_meteo_data()

            if self.climate_data_source == 'CHELSA':

                tasmax_period = tasmax_nc.tasmax.sel(time=slice(self.start_date, self.end_date)) - 273.15
                tasmin_period = tasmin_nc.tasmin.sel(time=slice(self.start_date, self.end_date)) - 273.15
                tmean_period = tmean_nc.tas.sel(time=slice(self.start_date, self.end_date)) - 273.15
                rf = prep_nc.pr.sel(time=slice(self.start_date, self.end_date)) * 86400  # Conversion from kg/m2/s to mm/day
                rf = rf.astype(np.float32).assign_coords(lat=rf['lat'].astype(np.float32), lon=rf['lon'].astype(np.float32))
                self.rf = rf

            elif self.climate_data_source == 'ERA5':

                tasmax_period = tasmax_nc.tasmax.sel(time=slice(self.start_date, self.end_date)) - 273.15
                tasmin_period = tasmin_nc.tasmin.sel(time=slice(self.start_date, self.end_date)) - 273.15
                tmean_period = tmean_nc.tas.sel(time=slice(self.start_date, self.end_date)) - 273.15
                rf = prep_nc.pr.sel(time=slice(self.start_date, self.end_date)) * 1000
                rf = rf.astype(np.float32).assign_coords(lat=rf['lat'].astype(np.float32), lon=rf['lon'].astype(np.float32))
                self.rf = rf
                #rf = rf.astype(np.float32).assign_coords(lat=rf['lat'].astype(np.float32), lon=rf['lon'].astype(np.float32))
            elif self.climate_data_source == 'CHIRPS':
                
                tasmax_period = tasmax_nc.tasmax.sel(time=slice(self.start_date, self.end_date)) - 273.15
                tasmin_period = tasmin_nc.tasmin.sel(time=slice(self.start_date, self.end_date)) - 273.15
                tmean_period = tmean_nc.tas.sel(time=slice(self.start_date, self.end_date)) - 273.15
                rf = prep_nc.pr.sel(time=slice(self.start_date, self.end_date))
                rf = rf.astype(np.float32).assign_coords(lat=rf['lat'].astype(np.float32), lon=rf['lon'].astype(np.float32))
                self.rf = rf
            
            td = np.sqrt(tasmax_period - tasmin_period)
            pet_params = 0.408 * 0.0023 * (tmean_period + 17.8) * td
            pet_params = pet_params.astype(np.float32)
            self.pet_params = pet_params.assign_coords(
                lat=pet_params['lat'].astype(np.float32),
                lon=pet_params['lon'].astype(np.float32)
            )

            # Extract latitude and expand dimensions to match the lon dimension
            latsg = tmean_period[0]['lat']
            latsg = latsg.astype(np.float32)
            self.latgrids = latsg.expand_dims(lon=tmean_period[0]['lon'], axis=[1]).values
            
            # Initial soil moisture condition
            soil_moisture = rf[0] * 0   #initialize soil moisture
            soil_moisture = self.uw.align_rasters(soil_moisture, israster=False)

            pickle_file_path = f'{self.working_dir}/ndvi/daily_ndvi_climatology.pkl'
            with open(pickle_file_path, 'rb') as f:
                ndvi_array = pickle.load(f)
            
            water_holding_capacity = self.uw.align_rasters(f'{self.working_dir}/soil/clipped_AWCh3_M_sl6_1km_ll.tif', israster=True) * 10
            max_allowable_depletion = 0.5 * water_holding_capacity[0]

            tree_cover_tiff = f'{self.working_dir}/vcf/mean_tree_cover.tif'
            herb_cover_tiff = f'{self.working_dir}/vcf/mean_herb_cover.tif'
            tree_cover = self.uw.align_rasters(tree_cover_tiff, israster=True)[0]
            herb_cover = self.uw.align_rasters(herb_cover_tiff, israster=True)[0]

            tree_cover = np.where(tree_cover > 100, 0, tree_cover)
            herb_cover = np.where(herb_cover > 100, 0, herb_cover)

            interception = ((0.15 * tree_cover) + (0.1 * herb_cover))/100
            total_ETa = 0
            total_ETc = 0

            # Initialize runoff router and compute flow direction
            rout = RunoffRouter(self.working_dir, self.clipped_dem, self.routing_method)
            fdir, acc = rout.compute_flow_dir()
            
            facc_thresh = np.nanmax(acc) * 0.0001
            facc_mask = np.where(acc < facc_thresh, 0, 1)

            # Lists for storing results
            self.wacc_list = []
            self.mam_ro, self.jja_ro, self.son_ro, self.djf_ro = 0, 0, 0, 0
            self.mam_wfa, self.jja_wfa, self.son_wfa, self.djf_wfa = 0, 0, 0, 0
            this_date = datetime.strptime(self.start_date, '%Y-%m-%d')

            start = datetime.strptime(self.start_date, "%Y-%m-%d")
            end = datetime.strptime(self.end_date, "%Y-%m-%d")
            date_list = [(start + timedelta(days=i)).strftime("%Y-%m-%d")
                        for i in range((end - start).days + 1)]
            
            print('\n')
            for count, date in tqdm(enumerate(date_list), desc="     Simulating and routing runoff", unit="day", total=len(date_list)):
                if count % 365 == 0:
                    year_num = (count // 365) + 1
                    print(f'    Computing surface runoff and routing flow to river channels in year {year_num}')
                count2 = count+1
                this_rf = rf[count]
                this_rf = self.uw.align_rasters(this_rf, israster=False)
                eff_rain = this_rf * (1- interception)
                eff_rain = np.where(eff_rain<0, 0, eff_rain)

                this_et = eto.compute_PET(self.pet_params[count], self.latgrids, tmean_period[count])
                this_et = self.uw.align_rasters(this_et, israster=False)

                day_num = tmean_period[count]['time'].dt.dayofyear
                day_num = day_num.values.item()

                ndvi_day = ndvi_array[day_num] * 0.0001
                ndvi_day = self.uw.align_rasters(ndvi_day, israster=False)

                this_kcp = np.where(ndvi_day>0.4, (1.25*ndvi_day+0.2), (1.25*ndvi_day))

                ks = np.where(soil_moisture < max_allowable_depletion, (soil_moisture/max_allowable_depletion), 1)
                ETa = this_et * ks * this_kcp
                soil_moisture = soil_moisture + eff_rain - ETa
                soil_moisture = soil_moisture.values
                soil_moisture[np.isinf(soil_moisture) | np.isnan(soil_moisture)] = 0
                q_surf = np.where(soil_moisture > water_holding_capacity, (soil_moisture - water_holding_capacity), 0)
                q_surf[np.isinf(q_surf) | np.isnan(q_surf)] = 0

                ETc = this_kcp * this_et
                total_ETa = total_ETa + ETa
                total_ETc = total_ETc + ETc

                # Use Pysheds for weighted flow accumulation            
                ro_tiff = rout.convert_runoff_layers(q_surf[0])
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
                #sparse_series.append({"time": date, "matrix": sparse_matrix})
                self.wacc_list.append({"time": date, "matrix": wacc})            
                
            # Save station weighted flow accumulation data
            filename = f'{self.working_dir}/runoff_output/wacc_sparse_arrays.pkl'
            
            with open(filename, 'wb') as f:
                pickle.dump(self.wacc_list, f)
            print(f'Completed. Routed runoff data saved to {self.working_dir}/runoff_output/wacc_sparse_arrays.pkl')
        else:
            print(f'Routed runoff data exists in {self.working_dir}/runoff_output/wacc_sparse_arrays.pkl. Skipping processing')