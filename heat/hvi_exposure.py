import pandas as pd
import numpy as np
import os
import glob
import rasterio
from pyproj import Transformer
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
from scipy.ndimage import distance_transform_edt
from heat.process_lst import Process_LST
from heat.process_ndvi import Process_NDVI
from heat.process_ndbi import Process_NDBI
from heat.process_ndwi import Process_NDWI
from heat.raster_clipper import RasterClipper
from heat.landsat_downloader import LandsatDownloader
from heat.process_climatedata import Process_ClimateData
from heat.process_static_predictors  import Process_StaticPredictors
from pathlib import Path
#import getPass

class ExposurePreprocessor:
    def __init__(self, project_name, local_climatedata_available=True):
        self.working_dir = project_name
        rootdir = f'./projects/{self.working_dir}/input_data'
        self.local_climatedata_available = local_climatedata_available
        self.rootdir = Path(rootdir)
        self.lst_folder = Path(f'./projects/{self.working_dir}/input_data/lst')
        
        #self.study_area= study_area_path
        self.lst_list = list(map(str, self.lst_folder.glob('L*B*.TIF')))
        
        rc = RasterClipper()
        self.ras_template = list(map(str, self.lst_folder.glob('L*B*.TIF')))[0]
        self.study_area = self.ras_template
        #rc.clip(self.lst_list[0], self.study_area, out_path=self.ras_template, save_output=True)
        with rasterio.open(self.ras_template) as rt:
            rt_data = rt.read(1)
            bbox = rt.bounds
            original_crs = rt.crs
        
            # Set up a transformer to convert coordinates to EPSG:4326
            transformer = Transformer.from_crs(original_crs, 'EPSG:4326', always_xy=True)
            
            # Transform the bounding box coordinates to EPSG:4326
            minx, miny = transformer.transform(bbox.left, bbox.bottom)
            maxx, maxy = transformer.transform(bbox.right, bbox.top)
            self.bbox = [minx, miny, maxx, maxy]
        self.grids = range(1,(rt_data.shape[0]*rt_data.shape[1]+1))
        self.landsat_indices_dir = Path(f'./projects/{self.working_dir}/input_data/landsat_indices')

        os.makedirs(f'./projects/{self.working_dir}/input_data/static_data', exist_ok=True)
        self.static_data_dir = Path(f'./projects/{self.working_dir}/input_data/static_data')

        os.makedirs(f'./projects/{self.working_dir}/input_data/tasmax', exist_ok=True)
        self.tasmax_dir = Path(f'./projects/{self.working_dir}/input_data/tasmax')

        os.makedirs(f'./projects/{self.working_dir}/input_data/tasmin', exist_ok=True)
        self.tasmin_dir = Path(f'./projects/{self.working_dir}/input_data/tasmin')

        os.makedirs(f'./projects/{self.working_dir}/input_data/srad', exist_ok=True)
        self.srad_dir = Path(f'./projects/{self.working_dir}/input_data/srad')

        # if self.local_climatedata_available == True:
        #     os.makedirs('./input_data/tasmax', exist_ok=True)
        #     self.tasmax_dir = Path('./input_data/tasmax')

        #     os.makedirs('./input_data/tasmin', exist_ok=True)
        #     self.tasmin_dir = Path('./input_data/tasmin')

        #     os.makedirs('./input_data/srad', exist_ok=True)
        #     self.srad_dir = Path('./input_data/srad')
            
            #self.tasmax_dir = tasmax_folder
            #self.tasmin_dir = tasmin_folder
            #self.srad_dir = srad_folder

    
    def get_daily_vars(self, lst_file):
        all_vars =  pd.DataFrame(data=self.grids, columns=['id'])
        #composite and clip lst data
        
        day_lst, this_date = self.plst.preprocess(lst_file)
        print('Preprocessing data for ' + this_date)

        print('     Computing NDVI')
        self.pndvi.compute_ndvi(this_date)

        print('     Computing NDBI')
        self.pndbi.compute_ndbi(this_date)

        print('     Computing NDWI')
        self.pndwi.compute_ndwi(this_date)

        if self.local_climatedata_available == False:
            print('     Extracting tasmax')
            day_tasmax = self.pcd.select_daily_clim('tasmax', this_date)
    
            print('     Extracting tasmin')
            day_tasmin = self.pcd.select_daily_clim('tasmin', this_date)
    
            print('     Extracting srad')
            day_srad = self.pcd.select_daily_clim('rsds', this_date)
        else:
            print('     Extracting tasmax')
            day_tasmax = self.pcd.select_daily_clim_local(self.tasmax_dir, this_date)
    
            print('     Extracting tasmin')
            day_tasmin = self.pcd.select_daily_clim_local(self.tasmin_dir, this_date)
    
            print('     Extracting srad')
            day_srad = self.pcd.select_daily_clim_local(self.srad_dir, this_date)

        all_vars['lst'] = np.where(day_lst <-20,np.nan, day_lst).flatten()
        all_vars['ndvi'] = np.where(self.pndvi.corrected_ndvi <-20,np.nan, self.pndvi.corrected_ndvi).flatten()
        all_vars['ndvi_dist'] = self.pndvi.ndvi_dist.flatten()
        all_vars['ndbi'] = np.where(self.pndbi.corrected_ndbi <-20,np.nan, self.pndbi.corrected_ndbi).flatten()
        all_vars['ndbi_dist'] = self.pndbi.ndbi_dist.flatten()
        all_vars['ndwi_dist'] = self.pndwi.ndwi_dist.flatten()
        all_vars['tasmax'] = day_tasmax.values.flatten()
        all_vars['tasmin'] = day_tasmin.values.flatten()
        all_vars['srad'] = day_srad.values.flatten()
        #all_vars['waterbodies'] = psp.waterbodies_data.flatten()
        #all_vars['water_dist'] = psp.waterbodies_dist.flatten()
        #all_vars['aspect'] = psp.aspect_data.flatten()
        return all_vars

    def get_all_days(self):
        self.plst = Process_LST(self.lst_folder, self.study_area)
        self.pndvi = Process_NDVI(self.landsat_indices_dir, self.study_area)
        self.pndbi = Process_NDBI(self.landsat_indices_dir, self.study_area)
        self.pndwi = Process_NDWI(self.landsat_indices_dir, self.study_area)
        self.pcd = Process_ClimateData(self.working_dir, self.bbox)
        #self.psp = Process_StaticPredictors(static_data_dir, self.study_area)

        #print('Concatenating climate data')
        #self.tasmax_nc = self.pcd.concat_nc(tasmax_dir, self.ras_template)
        #self.tasmin_nc = self.pcd.concat_nc(tasmin_dir, self.ras_template)
        #self.srad_nc = self.pcd.concat_nc(srad_dir, self.ras_template)

        day_df_list = []
        
        for file in self.lst_list:
            day_df = self.get_daily_vars(file)
            day_df = day_df.dropna()
            day_df_list.append(day_df.astype('float32'))
            
        full_table = pd.concat(day_df_list)
        return full_table

#==========================================================================================================================================

class LstTrainer:
    def __init__(self):
        full_table = pd.read_csv(f'./projects/{text_input.value}/output_data/heat_model_inputs.csv')  
        self.full_table = full_table.dropna()

    def train_model(self):
        X = self.all_vars.drop(['lst'], axis=1)
        y = self.all_vars['lst']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

        # Define parameter distributions
        n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=10)]
        grow_policy = ['depthwise', 'lossguide']
        booster = ['gbtree']
        tree_method = ['exact']
        learning_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        gamma = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        subsample = [0.7, 0.8, 0.9, 1.0]  # Values between 0 and 1
        sampling_method = ['uniform']
        colsample_bytree = [0.7, 0.8, 0.9, 1.0]  # Values between 0 and 1
        max_leaves = [None]  # Adjust based on problem
        min_child_weight = [1, 2, 3, 4, 5]
        max_depth = [2, 4, 6, 8, 10, 12, 14]
        alpha = [0.1, 0.2, 0.3, 0.4, 0.5]  # Positive values for L1 regularization
        lambda1 = [0.1, 0.2, 0.3, 0.4, 0.5]  # Positive values for L2 regularization

        # Define the parameter grid for RandomizedSearchCV
        param_dist = {
            'n_estimators': n_estimators,
            'tree_method': tree_method,
            'subsample': subsample,
            'booster': booster,
            'learning_rate': learning_rate,
            'gamma': gamma,
            'sampling_method': sampling_method,
            'colsample_bytree': colsample_bytree,
            'max_leaves': max_leaves,
            'min_child_weight': min_child_weight,
            'max_depth': max_depth,
            'alpha': alpha,
            'lambda': lambda1,
            'grow_policy': grow_policy,
        }

        xgb_regressor = xgb.XGBRegressor()
        cv_params = RandomizedSearchCV(xgb_regressor, param_distributions=param_dist, 
                                       n_iter=100, cv=10, random_state=42, n_jobs=6)
        cv_params.fit(self.X_train, self.y_train)
        best_params = cv_params.best_params_
        xgb_model = XGBRegressor(**best_params)
        self.xgb_model = xgb_model.fit(self.X_train, self.y_train)

        model_filepath = f'./projects/{self.working_dir}/models/lst_model_{self.working_dir}.h5'
        self.xgb_model.save_model(model_filepath)

    def evaluate_model(self):
        observedTestFlat = np.array(self.y_test).flatten()
        predicted = self.xgb_model.predict(self.y_test)
        
        predictedTestFlat = predicted.flatten()
        #residualTest = np.array(Y_test) - gb_pred
        #residualTestFlat = residualTest.flatten()
        

    
        