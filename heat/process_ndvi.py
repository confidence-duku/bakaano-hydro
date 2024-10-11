import numpy as np
import pandas as pd
import random
#import matplotlib.pyplot as plt
from netCDF4 import Dataset, date2num, num2date
import xarray as xr
import time, sys
from datetime import datetime, timedelta
import rasterio as rio
from scipy.ndimage import distance_transform_edt
#from rasterio.windows import window
#from rasterio.warp import Resampling
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
from rasterio.enums import Resampling
import os
import rioxarray
import glob
import matplotlib as mpl
from heat.raster_clipper import RasterClipper
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
#====================================================================================================================

class Process_NDVI:
    
    def __init__(self, ndvi_dir, study_area):
        self.ndvi_dir = ndvi_dir
        self.rc = RasterClipper()
        #self.corrected_ndvi_list = []
        self.study_area=study_area
    
    def compute_ndvi(self, lst_date):
        b5_list = list(map(str, self.ndvi_dir.glob('LC*B5*.TIF')))
        b4_list = list(map(str, self.ndvi_dir.glob('LC*B4*.TIF')))
        qa_list = list(map(str, self.ndvi_dir.glob('LC*QA*.TIF')))

        b5_match = [s for s in b5_list if lst_date in s]
        b4_match = [s for s in b4_list if lst_date in s]
        qa_match = [s for s in qa_list if lst_date in s]

        if len(qa_match) > 0 and len(b4_match)>0 and len(b5_match)>0:
            #lst_data = lst_data
            outpath = 'clipped.tif'
            
            b5_data= self.rc.clip(b5_match[0], self.study_area)
            b4_data= self.rc.clip(b4_match[0], self.study_area)
            qa_data = self.rc.clip(qa_match[0], self.study_area)
            #lst_qa_data = rio.open(outpath).read(1)
            #lst_qa_data = rio.open(lst_qa_match[0]).read(1)
            ndvi = (b5_data - b4_data ) / (b5_data + b4_data)

            qa_data = np.where(qa_data > 21824, np.nan, 1)
            self.corrected_ndvi = ndvi * qa_data

        else:
            b3_list = list(map(str, self.ndvi_dir.glob('LE*B3*.TIF')))
            b4_list = list(map(str, self.ndvi_dir.glob('LE*B4*.TIF')))
            qa_list = list(map(str, self.ndvi_dir.glob('LE*QA*.TIF')))
    
            b3_match = [s for s in b3_list if lst_date in s]
            b4_match = [s for s in b4_list if lst_date in s]
            qa_match = [s for s in qa_list if lst_date in s]

            if len(qa_match) > 0 and len(b4_match)>0 and len(b3_match)>0:
                #lst_data = lst_data
                outpath = 'clipped.tif'
                print('Processing Landsat image for NDVI ' + b5_filename)
                b3_data= self.rc.clip(b3_match[0], self.study_area)
                b4_data= self.rc.clip(b4_match[0], self.study_area)
                qa_data = self.rc.clip(qa_match[0], self.study_area)
                #lst_qa_data = rio.open(outpath).read(1)
                #lst_qa_data = rio.open(lst_qa_match[0]).read(1)
                ndvi = (b4_data - b3_data ) / (b4_data + b3_data)
    
                qa_data = np.where(qa_data > 21824, np.nan, 1)
                self.corrected_ndvi = ndvi * qa_data             

        ndvi_dist_threshold = np.where(self.corrected_ndvi > 0.5, 0, 1)
        self.ndvi_dist = distance_transform_edt(input=ndvi_dist_threshold, return_distances=True)
        


