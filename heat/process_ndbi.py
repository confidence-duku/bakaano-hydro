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

class Process_NDBI:
    
    def __init__(self, ndbi_dir, clip_extent):
        self.ndbi_dir = ndbi_dir
        self.rc = RasterClipper()
        #self.corrected_ndvi_list = []
        self.clip_extent=clip_extent
    
    def compute_ndbi(self, lst_date):
        b6_list = list(map(str, self.ndbi_dir.glob('LC*B6*.TIF')))
        b5_list = list(map(str, self.ndbi_dir.glob('LC*B5*.TIF')))
        qa_list = list(map(str, self.ndbi_dir.glob('LC*QA*.TIF')))

        b5_match = [s for s in b5_list if lst_date in s]
        b6_match = [s for s in b6_list if lst_date in s]
        qa_match = [s for s in qa_list if lst_date in s]

        if len(qa_match) > 0 and len(b6_match)>0 and len(b5_match)>0:
            #lst_data = lst_data
            outpath = 'clipped.tif'
            #print('     Computing NDBI for ' + lst_date)
            #print('Processing Landsat image for NDVI ' + b5_filename)
            b5_data= self.rc.clip(b5_match[0], self.clip_extent)
            b6_data= self.rc.clip(b6_match[0], self.clip_extent)
            qa_data = self.rc.clip(qa_match[0], self.clip_extent)
            #lst_qa_data = rio.open(outpath).read(1)
            #lst_qa_data = rio.open(lst_qa_match[0]).read(1)
            ndbi = (b6_data - b5_data ) / (b6_data + b5_data)

            qa_data = np.where(qa_data > 21824, np.nan, 1)
            self.corrected_ndbi = ndbi * qa_data

        else:
            b4_list = list(map(str, self.ndbi_dir.glob('LE*B4*.TIF')))
            b5_list = list(map(str, self.ndbi_dir.glob('LE*B5*.TIF')))
            qa_list = list(map(str, self.ndbi_dir.glob('LE*QA*.TIF')))
    
            b5_match = [s for s in b5_list if lst_date in s]
            b4_match = [s for s in b4_list if lst_date in s]
            qa_match = [s for s in qa_list if lst_date in s]

            if len(qa_match) > 0 and len(b4_match)>0 and len(b5_match)>0:
                #lst_data = lst_data
                outpath = 'clipped.tif'
                #print('Processing Landsat image for NDVI ' + b5_filename)
                b5_data= self.rc.clip(b5_match[0], self.clip_extent)
                b4_data= self.rc.clip(b4_match[0], self.clip_extent)
                qa_data = self.rc.clip(qa_match[0], self.clip_extent)
                #lst_qa_data = rio.open(outpath).read(1)
                #lst_qa_data = rio.open(lst_qa_match[0]).read(1)
                ndbi = (b5_data - b4_data ) / (b5_data + b4_data)
    
                qa_data = np.where(qa_data > 21824, np.nan, 1)
                self.corrected_ndbi = ndbi * qa_data             

        ndbi_dist_threshold = np.where(self.corrected_ndbi > 0.5, 0, 1)
        self.ndbi_dist = distance_transform_edt(input=ndbi_dist_threshold, return_distances=True)
        


