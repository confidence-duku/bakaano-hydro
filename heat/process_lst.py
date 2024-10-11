import numpy as np
import pandas as pd
import random
#import matplotlib.pyplot as plt
from netCDF4 import Dataset, date2num, num2date
import xarray as xr
import time, sys
from datetime import datetime, timedelta
import rasterio as rio
#from rasterio.windows import window
#from rasterio.warp import Resampling
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
import rasterio
from rasterio.enums import Resampling
import os
import rioxarray
import glob
import matplotlib as mpl
from heat.raster_clipper import RasterClipper
from pathlib import Path
#====================================================================================================================

class Process_LST:
    
    def __init__(self, lst_dir, study_area):
        self.lst_dir = lst_dir
        self.rc = RasterClipper()
        self.corrected_lst_list = []
        self.study_area=study_area
    
    def preprocess(self, lst_file):
        #lst_list = glob.glob(self.datadir + 'L*B*.TIF')
        lst_qa_list = list(map(str, self.lst_dir.glob('L*QA*.TIF')))
        lst_filename = os.path.split(lst_file)[1]
        self.lst_date = lst_filename[17:][:-26]
        lst_qa_match = [s for s in lst_qa_list if self.lst_date in s]
            
        if len(lst_qa_match) > 0:
            #lst_data = lst_data
            outpath = 'clipped.tif'
            #print('Processing Landsat image ' + lst_filename)
            lst_data= self.rc.clip(lst_file, self.study_area)
            #with rasterio.open(lst_file) as lt:
            #    lst_data = lt.read(1)
            lst_data = (lst_data * 0.00341802) + 149.0

            lst_qa_data = self.rc.clip(lst_qa_match[0], self.study_area)
            #with rasterio.open(lst_qa_match[0]) as qa:
            #    lst_qa_data = qa.read(1)
            lst_qa_data = np.where(lst_qa_data > 300, np.nan, 1)
                
            lst_corrected = lst_qa_data * lst_data
        return lst_corrected, self.lst_date


