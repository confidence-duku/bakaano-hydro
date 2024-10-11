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
from pathlib import Path
#====================================================================================================================

class Process_StaticPredictors:
    
    def __init__(self, sd_dir, study_area):
        self.static_dir = Path(sd_dir)
        self.rc = RasterClipper()
        #self.corrected_ndvi_list = []
        self.study_area=study_area

    def preprocess_static_predictors(self, waterbodies_file, aspect_file):
        waterbodies = self.static_dir / waterbodies_file
        aspect = self.static_dir / aspect_file
        
        rc = RasterClipper()
        with rasterio.open(waterbodies) as wbd:
            self.waterbodies_data = wbd.read(1)

        waterbodies_thresh = np.where(self.waterbodies_data == 1, 0, 1)
        self.waterbodies_dist = distance_transform_edt(input=waterbodies_thresh, return_distances=True)

        with rasterio.open(aspect) as src:
            self.aspect_data = src.read(1)


