
import os

from utils import Utils
import glob
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
from esa_worldcover_downloader import WorldCoverDownloader
import xarray as xr
import rioxarray

class LandCover:
    def __init__(self, project_name, study_area):
        self.study_area = study_area
        self.project_name = project_name
        self.cwd = os.path.dirname(os.getcwd())
        self.out_fp = os.path.join(self.cwd, f'{self.project_name}/land_cover/lc_mosaic.tif')

    def download_lc(self):
        #print('     - Downloading land cover data')
        uw = Utils(self.project_name, self.study_area)
        uw.get_bbox('EPSG:4326')
        bounds = [str(uw.minx), str(uw.miny), str(uw.maxx), str(uw.maxy)]
        wd = os.getcwd()
        thisdir = os.path.join(wd, f'../{self.project_name}/land_cover')
        downloader = WorldCoverDownloader(output_folder=thisdir, bounds=bounds, shapefile=self.study_area)
        downloader.download()
        
    def mosaic_lc_old(self):
        print('     - Mosaicking land cover data')
        lc_list = glob.glob(f'../{self.project_name}/land_cover/ESA_WorldCover*.tif')
        ftm = []
        for k in lc_list:
            src = rasterio.open(k)
            ftm.append(src)

        self.mosaic, out_trans = merge(ftm)
        
        out_meta = src.meta.copy()

        out_meta.update({"driver": "GTiff","height": self.mosaic.shape[1],
                         "width": self.mosaic.shape[2],"transform": out_trans, 
                         "dtype": 'uint8',
                         "compress": "lzw"
                        }
                       )
        
        with rasterio.open(self.out_fp, "w", **out_meta) as dest:
            dest.write(self.mosaic)
        #return self.out_fp

    def plot_lc(self):
        show(self.mosaic, cmap='terrain')

    def mosaic_lc(self):
        print('     - Mosaicking land cover data')
        
        lc_list = glob.glob(f'{self.cwd}/{self.project_name}/land_cover/ESA_WorldCover*.tif')
        datalist = []
        for k in lc_list:
            data = rioxarray.open_rasterio(k)
            datalist.append(data)
        
        mosaic = xr.combine_by_coords(datalist, combine_attrs="override")
        mosaic.rio.to_raster(self.out_fp)

    
