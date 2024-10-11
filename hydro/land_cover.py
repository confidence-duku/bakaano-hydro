import subprocess
import os
from hydro.utils import Utils
import glob
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
from hydro.esa_worldcover_downloader import WorldCoverDownloader

class LandCover:
    def __init__(self, project_name, study_area):
        self.study_area = study_area
        self.project_name = project_name
        self.out_fp = f'./projects/{self.project_name}/land_cover/lc_mosaic.tif'

    def download_lc(self):
        #print('     - Downloading land cover data')
        uw = Utils(self.project_name, self.study_area)
        #uw.get_bbox('EPSG:4326')
        bounds = [str(self.study_area[0]), str(self.study_area[1]), str(self.study_area[2]), str(self.study_area[3])]
        wd = os.getcwd()
        thisdir = os.path.join(wd, f'./projects/{self.project_name}/land_cover')
        downloader = WorldCoverDownloader(output_folder=thisdir, bounds=bounds)
        downloader.download()
        
    def mosaic_lc(self):
        print('     - Mosaicking land cover data')
        lc_list = glob.glob(f'./projects/{self.project_name}/land_cover/ESA_WorldCover*.tif')
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
    
