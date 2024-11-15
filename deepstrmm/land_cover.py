
import os
from hydro.utils import Utils
import glob
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
from hydro.esa_worldcover_downloader import WorldCoverDownloader
import xarray as xr
import rioxarray

class LandCover:
    def __init__(self, working_dir, study_area, local_data=False, local_data_path=None):
        self.study_area = study_area
        self.working_dir = working_dir
        #self.cwd = os.path.dirname(os.getcwd())
        self.lc_path = f'{self.working_dir}/land_cover/lc_mosaic.tif'
        self.clipped_lc = f'{self.working_dir}/land_cover/lc_clipped.tif'
        self.dem_path = f'{self.working_dir}/land_cover/dem_clipped.tif'
        self.local_data = local_data
        self.local_data_path = local_data_path

    def download_lc(self):
        if self.local_data is False:
            if not os.path.exists(self.lc_path):
                print('     - Downloading land cover data')
                uw = Utils(self.working_dir, self.study_area)
                uw.get_bbox('EPSG:4326')
                bounds = [str(uw.minx), str(uw.miny), str(uw.maxx), str(uw.maxy)]
                wd = os.getcwd()
                thisdir = f'{self.working_dir}/land_cover'
                downloader = WorldCoverDownloader(output_folder=thisdir, bounds=bounds, shapefile=self.study_area)
                downloader.download()
            else:
                print(f"     - Land cover data already exists in {self.lc_path}; skipping download.")
        
    def mosaic_lc_old(self):
        
        print('     - Mosaicking land cover data')
        lc_list = glob.glob(f'{self.working_dir}/land_cover/ESA_WorldCover*.tif')
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
        if self.local_data is False:
            if not os.path.exists(self.lc_path):
                print('     - Mosaicking land cover data')
                
                lc_list = glob.glob(f'{self.working_dir}/land_cover/ESA_WorldCover*.tif')
                datalist = []
                for k in lc_list:
                    data = rioxarray.open_rasterio(k)
                    datalist.append(data)
                
                mosaic = xr.combine_by_coords(datalist, combine_attrs="override")
                mosaic.rio.to_raster(self.lc_path)

                self.preprocess()
            else:
                print(f"     - Land cover data already exists in {self.lc_path}; skipping mosaicking.")
        else:
            print(f"     - Local Land cover data already provided")

            lc = self.uw.align_rasters(self.local_data_path, self.dem_path)[0]
            lc.rio.to_raster(self.clipped_lc, dtype='float32')

    def preprocess(self):
        lc = self.uw.align_rasters(self.lc_path, self.dem_path)[0]
        lc.rio.to_raster(self.clipped_lc, dtype='float32')

    def get_landcover_data(self):
        self.download_lc()
        self.mosaic_lc()
