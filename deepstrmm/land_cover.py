
import os
import glob
from deepstrmm.utils import Utils
from deepstrmm.esa_worldcover_downloader import WorldCoverDownloader
import rioxarray

class LandCover:
    def __init__(self, working_dir, study_area, local_data=False, local_data_path=None):
        """
        Initialize a LandCover object.

        Args:
            working_dir (str): The working directory where files and outputs will be stored.
            study_area (str): The path to the shapefile of the river basin or watershed.
            local_data (bool, optional): Flag indicating whether to use local data instead of downloading new data. Defaults to False.
            local_data_path (str, optional): Path to the local land cover geotiff tile if `local_data` is True. Defaults to None. Local landcover provided should have same land cover classes as ESA WorldCover. See documentation for more details
        
        Returns:
            A land cover geotiff clipped to the study area extent, reprojected to the correct CRS, resampled to match DEM resolution and to be stored in "{working_dir}/land_cover" directory
        """
        self.study_area = study_area
        self.working_dir = working_dir
        os.makedirs(f'{self.working_dir}/land_cover', exist_ok=True)
        self.lc_path = f'{self.working_dir}/land_cover/lc_mosaic.tif'
        self.clipped_lc = f'{self.working_dir}/land_cover/lc_clipped.tif'
        self.dem_path = f'{self.working_dir}/land_cover/dem_clipped.tif'
        self.local_data = local_data
        self.local_data_path = local_data_path
        self.uw = Utils(self.working_dir, self.study_area)

    def download_lc(self):
        if self.local_data is False:
            if not os.path.exists(self.clipped_lc):
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
        else:
            #print("     - Local land cover data provided.")
            try:
                if not self.lc_path:
                    raise ValueError("Local land cover path must be provided when 'local_data' is set to True.")
                if not os.path.exists(self.lc_path):
                    raise FileNotFoundError(f"The specified local land cover file '{self.lc_path}' does not exist.")
                if not self.lc_path.endswith('.tif'):
                    raise ValueError("The local land cover file must be a GeoTIFF (.tif) file.")
                #print(f"Processing local land cover data from '{self.lc_path}'.")
                self.uw.process_local_land_cover(self.lc_path)
            except (ValueError, FileNotFoundError) as e:
                print(f"Error: {e}")

    def get_landcover_data(self):
        self.download_lc()
        self.mosaic_lc()

    def mosaic_lc(self):
        if self.local_data is False:
            if not os.path.exists(self.clipped_lc):
                lc_list = glob.glob(f'{self.working_dir}/land_cover/ESA_WorldCover*.tif')
                combined_lc = 0
                for k in lc_list:
                    lc = self.uw.align_rasters(k, self.dem_path)[0]
                    combined_lc = combined_lc + lc
                combined_lc.rio.to_raster(self.clipped_lc, dtype='float32')
                self.uw.clip(raster_path=self.clipped_lc, out_path=self.clipped_lc, save_output=True)
            else:
                print(f"     - Land cover data already exists in {self.clipped_lc}; skipping mosaicking.")
        else:
            if not os.path.exists(self.local_data):
                print(f"Error: The specified local data path '{self.local_data}' does not exist.")
            elif not self.local_data.endswith('.tif'):
                print(f"Error: The specified local data file '{self.local_data}' is not a .tif file.")
            else:
                lc = self.uw.align_rasters(self.local_data, self.dem_path)[0]
                lc.rio.to_raster(self.clipped_lc, dtype='float32')


    def plot_landcover(self):
        lc_data = rioxarray.open_rasterio(self.clipped_lc)
        lc_data = lc_data.where(lc_data > 0)
        lc_data.plot(cmap='Set2')
        
        
