import rioxarray
import rasterio
import numpy as np
from owslib.wcs import WebCoverageService
from bakaano.utils import Utils
from rasterio.io import MemoryFile
from pathlib import Path
import os


class Soil:
    def __init__(self, working_dir, study_area):
        """
        Initialize a Soil object.

        Args:
            working_dir (str): The working directory where files and outputs will be stored.
            study_area (str): The path to the shapefile of the river basin or watershed.

        Returns:
            Geotiff files of sand and clay contents for different soil depths to be stored in "{working_dir}/soil" directory
        """
        self.study_area = study_area
        self.working_dir = working_dir
        os.makedirs(f'{self.working_dir}/soil', exist_ok=True)
        self.uw = Utils(self.working_dir, self.study_area)
        self.uw.get_bbox('EPSG:4326')
        self.hsg_filename = f'{self.working_dir}/soil/hsg.tif'
        self.hsg_filename_prj = f'{self.working_dir}/soil/hsg_clipped.tif'
        self.dem_path = f'{self.working_dir}/land_cover/dem_clipped.tif'

    def get_soil_data(self):
        soil_folder = Path(f'{self.working_dir}/soil/')
        if not any(soil_folder.iterdir()):
        
            #print('     - Downloading soil (clay & sand) data')
            clay_wcs = WebCoverageService('http://maps.isric.org/mapserv?map=/map/clay.map', version='2.0.1')
            sand_wcs = WebCoverageService('http://maps.isric.org/mapserv?map=/map/sand.map', version='2.0.1')

            clay_list = [element for element in clay_wcs.contents if 'mean' in element]
            sand_list = [element for element in sand_wcs.contents if 'mean' in element]
            #soil_list = clay_list + sand_list

            uw = Utils(self.working_dir, self.study_area)
            uw.get_bbox('ESRI:54052')
            bbox = [('X', uw.minx, uw.maxx), ('Y', uw.miny, uw.maxy)]
            crs = "http://www.opengis.net/def/crs/EPSG/0/152160"
            sformats = clay_wcs.contents[clay_list[0]].supportedFormats[0]
            #crs = rasterio.crs.CRS({"init": "epsg:32631"})

            #soil_data = []
            #print('Downloading clay data')
            for k in clay_list:
                response = clay_wcs.getCoverage(
                    identifier=[k], 
                    crs=crs,
                    subsets=bbox, 
                    resx=250, resy=250, 
                    format=sformats)

                filename = k + '.tif'
                with MemoryFile(response.read()) as memfile:
                    with memfile.open() as dataset:
                        # Get metadata from the original dataset
                        meta = dataset.meta.copy()
                
                        # Specify the output filename
                        filename = f'{self.working_dir}/soil/' + k + '.tif'
                
                        # Write the data to a new GeoTIFF file
                        with rasterio.open(filename, 'w', **meta) as dst:
                            dst.write(dataset.read())

                    
            #print('Downloading sand data')
            for k in sand_list:
                response = sand_wcs.getCoverage(
                    identifier=[k], 
                    crs=crs,
                    subsets=bbox, 
                    resx=250, resy=250, 
                    format=sformats)

                filename = k + '.tif'
                with MemoryFile(response.read()) as memfile:
                    with memfile.open() as dataset:
                        # Get metadata from the original dataset
                        meta = dataset.meta.copy()
                
                        # Specify the output filename
                        filename = f'{self.working_dir}/soil/' + k + '.tif'
                
                        # Write the data to a new GeoTIFF file
                        with rasterio.open(filename, 'w', **meta) as dst:
                            dst.write(dataset.read())
        else:
                print(f"     - Soil data already exists in {soil_folder}; skipping download.") 

        self.compute_HSG()

    def compute_HSG(self):
        soil_dir = Path(f'{self.working_dir}/soil')
        clay_list = list(map(str, soil_dir.glob('clay*.tif')))
        sand_list = list(map(str, soil_dir.glob('sand*.tif')))
        hsg_list = []
        for cl, sd in zip(clay_list, sand_list):
            with rasterio.open(cl) as src:
                clay = src.read(1)
                clay = np.divide(clay, 10)

            with rasterio.open(sd) as src1:
                sand = src1.read(1)
                sand = np.divide(sand, 10)
                
            # Conditions for each group
            conditions = [
                (sand > 90) & (clay < 10),  # Group A
                (sand >= 50) & (sand <= 90) & (clay >= 10) & (clay < 20),  # Group B
                (sand < 50) & (clay >= 20) & (clay <= 40),  # Group C
                (sand < 50) & (clay > 40)  # Group D
            ]

            # Corresponding group numbers
            choices = [1, 2, 3, 4]

            # Use np.select to classify the samples
            this_hsg = np.select(conditions, choices, default=0)
            hsg_list.append(this_hsg)

        hsg1 = np.max(np.array(hsg_list), axis=0)
        hsg1 = np.where(hsg1 < 0, 0, hsg1)
       

        with rasterio.open(clay_list[0]) as slp:
            slp_meta = slp.profile

        out_meta = slp_meta.copy()
        out_meta.update({            
             "compress": "lzw"
        })

        with rasterio.open(self.hsg_filename, 'w', **out_meta) as dst:
            dst.write(hsg1, indexes=1)

        self.uw.reproject_raster(self.hsg_filename, self.hsg_filename_prj)
        hsg2 = self.uw.align_rasters(self.hsg_filename_prj, self.dem_path)[0]
        hsg2.rio.to_raster(self.hsg_filename_prj, dtype='float32')
        self.uw.clip(raster_path=self.hsg_filename_prj, out_path=self.hsg_filename_prj, save_output=True)
        #self.hsg = np.where(self.lc !=50, self.hsg, 4) #adjusting soil for urban areas

    def plot_soil(self):
        soil_data = rioxarray.open_rasterio(self.hsg_filename_prj)
        soil_data = soil_data.where(soil_data > 0)
        soil_data.plot(cmap='Set1')



        
    