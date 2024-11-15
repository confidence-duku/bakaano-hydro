import rasterio
from owslib.wcs import WebCoverageService
from deepstrmm.utils import Utils
from rasterio.io import MemoryFile
from pathlib import Path


class Soil:
    def __init__(self, working_dir, study_area):
        self.study_area = study_area
        self.working_dir = working_dir
        self.uw = Utils(self.working_dir, self.study_area)
        self.uw.get_bbox('EPSG:4326')

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



        
    