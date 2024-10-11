import rasterio
from rasterio import plot
from owslib.wcs import WebCoverageService
from utils import Utils
from rasterio.io import MemoryFile
from io import BytesIO

class SoilGridsData:
    def __init__(self, project_name, study_area):
        self.study_area = study_area
        self.project_name = project_name

    def get_soil_data(self):
        #print('     - Downloading soil (clay & sand) data')
        clay_wcs = WebCoverageService('http://maps.isric.org/mapserv?map=/map/clay.map', version='2.0.1')
        sand_wcs = WebCoverageService('http://maps.isric.org/mapserv?map=/map/sand.map', version='2.0.1')

        clay_list = [element for element in clay_wcs.contents if 'mean' in element]
        sand_list = [element for element in sand_wcs.contents if 'mean' in element]
        #soil_list = clay_list + sand_list

        uw = Utils(self.project_name, self.study_area)
        #uw.get_bbox('ESRI:54052')
        bbox = [('X', self.study_area[0], self.study_area[2]), ('Y', self.study_area[1], self.study_area[3])]
        #bbox = [('X', -1784000, -1140000), ('Y', 1356000, 1863000)]
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
                    filename = f'./{self.project_name}/soil/' + k + '.tif'
            
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
                    filename = f'./{self.project_name}/soil/' + k + '.tif'
            
                    # Write the data to a new GeoTIFF file
                    with rasterio.open(filename, 'w', **meta) as dst:
                        dst.write(dataset.read())
        



        
    