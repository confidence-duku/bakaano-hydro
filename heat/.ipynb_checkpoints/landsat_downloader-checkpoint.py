import heat.landsatxplore
import json
from heat.landsatxplore.api import API
from heat.landsatxplore.earthexplorer import EarthExplorer
import os
import tarfile
from pathlib import Path
import geopandas as gpd
import rasterio
import rasterio.warp
from rasterio.crs import CRS
from shapely.geometry import shape, Polygon
import warnings

# Suppress specific UserWarning
warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS.*")

class LandsatDownloader:
    def __init__(self, project_name,  study_area):
        
        
        os.makedirs(f'./projects/{project_name}/input_data', exist_ok=True)
        self.output_dir = f'./projects/{project_name}/input_data'
        self.username = 'conficonfi'
        self.password = '1984@ConfiConfi'
        self.working_dir = project_name
        #self.dataset_id = landsat_dataset_id
       
        self.study_area = study_area

        # Dataset Name	Dataset ID
        # Landsat 5 TM Collection 2 Level 1	landsat_tm_c2_l1
        # Landsat 5 TM Collection 2 Level 2	landsat_tm_c2_l2
        # Landsat 7 ETM+ Collection 2 Level 1	landsat_etm_c2_l1
        # Landsat 7 ETM+ Collection 2 Level 2	landsat_etm_c2_l2
        # Landsat 8 Collection 2 Level 1	landsat_ot_c2_l1
        # Landsat 8 Collection 2 Level 2	landsat_ot_c2_l2
        # Landsat 9 Collection 2 Level 1	landsat_ot_c2_l1
        # Landsat 9 Collection 2 Level 2	landsat_ot_c2_l2
    def getlatlng(self):
        bbox_polygon = Polygon([(self.study_area[0], self.study_area[1]), (self.study_area[0], self.study_area[3]), 
                                (self.study_area[2], self.study_area[3]), (self.study_area[2], self.study_area[1]), (self.study_area[0], self.study_area[1])])
        shp = gpd.GeoDataFrame([1], geometry=[bbox_polygon], crs="EPSG:4326")
        
        #shp = gpd.read_file(self.study_area_shp)
            # Calculate the centroid
        centroid = shp.geometry.centroid
        
        # Extract the coordinates
        lon = centroid.x[0]
        lat = centroid.y[0]
    
            
        return lat, lon

    def downloader(self):
        ee = EarthExplorer(self.username, self.password)
        api = API(self.username, self.password)
        
        os.makedirs(f'./projects/{self.working_dir}/input_data/lst', exist_ok=True)
        lst_path = Path(f'./projects/{self.working_dir}/input_data/lst')
        
        os.makedirs(f'./projects/{self.working_dir}/input_data/landsat_indices', exist_ok=True)
        landsat_indices_path = Path(f'./projects/{self.working_dir}/input_data/landsat_indices')

        lat, lon = self.getlatlng()
        
        # Search for Landsat 8 and 9 scenes
        print('Searching and downloading landsat 8 and 9 scenes')
        scenes = api.search(
            dataset='landsat_ot_c2_l2',
            latitude=lat,
            longitude=lon,
            start_date='2001-01-01',
            end_date='2016-12-31',
            max_cloud_cover=20
        )
        print(    f"{len(scenes)} scenes found.")

        #landsat 8 and 9
        if len(scenes) > 0:
            for scene in scenes:
                did = scene['display_id']
                if 'L2SP' in did:
                    print('        downloading scene ' + did)
                    ee.download(identifier=did, output_dir=self.output_dir)
                    tar_file = self.output_dir + '/' + did + '.tar'

                    files_to_extract = [did + '_ST_B10.TIF', did + '_ST_QA.TIF']
                    self.extract_specific_files(tar_file, lst_path, files_to_extract)

                    files_to_extract = [did + '_SR_B6.TIF', did + '_SR_B5.TIF', did + '_SR_B4.TIF', did + '_SR_B3.TIF', did + '_QA_PIXEL.TIF']
                    self.extract_specific_files(tar_file, landsat_indices_path, files_to_extract)
                    os.remove(tar_file)
        else:
            print(f"{len(scenes)} scenes found.")

#================================================================================================================================
    # extract the surface temperature and QA bands from the landsat tar file and delete the tar file
    def extract_specific_files(self, tar_file, extracted_dir, files_to_extract):
        # Open the tar file
        with tarfile.open(tar_file, 'r') as tar:
            # Extract specific files/directories to the target directory
            for member in tar.getmembers():
                if member.name in files_to_extract:
                    tar.extract(member, path=extracted_dir)
        