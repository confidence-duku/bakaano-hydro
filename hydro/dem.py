import requests as r
import getpass, pprint, time, os, cgi, json
import geopandas as gpd
import glob
from shapely.geometry import Polygon

class DEMDownloader:
    def __init__(self, project_name, study_area, earthexplorer_username, earthexplorer_password):
        self.study_area = study_area
        self.username = earthexplorer_username
        self.password = earthexplorer_password
        self.project_name = project_name

        bbox_polygon = Polygon([(study_area[0], study_area[1]), (study_area[0], study_area[3]), (study_area[2], study_area[3]), (study_area[2], study_area[1]), (study_area[0], study_area[1])])
        self.nps = gpd.GeoDataFrame([1], geometry=[bbox_polygon], crs="EPSG:4326")

    def download_dem(self):
        api = 'https://appeears.earthdatacloud.nasa.gov/api/'  # Set the AρρEEARS API to a variable
        token_response = r.post('{}login'.format(api), auth=(self.username, self.password)).json() # Insert API URL, call login service, provide credentials & return json
        del self.username, self.password  
        product_response = r.get('{}product'.format(api)).json()                         # request all products in the product service
        products = {p['ProductAndVersion']: p for p in product_response} # Create a dictionary indexed by product name & version
        products['SRTMGL1_NC.003'] 
        prods = ['SRTMGL1_NC.003']
        dem_response = r.get('{}product/{}'.format(api, prods[0])).json() 
        layers = [(prods[0],'SRTMGL1_DEM')]
        prodLayer = []
        for l in layers:
            prodLayer.append({
                    "layer": l[1],
                    "product": l[0]
                  })
        token = token_response['token']                      # Save login token to a variable
        head = {'Authorization': 'Bearer {}'.format(token)}
        #nps = gpd.read_file(self.study_area) # Read in shapefile as dataframe using geopandas
        saj = self.nps.to_json()
        saj = json.loads(saj)
        projections = r.get('{}spatial/proj'.format(api)).json()  # Call to spatial API, return projs as json
        projs = {}                                  # Create an empty dictionary
        for p in projections: projs[p['Name']] = p  # Fill dictionary with `Name` as keys
        task_name = 'niger2'
        task_type = ['area']        # Type of task, area or point
        proj = projs['geographic']['Name']  # Set output projection 
        outFormat = ['geotiff']  # Set output file format type
        startDate = '01-01-1990'            # Start of the date range for which to extract data: MM-DD-YYYY
        endDate = '07-31-2017'              # End of the date range for which to extract data: MM-DD-YYYY
        recurring = False 

        task = {
            'task_type': task_type[0],
            'task_name': task_name,
            'params': {
                 'dates': [
                 {
                     'startDate': startDate,
                     'endDate': endDate
                 }],
                 'layers': prodLayer,
                 'output': {
                         'format': {
                                 'type': outFormat[0]}, 
                                 'projection': proj},
                 'geo': saj,
            }
        }

        task_response = r.post('{}task'.format(api), json=task, headers=head).json()  # Post json to the API task service, return response as json
        params = {'limit': 1, 'pretty': True} # Limit API response to 2 most recent entries, return as pretty json
        tasks_response = r.get('{}task'.format(api), params=params, headers=head).json() # Query task service, setting params and header 
        task_id = task_response['task_id']                                               # Set task id from request submission
        status_response = r.get('{}status/{}'.format(api, task_id), headers=head).json() # Call status service with specific task ID & user credentials

        # Ping API until request is complete, then continue to Section 4
        starttime = time.time()
        while r.get('{}task/{}'.format(api, task_id), headers=head).json()['status'] != 'done':
            print(r.get('{}task/{}'.format(api, task_id), headers=head).json()['status'])
            time.sleep(20.0 - ((time.time() - starttime) % 20.0))
        print(r.get('{}task/{}'.format(api, task_id), headers=head).json()['status'])

        mydir = os.getcwd()
        destDir = os.path.join(mydir, f'./projects/{self.project_name}/elevation')    
        if not os.path.exists(destDir):
            os.makedirs(destDir)

        bundle = r.get('{}bundle/{}'.format(api,task_id), headers=head).json()  # Call API and return bundle contents for the task_id as json   
        files = {}                                                       # Create empty dictionary
        for f in bundle['files']: 
            files[f['file_id']] = f['file_name']   # Fill dictionary with file_id as keys and file_name as values

        for f in files:
            dl = r.get('{}bundle/{}/{}'.format(api, task_id, f), headers=head, stream=True, allow_redirects = 'True')                                # Get a stream to the bundle file
            if files[f].endswith('.tif'):
                filename = files[f].split('/')[1]
            else:
                filename = files[f] 
            filepath = os.path.join(destDir, filename)                                                       # Create output file path
            with open(filepath, 'wb') as f:                                                                  # Write file to dest dir
                for data in dl.iter_content(chunk_size=8192): f.write(data) 
        print('Downloaded files can be found at: {}'.format(destDir))
        dem_file = glob.glob(f'./projects/{self.project_name}/elevation/SRTMGL1_NC.003*.tif')
        return dem_file

        