
import numpy as np
import rasterio
import glob
from rasterio.transform import xy
from rasterio.features import shapes
from rasterio.mask import mask
from pyproj import Proj, transform
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt
from rasterio.transform import rowcol
from shapely.geometry import Polygon
from pyproj import Proj, transform
from scipy.optimize import curve_fit
import scipy.stats as stats
import lmoments3 as lmom
from lmoments3 import distr
from whitebox import WhiteboxTools
import requests
import os
import pickle
wbt = WhiteboxTools()
wbt.verbose = False

class FloodMapper:
    def __init__(self, working_dir):
        
        # Define the original and target projections
        self.working_dir = working_dir
        self.original_proj = Proj('epsg:4326')  # WGS84
        self.target_proj = Proj('epsg:3857')# Web Mercator
        self.water_levels = [0.1, 0.23, 0.4, 0.75, 1, 1.36, 1.55, 1.79, 2, 2.29, 2.67, 2.88, 3, 
                             3.3, 3.66, 3.8, 4, 4.33, 4.57, 7]
        self.mannings_n = 0.05
        self.dem_filepath = f'{self.working_dir}/elevation/dem_clipped.tif'       

        os.makedirs(f'{self.working_dir}/climate_normals', exist_ok=True)
        os.makedirs(f'{self.working_dir}/scenarios/subbasin_flood_maps', exist_ok=True)
        os.makedirs(f'{self.working_dir}/scratch', exist_ok=True)
        os.makedirs(f'{self.working_dir}/glofas_flood_hazards', exist_ok=True)
#==========================================================================================================================================
    
    def reproject_raster(self):
        dst_crs = 'EPSG:3857'

        with rasterio.open(self.dem_filepath1) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            with rasterio.open(self.dem_filepath, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear)
#==========================================================================================================================================   
    # Function to transform a (lon, lat) tuple
    def transform_coordinates(self, lon, lat):
        x, y = transform(self.original_proj, self.target_proj, lon, lat)
        return x, y
#==========================================================================================================================================
    def extract_grid_value(self, lon, lat, grid_data, dem_transform):
        row, col = rowcol(dem_transform, lon, lat)
        value = grid_data[row, col]
        return value
#==========================================================================================================================================
    def convert_to_latlng(self, row, col, dem):
        with rasterio.open(dem) as dataset:
            # Get the affine transformation of the dataset
            transform = dataset.transform

            # Convert row and column to lat and lon
            lon, lat = xy(transform, row, col)
            return lat, lon
#==========================================================================================================================================
    def convert_coords(self, x, y):
        # Define the projections
        proj_3857 = Proj(init='epsg:3857')
        proj_4326 = Proj(init='epsg:4326')

        # Perform the transformation
        lon, lat = transform(proj_3857, proj_4326, x, y)
        return lat, lon

#==========================================================================================================================================    

    def rating_curve(self, Q, a, b, c):
        return a * Q**b + c

#=================================================================================================================================================

    def compute_watershed_features_glofas(self):
        wbt = WhiteboxTools()
        wbt.verbose = True
        
        print('Filling depressions and resolving flats')
        fil = f'{self.working_dir}/scratch/fil.tif'
        if os.path.exists(fil):
            print('    File already exists')
        else:
            wbt.fill_depressions(self.dem_filepath, fil)
        
        with rasterio.open(fil) as fl:
            fil_data = fl.read(1)
            fil_data = np.where(fil_data == fl.nodata, np.nan, fil_data)

        
        print('Computing flow direction')        
        fdr = f'{self.working_dir}/scratch/fdr.tif'
        if os.path.exists(fdr):
            print('    File already exists')
        else:
            wbt.d8_pointer(fil, fdr)
        
        print('Computing flow accumulation')
        facc = f'{self.working_dir}/scratch/facc.tif'
        if os.path.exists(facc):
            print('    File already exists')
        else:
            wbt.d8_flow_accumulation(fil, facc)        
        
        #get threshold for stream delineation
        with rasterio.open(facc) as src:
            facc_data = src.read(1)
            facc_data = np.where(facc_data == src.nodata, np.nan, facc_data)
            facc_threshold = np.nanmax(facc_data) * 0.05
        
        print('Extracting stream network')
        streams = f'{self.working_dir}/scratch/str.tif'
        if os.path.exists(streams):
            print('    File already exists')
        else:
            wbt.extract_streams(facc, streams, facc_threshold)
        
        print('Computing subbasin')
        subbasins = f'{self.working_dir}/scratch/subbasins.tif'
        if os.path.exists(subbasins):
            print('    File already exists')
        else:
            wbt.subbasins(fdr, streams, subbasins)
        
        with rasterio.open(subbasins) as sb:
            subbasin_data = sb.read(1)
            subbasin_ids = list(np.unique(subbasin_data))
            nodata_value = sb.nodata
            subbasin_ids.remove(nodata_value)

        return subbasin_data, subbasin_ids, facc_data

#==========================================================================================================================================
    def compute_baseline_flood_thresholds(self, num_years, start_year):
        
        flood_threshold_dict ={}

        subbasin_data, subbasin_ids, facc_data = self.compute_watershed_features_glofas()
        
        from deepstrmm.main import DeepSTRMM
        ds = DeepSTRMM(
            working_dir = self.working_dir,
            study_area = f'common_data/{self.working_dir}.shp',
            start_date = "1981-01-01",
            end_date = "2016-12-31",
            cncoef = -1
            )

        for num in subbasin_ids:
            print(f'Developing flood thresholds for reach {num}')
            this_subbasin = np.where(subbasin_data == num, 1, np.nan)
            if np.nansum(this_subbasin) > 10:
            
                outlet = this_subbasin * facc_data
                orow, ocol = np.unravel_index(np.nanargmax(outlet), outlet.shape)
                outlet_lat, outlet_lon = self.convert_to_latlng(orow, ocol, self.dem_filepath1)

            lat =  outlet_lat
            lon = outlet_lon

            predicted_streamflow = ds.simulate_streamflow_latlng(f'{self.working_dir}/models/model_tcn360.keras', lat, lon)

            return_period_list = [10,20,50,75,100,200, 500]
            flood_threshold_list = []

            for return_period in return_period_list:
            
                if len(predicted_streamflow)>0:
                    if np.max(predicted_streamflow)>0:
                        print(f'     Computing threshold for return period {return_period}')
                        total_years = num_years - 1
                        annual_peaks = self.extract_annual_peaks(predicted_streamflow[360:], total_years, start_year)
                        flood_threshold = self.return_period_discharge(annual_peaks, return_period)
                        flood_threshold_list.append(flood_threshold)
                else:
                    continue
                
            if len(flood_threshold_list) >= 7:
                flood_threshold_dict[num] = {
                    'subbasin_id': num,
                    'RP10': flood_threshold_list[0],
                    'RP20': flood_threshold_list[1],
                    'RP50': flood_threshold_list[2],
                    'RP75': flood_threshold_list[3],
                    'RP100': flood_threshold_list[4],
                    'RP200': flood_threshold_list[5],
                    'RP500': flood_threshold_list[6],
                    'outlet_lat': lat,
                    'outlet_lon': lon
                }
            else:
                print(f"Warning: flood_threshold_list for subbasin {num} does not have enough elements.")


        with open(f'{self.working_dir}/models/baseline_flood_threshold_dict.pkl', 'wb') as file:
            pickle.dump(flood_threshold_dict, file)
        return flood_threshold_dict

#=====================================================================================================================================================
    def select_glofas_flood_hazard_file(self, lat, lon, return_period):
        # Determine the northern/southern latitude bound
        if lat >= -1:  # Northern Hemisphere
            lat_prefix = "N"
            if lat < 9:
                north_lat = 9  # For latitudes between 0 and 9
            else:
                north_lat = (int(np.ceil(lat)) // 10) * 10 + 9  # N19, N29, etc.
                
        
        else:
            lat_prefix = "S"
            if lat >= -11:
                north_lat = 1  # For latitudes between -1 and -11, use S1
            else:
                north_lat = abs(int(lat) // 10) * 10 + 1 - 10 # For latitudes -11 and below, use S11, S21, etc.
    
       # Determine the western/eastern longitude bound
        if lon >= 10:  # Eastern Hemisphere
            lon_prefix = "E"
            west_lon = (int(lon) // 10) * 10  # Minimum E tile (e.g., E10)
        elif 0 <= lon < 10:  # Special case for longitudes between 0 and 10
            lon_prefix = "W"
            west_lon = 0  # Use W0 for longitudes between 0 and 10
        else:  # Western Hemisphere
            lon_prefix = "W"
            west_lon = abs((int(lon) // 10 ) * 10)  # W10, W20, etc., for negative longitudes
        
        # Construct the file name
        file_name = f"{lat_prefix}{north_lat}_{lon_prefix}{west_lon}_{return_period}_depth.tif"
        return file_name
    
#================================================================================================================================
    def get_glofas_flood_hazard(self, file_url, rp):
        base_url = f'https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard/{rp}/'
        os.makedirs(f'{self.working_dir}/glofas_flood_hazards/{rp}', exist_ok=True)
        download_file_url = base_url + file_url
        local_file_path = os.path.join(f'{self.working_dir}/glofas_flood_hazards/{rp}', file_url)

        # Download the file
        print(f'Downloading {file_url}...')
        with requests.get(download_file_url, stream=True) as r:
            with open(local_file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_file_path
#=====================================================================================================================================================
    def mosaic_glofas_flood_hazard_files(self):
        reference_raster = f'{self.working_dir}/scratch/subbasins.tif'

        """Get the bounding box of the reference raster."""
        with rasterio.open(reference_raster) as src:
            reference_bounds = src.bounds
    
        min_lat, min_lon, max_lat, max_lon = reference_bounds.bottom, reference_bounds.left, reference_bounds.top, reference_bounds.right
        
        lats = [min_lat, max_lat]
        lons = [min_lon, max_lon]
        lat_list = []
        lon_list = []
        for y in lats:
            if y >= -1:  # Northern Hemisphere
                lat_prefix = "N"
                if y < 9:
                    north_lat = 9  # For latitudes between 0 and 9
                else:
                    north_lat = (int(np.ceil(y)) // 10) * 10 + 9  # N19, N29, etc.
                values = list(range(north_lat, 8, -10))
                prefixed_values = [f"{lat_prefix}{abs(v)}" for v in values]
                lat_list = lat_list + prefixed_values
            else:
                lat_prefix = "S"
                if y >= -11:
                    north_lat = 1  # For latitudes between -1 and -11, use S1
                else:
                    north_lat = abs(int(y) // 10) * 10 + 1 - 10  # For latitudes -11 and below, use S11, S21, etc.
                values = list(range(north_lat, 0, 10)) 
                prefixed_values = [f"{lat_prefix}{abs(v)}" for v in values]
                lat_list = lat_list + prefixed_values
        
        for x in lons:
            # Determine the western/eastern longitude bound
            if x >= 10:  # Eastern Hemisphere
                lon_prefix = "E"
                west_lon = (int(x) // 10) * 10  # Minimum E tile (e.g., E10)
                values = list(range(west_lon, int(x)+10, 10))
                prefixed_values = [f"{lon_prefix}{abs(v)}" for v in values]
                lon_list = lon_list + prefixed_values
            elif 0 <= x < 10:  # Special case for longitudes between 0 and 10
                lon_prefix = "W"
                west_lon = 0  # Use W0 for longitudes between 0 and 10
                prefixed_values = ['W0']
                lon_list = lon_list + prefixed_values
            else:  # Western Hemisphere
                lon_prefix = "W"
                west_lon = abs((int(x) // 10 ) * 10)  # W10, W20, etc., for negative longitudes
                values = list(range(0, west_lon+10, 10))
                prefixed_values = [f"{lon_prefix}{abs(v)}" for v in values]
                lon_list = lon_list + prefixed_values
        
        lat_list = list(set(lat_list))
        lon_list = list(set(lon_list))

        
        rp_list = ['RP10', 'RP20', 'RP50', 'RP75', 'RP100', 'RP200', 'RP500']
        for rp in rp_list:
            files = []
            for y in lat_list:
                for x in lon_list:
                    file_name = f"{y}_{x}_{rp}_depth.tif"
                    this_file = self.get_glofas_flood_hazard(file_name, rp)
                    files.append(this_file[0])
            #print(files)
            full_flood_map = f'{self.working_dir}/scratch/subbasin_flood_maps/glofas_flood_map_{rp}.tif'
            # Check if the full flood map file already exists
            if not os.path.exists(full_flood_map):
                if files:  # Proceed with mosaicking if there are files to mosaic
                    wbt.mosaic(full_flood_map, files, method="nn")
                else:
                    print(f"No files found for return period {rp}. Skipping mosaic.")
            else:
                print(f"Flood map for return period {rp} already exists. Skipping.")
            
        
#==========================================================================================================================================
    def compute_runoff_glofas(self, temp_change, rf_change, scenario_name):
        wbt = WhiteboxTools()
        wbt.verbose = False
        os.makedirs(f'{self.working_dir}/scratch/subbasin_flood_maps', exist_ok=True)

        from deepstrmm.hydro_scenarios import DeepSTRMM
        ds = DeepSTRMM(
            working_dir = self.working_dir,
            study_area = f'common_data/{self.working_dir}.shp',
            cncoef = -1
            )
        if scenario_name == 'baseline':
            ds.soil_lc_baseline(temp_change, rf_change, scenario_name)
        else:
            ds.soil_lc_scenarios(temp_change, rf_change, scenario_name)

#===================================================================================================================================================

    def clip_to_subbasin(self, reference_raster_path, target_raster_path, subbasin_id):
        # Open the reference raster
        with rasterio.open(reference_raster_path) as ref_raster:
            ref_data = ref_raster.read(1)  # Reading the first band (you can adjust if needed)
            ref_data = np.where(ref_data == subbasin_id, 1, np.nan)
                
            # Create a mask for non-NaN values
            non_nan_mask = np.where(~np.isnan(ref_data), 1, 0).astype(np.uint8)
    
            # Generate polygons for the non-NaN areas
            shapes_gen = shapes(non_nan_mask, transform=ref_raster.transform)
            geometries = [Polygon(shape[0]["coordinates"][0]) for shape in shapes_gen if shape[1] == 1]
    
            # If there are no geometries, return early
            if not geometries:
                raise ValueError("No non-NaN values found in the reference raster.")
    
            # Clip the target raster using the non-NaN geometries
            with rasterio.open(target_raster_path) as target_raster:
                out_image, out_transform = mask(target_raster, geometries, crop=True)
    
                # Update metadata for the clipped image
                out_meta = target_raster.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })
    
                # Save the clipped raster
                output_path =  f'{self.working_dir}/scratch/subbasin_flood_maps/subbasin_' + str(subbasin_id)  + '.tif'
                with rasterio.open(output_path, "w", **out_meta) as dest:
                    dest.write(out_image)
    
    
#=======================================================================================================================================================
    def map_inundated_areas_glofas(self,  flood_threshold_dict, scenario_name, runoff_data_path, tchange, rchange):
        from deepstrmm.hydro_scenarios import DeepSTRMM
        ds = DeepSTRMM(
            working_dir = self.working_dir,
            study_area = f'common_data/{self.working_dir}.shp',
            cncoef = -1
            )
        
        with open(flood_threshold_dict, 'rb') as file:
            flood_thresholds = pickle.load(file)
            subbasins = [data['subbasin_id'] for data in flood_thresholds.values()]
            
        subbasin_data, subbasin_ids, facc_data, fil_data = self.compute_watershed_features_glofas()
       
        for num in subbasins:
            self.sub_rp = None
            print(f'processing subbasin  {num}')
            this_subbasin = np.where(subbasin_data == num, 1, np.nan)
            if np.nansum(this_subbasin) > 10:
            
                outlet = this_subbasin * facc_data
                #outlet_value = np.nanmin(outlet)
                orow, ocol = np.unravel_index(np.nanargmax(outlet), outlet.shape)
                #print(facc_data[orow, ocol])

                outlet_lat, outlet_lon = self.convert_to_latlng(orow, ocol, self.dem_filepath1)

            #self.lat, self.lon = self.convert_coords(outlet_lon, outlet_lat)
            self.lat =  outlet_lat
            self.lon = outlet_lon

            #predict streamflow and get values for specific return period as Q
           
            self.predicted_streamflow = ds.simulate_streamflow_latlng(f'{self.working_dir}/models/model_tcn360.keras', 
                                                                     self.lat, self.lon, scenario_name, runoff_data_path)
            
            
            peak_flow = np.nanmax(self.predicted_streamflow)
            print('peak flow is ' + str(peak_flow))
            

            if len(flood_thresholds) >= 7:
                sub_flow_info = flood_thresholds[num]
                rp_values = {
                    'RP10': sub_flow_info['RP10'],
                    'RP20': sub_flow_info['RP20'],
                    'RP50': sub_flow_info['RP50'],
                    'RP75': sub_flow_info['RP75'],
                    'RP100': sub_flow_info['RP100'],
                    'RP200': sub_flow_info['RP200'],
                    'RP500': sub_flow_info['RP500'],
                }
        
            # Sort the return periods by their discharge values (ascending)
            self.sorted_rps = sorted(rp_values.items(), key=lambda x: x[1])
        
            # Find between which two RPs the discharge falls
            for i in range(len(self.sorted_rps) - 1):
                rp_lower, discharge_lower = self.sorted_rps[i]
                rp_upper, discharge_upper = self.sorted_rps[i + 1]
                
                if not np.isnan(discharge_lower) and not np.isnan(discharge_upper) and not np.isnan(peak_flow):
                    if discharge_lower <= peak_flow <= discharge_upper:
                        self.sub_rp = rp_lower
                        break
        
            # If discharge is less than the minimum or greater than the maximum
            if not np.isnan(self.sorted_rps[0][1]) and not np.isnan(self.sorted_rps[-1][1]):
                if peak_flow < self.sorted_rps[0][1]:
                    self.sub_rp = self.sorted_rps[0][0]
                    #f"The discharge {discharge} is lower than the smallest return period threshold {sorted_rps[0][0]} ({sorted_rps[0][1]})"
                elif peak_flow > self.sorted_rps[-1][1]:
                    self.sub_rp = self.sorted_rps[-1][0]
                #f"The discharge {discharge} is higher than the largest return period threshold {sorted_rps[-1][0]} ({sorted_rps[-1][1]})"

            if self.sub_rp is not None:
                glofas_flood_reference = glob.glob(f'{self.working_dir}/scratch/subbasin_flood_maps/glofas_flood_map_{self.sub_rp}.tif')
   
                subbasins = f'{self.working_dir}/scratch/subbasins.tif'
                self.clip_to_subbasin(subbasins, glofas_flood_reference[0], num)
                    
        files = glob.glob(f'{self.working_dir}/scratch/subbasin_flood_maps/subbasin*.tif')
        full_flood_map = f'{self.working_dir}/scratch/subbasin_flood_maps/floodmap_{scenario_name}_t{tchange}_r{rchange}.tif'
        wbt.mosaic(full_flood_map, files, method="nn")
#==========================================================================================================================================================        
    def extract_annual_peaks(self, streamflow_data, total_years, start_year):
        # Reshape the data to have one year per row
        def is_leap_year(year):
            return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

        # Calculate the number of days per year considering leap years
        days_per_year = [365 + is_leap_year(year) for year in range(start_year, start_year + total_years)]

        # Initialize variables
        annual_peaks = []
        start_idx = 0

        # Extract annual peaks considering leap years
        for days in days_per_year:
            end_idx = start_idx + days
            annual_peak = np.max(streamflow_data[start_idx:end_idx])
            annual_peaks.append(annual_peak)
            start_idx = end_idx

        #return pd.DataFrame(np.array(annual_peaks), columns=['annual_peaks'])
        return annual_peaks

   
#===========================================================================================================================================
    # Step 2: Fit Gumbel Distribution using L-Moments
    def fit_gumbel_l_moments(self, annual_peaks):
        """Fit a Gumbel distribution to the annual maxima using L-moments."""
        # Compute the L-moments of the data
        
        lmoments = lmom.lmom_ratios(annual_peaks, nmom=3)
        
        # Estimate the Gumbel distribution parameters using L-moments
        gumbel_params = distr.gum.lmom_fit(lmoments)
        return gumbel_params
    
    # Step 3: Calculate Return Period Discharges
    def return_period_discharge(self, annual_peaks, return_period):
        """Calculate the discharge magnitude for a specific return period."""
        # Return period formula: F = 1 - 1/return_period
        gumbel_params = self.fit_gumbel_l_moments(annual_peaks)
        F = 1 - 1 / return_period
        # Gumbel inverse CDF (percent point function) to calculate discharge
        discharge = stats.gumbel_r.ppf(F, loc=gumbel_params['loc'], scale=gumbel_params['scale'])
        return discharge

    def count_exceedances(self, annual_peaks, threshold, last_n_years):
        # Count the number of years exceeding the threshold in the last n years
        return np.sum(annual_peaks[-last_n_years:] > threshold)       
  

    def is_leap_year(self, year):
        """Check if a year is a leap year."""
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    

    

            



            
        

            
            