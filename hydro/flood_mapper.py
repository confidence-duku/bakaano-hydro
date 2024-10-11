import pysheds.grid
import numpy as np
import pandas as pd
import rasterio
import rasterio
from rasterio.transform import Affine
from rasterio.transform import xy
from pyproj import Proj, transform
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib
from rasterio.transform import rowcol
from shapely.geometry import shape, LineString
from pyproj import Proj, transform
from scipy.optimize import curve_fit
import scipy.stats as stats
import richdem as rd
from whitebox import WhiteboxTools
import os
import pickle

class FloodMapper:
    def __init__(self, dem):
        
        # Define the original and target projections
        self.original_proj = Proj('epsg:4326')  # WGS84
        self.target_proj = Proj('epsg:3857')# Web Mercator
        #self.corrected_dem_name  = './niger/elevation/corrected_dem.tif'
        self.water_levels = [0.1, 0.23, 0.4, 0.75, 1, 1.36, 1.55, 1.79, 2, 2.29, 2.67, 2.88, 3, 
                             3.3, 3.66, 3.8, 4, 4.33, 4.57]
        self.mannings_n = 0.05
        self.ddir2 = os.getcwd() + '/niger/'
        self.ddir = '/lustre/backup/WUR/ESG/duku002/NBAT/hydro/niger/'
        self.dem_filepath1 = self.ddir2 + dem
        self.dem_filepath = self.ddir2 + 'elevation/dem_niger_prj.tif'

    #non linear least square fitting to develop rating curve
    # formula is Q = c(h+a)to the power n
#     def rating_curve_eq(self, h, c, a, n):
#         return c * ((h + a) ** n)

    # Function to predict h given Q
    def get_stage_height(self, Q, a,b,c,d):
        sh = (a * (Q**3)) + (b * (Q**2)) + (c*Q) + d
        return sh

    # Flip the DEM data along the vertical axis (latitude)
    def correct_inverted_dem(self, dem_data, profile):
        dem_flipped = np.flipud(dem_data)
        
        # Update the profile to reflect the new bounding box
        # Adjust the transform to reflect the flipping of latitude
        new_transform = rasterio.Affine(
            profile['transform'][0], profile['transform'][1], profile['transform'][2],
            profile['transform'][3], -profile['transform'][4], profile['transform'][5] + profile['transform'][4] * dem_flipped.shape[0]
        )        
        profile.update({
            'transform': new_transform
        })    
        return dem_flipped, profile

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
                        resampling=Resampling.nearest)
    
    # Function to transform a (lon, lat) tuple
    def transform_coordinates(self, lon, lat):
        x, y = transform(self.original_proj, self.target_proj, lon, lat)
        return x, y

    def extract_grid_value(self, lon, lat, grid_data, dem_transform):
        row, col = rowcol(dem_transform, lon, lat)
        value = grid_data[row, col]
        return value
    def convert_to_latlng(self, row, col):
        with rasterio.open(self.dem_filepath) as dataset:
            # Get the affine transformation of the dataset
            transform = dataset.transform

            # Convert row and column to lat and lon
            lon, lat = xy(transform, row, col)
            return lat, lon

    def convert_coords(self, x, y):
        # Define the projections
        proj_3857 = Proj(init='epsg:3857')
        proj_4326 = Proj(init='epsg:4326')

        # Perform the transformation
        lon, lat = transform(proj_3857, proj_4326, x, y)
        return lat, lon

    def compute_rating_curve_pysheds(self):
        with rasterio.open(self.dem_filepath) as src:
            dem_data = src.read(1)
            dem_transform = src.transform
            bbox = src.bounds
            profile = src.profile
        #print(bbox)
        print('Correcting flipped dem')
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            grid = pysheds.grid.Grid.from_raster(self.dem_filepath, nodata=32767)
            dem = grid.read_raster(self.dem_filepath, nodata=32767)        
        else:
            corrected_dem, corrected_profile = self.correct_inverted_dem(dem_data, profile)
        
            with rasterio.open(self.corrected_dem_name, 'w', **corrected_profile) as dst:
                dst.write(corrected_dem, 1)
        
            grid = pysheds.grid.Grid.from_raster(self.corrected_dem_name, nodata=32767)
            dem = grid.read_raster(self.corrected_dem_name, nodata=32767)
            
        
        #grid = pysheds.grid.Grid.from_raster(self.dem_filepath, nodata=-9999)
        #dem = grid.read_raster(self.dem_filepath, nodata=-9999)
        
        print('Filling depressions and resolving flats')
        flooded_dem = grid.fill_depressions(dem)
        inflated_dem = grid.resolve_flats(flooded_dem)
        
        print('Computing flow direction and flow accumulation')
        fdir = grid.flowdir(inflated_dem)
        acc = grid.accumulation(fdir=fdir)
        
        print('Extracting river network')
        facc_threshold = np.nanmax(acc) * 0.01
        self.river_grid = grid.extract_river_network(fdir, acc > facc_threshold)
        
        print('Computing HAND raster')
        hand = grid.compute_hand(fdir, inflated_dem, acc > facc_threshold)
        initial_guess = [1, 1, 1]  # Initial guesses for c, a, n
        
        print('Computing slope')
        slope = rd.TerrainAttribute(rd.rdarray(dem_data, no_data=-32768), attrib='slope_riserun')

        #rating curve dictionary
        rating_curve_dict = {}
        
        for feature in self.river_grid['features']:
            id_num = feature['id']
            print(f'Developing rating curve for reach {id_num}')
            discharge_list = []
            for level in self.water_levels:
                print(f'    Rating curve at stage height {level}m')
                coord = feature['geometry']['coordinates']
                #print(coord)
                first_coord = coord[0]
                last_coord = coord[-1]
            
                first_pt = self.extract_grid_value(first_coord[0], first_coord[1], acc, dem_transform)
                last_pt = self.extract_grid_value(last_coord[0], last_coord[1], acc, dem_transform)
            
                if first_pt > last_pt:
                    outlet_coords = first_coord
                else:
                    outlet_coords = last_coord
                    
                if not (bbox.left <= outlet_coords[0] <= bbox.right and bbox.bottom <= outlet_coords[1] <= bbox.top):
                    print(f"     Skipping iteration as it is outside the bounding box.")
                    status = -1
                    break
                    
                catch = grid.catchment(x=outlet_coords[0], y=outlet_coords[1], fdir=fdir, xytype='coordinate') 
                catch = np.where(catch>0, 1, np.nan)
                catch_hand = catch * hand #select hand cells for a specific catchment
            
                #select hand grid cells with values lower than a given water level
                inundation_area = np.where(catch_hand<level, 1, np.nan)
                inundation_depth = level - (catch_hand / inundation_area)
                surface_area = np.nansum(inundation_area) * 30 * 30
        
                volume  = np.nanmean(inundation_depth) * surface_area
                slope_roi = slope * inundation_area
                wetted_bed_area = np.sqrt(1+(slope_roi/100) * (slope_roi/100))
                bed_area = np.nanmean(wetted_bed_area) * surface_area
        
                length = self.calculate_length(feature)

                z1 = self.extract_grid_value(first_coord[0], first_coord[1], fil, dem_transform)
                z2 = self.extract_grid_value(last_coord[0], last_coord[1], fil, dem_transform)
                bed_slope = abs(z2-z1)/length
                A = volume/length
                P = bed_area / length
                R = A / P
                
                discharge = (1/self.mannings_n) * A * (R**(2/3)) * (bed_slope**0.5) * 35.3  #output unit is m3/s
                discharge_list.append(discharge)
                
            if status != -1:    
                h = np.array(self.water_levels)
                Q = np.array(discharge_list)
                popt, pcov = curve_fit(self.rating_curve_eq, h, Q, p0=initial_guess)

                # Extract the fitted parameters
                c, a, n = popt
                rating_curve_dict[feature['id']] = {
                    'c_param':c,
                    'a_param':a,
                    'n_param':n,
                    'outlet_lat':outlet_coords[1],
                    'outlet_lon':outlet_coords[0]
                }            
        return rating_curve_dict
    
    def compute_rating_curve_wbt(self):
        wbt = WhiteboxTools()
        wbt.verbose = False
        
        print('Projecting DEM')
        if os.path.exists(self.dem_filepath):
            print('    File already exists')
        else:
            self.reproject_raster()
        
        print('Filling depressions and resolving flats')
        fil = self.ddir + 'scratch/fil.tif'
        if os.path.exists(fil):
            print('    File already exists')
        else:
            wbt.fill_depressions(self.dem_filepath, fil)
        
        with rasterio.open(fil) as fl:
            fil_data = fl.read(1)
            fil_data = np.where(fil_data == fl.nodata, np.nan, fil_data)
            transform = fl.transform

            # Extract the resolution
            resolution = transform.a 
        
        print('Computing flow direction')        
        fdr = self.ddir + 'scratch/fdr.tif'
        if os.path.exists(fdr):
            print('    File already exists')
        else:
            wbt.d8_pointer(fil, fdr)
        
        print('Computing flow accumulation')
        facc = self.ddir + 'scratch/facc.tif'
        if os.path.exists(facc):
            print('    File already exists')
        else:
            wbt.d8_flow_accumulation(fil, facc)        
        
        #get threshold for stream delineation
        with rasterio.open(facc) as src:
            facc_data = src.read(1)
            facc_data = np.where(facc_data == src.nodata, np.nan, facc_data)
            facc_threshold = np.nanmax(facc_data) * 0.01
        
        print('Extracting stream network')
        streams = self.ddir + 'scratch/str.tif'
        if os.path.exists(streams):
            print('    File already exists')
        else:
            wbt.extract_streams(facc, streams, facc_threshold)
        
        print('Computing stream link identifier')        
        strlnk = self.ddir + 'scratch/strlnk.tif'
        if os.path.exists(strlnk):
            print('    File already exists')
        else:
            wbt.stream_link_identifier(fdr, streams, strlnk)
            
        with rasterio.open(strlnk) as slk:
            strlnk_data = slk.read(1)
            self.strlnk_data = np.where(strlnk_data == slk.nodata, np.nan, strlnk_data)
        
        print('Computing HAND raster')
        hand = self.ddir + 'scratch/hand.tif'
        if os.path.exists(hand):
            print('    File already exists')
        else:
            wbt.elevation_above_stream(fil, streams, hand) 
        
        with rasterio.open(hand) as do:
            self.hand_data = do.read(1)
            hand_data = np.where(self.hand_data == do.nodata, np.nan, self.hand_data)
        
        print('Computing length of stream network reaches')
        strlnk_length = self.ddir + 'scratch/strlnk_length.tif'
        if os.path.exists(strlnk_length):
            print('    File already exists')
        else:
            wbt.stream_link_length(fdr, strlnk,strlnk_length)
        
        with rasterio.open(strlnk_length) as sl:
            strlnk_length_data = sl.read(1)
            strlnk_length_data = np.where(strlnk_length_data == sl.nodata, np.nan, strlnk_length_data)
        
        print('Computing subbasin')
        subbasins = self.ddir + 'scratch/subbasins.tif'
        if os.path.exists(subbasins):
            print('    File already exists')
        else:
            wbt.subbasins(fdr, streams, subbasins)
        
        with rasterio.open(subbasins) as sb:
            self.subbasin_data = sb.read(1)
            self.subbasin_ids = list(np.unique(self.subbasin_data))
            nodata_value = sb.nodata
            self.subbasin_ids.remove(nodata_value)
            
        with rasterio.open(self.dem_filepath) as dm:
            dem_data = dm.read(1)

        initial_guess = [1, 1, 1]  # Initial guesses for c, a, n
        
        print('Computing slope')
        slope = rd.TerrainAttribute(rd.rdarray(dem_data, no_data=-32768), attrib='slope_riserun')

        #rating curve dictionary
        self.rating_curve_dict = {}
        
        for num in self.subbasin_ids:
            print(f'Developing rating curve for reach {num}')
            this_subbasin = np.where(self.subbasin_data == num, 1, np.nan)
            this_str = np.where(self.strlnk_data == num, 1, np.nan)
            if np.nansum(this_subbasin) > 10:
            
                outlet = this_subbasin * facc_data
                #outlet_value = np.nanmin(outlet)
                orow, ocol = np.unravel_index(np.nanargmax(outlet), outlet.shape)
                #print(facc_data[orow, ocol])

                outlet_lat, outlet_lon = self.convert_to_latlng(orow, ocol)
                #print(outlet_lat, outlet_lon)

                urow, ucol = np.unravel_index(np.nanargmin(outlet), outlet.shape)
                #print(urow, ucol)
                z1 = fil_data[orow, ocol]
                z2 = fil_data[urow, ucol]
                #print(z2)

                length = strlnk_length_data[orow, ocol]
                #print(length)

                catch_hand = this_subbasin * self.hand_data
                str_hand = this_str * self.hand_data

                discharge_list = []
                for level in self.water_levels:
                    #print(f'    Rating curve at stage height {level}m')

                    #select hand grid cells with values lower than a given water level
                    inundation_area = np.where(catch_hand<level, 1, np.nan)
                    inundation_depth = level - (catch_hand * inundation_area)
                    inundatation_depth = np.where(inundation_depth<0, 0, inundation_depth)
                    surface_area = np.nansum(inundation_area) * resolution * resolution
                    #print(surface_area)

                    volume  = np.nanmean(inundation_depth) * surface_area
                    
                    slope_roi = np.array(slope) * inundation_area
                    #print(np.nanmean(slope_roi))

                    wetted_bed_area = np.sqrt(1+(slope_roi * slope_roi))
                    bed_area = np.nanmean(wetted_bed_area) * surface_area

                    bed_slope = abs(z2-z1)/length
                    A = volume/length
                    P = bed_area / length
                    R = A / P

                    #print(A, R, bed_slope)
                    discharge = (1/self.mannings_n) * A * (R**(2/3)) * (bed_slope**0.5)  #output unit is m3/s
                    discharge_list.append(discharge)
                    
                h = np.array(self.water_levels)
                Q = np.array(discharge_list)
                # Check for NaNs and infinities
                try:
                    #popt, pcov = curve_fit(self.rating_curve_eq, h, Q, p0=initial_guess)
                    a, b, c, d = np.polyfit(Q, h, 3)
                    
                except RuntimeError as e:
                    # If optimal parameters are not found, print a message and skip this iteration
                    print(f"Skipping iteration due to: {e}")
                    continue

                # Extract the fitted parameters
                #c, a, n = popt
                self.rating_curve_dict[num] = {
                    'subbasin_id': num,
                    'a_param':a,
                    'b_param':b,
                    'c_param':c,
                    'd_param':d,
                    'outlet_lat':outlet_lat,
                    'outlet_lon':outlet_lon,
                    'stage_height_list': self.water_levels,
                    'discharge_list': discharge_list
                }            
            else:
                continue
                             
        return self.rating_curve_dict

    def map_inundated_areas(self, return_period):
#         with open('./rc_dict.pkl', 'rb') as file:
#             self.rating_curve_dict = pickle.load(file)
            
        from deepstrmm.hydro import DeepSTRMM
        ds = DeepSTRMM(
            project_name = 'niger',
            study_area = 'common_data/niger.shp',
            start_date = "1981-01-01",
            end_date = "2016-12-31",
            cncoef = -1
            )
        
        self.catch_flood = []
        rating_curve_keys = list(self.rating_curve_dict.keys())
        self.flood_frequency_list = []
        for sid in rating_curve_keys:  
            sbbasin = self.rating_curve_dict[sid]
            print(f'Mapping inundated areas for subbasin {sid}')
            a = sbbasin['a_param']
            b = sbbasin['b_param']
            c = sbbasin['c_param']
            d = sbbasin['d_param']
            lat = sbbasin['outlet_lat']
            lon = sbbasin['outlet_lon']
            
            lat, lon = self.convert_coords(lon, lat)

            #predict streamflow and get values for specific return period as Q
            predicted_streamflow = ds.simulate_streamflow_latlng('niger/models/niger_model_tcn360.h5', lat, lon)
            
            if len(predicted_streamflow)>0:
                if np.max(predicted_streamflow)>0:
                    mean_annual_flow = predicted_streamflow.sum(axis=0)/35

                    total_years = 35
                    #flood analysis
                    annual_peaks = self.extract_annual_peaks(predicted_streamflow[360:], total_years, 1982)
                    flood_threshold = self.flood_frequency_analysis(annual_peaks, return_period)
                    #flood_frequency = self.count_exceedances(annual_peaks, flood_threshold, 35)
            else:
                continue
                
            h = self.get_stage_height(flood_threshold, a,b,c,d)
            print(f'flood threshold is {flood_threshold}')
            print(f'Stage height is {h}')
            
            this_subbasin = np.where(self.subbasin_data == sid, 1, np.nan)
            catch_hand = this_subbasin * self.hand_data
            inundation_area = np.where(catch_hand<h, 1, np.nan)
            inundation_depth = h - (catch_hand / inundation_area)
            inundatation_depth = np.nan_to_num(inundation_depth, 0)
            self.catch_flood.append(inundation_depth)
        self.full_inundated_areas = np.sum(np.array(self.catch_flood), axis=0)
        #self.flood_frequency_list.append(flood_frequency)
              
        
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

        return pd.DataFrame(np.array(annual_peaks), columns=['annual_peaks'])

    def flood_frequency_analysis(self, annual_peaks, return_period):
        # Fit the data to a Gumbel distribution
        annual_peaks_sorted = annual_peaks.sort_values(by='annual_peaks')
        n = annual_peaks_sorted.shape[0]
        annual_peaks_sorted.insert(0, 'rank', range(1, 1+n))
        annual_peaks_sorted["exceedance_prob"] = (n - annual_peaks_sorted["rank"] + 1) / (n + 1)
        annual_peaks_sorted["return_period"] = 1 / annual_peaks_sorted["exceedance_prob"]
        x = annual_peaks_sorted['return_period'].values
        y = annual_peaks_sorted['annual_peaks'].values
        f = np.polyfit(np.log10(x), y, 2, w=np.sqrt(y))
        flow_rate = f[0]*(np.log10(return_period)*np.log10(return_period)) + f[1]*np.log10(return_period) + f[2]                       
        return flow_rate

    def count_exceedances(self, annual_peaks, threshold, last_n_years):
        # Count the number of years exceeding the threshold in the last n years
        return np.sum(annual_peaks[-last_n_years:] > threshold)       
  

    def is_leap_year(self, year):
        """Check if a year is a leap year."""
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    

    

            



            
        

            
            