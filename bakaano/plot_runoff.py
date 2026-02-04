"""Plotting utilities for routed runoff outputs.

Role: Visualize routed runoff maps and time series.
"""

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import xarray as xr
import rasterio
from rasterio.transform import rowcol
from scipy.spatial.distance import cdist
import geopandas as gpd
from bakaano.utils import Utils

class RoutedRunoff:
    def __init__(self, working_dir, study_area):
        """Role: Provide visualization utilities for routed runoff.

        Initialize helper for plotting routed runoff outputs.

        Args:
            working_dir (str): Working directory containing runoff outputs.
            study_area (str): Path to the basin/watershed shapefile.
        """
        self.working_dir = working_dir
        self.study_area = study_area
        self.uw = Utils(self.working_dir, self.study_area)
        self.out_path = f'{self.working_dir}/elevation/dem_clipped.tif'


    def map_routed_runoff(self, date, vmax=8):
        """Map routed runoff for a specific date.

        Args:
            date (str): Date string (YYYY-MM-DD) matching runoff output keys.
            vmax (float): Max value for color scaling (log1p).

        Returns:
            None. Displays a matplotlib plot.
        """
        # Function to map routed runoff for a specific date
        data = sorted(glob.glob(f'{self.working_dir}/runoff_output/*.pkl'))[0]
        if os.path.exists(data) is False:
            raise FileNotFoundError("Routed runoff output directory not found. Please run veget module first.")
        else:
            with open(data, 'rb') as f:
                wfa_list = pickle.load(f)

        entry = next((item for item in wfa_list if item['time'] == date), None)
        del wfa_list

        if entry is None:
            raise ValueError(f"No matrix found for date {date}")

        # Extract sparse matrix and convert to dense
        mat = entry['matrix'].toarray()

        dem_data = self.uw.clip(raster_path=self.out_path, out_path=None, save_output=False, crop_type=True)[0]
        dem_data = np.where(dem_data > 0, 1, np.nan)
        dem_data = np.where(dem_data < 32000, 1, np.nan)

        ro = dem_data * mat
        # Plot
        plt.figure(figsize=(7, 5))
        plt.imshow(np.log1p(ro), cmap='viridis', vmax=vmax)
        plt.colorbar(label='Value')
        plt.title(f"Routed runoff for {date}")
        plt.xlabel("X index")
        plt.ylabel("Y index")
        plt.show()

    def plot_routed_runoff_timeseries(
        self,
        start_date,
        end_date,
        station_id=None,
        lat=None,
        lon=None,
        grdc_netcdf=None,
        lookup_csv=None,
        id_col="id",
        lat_col="latitude",
        lon_col="longitude",
    ):
        """
        Plot routed runoff time series for one or more stations or coordinates.

        Provide either:
          - station_id with grdc_netcdf OR lookup_csv, or
          - lat/lon directly.

        Returns:
            None. Displays a matplotlib plot.
        """
        if station_id is None and (lat is None or lon is None):
            raise ValueError("Provide station_id or lat/lon.")

        def _snap_coordinates(la, lo, grid_path):
            """Snap a coordinate to the nearest river grid cell."""
            if not os.path.exists(grid_path):
                return la, lo
            coordinate_to_snap = (lo, la)
            with rasterio.open(grid_path) as src:
                grid = src.read(1)
                transform = src.transform

                river_coords = []
                for py in range(grid.shape[0]):
                    for px in range(grid.shape[1]):
                        if grid[py, px] == 1:
                            river_coords.append(transform * (px + 0.5, py + 0.5))

                if not river_coords:
                    return la, lo

                distances = cdist([coordinate_to_snap], river_coords)
                nearest_index = np.argmin(distances)
                snap_point = river_coords[nearest_index]
                return snap_point[1], snap_point[0]


        river_grid_path = f"{self.working_dir}/catchment/river_grid.tif"

        with rasterio.open(self.out_path) as src:
            ref_transform = src.transform
            ref_shape = (src.height, src.width)

        stations = []
        if station_id is not None:
            station_ids = station_id if isinstance(station_id, (list, tuple)) else [station_id]
            if lookup_csv:
                lookup = pd.read_csv(lookup_csv)
                if id_col not in lookup.columns or lat_col not in lookup.columns or lon_col not in lookup.columns:
                    raise ValueError("lookup_csv must include id, latitude, and longitude columns.")
                for sid in station_ids:
                    row = lookup.loc[lookup[id_col].astype(str) == str(sid)]
                    if row.empty:
                        raise ValueError(f"Station id not found in lookup CSV: {sid}")
                    la = float(row[lat_col].values[0])
                    lo = float(row[lon_col].values[0])
                    la, lo = _snap_coordinates(la, lo, river_grid_path)
                    stations.append((str(sid), la, lo))
            elif grdc_netcdf:
                ds = xr.open_dataset(grdc_netcdf)
                if "id" not in ds.dims:
                    raise ValueError("GRDC NetCDF missing 'id' dimension.")
                ds_id_vals = ds["id"].values
                ds_ids = set([str(s) for s in ds_id_vals.tolist()])
                id_dtype = ds_id_vals.dtype
                for sid in station_ids:
                    if str(sid) not in ds_ids:
                        raise ValueError(f"Station id not found in GRDC NetCDF: {sid}")
                    if np.issubdtype(id_dtype, np.number):
                        sid_sel = np.array(sid, dtype=id_dtype).item()
                    else:
                        sid_sel = str(sid)
                    sx = ds["geo_x"].sel(id=sid_sel).values
                    sy = ds["geo_y"].sel(id=sid_sel).values
                    la = float(np.nanmax(sy))
                    lo = float(np.nanmax(sx))
                    la, lo = _snap_coordinates(la, lo, river_grid_path)
                    stations.append((str(sid), la, lo))
            else:
                raise ValueError("Provide grdc_netcdf or lookup_csv when using station_id.")
        else:
            lats = lat if isinstance(lat, (list, tuple)) else [lat]
            lons = lon if isinstance(lon, (list, tuple)) else [lon]
            if len(lats) != len(lons):
                raise ValueError("lat and lon must have the same length.")
            for i, (la, lo) in enumerate(zip(lats, lons)):
                la = float(la)
                lo = float(lo)
                la, lo = _snap_coordinates(la, lo, river_grid_path)
                stations.append((f"lat{la}_lon{lo}", la, lo))

        data_files = sorted(glob.glob(f'{self.working_dir}/runoff_output/*.pkl'))
        if not data_files:
            raise FileNotFoundError("Routed runoff output not found. Run veget module first.")

        wfa_list = []
        for fp in data_files:
            with open(fp, "rb") as f:
                wfa_list += pickle.load(f)

        wfa_dict = {item["time"]: item["matrix"] for item in wfa_list}
        date_index = pd.date_range(start=start_date, end=end_date, freq="D")

        transform = ref_transform

        series = {}
        for label, la, lo in stations:
            r, c = rowcol(transform, lo, la)
            values = []
            for dt in date_index:
                key = dt.strftime("%Y-%m-%d")
                mat = wfa_dict.get(key)
                if mat is None:
                    values.append(np.nan)
                else:
                    # coo_array is not subscriptable; index via CSR
                    if hasattr(mat, "tocoo"):
                        mat = mat.tocsr()
                    # Guard against row/col outside matrix bounds
                    if r < 0 or c < 0 or r >= mat.shape[0] or c >= mat.shape[1]:
                        values.append(np.nan)
                    else:
                        values.append(float(mat[r, c]))
            series[label] = values

        df = pd.DataFrame(series, index=date_index)
        df.plot(figsize=(9, 4))
        plt.title("Routed runoff time series")
        plt.xlabel("Date")
        plt.ylabel("Routed runoff")
        plt.legend()
        plt.tight_layout()
        plt.show()
        #return df

    def interactive_plot_routed_runoff_timeseries(
        self,
        start_date,
        end_date,
        grdc_netcdf=None,
        lookup_csv=None,
        id_col="id",
        lat_col="latitude",
        lon_col="longitude",
    ):
        """
        Interactive wrapper: lists available station_ids and prompts user to select one.

        Args:
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).
            grdc_netcdf (str, optional): GRDC NetCDF path with station coordinates.
            lookup_csv (str, optional): Station lookup CSV path.
            id_col (str): Station id column name in lookup CSV.
            lat_col (str): Latitude column name in lookup CSV.
            lon_col (str): Longitude column name in lookup CSV.

        Returns:
            None. Displays a matplotlib plot.
        """
        station_ids = []
        if lookup_csv:
            lookup = pd.read_csv(lookup_csv)
            if id_col not in lookup.columns or lat_col not in lookup.columns or lon_col not in lookup.columns:
                raise ValueError("lookup_csv must include id, latitude, and longitude columns.")
            stations_df = pd.DataFrame({
                "id": lookup[id_col].astype(str).values,
                "geo_x": lookup[lon_col].astype(float).values,
                "geo_y": lookup[lat_col].astype(float).values,
            })
        elif grdc_netcdf:
            ds = xr.open_dataset(grdc_netcdf)
            if "id" not in ds.dims:
                raise ValueError("GRDC NetCDF missing 'id' dimension.")
            stations_df = pd.DataFrame({
                "id": [str(s) for s in ds["id"].values.tolist()],
                "geo_x": ds["geo_x"].values,
                "geo_y": ds["geo_y"].values,
            })
        else:
            raise ValueError("Provide grdc_netcdf or lookup_csv.")

        if stations_df.empty:
            raise ValueError("No station ids found.")

        # Filter stations to those within the study area shapefile
        region_shape = gpd.read_file(self.study_area)
        stations_gdf = gpd.GeoDataFrame(
            stations_df,
            geometry=gpd.points_from_xy(stations_df["geo_x"], stations_df["geo_y"]),
            crs="EPSG:4326",
        )
        stations_in_region = gpd.sjoin(
            stations_gdf,
            region_shape,
            how="inner",
            predicate="intersects",
        )
        station_ids = stations_in_region["id"].astype(str).unique().tolist()
        if not station_ids:
            raise ValueError("No station ids found within the study area.")

        print("Available station_ids:")
        print(", ".join(station_ids))
        user_id = input("Enter station_id: ").strip()
        if user_id not in station_ids:
            raise ValueError(f"Station id not found: {user_id}")

        return self.plot_routed_runoff_timeseries(
            start_date=start_date,
            end_date=end_date,
            station_id=user_id,
            grdc_netcdf=grdc_netcdf,
            lookup_csv=lookup_csv,
            id_col=id_col,
            lat_col=lat_col,
            lon_col=lon_col,
        )
