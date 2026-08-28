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
from bakaano.core.utils import Utils


def _open_dataset_with_fallback(nc_path):
    """Open NetCDF with a backend fallback for Colab compatibility."""
    open_errors = []
    for engine in (None, "h5netcdf"):
        try:
            if engine is None:
                return xr.open_dataset(nc_path)
            return xr.open_dataset(nc_path, engine=engine)
        except Exception as e:
            name = "netcdf4(default)" if engine is None else engine
            open_errors.append(f"{name}: {str(e)}")

    raise OSError(
        "Unable to open NetCDF with available backends.\n" + "\n".join(open_errors)
    )


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
        self.uw.require_study_area_gdf()
        self.out_path = f'{self.working_dir}/elevation/dem_clipped.tif'

    def _require_runoff_output_files(self):
        """Return routed-runoff pickle files or raise a clear setup error."""
        runoff_dir = f'{self.working_dir}/runoff_output'
        self.uw.require_existing_path(runoff_dir, "Routed runoff output", path_type="dir")
        data_files = sorted(glob.glob(f'{runoff_dir}/*.pkl'))
        if not data_files:
            raise FileNotFoundError(
                "No routed runoff pickle files were found in "
                f"{runoff_dir}. Run the runoff-routing workflow first."
            )
        return data_files

    def _load_runoff_entries(self):
        """Load routed-runoff entries with context-rich error reporting."""
        wfa_list = []
        for fp in self._require_runoff_output_files():
            try:
                with open(fp, "rb") as f:
                    loaded = pickle.load(f)
            except Exception as exc:
                raise ValueError(
                    f"Failed to read routed runoff pickle file: {fp}. "
                    "The file may be corrupted or incompatible."
                ) from exc

            if not isinstance(loaded, list):
                raise ValueError(
                    f"Unexpected runoff output format in {fp}. Expected a list of daily entries."
                )
            wfa_list.extend(loaded)

        if not wfa_list:
            raise ValueError(
                "Routed runoff output files were found, but they did not contain any daily runoff entries."
            )
        return wfa_list

    def _require_dem_reference(self):
        """Ensure the clipped DEM needed for plotting exists."""
        return self.uw.require_dem_reference()

    def _station_dataframe_from_source(
        self,
        grdc_netcdf=None,
        lookup_csv=None,
        id_col="id",
        lat_col="latitude",
        lon_col="longitude",
        name_col=None,
    ):
        """Build a station coordinate table from GRDC NetCDF or a lookup CSV."""
        if lookup_csv:
            self.uw.require_existing_path(lookup_csv, "Station lookup CSV", path_type="file")
            lookup = pd.read_csv(lookup_csv)
            required = [id_col, lat_col, lon_col]
            self.uw.require_columns(lookup, required, "lookup_csv")
            stations_df = pd.DataFrame({
                "id": lookup[id_col].astype(str).values,
                "geo_x": lookup[lon_col].astype(float).values,
                "geo_y": lookup[lat_col].astype(float).values,
            })
            if name_col and name_col in lookup.columns:
                stations_df["station_name"] = lookup[name_col].astype(str).values
            else:
                stations_df["station_name"] = stations_df["id"]
            return stations_df

        if grdc_netcdf:
            self.uw.require_existing_path(grdc_netcdf, "GRDC NetCDF", path_type="file")
            ds = _open_dataset_with_fallback(grdc_netcdf)
            if "id" not in ds.dims:
                raise ValueError("GRDC NetCDF missing 'id' dimension.")
            missing_vars = [name for name in ("geo_x", "geo_y") if name not in ds.variables]
            if missing_vars:
                raise ValueError(
                    "GRDC NetCDF is missing required coordinate variables: "
                    + ", ".join(missing_vars)
                )
            stations_df = pd.DataFrame({
                "id": [str(s) for s in ds["id"].values.tolist()],
                "geo_x": ds["geo_x"].values,
                "geo_y": ds["geo_y"].values,
            })
            if "station_name" in ds.variables:
                stations_df["station_name"] = [
                    str(s) for s in np.asarray(ds["station_name"].values).tolist()
                ]
            else:
                stations_df["station_name"] = stations_df["id"]
            return stations_df

        raise ValueError("Provide grdc_netcdf or lookup_csv.")

    def _stations_in_study_area(self, stations_df):
        """Return station rows that intersect the study-area geometry."""
        if stations_df.empty:
            raise ValueError("No station ids found.")

        region_shape = self.uw.require_study_area_gdf()
        if region_shape.crs is None:
            region_shape = region_shape.set_crs("EPSG:4326")
        elif str(region_shape.crs) != "EPSG:4326":
            region_shape = region_shape.to_crs("EPSG:4326")

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
        if stations_in_region.empty:
            raise ValueError("No station ids found within the study area.")
        return stations_in_region.drop_duplicates(subset=["id"])

    def _station_observed_series(
        self,
        station_id,
        grdc_netcdf=None,
        csv_dir=None,
        date_col="date",
        discharge_col="discharge",
        file_pattern="{id}.csv",
    ):
        """Load an observed discharge series for one station."""
        if grdc_netcdf:
            ds = _open_dataset_with_fallback(grdc_netcdf)
            if "runoff_mean" not in ds.variables:
                raise ValueError("GRDC NetCDF is missing observed data variable 'runoff_mean'.")
            ds_id_vals = ds["id"].values
            ds_ids = {str(s) for s in ds_id_vals.tolist()}
            if str(station_id) not in ds_ids:
                raise ValueError(f"Station id not found in GRDC NetCDF: {station_id}")
            if np.issubdtype(ds_id_vals.dtype, np.number):
                sid_sel = np.array(station_id, dtype=ds_id_vals.dtype).item()
            else:
                sid_sel = str(station_id)
            series = ds["runoff_mean"].sel(id=sid_sel).to_series()
            series.index = pd.to_datetime(series.index)
            series = pd.to_numeric(series, errors="coerce").dropna()
            series.name = "Observed discharge"
            return series

        if csv_dir:
            pattern = file_pattern.format(id=station_id)
            matches = sorted(glob.glob(os.path.join(csv_dir, pattern)))
            if not matches:
                raise FileNotFoundError(f"No observed CSV found for station id {station_id}: {pattern}")
            df = pd.read_csv(matches[0])
            missing = [col for col in (date_col, discharge_col) if col not in df.columns]
            if missing:
                raise ValueError(
                    f"Station CSV for id={station_id} is missing columns: "
                    + ", ".join(missing)
                )
            df[date_col] = pd.to_datetime(df[date_col])
            series = pd.to_numeric(df[discharge_col], errors="coerce")
            series.index = df[date_col]
            series = series.sort_index().dropna()
            series.name = "Observed discharge"
            return series

        raise ValueError("Observed data requires either grdc_netcdf or csv_dir.")

    def interactive_station_map(
        self,
        grdc_netcdf=None,
        csv_dir=None,
        lookup_csv=None,
        id_col="id",
        lat_col="latitude",
        lon_col="longitude",
        name_col=None,
        date_col="date",
        discharge_col="discharge",
        file_pattern="{id}.csv",
    ):
        """Open an interactive map of stations, study area, and river network.

        Clicking a station marker updates a hydrograph panel below the map.
        Use ``grdc_netcdf`` for GRDC observations, or ``lookup_csv`` plus
        ``csv_dir`` for per-station CSV observations.
        """
        from IPython.display import display
        from ipyleaflet import (
            GeoJSON,
            LayersControl,
            Map,
            Marker,
            Popup,
            basemaps,
        )
        from ipywidgets import HTML, Output, VBox

        stations_df = self._station_dataframe_from_source(
            grdc_netcdf=grdc_netcdf,
            lookup_csv=lookup_csv,
            id_col=id_col,
            lat_col=lat_col,
            lon_col=lon_col,
            name_col=name_col,
        )
        stations_in_region = self._stations_in_study_area(stations_df)

        center_lat = float(stations_in_region["geo_y"].mean())
        center_lon = float(stations_in_region["geo_x"].mean())
        m = Map(
            center=(center_lat, center_lon),
            zoom=7,
            basemap=basemaps.Esri.WorldImagery,
            scroll_wheel_zoom=True,
        )

        study_area_gdf = self.uw.require_study_area_gdf()
        if study_area_gdf.crs is None:
            study_area_gdf = study_area_gdf.set_crs("EPSG:4326")
        elif str(study_area_gdf.crs) != "EPSG:4326":
            study_area_gdf = study_area_gdf.to_crs("EPSG:4326")
        m.add_layer(GeoJSON(data=study_area_gdf.__geo_interface__, name="Study area"))

        west = float(study_area_gdf.total_bounds[0])
        south = float(study_area_gdf.total_bounds[1])
        east = float(study_area_gdf.total_bounds[2])
        north = float(study_area_gdf.total_bounds[3])
        m.fit_bounds([[south, west], [north, east]])

        hydrograph_output = Output(
            layout={
                "width": "100%",
                "min_height": "420px",
                "border": "1px solid #ddd",
                "padding": "8px",
            }
        )

        def _station_header(row):
            station_id = str(row["id"])
            station_name = str(row.get("station_name", station_id))
            return (
                f"<b>{station_name}</b><br>"
                f"Station ID: {station_id}<br>"
                f"Lon/lat: {float(row['geo_x']):.4f}, {float(row['geo_y']):.4f}"
            )

        def _plot_station_below_map(row):
            station_id = str(row["id"])
            station_name = str(row.get("station_name", station_id))
            hydrograph_output.clear_output(wait=True)
            with hydrograph_output:
                display(HTML(value=_station_header(row)))
                try:
                    series = self._station_observed_series(
                        station_id,
                        grdc_netcdf=grdc_netcdf,
                        csv_dir=csv_dir,
                        date_col=date_col,
                        discharge_col=discharge_col,
                        file_pattern=file_pattern,
                    )
                    if series.empty:
                        print(f"No observed values found for station {station_id}.")
                        return

                    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=120)
                    series.plot(ax=ax, color="#1f77b4", linewidth=1.2)
                    ax.set_title(f"{station_name} ({station_id})")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Observed discharge")
                    ax.grid(True, alpha=0.25)
                    fig.tight_layout()
                    display(fig)
                    plt.close(fig)
                except Exception as exc:
                    print(f"Could not plot observed data for station {station_id}: {exc}")

        def _popup_for_station(row):
            body = (
                f"{_station_header(row)}<br><br>"
                "Hydrograph displayed below the map."
            )
            return HTML(value=body)

        def _make_click_handler(row, location):
            def _on_click(**kwargs):
                _plot_station_below_map(row)
                popup = Popup(
                    location=location,
                    child=_popup_for_station(row),
                    close_button=True,
                    auto_close=True,
                    close_on_escape_key=True,
                )
                m.add_layer(popup)
            return _on_click

        for _, row in stations_in_region.iterrows():
            location = (float(row["geo_y"]), float(row["geo_x"]))
            marker = Marker(
                location=location,
                title=str(row["id"]),
                draggable=False,
            )
            marker.on_click(_make_click_handler(row, location))
            m.add_layer(marker)

        m.add_control(LayersControl(position="topright"))
        hydrograph_output.append_stdout("Click a station marker to display its observed hydrograph here.")
        return VBox([m, hydrograph_output])


    def map_routed_runoff(self, date, vmax=8):
        """Map routed runoff for a specific date.

        Args:
            date (str): Date string (YYYY-MM-DD) matching runoff output keys.
            vmax (float): Max value for color scaling (log1p).

        Returns:
            None. Displays a matplotlib plot.
        """
        # Function to map routed runoff for a specific date
        self.uw.validate_date_window(date, date, "date", "date")
        self._require_dem_reference()
        data = self._require_runoff_output_files()[0]
        try:
            with open(data, 'rb') as f:
                wfa_list = pickle.load(f)
        except Exception as exc:
            raise ValueError(
                f"Failed to read routed runoff pickle file: {data}. "
                "The file may be corrupted or incomplete."
            ) from exc

        if not isinstance(wfa_list, list):
            raise ValueError(
                f"Unexpected routed runoff format in {data}. Expected a list of daily entries."
            )

        entry = next((item for item in wfa_list if item['time'] == date), None)
        del wfa_list

        if entry is None:
            raise ValueError(
                f"No routed runoff matrix was found for {date}. "
                "Verify that the requested date falls within the routed-runoff output period."
            )
        if "matrix" not in entry:
            raise ValueError(
                f"Runoff entry for {date} is missing the 'matrix' field."
            )

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
        self.uw.validate_date_window(start_date, end_date, "start_date", "end_date")
        self._require_dem_reference()
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
                self.uw.require_existing_path(lookup_csv, "Station lookup CSV", path_type="file")
                lookup = pd.read_csv(lookup_csv)
                self.uw.require_columns(
                    lookup,
                    [id_col, lat_col, lon_col],
                    "lookup_csv",
                )
                for sid in station_ids:
                    row = lookup.loc[lookup[id_col].astype(str) == str(sid)]
                    if row.empty:
                        raise ValueError(f"Station id not found in lookup CSV: {sid}")
                    la = float(row[lat_col].values[0])
                    lo = float(row[lon_col].values[0])
                    la, lo = _snap_coordinates(la, lo, river_grid_path)
                    stations.append((str(sid), la, lo))
            elif grdc_netcdf:
                self.uw.require_existing_path(grdc_netcdf, "GRDC NetCDF", path_type="file")
                ds = _open_dataset_with_fallback(grdc_netcdf)
                if "id" not in ds.dims:
                    raise ValueError("GRDC NetCDF missing 'id' dimension.")
                missing_vars = [name for name in ("geo_x", "geo_y") if name not in ds.variables]
                if missing_vars:
                    raise ValueError(
                        "GRDC NetCDF is missing required coordinate variables: "
                        + ", ".join(missing_vars)
                    )
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

        wfa_list = self._load_runoff_entries()

        wfa_dict = {item["time"]: item["matrix"] for item in wfa_list}
        date_index = pd.date_range(start=start_date, end=end_date, freq="D")
        if not any(dt.strftime("%Y-%m-%d") in wfa_dict for dt in date_index):
            available_dates = sorted(wfa_dict.keys())
            available_window = "unknown"
            if available_dates:
                available_window = f"{available_dates[0]} to {available_dates[-1]}"
            raise ValueError(
                "No routed runoff data is available for the requested period "
                f"{start_date} to {end_date}. Available period: {available_window}."
            )

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
        self.uw.validate_date_window(start_date, end_date, "start_date", "end_date")
        station_ids = []
        if lookup_csv:
            self.uw.require_existing_path(lookup_csv, "Station lookup CSV", path_type="file")
            lookup = pd.read_csv(lookup_csv)
            self.uw.require_columns(
                lookup,
                [id_col, lat_col, lon_col],
                "lookup_csv",
            )
            stations_df = pd.DataFrame({
                "id": lookup[id_col].astype(str).values,
                "geo_x": lookup[lon_col].astype(float).values,
                "geo_y": lookup[lat_col].astype(float).values,
            })
        elif grdc_netcdf:
            self.uw.require_existing_path(grdc_netcdf, "GRDC NetCDF", path_type="file")
            ds = _open_dataset_with_fallback(grdc_netcdf)
            if "id" not in ds.dims:
                raise ValueError("GRDC NetCDF missing 'id' dimension.")
            missing_vars = [name for name in ("geo_x", "geo_y") if name not in ds.variables]
            if missing_vars:
                raise ValueError(
                    "GRDC NetCDF is missing required coordinate variables: "
                    + ", ".join(missing_vars)
                )
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
        region_shape = self.uw.require_study_area_gdf()
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
