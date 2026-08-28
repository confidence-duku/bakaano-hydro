"""Flood mapping utilities built on routed terrain analysis.

Role: Build reach-scale rating curves and map inundation depth without RichDEM.
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from pyproj import CRS, Transformer
from rasterio.transform import xy
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.optimize import curve_fit
from shapely.geometry import Point
from pysheds.grid import Grid
from whitebox import WhiteboxTools

class FloodMapper:
    """Build flood rating curves and inundation maps for a Bakaano project.

    The class reuses the existing Bakaano project layout under ``working_dir`` and
    stores flood outputs under ``{working_dir}/flood``.

    Notes
    -----
    This implementation is designed for rapid reach-scale flood mapping rather
    than full hydraulic modelling. It uses HAND, simplified Manning-based
    hydraulics, and empirical flood-frequency analysis, so outputs should be
    treated as screening-level products unless locally validated.
    """

    def __init__(
        self,
        working_dir,
        study_area,
        climate_data_source=None,
        dem_path=None,
        routing_method="mfd",
        mannings_n=0.05,
        water_levels=None,
        stream_threshold_ratio=0.1,
    ):
        if working_dir is None or str(working_dir).strip() == "":
            raise ValueError("working_dir must be a non-empty path.")
        if study_area is None or str(study_area).strip() == "":
            raise ValueError("study_area must be a non-empty vector-data path.")

        self.working_dir = str(Path(working_dir).expanduser().resolve())
        self.study_area = str(Path(study_area).expanduser().resolve())
        self.climate_data_source = climate_data_source
        self.routing_method = routing_method
        self.mannings_n = float(mannings_n)
        self.water_levels = list(water_levels) if water_levels is not None else None
        self.stream_threshold_ratio = float(stream_threshold_ratio)
        dem_candidate = Path(dem_path).expanduser() if dem_path is not None else Path(self.working_dir) / "elevation" / "dem_clipped.tif"
        self.dem_path = str(dem_candidate.resolve())
        self.flood_dir = Path(self.working_dir) / "flood"
        self.scratch_dir = self.flood_dir / "scratch"
        self.flood_dir.mkdir(parents=True, exist_ok=True)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self._validate_configuration()

        self.analysis_dem_path = None
        self.analysis_dem_transform = None
        self.analysis_dem_crs = None
        self.analysis_dem_profile = None

        self.hand_data = None
        self.subbasin_data = None
        self.subbasin_ids = []
        self.rating_curve_dict = {}
        self.full_inundated_areas = None
        self.last_skip_reasons = {}
        self.last_skip_summary = pd.DataFrame(columns=["subbasin_id", "reason"])

    def _validate_configuration(self):
        """Validate constructor inputs before filesystem work starts."""
        working_dir_path = Path(self.working_dir)
        if not working_dir_path.exists():
            raise FileNotFoundError(f"working_dir does not exist: {working_dir_path}")
        if not working_dir_path.is_dir():
            raise ValueError(f"working_dir is not a directory: {working_dir_path}")

        study_area_path = Path(self.study_area)
        if not study_area_path.exists():
            raise FileNotFoundError(f"study_area file was not found: {study_area_path}")

        try:
            study_area_gdf = gpd.read_file(study_area_path)
        except Exception as exc:
            raise ValueError(
                f"study_area could not be read as a vector dataset: {study_area_path}. "
                f"Original error: {exc}"
            ) from exc

        if study_area_gdf.empty:
            raise ValueError(f"study_area contains no features: {study_area_path}")
        if study_area_gdf.crs is None:
            raise ValueError(f"study_area has no CRS metadata: {study_area_path}")

        if self.mannings_n <= 0:
            raise ValueError(f"mannings_n must be > 0. Received: {self.mannings_n}")
        if self.water_levels is not None:
            if not self.water_levels:
                raise ValueError("water_levels must contain at least one stage height when provided.")
            if any(float(level) <= 0 for level in self.water_levels):
                raise ValueError("water_levels must all be positive values.")
        if self.stream_threshold_ratio <= 0:
            raise ValueError(
                f"stream_threshold_ratio must be > 0. Received: {self.stream_threshold_ratio}"
            )
        if str(self.routing_method).lower() not in {"d8", "dinf", "mfd"}:
            raise ValueError(
                f"routing_method must be one of 'd8', 'dinf', or 'mfd'. Received: {self.routing_method}"
            )
        if not os.access(self.scratch_dir, os.W_OK):
            raise PermissionError(f"FloodMapper scratch directory is not writable: {self.scratch_dir}")
        dem_path = Path(self.dem_path)
        if not dem_path.exists():
            raise FileNotFoundError(
                "FloodMapper requires a clipped DEM, but it was not found at:\n"
                f"  - {dem_path}\n"
                "Run DEM preprocessing first so working_dir/elevation/dem_clipped.tif "
                "exists, or pass an explicit dem_path."
            )

    def _require_rating_curves_available(self, path):
        """Validate that rating curves exist with a workflow-aware error message."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(
                "FloodMapper rating curves were not found at:\n"
                f"  - {path_obj}\n"
                "Run compute_rating_curves() first to generate "
                "working_dir/flood/rating_curves.pkl."
            )
        return self._require_existing_file(path_obj, "Rating curve file")

    def _require_existing_file(self, path, description):
        """Validate that a required file exists."""
        if path is None or str(path).strip() == "":
            raise ValueError(f"{description} was not provided.")
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"{description} was not found: {path_obj}")
        if not path_obj.is_file():
            raise ValueError(f"{description} is not a file: {path_obj}")
        return str(path_obj)

    def _ensure_raster_readable(self, path, description):
        """Validate that a raster exists and can be opened."""
        path_str = self._require_existing_file(path, description)
        try:
            with rasterio.open(path_str) as src:
                if src.count < 1:
                    raise ValueError(f"{description} contains no raster bands: {path_str}")
                _ = src.bounds
        except Exception as exc:
            raise ValueError(
                f"{description} exists but could not be read as a raster: {path_str}. "
                f"Original error: {exc}"
            ) from exc
        return path_str

    def _record_skip(self, subbasin_id, reason):
        """Track why a subbasin was skipped during rating-curve generation."""
        self.last_skip_reasons[int(subbasin_id)] = str(reason)

    def _update_skip_summary(self):
        """Materialize skip reasons as a dataframe for inspection."""
        if self.last_skip_reasons:
            rows = [
                {"subbasin_id": int(subbasin_id), "reason": reason}
                for subbasin_id, reason in sorted(self.last_skip_reasons.items())
            ]
            self.last_skip_summary = pd.DataFrame(rows)
        else:
            self.last_skip_summary = pd.DataFrame(columns=["subbasin_id", "reason"])

    def get_skip_summary(self):
        """Return the last recorded subbasin skip summary."""
        return self.last_skip_summary.copy()

    def get_stage_height(self, discharge, k, h0, m):
        """Predict stage height from discharge using Q = k (h - h0)^m."""
        raise NotImplementedError("Use the bounded get_stage_height signature below.")

    def _prepare_analysis_dem(self, overwrite=False):
        """Use a metric DEM for hydraulic calculations, reprojecting if needed."""
        dem_metric_path = self.scratch_dir / "dem_metric.tif"
        self._ensure_raster_readable(self.dem_path, "Input DEM")

        with rasterio.open(self.dem_path) as src:
            if src.crs is None:
                raise ValueError(f"DEM has no CRS: {self.dem_path}")

            if not src.crs.is_geographic:
                self.analysis_dem_path = self.dem_path
                self.analysis_dem_transform = src.transform
                self.analysis_dem_crs = src.crs
                self.analysis_dem_profile = src.profile.copy()
                self._sanitize_dem_file(self.analysis_dem_path)
                return self.analysis_dem_path

            if overwrite or not dem_metric_path.exists():
                target_crs = self._select_metric_crs(src)
                transform, width, height = calculate_default_transform(
                    src.crs,
                    target_crs,
                    src.width,
                    src.height,
                    *src.bounds,
                )
                profile = src.profile.copy()
                profile.update(
                    crs=target_crs,
                    transform=transform,
                    width=width,
                    height=height,
                )
                with rasterio.open(dem_metric_path, "w", **profile) as dst:
                    for band_idx in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, band_idx),
                            destination=rasterio.band(dst, band_idx),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=Resampling.bilinear,
                        )
                self._sanitize_dem_file(dem_metric_path)

        with rasterio.open(dem_metric_path) as src:
            self.analysis_dem_path = str(dem_metric_path)
            self.analysis_dem_transform = src.transform
            self.analysis_dem_crs = src.crs
            self.analysis_dem_profile = src.profile.copy()
        return self.analysis_dem_path

    def _sanitize_dem_file(self, path):
        """Mask implausible DEM values in-place after copy/reprojection."""
        path = Path(path)
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            nodata_val = src.nodata
            if nodata_val is not None:
                data = np.where(
                    np.isclose(data, float(nodata_val), rtol=0.0, atol=1e-3),
                    np.nan,
                    data,
                )
            # Remove common DEM sentinels and globally implausible elevations.
            data = np.where((data <= -12000) | (data >= 10000), np.nan, data)
            profile = src.profile.copy()
            profile.update(dtype=rasterio.float32, nodata=np.nan, compress=None)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)

    def _load_array(self, path):
        """Load a raster as float32 and convert nodata to NaN."""
        with rasterio.open(path) as src:
            raw = src.read(1, masked=True)
            original = np.asarray(raw.data)
            data = original.astype(np.float32, copy=False)
            mask = np.ma.getmaskarray(raw).copy()
            nodata = src.nodata
            if nodata is not None and np.isfinite(nodata):
                if np.issubdtype(original.dtype, np.integer):
                    mask |= original == original.dtype.type(nodata)
                else:
                    mask |= np.isclose(data, float(nodata), rtol=0.0, atol=1e-6)
            data = np.where(mask, np.nan, data)
            data = np.where(np.isfinite(data), data, np.nan)
            transform = src.transform
            crs = src.crs
            profile = src.profile.copy()
        return data, transform, crs, profile

    def _select_metric_crs(self, src):
        """Choose a local projected CRS for hydraulic calculations."""
        bounds = src.bounds
        transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        center_lon, center_lat = transformer.transform(
            (bounds.left + bounds.right) / 2.0,
            (bounds.bottom + bounds.top) / 2.0,
        )
        zone = int(np.floor((center_lon + 180.0) / 6.0) + 1)
        zone = max(1, min(zone, 60))
        epsg = 32600 + zone if center_lat >= 0 else 32700 + zone
        return CRS.from_epsg(epsg)

    def _compute_slope_riserun(self, dem_data, transform):
        """Compute slope as rise/run using NumPy gradients."""
        dem = np.asarray(dem_data, dtype=np.float32)
        valid = np.isfinite(dem)
        if not np.any(valid):
            return np.full_like(dem, np.nan, dtype=np.float32)

        fill_value = float(np.nanmedian(dem[valid]))
        dem_filled = np.where(valid, dem, fill_value)

        xres = abs(float(transform.a))
        yres = abs(float(transform.e))
        if xres == 0 or yres == 0:
            raise ValueError("DEM transform has zero spatial resolution.")

        dz_dy, dz_dx = np.gradient(dem_filled, yres, xres)
        slope = np.hypot(dz_dx, dz_dy).astype(np.float32)
        slope[~valid] = np.nan
        return slope

    def _estimate_bankfull_depth(self, drainage_area_km2):
        """Estimate bankfull depth from drainage area."""
        drainage_area_km2 = max(float(drainage_area_km2), 0.0)
        return 0.3 * (drainage_area_km2 ** 0.3)

    def _derive_stage_heights_for_subbasin(self, hand_values, drainage_area_km2):
        """
        Physically consistent stage generation:
        - dense near channel (controls rating curve shape)
        - coarser in floodplain
        """

        hand = np.asarray(hand_values, dtype=np.float32)
        hand = hand[np.isfinite(hand) & (hand >= 0)]

        if hand.size == 0:
            raise ValueError("No valid HAND values")

        bankfull = self._estimate_bankfull_depth(drainage_area_km2)

        # --- LOW FLOW: dense resolution (critical!)
        low = np.linspace(0.05, bankfull, 6)

        # --- MID FLOW: transition to floodplain
        mid = np.linspace(bankfull, bankfull * 2.5, 5)

        # --- HIGH FLOW: based on HAND distribution
        high_quantiles = np.quantile(hand, [0.6, 0.75, 0.9, 0.97])

        # --- EXTREME
        extreme = max(np.nanmax(hand), bankfull * 4.0)

        levels = np.concatenate([low, mid, high_quantiles, [extreme]])

        levels = levels[np.isfinite(levels) & (levels > 0)]
        levels = np.unique(np.round(levels, 3))

        return levels.astype(np.float32)

    def _rating_curve_model(self, stage_height, k, h0, m):
        """Stage-discharge model Q = k (h - h0)^m."""
        effective = np.maximum(np.asarray(stage_height, dtype=np.float64) - h0, 1e-9)
        return k * np.power(effective, m)

    def _fit_rating_curve(self, stage_height, discharge):
        """Fit rating-curve parameters with h0 as a bounded free parameter."""
        h = np.asarray(stage_height, dtype=np.float64)
        q = np.asarray(discharge, dtype=np.float64)
        valid = np.isfinite(h) & np.isfinite(q) & (q > 0)
        h = h[valid]
        q = q[valid]
        if h.size < 4 or np.ptp(q) <= 0:
            raise ValueError("Insufficient valid stage-discharge pairs to fit rating curve.")

        min_h = float(np.min(h))
        max_q = float(np.max(q))
        initial_h0 = max(min_h * 0.5, 0.0)
        initial_m = 1.5
        initial_k = max_q / max((float(np.max(h)) - initial_h0) ** initial_m, 1e-9)
        params, _ = curve_fit(
            self._rating_curve_model,
            h,
            q,
            p0=(max(initial_k, 1e-6), initial_h0, initial_m),
            bounds=(
                (1e-9, 0.0, 0.5),
                (np.inf, max(float(np.max(h)) * 0.99, 1e-6), 5.0),
            ),
            maxfev=20000,
        )
        k, h0, m = params
        if not np.isfinite(k) or not np.isfinite(h0) or not np.isfinite(m):
            raise ValueError("Rating-curve fitting produced non-finite parameters.")
        if k <= 0 or m <= 0:
            raise ValueError(f"Rating-curve fitting produced invalid parameters: k={k}, m={m}")
        if h0 < 0 or h0 >= float(np.max(h)):
            raise ValueError(
                f"Rating-curve fitting produced implausible h0={h0}; expected 0 <= h0 < max(stage)."
            )
        return float(k), float(h0), float(m)

    def get_stage_height(
        self,
        discharge,
        k,
        h0,
        m,
        discharge_min=None,
        discharge_max=None,
    ):
        """Predict stage height from discharge with bounded extrapolation."""
        discharge = max(float(discharge), 0.0)
        k = float(k)
        h0 = float(h0)
        m = float(m)
        if not np.isfinite(k) or not np.isfinite(h0) or not np.isfinite(m):
            raise ValueError("Rating-curve parameters must be finite.")
        if k <= 0 or m <= 0:
            raise ValueError(f"Invalid rating-curve parameters for inversion: k={k}, m={m}")

        used_discharge = discharge
        extrapolated_low = False
        extrapolated_high = False
        if discharge_min is not None and np.isfinite(discharge_min) and used_discharge < float(discharge_min):
            used_discharge = float(discharge_min)
            extrapolated_low = True
        if discharge_max is not None and np.isfinite(discharge_max) and used_discharge > float(discharge_max):
            used_discharge = float(discharge_max)
            extrapolated_high = True

        stage_height = float(((used_discharge / k) ** (1.0 / m)) + h0)
        if not np.isfinite(stage_height) or stage_height < h0:
            raise ValueError(
                f"Stage inversion produced invalid stage_height={stage_height} from k={k}, h0={h0}, m={m}."
            )
        return stage_height, {
            "input_discharge_m3s": discharge,
            "used_discharge_m3s": used_discharge,
            "extrapolated_low": extrapolated_low,
            "extrapolated_high": extrapolated_high,
            "extrapolated": extrapolated_low or extrapolated_high,
        }

    def _extract_subbasin_window(self, subbasin_data, subbasin_id):
        """Return a cropped boolean mask and slices for one subbasin."""
        rows, cols = np.where(subbasin_data == subbasin_id)
        if rows.size == 0:
            return None

        row_min = int(rows.min())
        row_max = int(rows.max()) + 1
        col_min = int(cols.min())
        col_max = int(cols.max()) + 1
        row_slice = slice(row_min, row_max)
        col_slice = slice(col_min, col_max)
        mask = subbasin_data[row_slice, col_slice] == subbasin_id
        return mask, row_slice, col_slice

    def _rowcol_to_latlon(self, row, col):
        """Convert a raster row/col on the analysis DEM to latitude/longitude."""
        xcoord, ycoord = xy(self.analysis_dem_transform, row, col)
        transformer = Transformer.from_crs(self.analysis_dem_crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(xcoord, ycoord)
        return float(lat), float(lon)

    def _filter_outlets_in_study_area(self, subbasin_ids, latlist, lonlist):
        """Return only outlet coordinates that fall inside or on the study-area boundary."""
        region_gdf = gpd.read_file(self.study_area)
        if region_gdf.crs is None:
            region_gdf = region_gdf.set_crs("EPSG:4326")
        elif str(region_gdf.crs) != "EPSG:4326":
            region_gdf = region_gdf.to_crs("EPSG:4326")

        valid = []
        for subbasin_id, lat, lon in zip(subbasin_ids, latlist, lonlist):
            point = Point(lon, lat)
            if region_gdf.geometry.covers(point).any():
                valid.append((subbasin_id, lat, lon))
            else:
                self._record_skip(subbasin_id, f"outlet point outside study area ({lat}, {lon})")
        return valid

    def _write_raster(self, output_path, data):
        """Write a single-band float32 raster aligned to the analysis DEM."""
        profile = self.analysis_dem_profile.copy()
        profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(np.asarray(data, dtype=np.float32), 1)

    def _write_grid_raster(self, output_path, data, dtype=None, nodata=np.nan):
        """Write a single-band raster aligned to the analysis DEM."""
        arr = np.asarray(data)
        profile = self.analysis_dem_profile.copy()
        profile.update(dtype=(dtype or arr.dtype), count=1, nodata=nodata)
        write_arr = arr.copy()
        if nodata is not None and not np.issubdtype(write_arr.dtype, np.floating):
            write_arr = np.where(np.isfinite(write_arr), write_arr, nodata)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(write_arr.astype(profile["dtype"]), 1)

    def _initialize_whitebox_tools(self):
        """Initialize WhiteboxTools and point it at the scratch directory."""
        exe_name = "whitebox_tools.exe" if os.name == "nt" else "whitebox_tools"
        candidates = []
        for site_part in (Path(__import__("whitebox").__file__).resolve().parent,):
            candidates.append(site_part / exe_name)
            candidates.append(site_part / "WBT" / exe_name)

        resolved = None
        for candidate in candidates:
            try:
                candidate = candidate.expanduser().resolve()
            except Exception:
                continue
            if candidate.exists() and candidate.is_file() and os.access(candidate, os.X_OK):
                resolved = candidate
                break

        if resolved is None:
            raise RuntimeError(
                "WhiteboxTools executable was not found. The Python `whitebox` package is "
                "installed, but the `whitebox_tools` binary is missing in this environment."
            )

        wbt = WhiteboxTools()
        wbt.set_whitebox_dir(str(resolved.parent))
        wbt.set_working_dir(str(self.scratch_dir))   # ← KEY FIX
        wbt.set_verbose_mode(False)
        return wbt

    def _run_wbt_tool(self, overwrite, output_path, func, *args):
        """Run a WhiteboxTools function only when needed."""
        if overwrite and os.path.exists(output_path):
            os.remove(output_path)
        if not os.path.exists(output_path):
            try:
                result = func(*args)
            except Exception as exc:
                raise RuntimeError(
                    f"WhiteboxTools failed while creating {output_path}: {exc}"
                ) from exc
            if result not in (None, 0) and result is not True:
                raise RuntimeError(
                    f"WhiteboxTools reported a non-success status while creating {output_path}: {result}"
                )
        self._ensure_raster_readable(output_path, "WhiteboxTools output raster")

    def _condition_dem_for_hydrology(self, source_dem, output_dem, overwrite=False):
        """Condition DEM using WhiteboxTools to breach depressions using least cost."""
        if not overwrite and Path(output_dem).exists():
            self._ensure_raster_readable(output_dem, "Conditioned DEM")
            return

        wbt = self._initialize_whitebox_tools()
        temp_dem = str(self.scratch_dir / "temp_dem.tif")
        self._write_wbt_safe_dem(source_dem, temp_dem)

        # Breach depressions using least cost with WhiteboxTools
        wbt.breach_depressions_least_cost(temp_dem, output_dem, dist=10000)
        self._ensure_raster_readable(output_dem, "Conditioned DEM")

    def _build_d8_downstream(self, fdir_data, resolution):
        """Return downstream indices and per-cell step lengths for D8 flow directions."""
        nrows, ncols = fdir_data.shape
        row_grid, col_grid = np.indices((nrows, ncols))
        down_row = np.full((nrows, ncols), -1, dtype=np.int32)
        down_col = np.full((nrows, ncols), -1, dtype=np.int32)
        step_length = np.zeros((nrows, ncols), dtype=np.float32)
        d8_map = {
            64: (-1, 0, resolution),
            128: (-1, 1, resolution * np.sqrt(2.0)),
            1: (0, 1, resolution),
            2: (1, 1, resolution * np.sqrt(2.0)),
            4: (1, 0, resolution),
            8: (1, -1, resolution * np.sqrt(2.0)),
            16: (0, -1, resolution),
            32: (-1, -1, resolution * np.sqrt(2.0)),
        }
        for code, (drow, dcol, length) in d8_map.items():
            mask = fdir_data == code
            if not np.any(mask):
                continue
            rr = row_grid[mask] + drow
            cc = col_grid[mask] + dcol
            valid = (rr >= 0) & (rr < nrows) & (cc >= 0) & (cc < ncols)
            if not np.any(valid):
                continue
            mr = row_grid[mask][valid]
            mc = col_grid[mask][valid]
            down_row[mr, mc] = rr[valid]
            down_col[mr, mc] = cc[valid]
            step_length[mr, mc] = length
        return down_row, down_col, step_length

    def _derive_channel_products_pysheds(self, filled_dem, fdir_data, facc_data, stream_mask, resolution):
        """Derive HAND, stream-link ids, link lengths, and subbasins from pysheds products."""
        nrows, ncols = filled_dem.shape
        size = nrows * ncols
        dem_flat = filled_dem.reshape(-1)
        stream_flat = stream_mask.reshape(-1)
        valid_dem = np.isfinite(dem_flat)

        down_row, down_col, step_length = self._build_d8_downstream(fdir_data, resolution)
        flat_index = np.arange(size, dtype=np.int64).reshape((nrows, ncols))
        down_flat = np.full(size, -1, dtype=np.int64)
        valid_down = down_row >= 0
        down_flat[valid_down.reshape(-1)] = flat_index[down_row[valid_down], down_col[valid_down]]

        upstream_stream_count = np.zeros(size, dtype=np.int32)
        stream_indices = np.flatnonzero(stream_flat)
        for idx in stream_indices:
            downstream_idx = down_flat[idx]
            if downstream_idx >= 0 and stream_flat[downstream_idx]:
                upstream_stream_count[downstream_idx] += 1

        link_ids = np.full(size, -1, dtype=np.int32)
        next_link_id = 1

        def assign_link(start_idx, link_id):
            current = int(start_idx)
            while True:
                if link_ids[current] != -1:
                    break
                link_ids[current] = int(link_id)
                downstream_idx = down_flat[current]
                if downstream_idx < 0 or not stream_flat[downstream_idx]:
                    break
                if upstream_stream_count[downstream_idx] != 1:
                    break
                current = int(downstream_idx)

        for idx in stream_indices:
            if upstream_stream_count[idx] != 1 and link_ids[idx] == -1:
                assign_link(idx, next_link_id)
                next_link_id += 1
        for idx in stream_indices:
            if link_ids[idx] == -1:
                assign_link(idx, next_link_id)
                next_link_id += 1

        facc_for_order = np.where(np.isfinite(facc_data.reshape(-1)), facc_data.reshape(-1), -np.inf)
        processing_order = np.argsort(facc_for_order)[::-1]
        first_stream_idx = np.full(size, -1, dtype=np.int64)
        for idx in processing_order:
            if not valid_dem[idx]:
                continue
            if stream_flat[idx]:
                first_stream_idx[idx] = idx
                continue
            downstream_idx = down_flat[idx]
            if downstream_idx >= 0:
                first_stream_idx[idx] = first_stream_idx[downstream_idx]

        hand_flat = np.full(size, np.nan, dtype=np.float32)
        valid_first_stream = first_stream_idx >= 0
        if np.any(valid_first_stream):
            hand_flat[valid_first_stream] = (
                dem_flat[valid_first_stream] - dem_flat[first_stream_idx[valid_first_stream]]
            ).astype(np.float32)
            hand_flat[valid_first_stream] = np.maximum(hand_flat[valid_first_stream], 0.0)

        subbasin_flat = np.full(size, np.nan, dtype=np.float32)
        valid_segment = valid_first_stream & (link_ids[first_stream_idx] > 0)
        if np.any(valid_segment):
            subbasin_flat[valid_segment] = link_ids[first_stream_idx[valid_segment]].astype(np.float32)

        link_length_flat = np.zeros(size, dtype=np.float32)
        step_length_flat = step_length.reshape(-1)
        for idx in stream_indices:
            downstream_idx = down_flat[idx]
            if downstream_idx >= 0 and stream_flat[downstream_idx]:
                link_length_flat[idx] = step_length_flat[idx]
            else:
                link_length_flat[idx] = resolution

        stream_link_flat = np.full(size, np.nan, dtype=np.float32)
        stream_link_flat[stream_indices] = link_ids[stream_indices].astype(np.float32)

        return {
            "hand": hand_flat.reshape((nrows, ncols)),
            "subbasins": subbasin_flat.reshape((nrows, ncols)),
            "stream_links": stream_link_flat.reshape((nrows, ncols)),
            "stream_link_length": link_length_flat.reshape((nrows, ncols)),
        }
    
    def _estimate_reach_roughness(self, drainage_area_km2, reach_length_m, mean_bed_slope):
        """Estimate Manning's n from geomorphic properties (stable SRC-style)."""
        area = max(float(drainage_area_km2), 1e-6)
        length_km = max(float(reach_length_m) / 1000.0, 1e-6)
        slope = max(float(mean_bed_slope), 1e-6)

        n = 0.06
        n -= 0.006 * np.log10(area)
        n -= 0.003 * np.log10(length_km)
        n += 0.004 * np.log10(1.0 / slope)

        return float(np.clip(n, 0.025, 0.12))

    def _write_wbt_safe_dem(self, src_path, dst_path):
        """Write a DEM with nodata cells set to NaN so WBT skips them correctly.

        WBT reads raw GeoTIFF values directly. If nodata is stored as a sentinel
        like -9999.0 without being masked, WBT treats it as valid elevation and
        panics during pit-finding.
        """
        with rasterio.open(src_path) as src:
            data = src.read(1).astype(np.float32)
            nodata_val = src.nodata
            if nodata_val is not None:
                data = np.where(
                    np.isclose(data, float(nodata_val), rtol=0.0, atol=1e-3),
                    np.nan,
                    data,
                )
            # Mask common DEM sentinel values and implausible elevations before
            # Whitebox conditioning. Positive sentinels like 32767 are common.
            data = np.where(data < -100, np.nan, data)
            data = np.where(data > 9000, np.nan, data)
            profile = src.profile.copy()
            profile.update(dtype=rasterio.float32, nodata=np.nan, compress=None)
            with rasterio.open(dst_path, "w", **profile) as dst:
                dst.write(data, 1)


    def _prepare_hydrology_products(self, analysis_dem, overwrite=False):
        """Prepare hydrology products using pysheds for depression filling."""
        fil = str(self.scratch_dir / "filled_dem.tif")
        fdr = str(self.scratch_dir / "flow_direction.tif")
        facc = str(self.scratch_dir / "flow_accumulation.tif")

        # Step 1: Fill depressions using pysheds
        self._condition_dem_for_hydrology(analysis_dem, fil, overwrite)

        # Step 2: D8 Flow Direction
        wbt = self._initialize_whitebox_tools()
        self._run_wbt_tool(overwrite, fdr, wbt.d8_pointer, fil, fdr)

        # Step 3: Flow Accumulation
        self._run_wbt_tool(overwrite, facc, wbt.d8_flow_accumulation, fil, facc)

        streams = str(self.scratch_dir / "streams.tif")
        strlnk = str(self.scratch_dir / "stream_links.tif")
        hand = str(self.scratch_dir / "hand.tif")
        strlnk_length = str(self.scratch_dir / "stream_link_length.tif")
        subbasins = str(self.scratch_dir / "subbasins.tif")

        facc_data, _, _, _ = self._load_array(facc)
        study_area_gdf = gpd.read_file(self.study_area)
        if study_area_gdf.crs is not None and self.analysis_dem_crs is not None and study_area_gdf.crs != self.analysis_dem_crs:
            study_area_gdf = study_area_gdf.to_crs(self.analysis_dem_crs)
        study_mask = geometry_mask(
            study_area_gdf.geometry,
            out_shape=facc_data.shape,
            transform=self.analysis_dem_transform,
            invert=True,
        )
        facc_data = np.where(study_mask, facc_data, np.nan).astype(np.float32)
        self._write_grid_raster(facc, facc_data, dtype=rasterio.float32, nodata=np.nan)
        facc_threshold = float(np.nanmax(facc_data) * self.stream_threshold_ratio)

        self._run_wbt_tool(overwrite, streams,
            lambda: wbt.extract_streams(flow_accum=facc, output=streams,
                                        threshold=facc_threshold))
        # WhiteboxTools exposes stream-link IDs and subbasins only for D8 pointers,
        # so even when accumulation/stream extraction uses D-infinity or FD8,
        # topology products below are still derived with the D8 pointer.
        self._run_wbt_tool(overwrite, strlnk,
            lambda: wbt.stream_link_identifier(d8_pntr=fdr, streams=streams,
                                               output=strlnk))
        self._run_wbt_tool(overwrite, hand,
            lambda: wbt.elevation_above_stream(dem=fil, streams=streams,
                                               output=hand))
        self._run_wbt_tool(overwrite, strlnk_length,
            lambda: wbt.stream_link_length(d8_pntr=fdr, linkid=strlnk,
                                        output=strlnk_length))
        self._run_wbt_tool(overwrite, subbasins,
            lambda: wbt.subbasins(d8_pntr=fdr, streams=streams,
                                output=subbasins))

        filled_data, transform, crs, profile = self._load_array(fil)
        hand_data, _, _, _ = self._load_array(hand)
        strlnk_length_data, _, _, _ = self._load_array(strlnk_length)
        subbasin_data, _, _, _ = self._load_array(subbasins)
        hand_data = np.where(study_mask, hand_data, np.nan).astype(np.float32)
        strlnk_length_data = np.where(study_mask, strlnk_length_data, np.nan).astype(np.float32)
        subbasin_data = np.where(study_mask, subbasin_data, np.nan).astype(np.float32)

        return (
            filled_data,
            facc_data,
            hand_data,
            strlnk_length_data,
            subbasin_data,
            transform,
            crs,
            profile,
        )

    def compute_rating_curves(self, overwrite=False, min_subbasin_cells=10):

        analysis_dem = self._prepare_analysis_dem(overwrite=overwrite)

        fil_data, facc_data, hand_data, strlnk_length_data, subbasin_data, transform, crs, profile = (
            self._prepare_hydrology_products(analysis_dem, overwrite=overwrite)
        )

        self.analysis_dem_transform = transform
        self.analysis_dem_crs = crs
        self.analysis_dem_profile = profile
        self.hand_data = hand_data
        self.subbasin_data = subbasin_data

        subbasin_ids = np.unique(subbasin_data[np.isfinite(subbasin_data)])
        self.subbasin_ids = [int(x) for x in subbasin_ids.tolist()]

        resolution = abs(float(transform.a))
        slope = self._compute_slope_riserun(fil_data, transform)

        rating_curve_dict = {}
        self.last_skip_reasons = {}

        for subbasin_id in self.subbasin_ids:

            window = self._extract_subbasin_window(subbasin_data, subbasin_id)
            if window is None:
                self._record_skip(subbasin_id, "no window")
                continue

            sub_mask, row_slice, col_slice = window
            cell_count = int(np.sum(sub_mask))

            if cell_count <= min_subbasin_cells:
                self._record_skip(subbasin_id, "too small")
                continue

            facc_window = facc_data[row_slice, col_slice]
            hand_window = hand_data[row_slice, col_slice]
            slope_window = slope[row_slice, col_slice]
            link_length_window = strlnk_length_data[row_slice, col_slice]

            # outlet
            outlet = np.where(sub_mask, facc_window, np.nan)
            if not np.isfinite(outlet).any():
                continue

            orow_local, ocol_local = np.unravel_index(np.nanargmax(outlet), outlet.shape)
            orow = int(row_slice.start + orow_local)
            ocol = int(col_slice.start + ocol_local)

            # reach length
            length = float(np.nansum(link_length_window[sub_mask]))
            if length <= 0:
                continue

            hand_values = hand_window[sub_mask]
            valid = np.isfinite(hand_values)

            if not np.any(valid):
                continue

            hand_values = hand_values[valid].astype(np.float32)
            slope_values = slope_window[sub_mask][valid]

            slope_values = slope_values[np.isfinite(slope_values)]
            if len(slope_values) == 0:
                continue

            mean_bed_slope = max(float(np.nanmean(slope_values)), 1e-6)

            drainage_area_km2 = cell_count * resolution * resolution / 1_000_000.0
            bankfull_depth = self._estimate_bankfull_depth(drainage_area_km2)

            levels = self._derive_stage_heights_for_subbasin(hand_values, drainage_area_km2)

            # --- HAND inundation ---
            wet_mask = hand_values[None, :] < levels[:, None]
            wetted_counts = wet_mask.sum(axis=1)

            if not np.any(wetted_counts):
                continue

            inundation_depth = np.where(
                wet_mask,
                np.maximum(levels[:, None] - hand_values[None, :], 0.0),
                np.nan
            )

            # --- geometry from HAND ---
            surface_area = wetted_counts * (resolution ** 2)

            depth_sum = np.nansum(inundation_depth, axis=1)
            mean_depth = np.divide(
                depth_sum,
                wetted_counts,
                out=np.zeros_like(depth_sum),
                where=wetted_counts > 0
            )

            # Width from inundation.
            width = np.sqrt(surface_area)

            area = width * mean_depth

            # --- roughness ---
            base_n = self._estimate_reach_roughness(
                drainage_area_km2,
                length,
                mean_bed_slope
            )

            roughness_values = base_n * np.where(
                hand_values < bankfull_depth,
                0.7,
                1.3
            )

            roughness_values = np.clip(roughness_values, 0.02, 0.15)

            # effective n
            effective_n = np.zeros_like(levels)

            for i in range(len(levels)):
                mask = wet_mask[i]
                if np.any(mask):
                    effective_n[i] = np.nanmean(roughness_values[mask])
                else:
                    effective_n[i] = base_n

            # --- hydraulics ---
            wetted_perimeter = width + 2.0 * mean_depth

            hydraulic_radius = np.where(
                wetted_perimeter > 0,
                area / wetted_perimeter,
                0.0
            )

            discharge = (1.0 / effective_n) * area * (hydraulic_radius ** (2.0 / 3.0)) * (mean_bed_slope ** 0.5)

            # Scale with basin size.
            discharge = discharge * (drainage_area_km2 ** 0.8)

            # fit curve
            valid = np.isfinite(discharge) & np.isfinite(levels)

            if np.sum(valid) < 4:
                continue

            h = levels[valid]
            q = discharge[valid]

            try:
                k, h0, m = self._fit_rating_curve(h, q)
            except:
                continue

            lat, lon = self._rowcol_to_latlon(orow, ocol)

            rating_curve_dict[subbasin_id] = {
                "k_param": k,
                "h0_param": h0,
                "m_param": m,
                "outlet_lat": lat,
                "outlet_lon": lon,
                "stage_height_list": h.tolist(),
                "discharge_list": q.tolist(),
                "drainage_area_km2": drainage_area_km2,
                "reach_length_m": length,
                "mean_bed_slope": mean_bed_slope,
                "bankfull_depth_m": bankfull_depth,
            }

        if not rating_curve_dict:
            raise ValueError("No rating curves generated")

        self.rating_curve_dict = rating_curve_dict

        with open(self.flood_dir / "rating_curves.pkl", "wb") as f:
            pickle.dump(rating_curve_dict, f)

        return rating_curve_dict
    
    

    def load_rating_curves(self, path=None):
        """Load previously computed rating curves from disk.

        Parameters
        ----------
        path : str or Path, optional
            Pickle path. Defaults to ``working_dir/flood/rating_curves.pkl``.
        """
        path = Path(path or self.flood_dir / "rating_curves.pkl")
        self._require_rating_curves_available(path)
        with open(path, "rb") as file:
            self.rating_curve_dict = pickle.load(file)
        if not isinstance(self.rating_curve_dict, dict):
            raise ValueError(f"Rating curve file has invalid format: {path}")
        return self.rating_curve_dict

    def _ensure_subbasin_layers_loaded(self):
        """Load hand/subbasin rasters if they are not already available in memory."""
        if self.subbasin_data is None or self.hand_data is None:
            subbasin_path = self.scratch_dir / "subbasins.tif"
            hand_path = self.scratch_dir / "hand.tif"
            if not subbasin_path.exists() or not hand_path.exists():
                self.compute_rating_curves()
            subbasin_data, transform, crs, profile = self._load_array(subbasin_path)
            hand_data, _, _, _ = self._load_array(hand_path)
            self.subbasin_data = subbasin_data
            self.hand_data = hand_data
            self.analysis_dem_transform = transform
            self.analysis_dem_crs = crs
            self.analysis_dem_profile = profile

    def describe_hand_subbasin(self, subbasin_id):
        """Return descriptive HAND statistics for one subbasin."""
        self._ensure_subbasin_layers_loaded()
        subbasin_id = int(subbasin_id)
        mask = self.subbasin_data == subbasin_id
        if not np.any(mask):
            raise KeyError(f"Subbasin id not found in subbasin raster: {subbasin_id}")
        values = np.asarray(self.hand_data[mask], dtype=np.float32)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(f"No finite HAND values found for subbasin {subbasin_id}.")
        quantiles = np.nanquantile(values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
        return pd.Series(
            {
                "subbasin_id": subbasin_id,
                "count": int(values.size),
                "min_m": float(quantiles[0]),
                "q05_m": float(quantiles[1]),
                "q25_m": float(quantiles[2]),
                "median_m": float(quantiles[3]),
                "q75_m": float(quantiles[4]),
                "q95_m": float(quantiles[5]),
                "q99_m": float(quantiles[6]),
                "max_m": float(quantiles[7]),
                "mean_m": float(np.nanmean(values)),
            }
        )

    def plot_hand_subbasin(self, subbasin_id, figsize=(12, 5)):
        """Plot HAND values for one subbasin as a map plus a histogram."""
        self._ensure_subbasin_layers_loaded()
        subbasin_id = int(subbasin_id)
        mask = self.subbasin_data == subbasin_id
        if not np.any(mask):
            raise KeyError(f"Subbasin id not found in subbasin raster: {subbasin_id}")

        hand_sub = np.where(mask, self.hand_data, np.nan).astype(np.float32)
        values = hand_sub[np.isfinite(hand_sub)]
        if values.size == 0:
            raise ValueError(f"No finite HAND values found for subbasin {subbasin_id}.")

        transform = self.analysis_dem_transform
        nrows, ncols = self.subbasin_data.shape
        left = transform.c
        top = transform.f
        right = left + (transform.a * ncols)
        bottom = top + (transform.e * nrows)
        extent = [min(left, right), max(left, right), min(bottom, top), max(bottom, top)]

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        im = axes[0].imshow(hand_sub, extent=extent, origin="upper", cmap="viridis")
        try:
            study_area_gdf = gpd.read_file(self.study_area)
            if study_area_gdf.crs is not None and self.analysis_dem_crs is not None and study_area_gdf.crs != self.analysis_dem_crs:
                study_area_gdf = study_area_gdf.to_crs(self.analysis_dem_crs)
            study_area_gdf.boundary.plot(ax=axes[0], color="white", linewidth=0.8)
        except Exception:
            pass
        axes[0].set_title(f"HAND for Subbasin {subbasin_id}")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        fig.colorbar(im, ax=axes[0], label="HAND (m)")

        axes[1].hist(values, bins=min(40, max(10, values.size // 50)), color="#1f77b4", edgecolor="black")
        axes[1].set_title(f"HAND Distribution for Subbasin {subbasin_id}")
        axes[1].set_xlabel("HAND (m)")
        axes[1].set_ylabel("Count")
        axes[1].grid(alpha=0.25)
        plt.tight_layout()
        return axes

    def plot_rating_curve(self, subbasin_id, num_curve_points=200, figsize=(12, 5)):
        """Plot sampled stage-discharge points and a map for one subbasin."""
        if not self.rating_curve_dict:
            self.load_rating_curves()

        subbasin_id = int(subbasin_id)
        if subbasin_id not in self.rating_curve_dict:
            raise KeyError(f"Subbasin id not found in rating_curve_dict: {subbasin_id}")

        self._ensure_subbasin_layers_loaded()

        curve = self.rating_curve_dict[subbasin_id]
        stages = np.asarray(curve["stage_height_list"], dtype=np.float32)
        discharges = np.asarray(curve["discharge_list"], dtype=np.float32)
        k = float(curve["k_param"])
        h0 = float(curve["h0_param"])
        m = float(curve["m_param"])

        stage_min = float(np.nanmin(stages))
        stage_max = float(np.nanmax(stages))
        curve_x = np.linspace(stage_min, stage_max, int(num_curve_points), dtype=np.float32)
        curve_q = np.where(curve_x > h0, k * np.power(np.maximum(curve_x - h0, 0.0), m), np.nan)

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax = axes[0]
        ax.scatter(stages, discharges, color="#1f77b4", s=36, label="Sampled points", zorder=3)
        ax.plot(curve_x, curve_q, color="#d62728", linewidth=2, label="Fitted curve")
        ax.vlines(stages, ymin=0.0, ymax=discharges, color="#9ecae1", linewidth=1, alpha=0.7)
        ax.set_title(f"Rating Curve for Subbasin {subbasin_id}")
        ax.set_xlabel("Stage height (m)")
        ax.set_ylabel("Discharge (m3/s)")
        ax.grid(alpha=0.25)
        ax.legend()

        summary = (
            f"area={curve.get('drainage_area_km2', np.nan):.2f} km2 | "
            f"length={curve.get('reach_length_m', np.nan):.0f} m | "
            f"slope={curve.get('mean_bed_slope', np.nan):.4f} | "
            f"bankfull={curve.get('bankfull_depth_m', np.nan):.2f} m"
        )
        ax.text(0.02, 0.98, summary, transform=ax.transAxes, va="top", ha="left", fontsize=9)

        map_ax = axes[1]
        subbasin_mask = np.where(self.subbasin_data == subbasin_id, 1.0, np.nan)
        transform = self.analysis_dem_transform
        nrows, ncols = self.subbasin_data.shape
        left = transform.c
        top = transform.f
        right = left + (transform.a * ncols)
        bottom = top + (transform.e * nrows)
        extent = [min(left, right), max(left, right), min(bottom, top), max(bottom, top)]
        study_area_gdf = None
        try:
            study_area_gdf = gpd.read_file(self.study_area)
            if study_area_gdf.crs is not None and self.analysis_dem_crs is not None and study_area_gdf.crs != self.analysis_dem_crs:
                study_area_gdf = study_area_gdf.to_crs(self.analysis_dem_crs)
            study_mask = geometry_mask(
                study_area_gdf.geometry,
                out_shape=self.subbasin_data.shape,
                transform=transform,
                invert=True,
            )
        except Exception:
            study_mask = np.isfinite(self.subbasin_data)

        background = np.where(study_mask, 1.0, np.nan)
        subbasin_mask = np.where(study_mask & (self.subbasin_data == subbasin_id), 1.0, np.nan)
        map_ax.imshow(background, extent=extent, origin="upper", cmap="Greys", alpha=0.12)
        map_ax.imshow(subbasin_mask, extent=extent, origin="upper", cmap="Blues", vmin=0.0, vmax=1.0, alpha=0.95)

        try:
            if study_area_gdf is None:
                study_area_gdf = gpd.read_file(self.study_area)
                if study_area_gdf.crs is not None and self.analysis_dem_crs is not None and study_area_gdf.crs != self.analysis_dem_crs:
                    study_area_gdf = study_area_gdf.to_crs(self.analysis_dem_crs)
            study_area_gdf.boundary.plot(ax=map_ax, color="black", linewidth=0.8)
        except Exception:
            pass

        outlet_lon = curve.get("outlet_lon")
        outlet_lat = curve.get("outlet_lat")
        if outlet_lon is not None and outlet_lat is not None and self.analysis_dem_crs is not None:
            try:
                transformer = Transformer.from_crs("EPSG:4326", self.analysis_dem_crs, always_xy=True)
                outlet_x, outlet_y = transformer.transform(outlet_lon, outlet_lat)
                map_ax.scatter(outlet_x, outlet_y, color="blue", s=28, edgecolor="white", linewidth=0.6, zorder=4)
            except Exception:
                pass

        map_ax.set_title(f"Subbasin {subbasin_id} Map")
        map_ax.set_xlabel("x")
        map_ax.set_ylabel("y")
        plt.tight_layout()
        return axes

    def plot_stage_levels(self, subbasin_id=None, figsize=(10, 5)):
        """Plot stage-height levels for one subbasin or summarize them across all subbasins."""
        if not self.rating_curve_dict:
            self.load_rating_curves()

        if subbasin_id is not None:
            subbasin_id = int(subbasin_id)
            if subbasin_id not in self.rating_curve_dict:
                raise KeyError(f"Subbasin id not found in rating_curve_dict: {subbasin_id}")
            stages = np.asarray(self.rating_curve_dict[subbasin_id]["stage_height_list"], dtype=np.float32)
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(np.arange(1, len(stages) + 1), stages, marker="o", color="#2ca02c")
            ax.set_title(f"Stage-height Levels for Subbasin {subbasin_id}")
            ax.set_xlabel("Level index")
            ax.set_ylabel("Stage height (m)")
            ax.grid(alpha=0.25)
            plt.tight_layout()
            return ax

        rows = []
        for sid, curve in self.rating_curve_dict.items():
            stages = np.asarray(curve["stage_height_list"], dtype=np.float32)
            if stages.size == 0:
                continue
            rows.append(
                {
                    "subbasin_id": int(sid),
                    "min_stage_m": float(np.nanmin(stages)),
                    "max_stage_m": float(np.nanmax(stages)),
                    "mean_stage_m": float(np.nanmean(stages)),
                    "n_levels": int(stages.size),
                    "drainage_area_km2": float(curve.get("drainage_area_km2", np.nan)),
                }
            )
        if not rows:
            raise ValueError("No stage-height levels are available to plot.")

        summary = pd.DataFrame(rows).sort_values("drainage_area_km2")
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].scatter(summary["drainage_area_km2"], summary["max_stage_m"], color="#1f77b4")
        axes[0].set_title("Max Stage vs Drainage Area")
        axes[0].set_xlabel("Drainage area (km2)")
        axes[0].set_ylabel("Max stage (m)")
        axes[0].grid(alpha=0.25)

        axes[1].hist(summary["max_stage_m"], bins=min(12, len(summary)), color="#ff7f0e", edgecolor="black")
        axes[1].set_title("Distribution of Max Stage Heights")
        axes[1].set_xlabel("Max stage (m)")
        axes[1].set_ylabel("Count")
        axes[1].grid(alpha=0.25)
        plt.tight_layout()
        return axes

    def _split_predictions_by_station(self, flat_predictions, station_window_counts):
        """Split flat model predictions into station-aligned chunks."""
        total = int(np.sum(station_window_counts))
        if total != len(flat_predictions):
            raise ValueError(
                "Prediction/window mismatch: "
                f"predictions={len(flat_predictions)} vs expected={total}"
            )

        chunks = []
        start = 0
        for count in station_window_counts:
            end = start + int(count)
            chunks.append(flat_predictions[start:end])
            start = end
        return chunks

    def simulate_streamflow_at_outlets(
        self,
        model_path,
        sim_start,
        sim_end,
        runoff_output_dir=None,
        **scaling_options,
    ):
        """Simulate Bakaano streamflow at the rating-curve outlet coordinates.

        Parameters
        ----------
        model_path : str
            Trained Bakaano model checkpoint.
        sim_start, sim_end : str
            Simulation window in ``YYYY-MM-DD`` format.
        runoff_output_dir : str, optional
            Override for routed runoff inputs. This is useful when coupling
            FloodMapper to scenario-specific runoff outputs.

        Returns
        -------
        dict
            Mapping of subbasin id to a daily streamflow DataFrame.

        Notes
        -----
        Rating curves must already exist, either in memory or on disk. The
        streamflow series start one year after ``sim_start`` because Bakaano's
        predictor windows require a 365-day lead-in period.
        Model scaling options are loaded automatically from the trained model's
        sidecar config when available.
        """
        area_normalize = scaling_options.pop("area_normalize", True)
        log_transform = scaling_options.pop("log_transform", True)
        if scaling_options:
            unknown = ", ".join(sorted(scaling_options))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")

        if not self.rating_curve_dict:
            self.load_rating_curves()
        if model_path is None or str(model_path).strip() == "":
            raise ValueError(
                "model_path was not provided. Provide a trained Bakaano model, "
                "typically working_dir/models/bakaano_model.keras."
            )
        if not Path(model_path).exists():
            raise FileNotFoundError(
                "Bakaano streamflow model was not found at:\n"
                f"  - {model_path}\n"
                "Train a streamflow model first so a checkpoint is available "
                "under working_dir/models."
            )
        self._require_existing_file(model_path, "Bakaano streamflow model")

        from bakaano.neuralnet.simulate import PredictDataPreprocessor, PredictStreamflow

        subbasin_ids = list(self.rating_curve_dict.keys())
        latlist = [self.rating_curve_dict[sid]["outlet_lat"] for sid in subbasin_ids]
        lonlist = [self.rating_curve_dict[sid]["outlet_lon"] for sid in subbasin_ids]
        valid_outlets = self._filter_outlets_in_study_area(subbasin_ids, latlist, lonlist)
        if not valid_outlets:
            raise ValueError("No rating-curve outlets fall inside the study area.")
        subbasin_ids = [sid for sid, _, _ in valid_outlets]
        latlist = [lat for _, lat, _ in valid_outlets]
        lonlist = [lon for _, _, lon in valid_outlets]
        raw_index_to_subbasin_id = {idx: sid for idx, sid in enumerate(subbasin_ids)}

        vdp = PredictDataPreprocessor(
            self.working_dir,
            self.study_area,
            sim_start,
            sim_end,
            self.routing_method,
            runoff_output_dir=runoff_output_dir,
        )
        rawdata = vdp.get_data_latlng(latlist, lonlist)
        if not rawdata:
            raise ValueError("No outlet predictor data could be prepared for flood mapping.")

        vmodel = PredictStreamflow(
            self.working_dir,
            area_normalize=area_normalize,
            log_transform=log_transform,
        )
        vmodel.load_model_config(model_path)
        vmodel.prepare_data_latlng(rawdata)
        vmodel.load_model(model_path)

        batch_size = max(1, len(latlist))
        station_preds = vmodel.predict_station_series(
            batch_size=batch_size,
            area_normalize=vmodel.area_normalize,
        )

        adjusted_start = pd.to_datetime(sim_start) + pd.DateOffset(days=365)
        streamflow_by_subbasin = {}
        for station_pred, raw_idx in zip(
            station_preds,
            vmodel.valid_entry_indices,
        ):
            station_pred = np.maximum(np.asarray(station_pred, dtype=np.float32).reshape(-1), 0.0)
            period = pd.date_range(adjusted_start, periods=len(station_pred), freq="D")
            if raw_idx not in raw_index_to_subbasin_id:
                raise ValueError(f"Unexpected outlet raw_idx={raw_idx}; cannot map prediction to subbasin id.")
            subbasin_id = raw_index_to_subbasin_id[raw_idx]
            streamflow_by_subbasin[subbasin_id] = pd.DataFrame(
                {
                    "time": period,
                    "streamflow_m3s": station_pred,
                }
            )
        if not streamflow_by_subbasin:
            raise ValueError("No outlet streamflow series were produced for flood mapping.")
        return streamflow_by_subbasin

    def extract_annual_peaks(self, streamflow_data, start_date=None):
        """Extract annual maximum discharge from a daily streamflow series.

        Accepts a DataFrame, Series, or array-like input. When the input lacks a
        datetime index, ``start_date`` is required so that annual aggregation
        can be constructed correctly.
        """
        if isinstance(streamflow_data, pd.DataFrame):
            if "time" in streamflow_data.columns:
                value_cols = [c for c in streamflow_data.columns if c != "time"]
                if not value_cols:
                    raise ValueError("Streamflow DataFrame must include a value column.")
                series = pd.Series(
                    streamflow_data[value_cols[0]].to_numpy(),
                    index=pd.to_datetime(streamflow_data["time"]),
                )
            else:
                series = streamflow_data.iloc[:, 0]
                if not isinstance(series.index, pd.DatetimeIndex):
                    if start_date is None:
                        raise ValueError("start_date is required when streamflow data has no datetime index.")
                    series.index = pd.date_range(pd.to_datetime(start_date), periods=len(series), freq="D")
        elif isinstance(streamflow_data, pd.Series):
            series = streamflow_data.copy()
            if not isinstance(series.index, pd.DatetimeIndex):
                if start_date is None:
                    raise ValueError("start_date is required when streamflow series has no datetime index.")
                series.index = pd.date_range(pd.to_datetime(start_date), periods=len(series), freq="D")
        else:
            values = np.asarray(streamflow_data, dtype=np.float32).reshape(-1)
            if start_date is None:
                raise ValueError("start_date is required when streamflow data is array-like.")
            index = pd.date_range(pd.to_datetime(start_date), periods=len(values), freq="D")
            series = pd.Series(values, index=index)

        annual_peaks = series.resample("YE").max().dropna()
        return annual_peaks.to_frame(name="annual_peaks")

    def flood_frequency_analysis(self, annual_peaks, return_period):
        """Estimate a return-period discharge from annual peaks.

        Parameters
        ----------
        annual_peaks : pandas.DataFrame
            Annual maxima with an ``annual_peaks`` column.
        return_period : float
            Return period in years. Must be greater than 1.

        Returns
        -------
        float
            Estimated discharge threshold in ``m3/s``.

        Notes
        -----
        This uses a simple quadratic fit in log-return-period space. It is a
        pragmatic approximation for short or moderately sized records and should
        not be treated as a substitute for a full frequency analysis study.
        """
        if return_period is None or float(return_period) <= 1:
            raise ValueError(f"return_period must be > 1 years. Received: {return_period}")
        annual_peaks_sorted = annual_peaks.sort_values(by="annual_peaks", ascending=False).copy()
        n = annual_peaks_sorted.shape[0]
        if n < 3:
            raise ValueError("At least three annual peaks are required for flood frequency analysis.")

        annual_peaks_sorted.insert(0, "rank", range(1, 1 + n))
        annual_peaks_sorted["exceedance_prob"] = (
            annual_peaks_sorted["rank"] - 0.44
        ) / (n + 0.12)
        annual_peaks_sorted["return_period"] = 1.0 / annual_peaks_sorted["exceedance_prob"]
        xvals = annual_peaks_sorted["return_period"].values
        yvals = annual_peaks_sorted["annual_peaks"].values
        weights = np.sqrt(np.maximum(yvals, 1e-6))
        coeffs = np.polyfit(np.log10(xvals), yvals, 2, w=weights)
        log_rp = np.log10(return_period)
        estimate = float((coeffs[0] * log_rp * log_rp) + (coeffs[1] * log_rp) + coeffs[2])
        if not np.isfinite(estimate):
            raise ValueError("Flood frequency analysis produced a non-finite discharge estimate.")
        metadata = {
            "return_period_extrapolated": bool(return_period < float(np.nanmin(xvals)) or return_period > float(np.nanmax(xvals))),
            "negative_estimate_fallback": False,
            "fitted_return_period_min_years": float(np.nanmin(xvals)),
            "fitted_return_period_max_years": float(np.nanmax(xvals)),
        }
        if estimate <= 0:
            warnings.warn(
                "Flood frequency analysis produced a non-positive discharge estimate. "
                "Using the maximum observed annual peak instead.",
                RuntimeWarning,
            )
            estimate = float(np.nanmax(yvals))
            metadata["negative_estimate_fallback"] = True
        return estimate, metadata

    def map_inundated_areas(
        self,
        return_period,
        model_path=None,
        sim_start=None,
        sim_end=None,
        predicted_streamflows=None,
        runoff_output_dir=None,
        output_path=None,
        **scaling_options,
    ):
        """Map inundation depth for a specified return period.

        Parameters
        ----------
        return_period : float
            Return period in years.
        model_path : str, optional
            Trained Bakaano model used to simulate outlet hydrographs.
        sim_start, sim_end : str, optional
            Simulation window used when ``model_path`` is supplied.
        predicted_streamflows : dict, optional
            Precomputed streamflow series keyed by subbasin id. When provided,
            these are used instead of running Bakaano simulation.
        runoff_output_dir : str, optional
            Routed runoff directory override, useful for scenario workflows.
        output_path : str or Path, optional
            Custom output GeoTIFF path.

        Returns
        -------
        dict
            Inundation depth array, flood metadata, output path, and skipped
            subbasins summary.

        Notes
        -----
        Provide either:

        - ``predicted_streamflows`` keyed by subbasin id, or
        - ``model_path`` with ``sim_start`` and ``sim_end`` so FloodMapper can
          simulate outlet hydrographs itself.

        The output raster is a screening-level inundation product based on HAND
        and rating curves, not a full 2D hydraulic simulation.
        Model scaling options are loaded automatically from the trained model's
        sidecar config when ``model_path`` is used.
        """
        area_normalize = scaling_options.pop("area_normalize", True)
        log_transform = scaling_options.pop("log_transform", True)
        if scaling_options:
            unknown = ", ".join(sorted(scaling_options))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")

        if self.hand_data is None or self.subbasin_data is None:
            self.compute_rating_curves()
        if not self.rating_curve_dict:
            self.load_rating_curves()

        if predicted_streamflows is None:
            if not model_path or not sim_start or not sim_end:
                raise ValueError(
                    "Flood mapping needs streamflow at rating-curve outlets. "
                    "Provide either:\n"
                    "  - predicted_streamflows keyed by subbasin id, or\n"
                    "  - model_path together with sim_start and sim_end so "
                    "FloodMapper can simulate outlet hydrographs."
                )
            predicted_streamflows = self.simulate_streamflow_at_outlets(
                model_path=model_path,
                sim_start=sim_start,
                sim_end=sim_end,
                area_normalize=area_normalize,
                log_transform=log_transform,
                runoff_output_dir=runoff_output_dir,
            )

        catch_flood = []
        flood_metadata = {}
        analysis_start = pd.to_datetime(sim_start) + pd.DateOffset(days=365) if sim_start else None

        for subbasin_id, curve in self.rating_curve_dict.items():
            streamflow_data = predicted_streamflows.get(subbasin_id)
            if streamflow_data is None:
                continue

            annual_peaks = self.extract_annual_peaks(streamflow_data, start_date=analysis_start)
            if annual_peaks.empty:
                continue

            flood_threshold, frequency_info = self.flood_frequency_analysis(annual_peaks, return_period)
            stage_height, stage_info = self.get_stage_height(
                flood_threshold,
                curve["k_param"],
                curve["h0_param"],
                curve["m_param"],
                discharge_min=curve.get("fit_discharge_min_m3s"),
                discharge_max=curve.get("fit_discharge_max_m3s"),
            )

            this_subbasin = np.where(self.subbasin_data == subbasin_id, 1.0, np.nan)
            catch_hand = np.where(np.isfinite(this_subbasin), self.hand_data, np.nan)
            wet_mask = np.isfinite(catch_hand) & (catch_hand < stage_height)
            inundation_depth = np.where(
                np.isfinite(catch_hand),
                np.where(wet_mask, np.maximum(stage_height - catch_hand, 0.0), 0.0),
                np.nan,
            ).astype(np.float32)

            catch_flood.append(inundation_depth)
            flood_metadata[subbasin_id] = {
                "flood_threshold_m3s": float(flood_threshold),
                "stage_height_m": float(stage_height),
                "frequency_analysis": frequency_info,
                "stage_inversion": stage_info,
            }

        if not catch_flood:
            raise ValueError("No inundated areas were mapped. Check streamflow inputs and rating curves.")

        # NaN-aware merge: keep NaN outside every mapped subbasin while taking the
        # maximum inundation depth where subbasins contribute valid values.
        self.full_inundated_areas = np.fmax.reduce(np.stack(catch_flood, axis=0)).astype(np.float32)
        self.flood_metadata = flood_metadata

        output_path = Path(output_path or (self.flood_dir / f"inundation_depth_{return_period}yr.tif"))
        self._write_raster(output_path, self.full_inundated_areas)
        self._ensure_raster_readable(output_path, "Flood inundation output raster")
        return {
            "inundation_depth": self.full_inundated_areas,
            "metadata": flood_metadata,
            "output_path": str(output_path),
            "skipped_subbasins": self.get_skip_summary(),
        }

    def _resolve_inundation_output_path(self, result=None, output_path=None):
        """Resolve an inundation GeoTIFF path from a result dict or explicit path."""
        if result is not None:
            if isinstance(result, dict) and result.get("output_path"):
                output_path = result["output_path"]
            else:
                raise ValueError("result must be the dictionary returned by map_inundated_areas(...).")
        output_path = Path(output_path) if output_path is not None else None
        if output_path is None:
            raise ValueError("Provide either result=map_inundated_areas(...) output or output_path=path/to/raster.")
        self._ensure_raster_readable(output_path, "Flood inundation output raster")
        return output_path

    def plot_inundation_map(
        self,
        result=None,
        output_path=None,
        figsize=(10, 8),
        cmap="Blues",
        overlay_study_area=True,
    ):
        """Plot the inundation depth raster with an optional study-area boundary."""
        output_path = self._resolve_inundation_output_path(result=result, output_path=output_path)

        with rasterio.open(output_path) as src:
            inundation = src.read(1).astype(np.float32)
            bounds = src.bounds
            nodata = src.nodata
            raster_crs = src.crs

        if nodata is not None and np.isfinite(nodata):
            inundation = np.where(np.isclose(inundation, float(nodata), rtol=0.0, atol=1e-6), np.nan, inundation)

        fig, ax = plt.subplots(figsize=figsize)
        img = ax.imshow(
            inundation,
            extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
            origin="upper",
            cmap=cmap,
        )
        if overlay_study_area:
            study_area_gdf = gpd.read_file(self.study_area)
            if study_area_gdf.crs is None:
                study_area_gdf = study_area_gdf.set_crs("EPSG:4326")
            if raster_crs is not None and study_area_gdf.crs != raster_crs:
                study_area_gdf = study_area_gdf.to_crs(raster_crs)
            study_area_gdf.boundary.plot(ax=ax, color="black", linewidth=1)

        ax.set_title("Inundation depth map")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(img, ax=ax, label="Depth (m)")
        return fig, ax

    def view_inundation_map_interactive(
        self,
        result=None,
        output_path=None,
        overlay_study_area=True,
        opacity=0.6,
        cmap="Blues",
    ):
        """Open an interactive ipyleaflet map with inundation over satellite imagery."""
        from ipyleaflet import GeoJSON, LayersControl, Map, basemaps
        from localtileserver import get_leaflet_tile_layer

        output_path = self._resolve_inundation_output_path(result=result, output_path=output_path)

        with rasterio.open(output_path) as src:
            bounds = src.bounds
            raster_crs = src.crs
            nodata = src.nodata

        center_lat = float((bounds.bottom + bounds.top) / 2.0)
        center_lon = float((bounds.left + bounds.right) / 2.0)
        m = Map(center=(center_lat, center_lon), zoom=9, basemap=basemaps.Esri.WorldImagery, scroll_wheel_zoom=True)
        tile_layer = get_leaflet_tile_layer(
            str(output_path),
            nodata=nodata,
            colormap=cmap,
            opacity=float(opacity),
        )
        tile_layer.name = "Inundation depth"
        m.add_layer(tile_layer)
        m.fit_bounds([[float(bounds.bottom), float(bounds.left)], [float(bounds.top), float(bounds.right)]])

        if overlay_study_area:
            study_area_gdf = gpd.read_file(self.study_area)
            if study_area_gdf.crs is None:
                study_area_gdf = study_area_gdf.set_crs("EPSG:4326")
            elif str(study_area_gdf.crs) != "EPSG:4326":
                study_area_gdf = study_area_gdf.to_crs("EPSG:4326")
            m.add_layer(GeoJSON(data=study_area_gdf.__geo_interface__, name="study_area"))

        m.add_control(LayersControl(position="topright"))
        return m
