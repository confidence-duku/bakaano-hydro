"""Scenario utilities for land-cover change experiments."""

from __future__ import annotations

import json
import os
import pickle
from collections import UserDict
from pathlib import Path
from collections.abc import Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from shapely.geometry import shape
from shapely import wkt

from bakaano.core.utils import Utils

try:
    from IPython.display import HTML, display
except ImportError:  # pragma: no cover - optional dependency fallback
    HTML = display = None


def _load_veget():
    """Import VegET only when scenario runoff recomputation is requested."""
    from bakaano.hydrology.veget import VegET

    return VegET


class ScenarioMetadata(UserDict):
    """Dict-like scenario metadata with cleaner notebook display."""

    def __str__(self):
        lines = ["Scenario metadata"]
        for key, value in self.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    def _repr_pretty_(self, printer, cycle):
        if cycle:
            printer.text("ScenarioMetadata(...)")
            return
        printer.text(str(self))

    def _repr_html_(self):
        summary_rows = []
        path_rows = []
        nested_sections = []
        path_keys = {"tree_cover_tif", "herb_cover_tif", "ndvi_pickle_path", "geometry_path", "runoff_output_dir", "predicted_streamflow_dir"}

        for key, value in self.items():
            if isinstance(value, dict):
                nested_df = pd.DataFrame(
                    [{"Key": str(subkey), "Value": str(subvalue)} for subkey, subvalue in value.items()]
                )
                nested_sections.append(f"<h4>{key}</h4>{nested_df.to_html(index=False, escape=True, border=0)}")
            elif key in path_keys:
                path_rows.append({"Name": str(key), "Path": str(value)})
            else:
                summary_rows.append({"Field": str(key), "Value": str(value)})

        sections = []
        if summary_rows:
            sections.append("<h4>Summary</h4>")
            sections.append(pd.DataFrame(summary_rows).to_html(index=False, escape=True, border=0))
        if path_rows:
            sections.append("<h4>Outputs</h4>")
            sections.append(pd.DataFrame(path_rows).to_html(index=False, escape=True, border=0))
        sections.extend(nested_sections)
        return "".join(sections)

    def _repr_mimebundle_(self, include=None, exclude=None):
        return {
            "text/plain": str(self),
            "text/html": self._repr_html_(),
        }

    def _ipython_display_(self):
        if HTML is not None:
            display(HTML(self._repr_html_()))
            return
        print(str(self))


class ScenarioManager:
    """Create and evaluate standalone vegetation-change scenarios.

    The scenario workflow is file-based: each scenario is written under
    ``{working_dir}/scenarios/{scenario_name}`` and contains modified vegetation
    rasters, a scenario-specific NDVI climatology, runoff outputs, and any
    downstream streamflow simulations.

    Notes
    -----
    - This class is intentionally independent from the neural-network training
      and simulation entry points.
    - Scenarios modify vegetation inputs only inside a user-supplied polygon.
    - NDVI changes use a threshold-based rule tied to VegET's ``NDVI > 0.4``
      breakpoint; this is operationally convenient but still a simplification.
    """

    def __init__(self, working_dir, study_area, climate_data_source=None):
        self.working_dir = os.fspath(working_dir)
        self.study_area = os.fspath(study_area)
        self.climate_data_source = climate_data_source
        self.uw = Utils(self.working_dir, self.study_area)
        self.uw.require_study_area_gdf()
        self.scenarios_root = Path(self.working_dir) / "scenarios"
        self.scenarios_root.mkdir(parents=True, exist_ok=True)

    def _normalize_scenario_name(self, scenario_name):
        """Return a safe scenario directory name."""
        if scenario_name is None or str(scenario_name).strip() == "":
            raise ValueError("scenario_name must be a non-empty string.")
        candidate = Path(str(scenario_name).strip())
        if candidate.name in {"", ".", ".."} or candidate != Path(candidate.name):
            raise ValueError(
                "scenario_name must be a simple directory name without path separators."
            )
        return candidate.name

    def _resolve_climate_data_source(self, climate_data_source):
        """Return the configured climate source or raise a clear error."""
        source = climate_data_source or self.climate_data_source
        if source is None or str(source).strip() == "":
            raise ValueError(
                "climate_data_source is required for scenario runoff recomputation. "
                "Pass it to ScenarioManager(...) or recompute_runoff(...)."
            )
        return source

    def _require_baseline_scenario_inputs(self):
        """Validate baseline preprocessing outputs needed for scenario creation."""
        baseline_tree = Path(self.working_dir) / "vcf" / "mean_tree_cover.tif"
        baseline_herb = Path(self.working_dir) / "vcf" / "mean_herb_cover.tif"
        baseline_ndvi = Path(self.working_dir) / "ndvi" / "daily_ndvi_climatology.pkl"

        if not baseline_tree.exists() or not baseline_herb.exists():
            raise FileNotFoundError(
                "Scenario creation requires baseline vegetation-cover rasters, but "
                "one or both of the following files are missing:\n"
                f"  - {baseline_tree}\n"
                f"  - {baseline_herb}\n"
                "Run tree-cover preprocessing first so mean tree and herb cover "
                "are available under working_dir/vcf."
            )
        if not baseline_ndvi.exists():
            raise FileNotFoundError(
                "Scenario creation requires the baseline NDVI climatology, but "
                f"it was not found at:\n  - {baseline_ndvi}\n"
                "Run NDVI preprocessing first so daily_ndvi_climatology.pkl is "
                "available under working_dir/ndvi."
            )
        return baseline_tree, baseline_herb, baseline_ndvi

    def _require_scenario_inputs(self, paths):
        """Validate scenario outputs needed for runoff or simulation."""
        missing = []
        if not Path(paths["tree_cover"]).exists():
            missing.append(str(paths["tree_cover"]))
        if not Path(paths["herb_cover"]).exists():
            missing.append(str(paths["herb_cover"]))
        if not Path(paths["ndvi_pickle"]).exists():
            missing.append(str(paths["ndvi_pickle"]))
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(
                "Scenario inputs are incomplete. The following scenario files are missing:\n"
                f"{formatted}\n"
                "Create or recreate the scenario with create_land_cover_scenario(...) "
                "before recomputing runoff or running scenario simulations."
            )

    def get_paths(self, scenario_name):
        scenario_name = self._normalize_scenario_name(scenario_name)
        scenario_dir = self.scenarios_root / scenario_name
        vcf_dir = scenario_dir / "vcf"
        ndvi_dir = scenario_dir / "ndvi"
        runoff_dir = scenario_dir / "runoff_output"
        predicted_dir = scenario_dir / "predicted_streamflow_data"
        return {
            "scenario_dir": scenario_dir,
            "vcf_dir": vcf_dir,
            "ndvi_dir": ndvi_dir,
            "runoff_dir": runoff_dir,
            "predicted_streamflow_dir": predicted_dir,
            "tree_cover": vcf_dir / "mean_tree_cover.tif",
            "herb_cover": vcf_dir / "mean_herb_cover.tif",
            "ndvi_pickle": ndvi_dir / "daily_ndvi_climatology.pkl",
            "geometry": scenario_dir / "scenario_geometry.geojson",
            "metadata": scenario_dir / "scenario_metadata.json",
        }

    def _compute_ndvi_threshold_delta(self, percent_change):
        """Map scenario intensity to an NDVI threshold-crossing shift."""
        fraction = np.clip(float(percent_change) / 100.0, 0.0, 1.0)
        min_delta = 0.05
        max_delta = 0.20
        return float(min_delta + ((max_delta - min_delta) * fraction))

    def _build_ndvi_scenario(self, baseline_ndvi, mask, percent_change, change_type):
        """Create a threshold-based NDVI scenario matching VegET's 0.4 breakpoint."""
        delta = self._compute_ndvi_threshold_delta(percent_change)
        threshold = 0.4
        upper_target = min(threshold + delta, 0.95)
        lower_target = max(threshold - delta, 0.05)

        scenario_ndvi = {}
        changed_cells_by_day = {}
        for doy, ndvi_day in baseline_ndvi.items():
            if hasattr(ndvi_day, "values"):
                base_values = np.asarray(ndvi_day.values, dtype=np.float32)
            else:
                base_values = np.asarray(ndvi_day, dtype=np.float32)

            scaled = base_values * 0.0001
            day_values = scaled.copy()
            if change_type == "reforestation":
                eligible = mask & np.isfinite(day_values) & (day_values <= threshold)
                day_values[eligible] = np.maximum(day_values[eligible] + delta, upper_target)
            else:
                eligible = mask & np.isfinite(day_values) & (day_values > threshold)
                day_values[eligible] = np.minimum(day_values[eligible] - delta, lower_target)

            day_values = np.clip(day_values, 0.0, 1.0)
            changed_cells_by_day[int(doy)] = int(np.count_nonzero(eligible))

            if hasattr(ndvi_day, "copy"):
                scenario_day = ndvi_day.copy(deep=True)
                scenario_day.values = (day_values / 0.0001).astype(np.float32)
            else:
                scenario_day = (day_values / 0.0001).astype(np.float32)
            scenario_ndvi[int(doy)] = scenario_day

        return scenario_ndvi, {
            "threshold": threshold,
            "delta": delta,
            "upper_target": upper_target,
            "lower_target": lower_target,
            "changed_cells_by_day": changed_cells_by_day,
        }

    def _scenario_mask_from_geometry(self, geometry_path, reference_raster):
        """Rasterize the stored scenario geometry onto a reference raster grid."""
        self.uw.require_existing_path(geometry_path, "Scenario geometry", path_type="file")
        self.uw.require_existing_path(reference_raster, "Reference raster", path_type="file")

        geometry_gdf = gpd.read_file(geometry_path)
        with rasterio.open(reference_raster) as src:
            raster_crs = src.crs
            transform = src.transform
            out_shape = src.shape

        if geometry_gdf.crs is None:
            geometry_gdf = geometry_gdf.set_crs("EPSG:4326")
        if raster_crs is not None:
            geometry_gdf = geometry_gdf.to_crs(raster_crs)

        return geometry_mask(
            geometry_gdf.geometry,
            transform=transform,
            out_shape=out_shape,
            invert=True,
        )

    def _build_ipyleaflet_draw_map(self):
        """Return an ipyleaflet map configured for drawing scenario polygons."""
        import json as _json

        from ipyleaflet import DrawControl, GeoJSON, LayersControl, Map

        study_area_gdf = gpd.read_file(self.study_area).to_crs("EPSG:4326")
        minx, miny, maxx, maxy = study_area_gdf.total_bounds
        center = [float((miny + maxy) / 2.0), float((minx + maxx) / 2.0)]

        m = Map(center=center, zoom=9, scroll_wheel_zoom=True)
        m.add_layer(GeoJSON(data=_json.loads(study_area_gdf.to_json()), name="study_area"))
        m.add_control(LayersControl(position="topright"))
        m.fit_bounds([[float(miny), float(minx)], [float(maxy), float(maxx)]])

        draw_control = DrawControl()
        draw_control.polygon = {"shapeOptions": {"color": "#2ca25f", "fillOpacity": 0.2}}
        draw_control.rectangle = {"shapeOptions": {"color": "#3182bd", "fillOpacity": 0.15}}
        draw_control.circle = {}
        draw_control.circlemarker = {}
        draw_control.polyline = {}
        draw_control.marker = {}

        m.user_roi = None
        m.st_draw_features = []

        def handle_draw(target, action, geo_json):
            if action in {"deleted", "remove"}:
                m.user_roi = None
                m.st_draw_features = []
            else:
                m.user_roi = geo_json
                m.st_draw_features = [geo_json]

        draw_control.on_draw(handle_draw)
        m.draw_control = draw_control
        m.add_control(draw_control)
        return m

    def _scenario_mask_from_data_array(self, geometry_path, data_array):
        """Rasterize the stored scenario geometry onto an x/y DataArray grid."""
        self.uw.require_existing_path(geometry_path, "Scenario geometry", path_type="file")
        geometry_gdf = gpd.read_file(geometry_path)

        if not hasattr(data_array, "coords") or "x" not in data_array.coords or "y" not in data_array.coords:
            raise ValueError("NDVI DataArray is missing x/y coordinates needed to align the scenario geometry.")

        x = np.asarray(data_array.coords["x"].values, dtype=np.float64)
        y = np.asarray(data_array.coords["y"].values, dtype=np.float64)
        if x.ndim != 1 or y.ndim != 1 or x.size < 2 or y.size < 2:
            raise ValueError("NDVI x/y coordinates must be one-dimensional with at least two cells each.")

        xres = float(np.abs(np.nanmean(np.diff(x))))
        yres = float(np.abs(np.nanmean(np.diff(y))))
        left = float(np.nanmin(x) - (xres / 2.0))
        right = float(np.nanmax(x) + (xres / 2.0))
        bottom = float(np.nanmin(y) - (yres / 2.0))
        top = float(np.nanmax(y) + (yres / 2.0))
        transform = from_bounds(left, bottom, right, top, len(x), len(y))

        if geometry_gdf.crs is None:
            geometry_gdf = geometry_gdf.set_crs("EPSG:4326")

        crs = None
        if hasattr(data_array, "rio") and getattr(data_array.rio, "crs", None) is not None:
            crs = data_array.rio.crs
        if crs is not None:
            geometry_gdf = geometry_gdf.to_crs(crs)

        return geometry_mask(
            geometry_gdf.geometry,
            transform=transform,
            out_shape=(len(y), len(x)),
            invert=True,
        )

    def create_tree_cover_scenario(
        self,
        scenario_name,
        geometry,
        percent_change,
        change_type="deforestation",
    ):
        """Create scenario-specific vegetation and NDVI inputs from a polygon.

        Parameters
        ----------
        scenario_name : str
            Simple directory name used under ``working_dir/scenarios``.
        geometry : GeoDataFrame | dict | str
            Scenario polygon as a GeoDataFrame, GeoJSON-like dict, vector path,
            or WKT string. Only cells intersecting this geometry are modified.
        percent_change : float
            Scenario intensity between 0 and 100. For tree/herb cover this is
            interpreted as a proportional change. For NDVI it controls the size
            of the threshold-crossing shift around ``NDVI = 0.4``.
        change_type : {"deforestation", "reforestation"}
            Direction of change inside the polygon.

        Returns
        -------
        dict
            Scenario metadata including output paths and the NDVI threshold rule.

        Notes
        -----
        This method expects baseline products to already exist under
        ``working_dir/vcf`` and ``working_dir/ndvi``.
        """
        if percent_change < 0 or percent_change > 100:
            raise ValueError("percent_change must be between 0 and 100.")

        change_type = str(change_type).lower().strip()
        if change_type not in {"deforestation", "reforestation"}:
            raise ValueError("change_type must be 'deforestation' or 'reforestation'.")

        paths = self.get_paths(scenario_name)
        paths["scenario_dir"].mkdir(parents=True, exist_ok=True)
        paths["vcf_dir"].mkdir(parents=True, exist_ok=True)
        paths["ndvi_dir"].mkdir(parents=True, exist_ok=True)
        paths["runoff_dir"].mkdir(parents=True, exist_ok=True)
        paths["predicted_streamflow_dir"].mkdir(parents=True, exist_ok=True)

        baseline_tree, baseline_herb, baseline_ndvi = self._require_baseline_scenario_inputs()

        gdf = self._load_geometry(geometry)

        with rasterio.open(baseline_tree) as src_tree, rasterio.open(baseline_herb) as src_herb:
            tree = src_tree.read(1).astype(np.float32)
            herb = src_herb.read(1).astype(np.float32)
            profile = src_tree.profile.copy()
            transform = src_tree.transform
            out_shape = src_tree.shape
            raster_crs = src_tree.crs

            gdf_proj = gdf.to_crs(raster_crs)
            mask = geometry_mask(
                gdf_proj.geometry,
                transform=transform,
                out_shape=out_shape,
                invert=True,
            )

            delta = percent_change / 100.0
            new_tree = tree.copy()
            new_herb = herb.copy()

            if change_type == "deforestation":
                new_tree[mask] = new_tree[mask] * (1.0 - delta)
                new_herb[mask] = new_herb[mask] * (1.0 + delta)
            else:
                new_tree[mask] = new_tree[mask] * (1.0 + delta)
                new_herb[mask] = new_herb[mask] * (1.0 - delta)

            new_tree = np.clip(new_tree, 0.0, 100.0).astype(np.float32)
            new_herb = np.clip(new_herb, 0.0, 100.0).astype(np.float32)

            profile.update(dtype=rasterio.float32, count=1)
            with rasterio.open(paths["tree_cover"], "w", **profile) as dst:
                dst.write(new_tree, 1)
            with rasterio.open(paths["herb_cover"], "w", **profile) as dst:
                dst.write(new_herb, 1)

        with open(baseline_ndvi, "rb") as f:
            baseline_ndvi_data = pickle.load(f)
        scenario_ndvi_data, ndvi_metadata = self._build_ndvi_scenario(
            baseline_ndvi=baseline_ndvi_data,
            mask=mask,
            percent_change=percent_change,
            change_type=change_type,
        )
        with open(paths["ndvi_pickle"], "wb") as f:
            pickle.dump(scenario_ndvi_data, f)

        metadata = ScenarioMetadata({
            "scenario_name": self._normalize_scenario_name(scenario_name),
            "change_type": change_type,
            "percent_change": float(percent_change),
            "tree_cover_tif": str(paths["tree_cover"]),
            "herb_cover_tif": str(paths["herb_cover"]),
            "ndvi_pickle_path": str(paths["ndvi_pickle"]),
            "geometry_path": str(paths["geometry"]),
            "runoff_output_dir": str(paths["runoff_dir"]),
            "predicted_streamflow_dir": str(paths["predicted_streamflow_dir"]),
            "ndvi_threshold_rule": ndvi_metadata,
        })
        geometry_to_save = gdf.to_crs("EPSG:4326")
        geometry_to_save.to_file(paths["geometry"], driver="GeoJSON")
        with open(paths["metadata"], "w", encoding="utf-8") as f:
            json.dump(dict(metadata), f, indent=2)

        return metadata

    def create_land_cover_scenario(
        self,
        scenario_name,
        geometry=None,
        percent_change=0,
        change_type="deforestation",
        map_obj=None,
        open_map_if_missing=True,
    ):
        """Create a scenario from explicit geometry or from a drawn map polygon.

        If ``geometry`` is omitted and ``map_obj`` is provided, the most recent
        drawn polygon on that specific map object is used. If no map is
        supplied and no geometry is provided, ``open_map_if_missing=True``
        returns a new draw map instead of creating the scenario immediately.
        """
        if geometry is None and map_obj is not None:
            geometry = self.get_last_drawn_geometry(map_obj)
            if geometry is None:
                raise ValueError(
                    "No drawn polygon was found on the provided map object. "
                    "Draw a polygon first, then rerun create_land_cover_scenario(...)."
                )

        if geometry is None:
            if open_map_if_missing:
                return self.build_draw_map()
            raise ValueError("No geometry provided and no drawn polygon found.")

        return self.create_tree_cover_scenario(
            scenario_name=scenario_name,
            geometry=geometry,
            percent_change=percent_change,
            change_type=change_type,
        )

    def recompute_runoff(
        self,
        scenario_name,
        sim_start,
        sim_end,
        routing_method="mfd",
        climate_data_source=None,
        force=False,
        resume=False,
    ):
        """Recompute routed runoff using the scenario-specific vegetation inputs.

        Parameters
        ----------
        scenario_name : str
            Name of a previously created scenario.
        sim_start, sim_end : str
            Simulation window in ``YYYY-MM-DD`` format.
        routing_method : {"mfd", "d8", "dinf"}, default "mfd"
            Routing method forwarded to VegET.
        climate_data_source : str, optional
            Climate source override. If omitted, the value passed to
            ``ScenarioManager(...)`` is used.
        force : bool, default False
            If True, delete an existing final runoff file before recomputing.
        resume : bool, default False
            Whether VegET should resume from any checkpoint files in the
            scenario runoff directory.

        Returns
        -------
        str
            Path to the final ``wacc_sparse_arrays.pkl`` file.

        Notes
        -----
        This method consumes the scenario tree cover, herb cover, and NDVI
        climatology written by ``create_tree_cover_scenario``.
        """
        paths = self.get_paths(scenario_name)
        self._require_scenario_inputs(paths)
        self.uw.validate_date_window(sim_start, sim_end, "sim_start", "sim_end")

        final_file = paths["runoff_dir"] / "wacc_sparse_arrays.pkl"
        if force and final_file.exists():
            os.remove(final_file)

        VegET = _load_veget()
        veg = VegET(
            self.working_dir,
            self.study_area,
            sim_start,
            sim_end,
            self._resolve_climate_data_source(climate_data_source),
            routing_method=routing_method,
            runoff_output_dir=str(paths["runoff_dir"]),
            tree_cover_tiff=str(paths["tree_cover"]),
            herb_cover_tiff=str(paths["herb_cover"]),
            ndvi_pickle_path=str(paths["ndvi_pickle"]),
        )
        veg.compute_veget_runoff_route_flow(resume=resume)
        return str(final_file)

    def _validate_model_path(self, model_path):
        """Validate a trained-model checkpoint path before inference."""
        return self.uw.require_existing_path(model_path, "Model checkpoint", path_type="file")

    def _coerce_prediction_array(self, predicted_streamflow):
        """Convert model outputs to a NumPy float array safe for NumPy ops."""
        return np.asarray(predicted_streamflow, dtype=np.float32)

    def _clip_negative_predictions(self, predicted_streamflow):
        """Clamp negative predictions to zero after normalizing dtype."""
        predicted_streamflow = self._coerce_prediction_array(predicted_streamflow)
        return np.maximum(predicted_streamflow, 0.0)

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

    def _load_predictor_classes(self):
        """Load simulation helpers lazily to avoid unnecessary heavy imports."""
        from bakaano.neuralnet.simulate import PredictDataPreprocessor, PredictStreamflow

        return PredictDataPreprocessor, PredictStreamflow

    def simulate_streamflow(
        self,
        scenario_name,
        model_path,
        sim_start,
        sim_end,
        latlist,
        lonlist,
        routing_method="mfd",
        area_normalize=True,
        log_transform=True,
        recompute_runoff=False,
        runoff_resume=True,
    ):
        """Simulate streamflow at user-provided coordinates for one scenario.

        Parameters
        ----------
        scenario_name : str
            Name of a previously created scenario.
        model_path : str
            Trained Bakaano model checkpoint.
        sim_start, sim_end : str
            Simulation window in ``YYYY-MM-DD`` format.
        latlist, lonlist : iterable of float
            Coordinates to simulate. The lists must have equal length.
        routing_method : {"mfd", "d8", "dinf"}, default "mfd"
            Routing method used for predictor generation.
        area_normalize : bool, default True
            Whether model outputs are interpreted as area-normalized and
            converted back to discharge.
        log_transform : bool, default True
            Whether the model was trained with log1p-transformed predictors and
            targets.
        recompute_runoff : bool, default False
            If True, force scenario runoff recomputation before simulation.
        runoff_resume : bool, default True
            Whether scenario runoff recomputation may resume from existing
            checkpoints when this method needs to trigger VegET.

        Returns
        -------
        dict
            Output directory and written CSV files.

        Notes
        -----
        This method writes CSV outputs under the scenario-specific
        ``predicted_streamflow_data`` directory. It does not infer FloodMapper
        outlet coordinates automatically; use explicit lat/lon values.
        """
        self._validate_model_path(model_path)
        paths = self.get_paths(scenario_name)
        self._require_scenario_inputs(paths)
        self.uw.validate_date_window(sim_start, sim_end, "sim_start", "sim_end")
        if not isinstance(latlist, Iterable) or isinstance(latlist, (str, bytes)):
            raise ValueError("latlist must be an iterable of latitude values.")
        if not isinstance(lonlist, Iterable) or isinstance(lonlist, (str, bytes)):
            raise ValueError("lonlist must be an iterable of longitude values.")

        latlist = list(latlist)
        lonlist = list(lonlist)
        if not latlist or not lonlist:
            raise ValueError("latlist and lonlist must not be empty.")
        if len(latlist) != len(lonlist):
            raise ValueError(
                f"latlist and lonlist must have the same length. Received {len(latlist)} and {len(lonlist)}."
            )

        runoff_file = paths["runoff_dir"] / "wacc_sparse_arrays.pkl"
        if recompute_runoff or (not runoff_file.exists()):
            self.recompute_runoff(
                scenario_name=scenario_name,
                sim_start=sim_start,
                sim_end=sim_end,
                routing_method=routing_method,
                climate_data_source=self.climate_data_source,
                force=recompute_runoff,
                resume=runoff_resume,
            )

        paths["predicted_streamflow_dir"].mkdir(parents=True, exist_ok=True)
        PredictDataPreprocessor, PredictStreamflow = self._load_predictor_classes()

        vdp = PredictDataPreprocessor(
            self.working_dir,
            self.study_area,
            sim_start,
            sim_end,
            routing_method,
            runoff_output_dir=str(paths["runoff_dir"]),
        )
        rawdata = vdp.get_data_latlng(latlist, lonlist)

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

        output_files = []
        adjusted_start_date = pd.to_datetime(sim_start) + pd.DateOffset(days=365)
        valid_pairs = [(latlist[i], lonlist[i]) for i in vmodel.valid_entry_indices]
        for predicted_streamflow, (lat, lon) in zip(
            station_preds,
            valid_pairs,
        ):
            predicted_streamflow = self._coerce_prediction_array(predicted_streamflow).reshape(-1)

            period = pd.date_range(adjusted_start_date, periods=len(predicted_streamflow), freq="D")
            df = pd.DataFrame(
                {
                    "time": period,
                    "streamflow (m3/s)": predicted_streamflow,
                }
            )
            output_path = paths["predicted_streamflow_dir"] / (
                f"predicted_streamflow_lat{lat}_lon{lon}.csv"
            )
            df.to_csv(output_path, index=False)
            output_files.append(str(output_path))

        return {
            "output_dir": str(paths["predicted_streamflow_dir"]),
            "files": output_files,
        }

    def simulate_grdc_csv_stations(
        self,
        scenario_name,
        model_path,
        sim_start,
        sim_end,
        grdc_netcdf=None,
        routing_method="mfd",
        csv_dir=None,
        lookup_csv=None,
        id_col="id",
        lat_col="latitude",
        lon_col="longitude",
        date_col="date",
        discharge_col="discharge",
        file_pattern="{id}.csv",
        area_normalize=True,
        log_transform=True,
        recompute_runoff=False,
        runoff_resume=True,
    ):
        """Simulate streamflow for GRDC or CSV stations for one scenario.

        Provide exactly one station source:
        - ``grdc_netcdf``, or
        - ``csv_dir`` plus ``lookup_csv``.

        Returns a dictionary with the scenario output directory and all written
        station CSV files. Outputs are isolated under the scenario directory and
        do not overwrite baseline simulation files.

        Parameters
        ----------
        runoff_resume : bool, default True
            Whether scenario runoff recomputation may resume from existing
            checkpoints when this method needs to trigger VegET.
        """
        self._validate_model_path(model_path)
        paths = self.get_paths(scenario_name)
        self._require_scenario_inputs(paths)
        self.uw.validate_date_window(sim_start, sim_end, "sim_start", "sim_end")
        csv_mode = bool(csv_dir and lookup_csv)
        grdc_mode = grdc_netcdf is not None
        if csv_mode == grdc_mode:
            raise ValueError(
                "Provide exactly one station source for simulation: "
                "either grdc_netcdf or csv_dir+lookup_csv."
            )
        if grdc_netcdf is not None:
            self.uw.require_existing_path(grdc_netcdf, "GRDC NetCDF", path_type="file")
        if csv_dir is not None:
            self.uw.require_existing_path(csv_dir, "Observed streamflow CSV directory", path_type="dir")
        if lookup_csv is not None:
            self.uw.require_existing_path(lookup_csv, "Observed streamflow lookup CSV", path_type="file")

        runoff_file = paths["runoff_dir"] / "wacc_sparse_arrays.pkl"
        if recompute_runoff or (not runoff_file.exists()):
            self.recompute_runoff(
                scenario_name=scenario_name,
                sim_start=sim_start,
                sim_end=sim_end,
                routing_method=routing_method,
                climate_data_source=self.climate_data_source,
                force=recompute_runoff,
                resume=runoff_resume,
            )

        paths["predicted_streamflow_dir"].mkdir(parents=True, exist_ok=True)
        PredictDataPreprocessor, PredictStreamflow = self._load_predictor_classes()

        vdp = PredictDataPreprocessor(
            self.working_dir,
            self.study_area,
            sim_start,
            sim_end,
            routing_method,
            grdc_netcdf,
            runoff_output_dir=str(paths["runoff_dir"]),
        )
        if csv_mode:
            vdp.load_observed_streamflow_from_csv_dir(
                csv_dir=csv_dir,
                lookup_csv=lookup_csv,
                id_col=id_col,
                lat_col=lat_col,
                lon_col=lon_col,
                date_col=date_col,
                discharge_col=discharge_col,
                file_pattern=file_pattern,
            )

        rawdata = vdp.get_data()
        vmodel = PredictStreamflow(
            self.working_dir,
            area_normalize=area_normalize,
            log_transform=log_transform,
        )
        vmodel.load_model_config(model_path)
        vmodel.prepare_data(rawdata)
        vmodel.load_model(model_path)

        batch_size = max(1, len(vdp.station_ids))
        station_preds = vmodel.predict_station_series(
            batch_size=batch_size,
            area_normalize=vmodel.area_normalize,
        )

        output_files = []
        adjusted_start_date = pd.to_datetime(sim_start) + pd.DateOffset(days=365)
        valid_station_ids = [vdp.station_ids[i] for i in vmodel.valid_entry_indices]
        for predicted_streamflow, station_id in zip(
            station_preds,
            valid_station_ids,
        ):
            predicted_streamflow = self._coerce_prediction_array(predicted_streamflow).reshape(-1)

            period = pd.date_range(adjusted_start_date, periods=len(predicted_streamflow), freq="D")
            df = pd.DataFrame(
                {
                    "time": period,
                    "streamflow (m3/s)": predicted_streamflow,
                }
            )
            output_path = paths["predicted_streamflow_dir"] / f"bakaano_{station_id}.csv"
            df.to_csv(output_path, index=False)
            output_files.append(str(output_path))

        return {
            "output_dir": str(paths["predicted_streamflow_dir"]),
            "files": output_files,
        }

    def build_draw_map(self, backend="ipyleaflet"):
        """Return an interactive map with draw controls enabled.

        The map is a convenience UI for collecting a polygon. The drawn
        geometry is not stored globally; pass the returned map object back to
        ``create_land_cover_scenario(..., map_obj=...)`` when you are ready to
        create the scenario.
        """
        backend = str(backend).lower().strip()
        if backend == "ipyleaflet":
            return self._build_ipyleaflet_draw_map()
        if backend != "leafmap":
            raise ValueError("backend must be 'ipyleaflet' or 'leafmap'.")

        from leafmap.foliumap import Map

        m = Map()
        try:
            study_area = gpd.read_file(self.study_area)
            m.add_gdf(study_area, layer_name="study_area")
        except Exception:
            pass

        if hasattr(m, "add_draw_control"):
            m.add_draw_control()
        return m

    def plot_scenario_change(self, scenario_name, figsize=(15, 4)):
        """Plot baseline, scenario, and difference rasters for tree cover.

        Parameters
        ----------
        scenario_name : str
            Name of a previously created scenario.
        figsize : tuple, default (15, 4)
            Matplotlib figure size.

        Returns
        -------
        tuple
            ``(fig, axes)`` for additional customization.

        Notes
        -----
        The scenario polygon boundary is overlaid when available.
        """
        paths = self.get_paths(scenario_name)
        baseline_path = Path(self.working_dir) / "vcf" / "mean_tree_cover.tif"
        scenario_path = paths["tree_cover"]

        self.uw.require_existing_path(baseline_path, "Baseline tree-cover raster", path_type="file")
        self.uw.require_existing_path(scenario_path, "Scenario tree-cover raster", path_type="file")

        with rasterio.open(baseline_path) as src_base, rasterio.open(scenario_path) as src_scenario:
            baseline = src_base.read(1).astype(np.float32)
            scenario = src_scenario.read(1).astype(np.float32)
            bounds = src_base.bounds
            raster_crs = src_base.crs
            nodata = src_base.nodata

        if nodata is not None and np.isfinite(nodata):
            baseline = np.where(baseline == nodata, np.nan, baseline)
            scenario = np.where(scenario == nodata, np.nan, scenario)
        value_label = "Percent cover"
        main_cmap = "Greens"
        value_limits = (0.0, 100.0)
        diff_label = "Percent points"

        diff = scenario - baseline
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

        fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
        baseline_img = axes[0].imshow(
            baseline, extent=extent, origin="upper", cmap=main_cmap, vmin=value_limits[0], vmax=value_limits[1]
        )
        axes[0].set_title("Baseline tree")
        scenario_img = axes[1].imshow(
            scenario, extent=extent, origin="upper", cmap=main_cmap, vmin=value_limits[0], vmax=value_limits[1]
        )
        axes[1].set_title("Scenario tree")
        max_abs = float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else 1.0
        diff_img = axes[2].imshow(
            diff,
            extent=extent,
            origin="upper",
            cmap="RdBu_r",
            vmin=-max(max_abs, 1.0),
            vmax=max(max_abs, 1.0),
        )
        axes[2].set_title("Tree-cover change")

        if paths["geometry"].exists():
            geometry_gdf = gpd.read_file(paths["geometry"])
            if geometry_gdf.crs is None:
                geometry_gdf = geometry_gdf.set_crs("EPSG:4326")
            if raster_crs is not None:
                geometry_gdf = geometry_gdf.to_crs(raster_crs)
            for ax in axes:
                geometry_gdf.boundary.plot(ax=ax, color="black", linewidth=1)

        for ax in axes:
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        fig.colorbar(baseline_img, ax=axes[0], fraction=0.046, pad=0.04, label=value_label)
        fig.colorbar(scenario_img, ax=axes[1], fraction=0.046, pad=0.04, label=value_label)
        fig.colorbar(diff_img, ax=axes[2], fraction=0.046, pad=0.04, label=diff_label)
        return fig, axes

    def plot_ndvi_scenario_timeseries(self, scenario_name, figsize=(10, 4)):
        """Plot baseline vs scenario NDVI seasonal means inside the scenario area."""
        paths = self.get_paths(scenario_name)
        baseline_path = Path(self.working_dir) / "ndvi" / "daily_ndvi_climatology.pkl"
        scenario_path = paths["ndvi_pickle"]

        self.uw.require_existing_path(baseline_path, "Baseline NDVI climatology", path_type="file")
        self.uw.require_existing_path(scenario_path, "Scenario NDVI climatology", path_type="file")

        with open(baseline_path, "rb") as f:
            baseline_ndvi = pickle.load(f)
        with open(scenario_path, "rb") as f:
            scenario_ndvi = pickle.load(f)

        common_days = sorted(set(int(day) for day in baseline_ndvi).intersection(int(day) for day in scenario_ndvi))
        if not common_days:
            raise ValueError("No overlapping day-of-year entries found between baseline and scenario NDVI climatologies.")
        mask = self._scenario_mask_from_data_array(paths["geometry"], baseline_ndvi[common_days[0]])

        baseline_mean = []
        scenario_mean = []
        diff_mean = []
        changed_area_fraction = []

        for day in common_days:
            baseline_values = np.asarray(getattr(baseline_ndvi[day], "values", baseline_ndvi[day]), dtype=np.float32) * 0.0001
            scenario_values = np.asarray(getattr(scenario_ndvi[day], "values", scenario_ndvi[day]), dtype=np.float32) * 0.0001

            baseline_masked = np.where(mask, baseline_values, np.nan)
            scenario_masked = np.where(mask, scenario_values, np.nan)
            diff_masked = scenario_masked - baseline_masked

            baseline_mean.append(float(np.nanmean(baseline_masked)))
            scenario_mean.append(float(np.nanmean(scenario_masked)))
            diff_mean.append(float(np.nanmean(diff_masked)))
            changed_area_fraction.append(float(np.nanmean(np.abs(diff_masked) > 1e-6) * 100.0))

        fig, axes = plt.subplots(2, 1, figsize=figsize, constrained_layout=True, sharex=True, height_ratios=[3, 1.4])

        axes[0].plot(common_days, baseline_mean, label="Baseline NDVI", color="#4c78a8", linewidth=2)
        axes[0].plot(common_days, scenario_mean, label="Scenario NDVI", color="#2ca02c", linewidth=2)
        axes[0].set_ylabel("Mean NDVI")
        axes[0].set_title("Scenario NDVI seasonal comparison")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(common_days, diff_mean, label="Scenario - baseline", color="#c44e52", linewidth=1.8)
        axes[1].fill_between(common_days, diff_mean, 0, color="#c44e52", alpha=0.15)
        axes[1].set_xlabel("Day of year")
        axes[1].set_ylabel("NDVI change")
        axes[1].grid(True, alpha=0.3)

        ax2 = axes[1].twinx()
        ax2.plot(common_days, changed_area_fraction, color="#8172b2", linestyle="--", linewidth=1.5, label="Changed area (%)")
        ax2.set_ylabel("Changed area (%)")

        lines1, labels1 = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1].legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        return fig, axes

    def get_last_drawn_geometry(self, map_obj):
        """Extract the most recent drawn geometry from a Leafmap map object."""
        # Leafmap/Streamlit style: single last draw.
        for attr in ("st_last_draw", "user_roi"):
            value = getattr(map_obj, attr, None)
            geom = self._coerce_geometry_dict(value)
            if geom is not None:
                return geom

        # Leafmap/Streamlit style: list of drawn features.
        drawn = getattr(map_obj, "st_draw_features", None)
        if isinstance(drawn, list) and drawn:
            for item in reversed(drawn):
                geom = self._coerce_geometry_dict(item)
                if geom is not None:
                    return geom

        return None

    def _coerce_geometry_dict(self, value):
        """Convert GeoJSON-like objects to a geometry dict."""
        if value is None:
            return None

        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                return None

        if isinstance(value, dict):
            vtype = value.get("type")
            if vtype == "Feature":
                return value.get("geometry")
            if vtype == "FeatureCollection":
                feats = value.get("features", [])
                if feats:
                    return feats[-1].get("geometry")
                return None
            if vtype in {"Polygon", "MultiPolygon"}:
                return value

        if hasattr(value, "__geo_interface__"):
            geo = value.__geo_interface__
            if isinstance(geo, dict):
                if geo.get("type") == "Feature":
                    return geo.get("geometry")
                return geo

        return None

    def _load_geometry(self, geometry):
        """Load user geometry into a GeoDataFrame (EPSG:4326)."""
        if isinstance(geometry, gpd.GeoDataFrame):
            gdf = geometry.copy()
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            return gdf

        if isinstance(geometry, dict):
            if geometry.get("type") == "FeatureCollection":
                gdf = gpd.GeoDataFrame.from_features(geometry)
            elif geometry.get("type") == "Feature":
                gdf = gpd.GeoDataFrame.from_features([geometry])
            else:
                gdf = gpd.GeoDataFrame(geometry=[shape(geometry)], crs="EPSG:4326")
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            return gdf

        if isinstance(geometry, str):
            if os.path.exists(geometry):
                gdf = gpd.read_file(geometry)
                if gdf.crs is None:
                    gdf = gdf.set_crs("EPSG:4326")
                return gdf
            # Fallback: treat string as WKT.
            geom = wkt.loads(geometry)
            return gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")

        raise TypeError("geometry must be a GeoDataFrame, dict/GeoJSON, path, or WKT string.")
