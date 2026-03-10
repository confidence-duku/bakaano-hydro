"""Scenario utilities for land-cover change experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import shape
from shapely import wkt


class ScenarioManager:
    """Create and manage vegetation-change scenarios."""

    def __init__(self, working_dir, study_area):
        self.working_dir = working_dir
        self.study_area = study_area
        self.scenarios_root = Path(working_dir) / "scenarios"
        self.scenarios_root.mkdir(parents=True, exist_ok=True)

    def get_paths(self, scenario_name):
        scenario_dir = self.scenarios_root / str(scenario_name)
        vcf_dir = scenario_dir / "vcf"
        runoff_dir = scenario_dir / "runoff_output"
        return {
            "scenario_dir": scenario_dir,
            "vcf_dir": vcf_dir,
            "runoff_dir": runoff_dir,
            "tree_cover": vcf_dir / "mean_tree_cover.tif",
            "herb_cover": vcf_dir / "mean_herb_cover.tif",
            "metadata": scenario_dir / "scenario_metadata.json",
        }

    def create_tree_cover_scenario(self, scenario_name, geometry, percent_change, change_type="deforestation"):
        """Create scenario-specific tree/herb cover rasters from a user geometry."""
        if percent_change < 0 or percent_change > 100:
            raise ValueError("percent_change must be between 0 and 100.")

        change_type = str(change_type).lower().strip()
        if change_type not in {"deforestation", "reforestation"}:
            raise ValueError("change_type must be 'deforestation' or 'reforestation'.")

        paths = self.get_paths(scenario_name)
        paths["scenario_dir"].mkdir(parents=True, exist_ok=True)
        paths["vcf_dir"].mkdir(parents=True, exist_ok=True)
        paths["runoff_dir"].mkdir(parents=True, exist_ok=True)

        baseline_tree = Path(self.working_dir) / "vcf" / "mean_tree_cover.tif"
        baseline_herb = Path(self.working_dir) / "vcf" / "mean_herb_cover.tif"
        if not baseline_tree.exists() or not baseline_herb.exists():
            raise FileNotFoundError(
                "Baseline tree/herb cover rasters not found in working_dir/vcf. "
                "Run tree cover preprocessing first."
            )

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

        metadata = {
            "scenario_name": str(scenario_name),
            "change_type": change_type,
            "percent_change": float(percent_change),
            "tree_cover_tif": str(paths["tree_cover"]),
            "herb_cover_tif": str(paths["herb_cover"]),
            "runoff_output_dir": str(paths["runoff_dir"]),
        }
        with open(paths["metadata"], "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def build_draw_map(self):
        """Return an interactive map with draw controls for scenario polygons."""
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
