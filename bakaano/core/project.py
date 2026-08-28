"""Project-level helpers for Bakaano working directories.

Role: Provide shared project context, path discovery, and readiness checks.
"""

from __future__ import annotations

import os
from collections import UserDict
from html import escape
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
try:
    from IPython.display import HTML, display
except ImportError:  # pragma: no cover - optional dependency fallback
    HTML = display = None

from bakaano.core.utils import Utils


class WorkflowOverview(UserDict):
    """Mapping-like workflow summary with reliable notebook display hooks."""

    def __str__(self):
        return self._to_text()

    def __repr__(self):
        return self._to_text()

    def _repr_pretty_(self, printer, cycle):
        if cycle:
            printer.text("WorkflowOverview(...)")
            return
        printer.text(self._to_text())

    def _ipython_display_(self):
        if HTML is not None:
            display(HTML(self._repr_html_()))
            return
        print(self._to_text())

    def _repr_mimebundle_(self, include=None, exclude=None):
        return {
            "text/plain": self._to_text(),
            "text/html": self._repr_html_(),
        }

    def _repr_html_(self):
        main_entry = self.get("main_entry_point") or self.get("project_helper", "")
        workflow_df = pd.DataFrame(
            [
                {
                    "Step": item.get("step"),
                    "Goal": item.get("goal"),
                    "Methods": ", ".join(str(method) for method in item.get("methods", [])),
                }
                for item in self.get("normal_workflow", [])
            ]
        )
        helper_df = pd.DataFrame(
            {"Method": [str(method) for method in self.get("normal_user_methods", [])]}
        )
        extension_df = pd.DataFrame(
            [
                {"Extension": str(name), "Module": str(target)}
                for name, target in self.get("advanced_extensions", {}).items()
            ]
        )

        sections = [f"<p><strong>Entry point:</strong> <code>{escape(str(main_entry))}</code></p>"]
        if not workflow_df.empty:
            sections.append("<h4>Workflow Steps</h4>")
            sections.append(workflow_df.to_html(index=False, escape=True, border=0))
        if not helper_df.empty:
            sections.append("<h4>Core Helper Methods</h4>")
            sections.append(helper_df.to_html(index=False, escape=True, border=0))
        if not extension_df.empty:
            sections.append("<h4>Advanced Extensions</h4>")
            sections.append(extension_df.to_html(index=False, escape=True, border=0))
        return "".join(sections)

    def _to_text(self):
        lines = ["Bakaano workflow overview"]

        main_entry = self.get("main_entry_point")
        if main_entry:
            lines.append(f"Entry point: {main_entry}")

        project_helper = self.get("project_helper")
        if project_helper:
            lines.append(f"Project helper: {project_helper}")

        lines.append("")
        lines.append("Recommended sequence")
        for item in self.get("normal_workflow", []):
            lines.append(f"{item.get('step')}. {item.get('goal')}")
            for method in item.get("methods", []):
                lines.append(f"   - {method}")

        helper_methods = self.get("normal_user_methods", [])
        if helper_methods:
            lines.append("")
            lines.append("Core helper methods")
            for method in helper_methods:
                lines.append(f" - {method}")

        extensions = self.get("advanced_extensions", {})
        if extensions:
            lines.append("")
            lines.append("Advanced extensions")
            for name, target in extensions.items():
                lines.append(f" - {name}: {target}")

        return "\n".join(lines)


class ProjectPathsView(UserDict):
    """Mapping-like project path listing with reliable notebook display."""

    def __str__(self):
        lines = ["Bakaano project paths"]
        for key, value in self.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    def _repr_pretty_(self, printer, cycle):
        if cycle:
            printer.text("ProjectPathsView(...)")
            return
        printer.text(str(self))

    def _repr_html_(self):
        paths_df = pd.DataFrame(
            [{"Name": str(name), "Path": str(value)} for name, value in self.items()]
        )
        return paths_df.to_html(index=False, escape=True, border=0)

    def _ipython_display_(self):
        if HTML is not None:
            display(HTML(self._repr_html_()))
            return
        print(str(self))

    def _repr_mimebundle_(self, include=None, exclude=None):
        return {
            "text/plain": str(self),
            "text/html": self._repr_html_(),
        }


class ProjectStatusFrame(pd.DataFrame):
    """DataFrame subclass with notebook HTML display for project artifact status."""

    @property
    def _constructor(self):
        return ProjectStatusFrame

    def _repr_html_(self):
        status_df = self.copy()
        if "detail" in status_df.columns:
            status_df["detail"] = status_df["detail"].replace("", "-")

        sections = [status_df.to_html(index=False, escape=True, border=0)]
        scenarios = self.attrs.get("available_scenarios", [])
        if scenarios:
            scenarios_df = pd.DataFrame({"scenario": [str(scenario) for scenario in scenarios]})
            sections.append("<h4>Available scenarios</h4>")
            sections.append(scenarios_df.to_html(index=False, escape=True, border=0))
        return "".join(sections)

    def _ipython_display_(self):
        if HTML is not None:
            display(HTML(self._repr_html_()))
            return
        print(self.to_string(index=False))


class ProjectContext:
    """Lightweight project helper for shared paths and readiness checks."""

    def __init__(self, working_dir, study_area, climate_data_source):
        if working_dir is None or str(working_dir).strip() == "":
            raise ValueError("working_dir must be a non-empty path.")
        if study_area is None or str(study_area).strip() == "":
            raise ValueError("study_area must be a path to a basin or watershed shapefile.")

        climate_data_source = str(climate_data_source).upper()
        valid_sources = {"CHELSA", "ERA5", "CHIRPS"}
        if climate_data_source not in valid_sources:
            raise ValueError(
                "climate_data_source must be one of CHELSA, ERA5, or CHIRPS. "
                f"Received: {climate_data_source}"
            )

        self.working_dir = os.fspath(working_dir)
        self.study_area = os.fspath(study_area)
        self.climate_data_source = climate_data_source
        self.uw = Utils(self.working_dir, self.study_area)
        self.uw.require_study_area_gdf()

        os.makedirs(f"{self.working_dir}/models", exist_ok=True)
        os.makedirs(f"{self.working_dir}/runoff_output", exist_ok=True)
        os.makedirs(f"{self.working_dir}/scratch", exist_ok=True)
        os.makedirs(f"{self.working_dir}/shapes", exist_ok=True)
        os.makedirs(f"{self.working_dir}/catchment", exist_ok=True)
        os.makedirs(f"{self.working_dir}/predicted_streamflow_data", exist_ok=True)

    def project_paths(self):
        """Return the standard project paths used by the package."""
        working_dir = Path(self.working_dir)
        return ProjectPathsView({
            "working_dir": str(working_dir),
            "study_area": self.study_area,
            "dem": str(working_dir / "elevation" / "dem_clipped.tif"),
            "soil_dir": str(working_dir / "soil"),
            "vcf_dir": str(working_dir / "vcf"),
            "ndvi": str(working_dir / "ndvi" / "daily_ndvi_climatology.pkl"),
            "meteo_dir": str(working_dir / self.climate_data_source),
            "alpha_earth_dir": str(working_dir / "alpha_earth"),
            "runoff": str(working_dir / "runoff_output" / "wacc_sparse_arrays.pkl"),
            "models_dir": str(working_dir / "models"),
            "model": str(working_dir / "models" / "bakaano_model.keras"),
            "predicted_streamflow_dir": str(working_dir / "predicted_streamflow_data"),
            "flood_dir": str(working_dir / "flood"),
            "scenarios_dir": str(working_dir / "scenarios"),
        })

    def workflow_overview(self):
        """Return the recommended module-level workflow."""
        return WorkflowOverview({
            "project_helper": "bakaano.core.project.ProjectContext",
            "normal_workflow": [
                {
                    "step": 1,
                    "goal": "Inspect project layout and readiness",
                    "methods": ["project_paths", "project_status", "validate_project"],
                },
                {
                    "step": 2,
                    "goal": "Preprocess DEM, vegetation, NDVI, soil, meteorology, and AlphaEarth inputs",
                    "methods": [
                        "bakaano.data.dem.DEM",
                        "bakaano.data.tree_cover.TreeCover",
                        "bakaano.data.ndvi.NDVI",
                        "bakaano.data.soil.Soil",
                        "bakaano.data.meteo.Meteo",
                        "bakaano.data.alpha_earth.AlphaEarth",
                    ],
                },
                {
                    "step": 3,
                    "goal": "Compute runoff and routing",
                    "methods": ["bakaano.hydrology.veget.VegET"],
                },
                {
                    "step": 4,
                    "goal": "Train the streamflow model",
                    "methods": ["bakaano.neuralnet.train.train_streamflow_model"],
                },
                {
                    "step": 5,
                    "goal": "Evaluate and simulate streamflow",
                    "methods": [
                        "bakaano.neuralnet.simulate.evaluate_streamflow_model_interactively",
                        "bakaano.neuralnet.simulate.simulate_streamflow",
                        "bakaano.neuralnet.simulate.simulate_grdc_csv_stations",
                    ],
                },
            ],
            "advanced_extensions": {
                "flood_mapping": "bakaano.extensions.flood_mapper.FloodMapper",
                "scenarios": "bakaano.extensions.scenario.ScenarioManager",
            },
        })

    def _project_artifact_map(self):
        """Return core project artifacts tracked by project_status()."""
        working_dir = Path(self.working_dir)
        meteo_dir = working_dir / self.climate_data_source
        if self.climate_data_source == "CHELSA":
            meteo_artifacts = {
                "meteo_precip": {"path": meteo_dir / "prep", "type": "dir"},
                "meteo_tasmax": {"path": meteo_dir / "tasmax", "type": "dir"},
                "meteo_tasmin": {"path": meteo_dir / "tasmin", "type": "dir"},
                "meteo_tmean": {"path": meteo_dir / "tmean", "type": "dir"},
            }
        else:
            meteo_artifacts = {
                "meteo_precip": {"path": meteo_dir / "prep" / "pr.nc", "type": "file"},
                "meteo_tasmax": {"path": meteo_dir / "tasmax" / "tasmax.nc", "type": "file"},
                "meteo_tasmin": {"path": meteo_dir / "tasmin" / "tasmin.nc", "type": "file"},
                "meteo_tmean": {"path": meteo_dir / "tmean" / "tas.nc", "type": "file"},
            }
        return {
            "study_area": {"path": Path(self.study_area), "type": "file"},
            "dem": {"path": working_dir / "elevation" / "dem_clipped.tif", "type": "file"},
            "soil": {"path": working_dir / "soil" / "clipped_AWCh3_M_sl6_1km_ll.tif", "type": "file"},
            "tree_cover": {"path": working_dir / "vcf" / "mean_tree_cover.tif", "type": "file"},
            "herb_cover": {"path": working_dir / "vcf" / "mean_herb_cover.tif", "type": "file"},
            "ndvi": {"path": working_dir / "ndvi" / "daily_ndvi_climatology.pkl", "type": "file"},
            **meteo_artifacts,
            "alpha_earth_dir": {"path": working_dir / "alpha_earth", "type": "dir"},
            "runoff": {"path": working_dir / "runoff_output" / "wacc_sparse_arrays.pkl", "type": "file"},
            "model": {"path": working_dir / "models" / "bakaano_model.keras", "type": "file"},
            "flood_rating_curves": {"path": working_dir / "flood" / "rating_curves.pkl", "type": "file"},
            "scenarios_dir": {"path": working_dir / "scenarios", "type": "dir"},
        }

    def _check_artifact(self, path, path_type):
        """Return a lightweight existence/readability status for one artifact."""
        path_obj = Path(path)
        exists = path_obj.is_dir() if path_type == "dir" else path_obj.is_file()
        status = "ok" if exists else "missing"
        detail = ""
        if exists and path_type == "file":
            try:
                if path_obj.suffix.lower() in {".tif", ".tiff"}:
                    with rasterio.open(path_obj):
                        pass
                elif path_obj.suffix.lower() == ".shp":
                    gpd.read_file(path_obj)
                elif path_obj.suffix.lower() == ".pkl":
                    with open(path_obj, "rb"):
                        pass
            except Exception as exc:
                status = "unreadable"
                detail = str(exc)
        return {
            "path": str(path_obj),
            "type": path_type,
            "status": status,
            "detail": detail,
        }

    def project_status(self):
        """Summarize which preprocessing and model artifacts exist."""
        rows = []
        for artifact, meta in self._project_artifact_map().items():
            result = self._check_artifact(meta["path"], meta["type"])
            result["artifact"] = artifact
            rows.append(result)

        scenarios_dir = Path(self.working_dir) / "scenarios"
        available_scenarios = []
        if scenarios_dir.is_dir():
            available_scenarios = sorted(
                p.name for p in scenarios_dir.iterdir() if p.is_dir()
            )

        df = ProjectStatusFrame(rows, columns=["artifact", "status", "type", "path", "detail"])
        df.attrs["available_scenarios"] = available_scenarios
        return df

    def validate_project(self, for_task="preprocess"):
        """Validate project readiness for a specific workflow task."""
        task = str(for_task).lower().strip()
        requirements = {
            "preprocess": ["study_area"],
            "train": [
                "study_area",
                "dem",
                "soil",
                "tree_cover",
                "herb_cover",
                "ndvi",
                "meteo_precip",
                "meteo_tasmax",
                "meteo_tasmin",
                "meteo_tmean",
                "alpha_earth_dir",
                "runoff",
            ],
            "evaluate": ["study_area", "dem", "runoff", "model"],
            "simulate": ["study_area", "dem", "runoff", "model"],
            "flood": ["study_area", "dem", "model"],
            "scenario": ["study_area", "dem", "tree_cover", "herb_cover", "ndvi"],
        }
        remediation = {
            "preprocess": "Add a valid study-area shapefile and initialize ProjectContext.",
            "train": "Run DEM, tree cover, NDVI, soil, meteorological preprocessing, AlphaEarth preparation, and runoff routing before training.",
            "evaluate": "Generate runoff outputs and train a model before evaluation.",
            "simulate": "Generate runoff outputs and train a model before simulation.",
            "flood": "Prepare the DEM and train a model first. Rating curves can be generated automatically later by FloodMapper.",
            "scenario": "Run vegetation and NDVI preprocessing first so baseline scenario inputs exist.",
        }
        if task not in requirements:
            valid = ", ".join(sorted(requirements))
            raise ValueError(f"Unknown validation task '{for_task}'. Expected one of: {valid}.")

        status_df = self.project_status()
        status_lookup = {
            row["artifact"]: row for _, row in status_df.iterrows()
        }
        missing = []
        for artifact in requirements[task]:
            row = status_lookup.get(artifact)
            if row is None or row["status"] != "ok":
                missing.append(artifact)

        if missing:
            lines = []
            for artifact in missing:
                row = status_lookup.get(artifact)
                if row is None:
                    lines.append(f"  - {artifact}: not tracked")
                else:
                    lines.append(f"  - {artifact}: {row['status']} ({row['path']})")
            raise FileNotFoundError(
                f"Project is not ready for task '{task}'. Missing or unreadable artifacts:\n"
                + "\n".join(lines)
                + "\n"
                + remediation[task]
            )

        return status_df
