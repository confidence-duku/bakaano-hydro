"""Fast checks for packaging and public API regressions."""

import ast
import importlib
from importlib.metadata import requires, version
import json
import sys
from pathlib import Path

import bakaano


ROOT = Path(__file__).resolve().parents[1]


def test_package_version_matches_project_metadata():
    assert bakaano.__version__ == version("bakaano-hydro")


def test_runtime_dependencies_cover_direct_accelerator_and_crs_imports():
    dependencies = "\n".join(requires("bakaano-hydro") or []).lower()
    assert "numba" in dependencies
    assert "pyproj" in dependencies


def test_subpackage_imports_are_lazy():
    heavy_modules = {"tensorflow", "ee", "geopandas", "rasterio"}
    before = heavy_modules & sys.modules.keys()
    for module_name in ("bakaano.data", "bakaano.extensions", "bakaano.hydrology", "bakaano.neuralnet"):
        importlib.import_module(module_name)
    after = heavy_modules & sys.modules.keys()
    assert after == before


def test_canonical_entry_points_remain_defined():
    expected = {
        ROOT / "bakaano/neuralnet/train.py": {"train_streamflow_model"},
        ROOT / "bakaano/neuralnet/simulate.py": {
            "evaluate_streamflow_model_interactively",
            "simulate_grdc_csv_stations",
            "simulate_streamflow",
        },
    }
    for path, names in expected.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        defined = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assert names <= defined


def test_notebooks_are_v4_json_without_saved_errors():
    for path in ROOT.glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        errors = [
            output
            for cell in notebook.get("cells", [])
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        assert not errors, f"{path.name} contains saved error output"
