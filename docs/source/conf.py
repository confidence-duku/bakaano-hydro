import os
import sys

project = "Bakaano-Hydro"
author = "Confidence Duku"
release = "1.3.1"

# Add project root to sys.path for autodoc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]

autosummary_generate = True

# Mock heavy optional dependencies for docs build
autodoc_mock_imports = [
    "numpy",
    "pandas",
    "tensorflow",
    "tensorflow_probability",
    "tf_keras",
    "keras",
    "tcn",
    "geemap",
    "leafmap",
    "localtileserver",
    "earthengine_api",
    "ee",
    "rasterio",
    "rioxarray",
    "fiona",
    "geopandas",
    "pysheds",
    "xarray",
    "netCDF4",
    "scipy",
    "sklearn",
    "matplotlib",
    "hydroeval",
    "shapely",
    "dask",
    "requests",
    "isimip_client",
    "numba",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "press"
html_static_path = ["_static"]
html_theme_options = {}
html_css_files = ["press_custom.css"]


def _ensure_press_toc_dict(app, env):
    if not hasattr(env, "toc_dict"):
        return
    for docname in env.found_docs:
        env.toc_dict.setdefault(docname, {"sections": [], "toctrees": []})


def setup(app):
    app.connect("env-updated", _ensure_press_toc_dict)
