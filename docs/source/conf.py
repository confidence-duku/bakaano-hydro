import os
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

project = "Bakaano-Hydro"
author = "Confidence Duku"


def _package_version():
    init_path = Path(__file__).resolve().parents[2] / "bakaano" / "__init__.py"
    init_text = init_path.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"Could not find __version__ in {init_path}")
    return match.group(1)


release = _package_version()
version = release

# Add project root to sys.path for autodoc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Keras can initialize an installed TensorFlow runtime before autodoc's import
# hook takes effect. Pre-seeding these modules keeps API documentation builds
# deterministic and independent of CUDA availability.
for module_name in (
    "tensorflow",
    "tensorflow.keras",
    "tensorflow.keras.callbacks",
    "tensorflow.keras.layers",
    "tensorflow.keras.models",
    "tensorflow.keras.utils",
    "tensorflow_probability",
    "tf_keras",
    "keras",
    "keras.models",
    "keras.utils",
    "tcn",
):
    sys.modules[module_name] = MagicMock(name=module_name)

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
    "tensorflow.keras",
    "tensorflow.keras.models",
    "tensorflow.keras.layers",
    "tensorflow.keras.callbacks",
    "tensorflow.keras.utils",
    "tensorflow_probability",
    "tf_keras",
    "keras",
    "keras.models",
    "keras.utils",
    "tcn",
    "geemap",
    "leafmap",
    "leafmap.foliumap",
    "localtileserver",
    "earthengine_api",
    "ee",
    "rasterio",
    "rasterio.transform",
    "rioxarray",
    "fiona",
    "geopandas",
    "pysheds",
    "pysheds.grid",
    "pyproj",
    "whitebox",
    "xarray",
    "netCDF4",
    "scipy",
    "scipy.spatial",
    "scipy.spatial.distance",
    "sklearn",
    "sklearn.preprocessing",
    "matplotlib",
    "matplotlib.pyplot",
    "hydroeval",
    "shapely",
    "shapely.geometry",
    "dask",
    "requests",
    "tqdm",
    "isimip_client",
    "numba",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "press"
html_static_path = ["_static"]
html_theme_options = {}
html_css_files = ["press_custom.css"]
html_baseurl = os.environ.get("DOCS_BASE_URL", "").strip()


def _ensure_press_toc_dict(app, env):
    if not hasattr(env, "toc_dict"):
        return
    for docname in env.found_docs:
        env.toc_dict.setdefault(docname, {"sections": [], "toctrees": []})


def setup(app):
    app.connect("env-updated", _ensure_press_toc_dict)
