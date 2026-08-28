"""Input-data preparation and preprocessing modules with lazy exports."""

from importlib import import_module

__all__ = ["AlphaEarth", "DEM", "Meteo", "NDVI", "Soil", "TreeCover"]

_EXPORTS = {
    name: (f"bakaano.data.{module}", name)
    for name, module in {
        "AlphaEarth": "alpha_earth", "DEM": "dem", "Meteo": "meteo",
        "NDVI": "ndvi", "Soil": "soil", "TreeCover": "tree_cover",
    }.items()
}


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
