"""Hydrology and routing modules with lazy public exports."""

from importlib import import_module

__all__ = [
    "PotentialEvapotranspiration",
    "RainfallFeatures",
    "RoutedRunoff",
    "RunoffRouter",
    "VegET",
]

_EXPORTS = {
    "PotentialEvapotranspiration": ("bakaano.hydrology.pet", "PotentialEvapotranspiration"),
    "RainfallFeatures": ("bakaano.hydrology.rainfall_features", "RainfallFeatures"),
    "RoutedRunoff": ("bakaano.hydrology.plot_runoff", "RoutedRunoff"),
    "RunoffRouter": ("bakaano.hydrology.router", "RunoffRouter"),
    "VegET": ("bakaano.hydrology.veget", "VegET"),
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
