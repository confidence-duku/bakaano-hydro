"""Optional extensions built on top of the core Bakaano modules."""

from importlib import import_module

__all__ = ["FloodMapper", "ScenarioManager"]

_EXPORTS = {
    "FloodMapper": ("bakaano.extensions.flood_mapper", "FloodMapper"),
    "ScenarioManager": ("bakaano.extensions.scenario", "ScenarioManager"),
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
