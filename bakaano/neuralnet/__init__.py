"""Neural-network training and simulation modules with lazy public exports."""

from importlib import import_module

__all__ = [
    "DataPreprocessor",
    "PredictDataPreprocessor",
    "PredictStreamflow",
    "StreamflowModel",
]

_EXPORTS = {
    "DataPreprocessor": ("bakaano.neuralnet.train", "DataPreprocessor"),
    "PredictDataPreprocessor": ("bakaano.neuralnet.simulate", "PredictDataPreprocessor"),
    "PredictStreamflow": ("bakaano.neuralnet.simulate", "PredictStreamflow"),
    "StreamflowModel": ("bakaano.neuralnet.train", "StreamflowModel"),
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
