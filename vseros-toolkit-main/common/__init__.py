from importlib import import_module

__all__ = [
    "cv",
    "features",
    "io",
    "models",
    "make_key",
    "load_feature_pkg",
    "save_feature_pkg",
    "set_global_seed",
    "StageTimer",
    "validators",
]

_ALIASES = {
    "make_key": ("common.cache", "make_key"),
    "load_feature_pkg": ("common.cache", "load_feature_pkg"),
    "save_feature_pkg": ("common.cache", "save_feature_pkg"),
    "set_global_seed": ("common.seed", "set_global_seed"),
    "StageTimer": ("common.timer", "StageTimer"),
    "validators": ("common", "validators"),
}


def __getattr__(name: str):
    if name in {"cv", "features", "io", "models", "validators"}:
        return import_module(f"common.{name}")
    if name in _ALIASES:
        module_name, attr_name = _ALIASES[name]
        return getattr(import_module(module_name), attr_name)
    raise AttributeError(f"module 'common' has no attribute {name}")
