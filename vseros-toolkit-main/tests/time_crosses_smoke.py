import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd


def _prepare_package():
    base = Path("common")
    common_pkg = types.ModuleType("common")
    common_pkg.__path__ = [str(base)]
    sys.modules.setdefault("common", common_pkg)

    features_pkg = types.ModuleType("common.features")
    features_pkg.__path__ = [str(base / "features")]
    sys.modules.setdefault("common.features", features_pkg)

    types_spec = importlib.util.spec_from_file_location("common.features.types", base / "features" / "types.py")
    types_mod = importlib.util.module_from_spec(types_spec)
    assert types_spec and types_spec.loader
    types_spec.loader.exec_module(types_mod)  # type: ignore[arg-type]
    sys.modules["common.features.types"] = types_mod

    cache_spec = importlib.util.spec_from_file_location("common.cache", base / "cache.py")
    cache_mod = importlib.util.module_from_spec(cache_spec)
    assert cache_spec and cache_spec.loader
    cache_spec.loader.exec_module(cache_mod)  # type: ignore[arg-type]
    sys.modules["common.cache"] = cache_mod


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


_prepare_package()
time_agg = _load_module("common.features.time_agg", Path("common/features/time_agg.py"))
crosses = _load_module("common.features.crosses", Path("common/features/crosses.py"))


def test_time_agg_global_dense():
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=5, freq="D"),
            "user": [1, 1, 1, 2, 2],
            "x": [1, 2, 3, 4, 5],
        }
    )
    pkg = time_agg.build(df, date_col="ts", group_cols=["user"], lags=(1,), rollings_count=(2,), rollings_time=(), ewm_spans=(), agg_funcs=("mean",), use_cache=False)
    assert pkg.train.shape == (5, 2)
    assert pkg.test.empty


def test_time_agg_oof_embargo():
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=4, freq="D"),
            "x": [1, 2, 3, 4],
        }
    )
    folds = [([0, 1, 2], [3])]
    pkg = time_agg.build(df, date_col="ts", folds=folds, lags=(1,), rollings_count=(2,), rollings_time=(), ewm_spans=(), embargo="1D", agg_funcs=("mean",), use_cache=False)
    last_row = pkg.train.iloc[3]
    assert not pd.isna(last_row.filter(like="lag")).all()


def test_crosses_dense_and_sparse():
    train = pd.DataFrame({"x1": [1.0, 2.0], "x2": [3.0, 4.0], "cat": ["a", "b"]})
    test = pd.DataFrame({"x1": [5.0], "x2": [6.0], "cat": ["a"]})

    dense_pkg = crosses.build(train, test, num_cols=["x1", "x2"], num_num_ops=("mul",), use_cache=False)
    assert dense_pkg.kind == "dense"
    assert dense_pkg.train.shape[1] == 1

    sparse_pkg = crosses.build(train, test, num_cols=["x1"], cat_cols=["cat"], num_cat=True, hash_buckets=8, use_cache=False)
    assert sparse_pkg.kind in {"sparse", "dense"}
    assert sparse_pkg.train.shape[0] == len(train)
