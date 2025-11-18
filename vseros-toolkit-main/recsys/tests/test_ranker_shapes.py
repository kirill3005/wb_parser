import pandas as pd

from recsys.rankers.trainer import train_ranker
from recsys.rankers import group as group_utils


def _load_features():
    X_train = pd.read_csv("recsys/tests/fixtures/ranker_X_train.csv")
    X_test = pd.read_csv("recsys/tests/fixtures/ranker_X_test.csv")
    feature_cols = [c for c in X_train.columns if c not in {"query_id", "item_id"}]
    return X_train, X_test, feature_cols


def test_ranker_shapes_and_preds():
    pairs = pd.read_csv("recsys/tests/fixtures/tiny_pairs_train.csv")
    y = pairs["label"].to_numpy()
    groups = group_utils.build_groups(pairs)
    X_train, X_test, feature_cols = _load_features()
    model_run = train_ranker(
        X_train[feature_cols].to_numpy(),
        y,
        groups=groups,
        backend="linear",
        params={"n_splits": 2, "features": feature_cols, "task": "rank"},
        eval_metric="auc",
        seed=123,
        n_jobs=1,
        X_test=X_test[feature_cols].to_numpy(),
    )
    assert model_run.oof_pred.shape[0] == len(pairs)
    assert model_run.test_pred.shape[0] == len(X_test)


def test_ranker_determinism():
    pairs = pd.read_csv("recsys/tests/fixtures/tiny_pairs_train.csv")
    y = pairs["label"].to_numpy()
    groups = group_utils.build_groups(pairs)
    X_train, X_test, feature_cols = _load_features()
    run1 = train_ranker(
        X_train[feature_cols].to_numpy(),
        y,
        groups=groups,
        backend="linear",
        params={"n_splits": 2, "features": feature_cols, "task": "rank"},
        eval_metric="auc",
        seed=7,
        n_jobs=1,
        X_test=X_test[feature_cols].to_numpy(),
    )
    run2 = train_ranker(
        X_train[feature_cols].to_numpy(),
        y,
        groups=groups,
        backend="linear",
        params={"n_splits": 2, "features": feature_cols, "task": "rank"},
        eval_metric="auc",
        seed=7,
        n_jobs=1,
        X_test=X_test[feature_cols].to_numpy(),
    )
    assert run1.cv_mean == run2.cv_mean
    assert (run1.oof_pred == run2.oof_pred).all()
    assert (run1.test_pred == run2.test_pred).all()
