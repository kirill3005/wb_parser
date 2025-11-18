import pandas as pd

from recsys.dataio.schema import Schema
from recsys.features.joiner import FeatureJoiner
from recsys.tools.run_features import load_yaml


def _load_pairs(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train["ts_query"] = pd.to_datetime(train["ts_query"])
    test["ts_query"] = pd.to_datetime(test["ts_query"])
    return train, test


def test_feature_format_alignment():
    schema = Schema.from_yaml("recsys/configs/schema.yaml")
    cfg = load_yaml("recsys/configs/features.yaml")
    profile = load_yaml("recsys/configs/profiles/scout.yaml").get("features", {})
    interactions = pd.read_csv("recsys/tests/fixtures/tiny_interactions.csv")
    interactions["ts"] = pd.to_datetime(interactions["ts"])
    items = pd.read_csv("recsys/tests/fixtures/tiny_items.csv")
    train_pairs, test_pairs = _load_pairs(
        "recsys/tests/fixtures/tiny_pairs_train.csv", "recsys/tests/fixtures/tiny_pairs_test.csv"
    )

    import numpy as np

    joiner = FeatureJoiner(cfg, profile, schema=schema, rng=np.random.RandomState(0))
    joiner.fit(interactions, items)
    outputs = joiner.transform(train_pairs, test_pairs)

    train_cols = list(outputs["train"].columns)
    test_cols = list(outputs["test"].columns)
    assert train_cols == test_cols
    assert outputs["train"].isna().sum().sum() == 0
    assert outputs["test"].isna().sum().sum() == 0
