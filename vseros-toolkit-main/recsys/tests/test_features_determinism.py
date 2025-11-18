import numpy as np
import pandas as pd

from recsys.dataio.schema import Schema
from recsys.features.joiner import FeatureJoiner
from recsys.tools.run_features import load_yaml


def test_deterministic_outputs():
    schema = Schema.from_yaml("recsys/configs/schema.yaml")
    cfg = load_yaml("recsys/configs/features.yaml")
    profile = load_yaml("recsys/configs/profiles/scout.yaml").get("features", {})
    interactions = pd.read_csv("recsys/tests/fixtures/tiny_interactions.csv")
    interactions["ts"] = pd.to_datetime(interactions["ts"])
    items = pd.read_csv("recsys/tests/fixtures/tiny_items.csv")
    train_pairs = pd.read_csv("recsys/tests/fixtures/tiny_pairs_train.csv")
    test_pairs = pd.read_csv("recsys/tests/fixtures/tiny_pairs_test.csv")

    rng = np.random.RandomState(42)
    joiner1 = FeatureJoiner(cfg, profile, schema=schema, rng=rng)
    joiner1.fit(interactions, items)
    out1 = joiner1.transform(train_pairs, test_pairs)

    rng2 = np.random.RandomState(42)
    joiner2 = FeatureJoiner(cfg, profile, schema=schema, rng=rng2)
    joiner2.fit(interactions, items)
    out2 = joiner2.transform(train_pairs, test_pairs)

    pd.testing.assert_frame_equal(out1["train"], out2["train"])
    pd.testing.assert_frame_equal(out1["test"], out2["test"])
