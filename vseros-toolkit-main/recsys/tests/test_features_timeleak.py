import numpy as np
import pandas as pd

from recsys.dataio.schema import Schema
from recsys.features.joiner import FeatureJoiner
from recsys.tools.run_features import load_yaml


def test_no_time_leakage():
    schema = Schema.from_yaml("recsys/configs/schema.yaml")
    cfg = load_yaml("recsys/configs/features.yaml")
    profile = load_yaml("recsys/configs/profiles/scout.yaml").get("features", {})

    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["i1", "i2", "i1"],
            "ts": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-05"]),
        }
    )
    items = pd.DataFrame({"item_id": ["i1", "i2"], "price": [10, 12]})
    train_pairs = pd.DataFrame({"query_id": ["u1"], "item_id": ["i1"], "ts_query": pd.to_datetime(["2023-01-03"])})
    test_pairs = pd.DataFrame({"query_id": ["u1"], "item_id": ["i2"], "ts_query": pd.to_datetime(["2023-01-04"])})

    joiner = FeatureJoiner(cfg, profile, schema=schema, rng=np.random.RandomState(0))
    joiner.fit(interactions, items)
    outputs = joiner.transform(train_pairs, test_pairs)

    recency_train = outputs["train"]["recency_item_h"].iloc[0]
    assert recency_train > 0  # uses ts<=2023-01-03 only
    # interaction on 2023-01-05 should not affect train recency (last seen i1 is 2023-01-01)
    assert recency_train == (pd.to_datetime("2023-01-03") - pd.to_datetime("2023-01-01")).total_seconds() / 3600
