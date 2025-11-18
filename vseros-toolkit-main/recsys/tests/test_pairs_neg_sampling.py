import numpy as np
import pandas as pd

from recsys.dataio.pairs import build_pairs


def test_negative_sampling_ratio_and_no_positive_collision():
    inter = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "item_id": ["i1", "i2", "i1", "i3"],
            "ts": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
        }
    )
    queries = pd.DataFrame({"query_id": ["u1", "u2"], "ts_query": pd.to_datetime(["2024-01-03", "2024-01-03"])})
    pairs = build_pairs(inter, queries, neg_pos_ratio=2, rng=np.random.RandomState(0))
    # two queries -> two positives + negatives
    assert (pairs["label"] == 1).sum() == 2
    assert (pairs["label"] == 0).sum() >= 4
    for qid, grp in pairs.groupby("query_id"):
        pos_items = set(grp[grp["label"] == 1]["item_id"])
        neg_items = set(grp[grp["label"] == 0]["item_id"])
        assert pos_items.isdisjoint(neg_items)
