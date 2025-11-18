import pandas as pd

from recsys.rerank.rules import apply_rules


def test_brand_dedupe_limit():
    df = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q1"],
            "item_id": ["i1", "i2", "i3"],
            "score_ranker": [0.9, 0.8, 0.7],
        }
    )
    items = pd.DataFrame(
        {
            "item_id": ["i1", "i2", "i3"],
            "brand": ["b1", "b1", "b2"],
            "price": [10, 12, 11],
        }
    )
    out = apply_rules(df, items, K=2, config={"dedupe_brand": {"n_per_brand": 1}})
    brands = out.merge(items, on="item_id")["brand"].tolist()
    assert len(out) == 2
    assert brands.count("b1") == 1  # only one per brand allowed
