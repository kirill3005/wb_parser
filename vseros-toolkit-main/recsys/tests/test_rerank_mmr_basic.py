import pandas as pd

from recsys.rerank.mmr import mmr


def test_mmr_prefers_diversity():
    df = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q1"],
            "item_id": ["i1", "i2", "i3"],
            "score_ranker": [0.9, 0.85, 0.8],
        }
    )
    items = pd.DataFrame({"item_id": ["i1", "i2", "i3"], "category": ["c1", "c1", "c2"]})
    out = mmr(df, items, K=2, lambda_div=0.5, sim_backend="jaccard")
    cats = out.merge(items, on="item_id")["category"].tolist()
    assert len(out) == 2
    assert len(set(cats)) > 1  # diversified categories
