import pandas as pd

from recsys.candidates.covis import CoVisGenerator
from recsys.dataio.adapters import load_datasets
from recsys.dataio.schema import Schema


def test_covis_respects_cutoff():
    schema = Schema.from_yaml("recsys/configs/schema.yaml")
    data = load_datasets(
        schema=schema,
        path_interactions="recsys/tests/fixtures/tiny_interactions.csv",
        path_items="recsys/tests/fixtures/tiny_items.csv",
        path_queries="recsys/tests/fixtures/tiny_queries.csv",
    )
    cutoff = pd.Timestamp("2024-01-02", tz="UTC")
    gen = CoVisGenerator()
    gen.fit(data.interactions, data.items, cutoff_ts=cutoff, schema=schema, rng=None)
    filtered = data.interactions[data.interactions["ts"] <= cutoff]
    total_hist = sum(len(v) for v in gen.history.values())
    assert total_hist == len(filtered)
