import pandas as pd

from recsys.dataio.adapters import load_datasets
from recsys.dataio.schema import Schema


def test_adapter_aligns_columns(tmp_path):
    schema = Schema.from_yaml("recsys/configs/schema.yaml")
    interactions = "recsys/tests/fixtures/tiny_interactions.csv"
    items = "recsys/tests/fixtures/tiny_items.csv"
    queries = "recsys/tests/fixtures/tiny_queries.csv"

    data = load_datasets(
        schema=schema,
        path_interactions=interactions,
        path_items=items,
        path_queries=queries,
    )

    assert set(["user_id", "item_id", "ts"]).issubset(data.interactions.columns)
    assert "item_id" in data.items.columns
    assert "query_id" in data.queries.columns
