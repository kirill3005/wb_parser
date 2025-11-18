import pandas as pd

from recsys.dataio.schema import Schema
from recsys.dataio.adapters import load_interactions


def test_schema_validation_and_timestamp(tmp_path):
    schema = Schema.from_yaml("recsys/configs/schema.yaml")
    path = tmp_path / "inter.csv"
    path.write_text("user_id,item_id,ts\nu1,i1,1704067200000\n", encoding="utf-8")
    inter = load_interactions(str(path), schema)
    assert "ts" in inter.columns
    assert inter.iloc[0]["ts"].tzinfo is not None
    assert inter.iloc[0]["ts"].year == 2024
