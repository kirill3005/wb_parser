import pandas as pd

from recsys.dataio.splits import SplitConfig, assign_time_splits


def test_split_respects_time_and_embargo():
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["i1", "i2", "i3"],
            "ts": ["2024-01-01", "2024-01-05", "2024-01-10"],
        }
    )
    cfg = SplitConfig(train_until="2024-01-04", val_until="2024-01-08", embargo="1D")
    out = assign_time_splits(df, cfg)
    assert set(out[out["split"] == "train"]["item_id"]) == {"i1"}
    assert set(out[out["split"] == "val"]["item_id"]) == {"i2"}
    assert set(out[out["split"] == "test"]["item_id"]) == {"i3"}
