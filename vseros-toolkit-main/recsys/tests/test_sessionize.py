import pandas as pd

from recsys.dataio.sessionize import build_sessions


def test_gap_based_sessionize():
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["i1", "i2", "i3"],
            "ts": ["2024-01-01T00:00:00Z", "2024-01-01T00:10:00Z", "2024-01-01T01:00:00Z"],
        }
    )
    out = build_sessions(df, session_gap_min=30)
    assert out.loc[0, "session_id"] == out.loc[1, "session_id"]
    assert out.loc[2, "session_id"] != out.loc[0, "session_id"]
