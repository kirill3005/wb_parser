import pandas as pd

from recsys.eval.offline_harness import evaluate_offline
from recsys.dataio.adapters import _read_table


def test_offline_harness_end_to_end():
    pairs = pd.read_csv("recsys/tests/fixtures/tiny_pairs_train.csv")
    preds = pd.read_csv("recsys/tests/fixtures/tiny_predictions.csv")
    report = evaluate_offline(pairs, preds, metrics_list=["recall@1", "ndcg@2"])
    assert "overall" in report
    assert report["overall"]["recall@1"] >= 0
    assert report["counts"]["queries"] == 2
