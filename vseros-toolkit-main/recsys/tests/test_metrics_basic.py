from recsys.eval import metrics


def test_metrics_values():
    y_true = ["i1"]
    y_pred = ["i1", "i2", "i3"]
    assert metrics.recall_at_k(y_true, y_pred, 3) == 1.0
    assert metrics.hitrate_at_k(y_true, y_pred, 1) == 1.0
    assert round(metrics.ndcg_at_k(y_true, y_pred, 3), 3) == 1.0
    assert round(metrics.ap_at_k(y_true, y_pred, 3), 3) == 1.0
    assert round(metrics.mrr_at_k(y_true, y_pred, 3), 3) == 1.0
