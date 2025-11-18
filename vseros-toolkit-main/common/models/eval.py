import numpy as np
from typing import Literal, Tuple, List, Callable
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    log_loss,
    accuracy_score,
)


def _prepare_preds_for_classification(y_pred):
    if y_pred.ndim == 1:
        return y_pred
    if y_pred.shape[1] == 1:
        return y_pred[:, 0]
    return y_pred


def _predict_labels_from_scores(y_pred):
    if y_pred.ndim == 1:
        return (y_pred >= 0.5).astype(int)
    return np.argmax(y_pred, axis=1)


def _map_at_k(y_true, y_scores, k: int) -> float:
    # y_true and y_scores are 2d
    topk_idx = np.argpartition(-y_scores, kth=min(k, y_scores.shape[1] - 1), axis=1)[:, :k]
    ap_list = []
    for i in range(y_scores.shape[0]):
        relevant = 0
        precisions = []
        labels_row = y_true[i]
        scores_indices = topk_idx[i]
        scores_sorted = scores_indices[np.argsort(-y_scores[i, scores_indices])]
        for rank, idx in enumerate(scores_sorted, start=1):
            if labels_row[idx] > 0:
                relevant += 1
                precisions.append(relevant / rank)
        ap_list.append(np.mean(precisions) if precisions else 0.0)
    return float(np.mean(ap_list))


def get_scorer(task: str, metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Возвращает функцию score(y_true, y_pred/proba).
    regression: mae|rmse|r2
    binary: roc_auc|pr_auc|f1|logloss
    multiclass: f1_macro|acc|logloss
    multilabel: f1_micro|f1_macro|map@k (тогда scorer ожидает scores и y_true binarized)
    """
    task_l = task.lower()
    metric_l = metric.lower()

    if task_l == "regression":
        if metric_l == "mae":
            return lambda y_true, y_pred: float(mean_absolute_error(y_true, y_pred))
        if metric_l == "rmse":
            return lambda y_true, y_pred: float(mean_squared_error(y_true, y_pred, squared=False))
        if metric_l == "r2":
            return lambda y_true, y_pred: float(r2_score(y_true, y_pred))
        raise ValueError(f"Unknown regression metric: {metric}")

    if task_l == "binary":
        if metric_l == "roc_auc":
            return lambda y_true, y_pred: float(roc_auc_score(y_true, _prepare_preds_for_classification(y_pred)))
        if metric_l == "pr_auc":
            return lambda y_true, y_pred: float(average_precision_score(y_true, _prepare_preds_for_classification(y_pred)))
        if metric_l == "f1":
            return lambda y_true, y_pred: float(
                f1_score(y_true, _predict_labels_from_scores(_prepare_preds_for_classification(y_pred)))
            )
        if metric_l == "logloss":
            return lambda y_true, y_pred: float(log_loss(y_true, _prepare_preds_for_classification(y_pred)))
        raise ValueError(f"Unknown binary metric: {metric}")

    if task_l == "multiclass":
        if metric_l == "f1_macro":
            return lambda y_true, y_pred: float(
                f1_score(y_true, _predict_labels_from_scores(y_pred), average="macro")
            )
        if metric_l == "acc":
            return lambda y_true, y_pred: float(accuracy_score(y_true, _predict_labels_from_scores(y_pred)))
        if metric_l == "logloss":
            return lambda y_true, y_pred: float(log_loss(y_true, y_pred))
        raise ValueError(f"Unknown multiclass metric: {metric}")

    if task_l == "multilabel":
        if metric_l == "f1_micro":
            return lambda y_true, y_pred: float(
                f1_score(y_true, (y_pred >= 0.5).astype(int), average="micro")
            )
        if metric_l == "f1_macro":
            return lambda y_true, y_pred: float(
                f1_score(y_true, (y_pred >= 0.5).astype(int), average="macro")
            )
        if metric_l.startswith("map@"):
            try:
                k = int(metric_l.split("@")[-1])
            except Exception as e:
                raise ValueError(f"Invalid map@k metric: {metric}") from e
            return lambda y_true, y_pred: _map_at_k(y_true, y_pred, k)
        raise ValueError(f"Unknown multilabel metric: {metric}")

    raise ValueError(f"Unknown task type: {task}")


def cv_scores_by_folds(y_true, y_pred, folds, scorer) -> List[float]:
    """Считает metric на каждом фолде, используя oof-маску."""
    scores = []
    for _, val_idx in folds:
        scores.append(float(scorer(y_true[val_idx], y_pred[val_idx])))
    return scores
