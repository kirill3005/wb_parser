import numpy as np
from typing import List, Dict, Any
from sklearn.linear_model import Ridge, LogisticRegression

from .types import ModelRun
from . import artifacts as A, eval as ME


def _stack_preds(runs: List[ModelRun]):
    oof_list = [r.oof_pred for r in runs]
    test_list = [r.test_pred for r in runs]
    oof_stack = np.stack(oof_list, axis=-1)
    test_stack = np.stack(test_list, axis=-1)
    return oof_stack, test_stack


def equal_weight(runs: List[ModelRun]) -> ModelRun:
    """OOF/Test усредняются (простой бленд)."""
    if not runs:
        raise ValueError("No runs provided")
    task = runs[0].task
    oof_true = runs[0].oof_true
    oof_pred = np.mean([r.oof_pred for r in runs], axis=0)
    test_pred = np.mean([r.test_pred for r in runs], axis=0)
    fold_scores = []
    cv_mean = float(np.nan)
    cv_std = float(np.nan)
    manifest = {"type": "equal_weight", "sources": [r.run_id for r in runs]}
    return ModelRun(
        run_id=A.make_run_id(task, "blend", "manual", 0),
        task=task,
        oof_true=oof_true,
        oof_pred=oof_pred,
        test_pred=test_pred,
        fold_scores=fold_scores,
        cv_mean=cv_mean,
        cv_std=cv_std,
        artifacts_path=A.path("blend"),
        manifest=manifest,
    )


def weight_search(runs: List[ModelRun], y_true, scorer, nonneg=True, sum_to_one=True) -> Dict[str, Any]:
    """Находит веса по OOF под scorer; возвращает dict с weights и метрикой."""
    if not runs:
        raise ValueError("No runs provided")
    oof_stack, _ = _stack_preds(runs)
    n_models = oof_stack.shape[-1]
    best_w = np.ones(n_models) / n_models
    best_score = -np.inf
    for _ in range(100):
        w = np.random.rand(n_models)
        if nonneg:
            w = np.abs(w)
        if sum_to_one:
            w = w / w.sum()
        blended = np.tensordot(oof_stack, w, axes=1)
        score = scorer(y_true, blended)
        if score > best_score:
            best_score = score
            best_w = w
    return {"weights": best_w.tolist(), "score": float(best_score)}


def ridge_level2(runs: List[ModelRun], y_true, alpha=1.0) -> ModelRun:
    """Тренирует Ridge/LogReg на стэке OOF, применяет к стэку Test; возвращает ModelRun со своими артефактами."""
    if not runs:
        raise ValueError("No runs provided")
    task = runs[0].task
    oof_stack, test_stack = _stack_preds(runs)

    if task == "regression":
        model = Ridge(alpha=alpha)
    else:
        model = LogisticRegression(max_iter=200)

    n_samples = oof_stack.shape[0]
    X_oof = oof_stack.reshape(n_samples, -1)
    X_test = test_stack.reshape(test_stack.shape[0], -1)

    model.fit(X_oof, y_true)
    oof_pred = model.predict(X_oof)
    test_pred = model.predict(X_test)

    run_id = A.make_run_id(task, "blend", "level2", 0)
    run_dir = A.path(run_id)
    manifest = {"type": "ridge_level2", "sources": [r.run_id for r in runs], "alpha": alpha}
    A.save_manifest(run_dir, manifest)
    A.save_array(run_dir, "test_pred.npy", test_pred)

    return ModelRun(
        run_id=run_id,
        task=task,
        oof_true=y_true,
        oof_pred=oof_pred,
        test_pred=test_pred,
        fold_scores=[],
        cv_mean=float(np.nan),
        cv_std=float(np.nan),
        artifacts_path=run_dir,
        manifest=manifest,
    )
