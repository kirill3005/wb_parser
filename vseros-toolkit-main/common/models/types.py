from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Dict, Any, List
import numpy as np

TaskType = Literal["regression", "binary", "multiclass", "multilabel"]


@dataclass
class ModelRun:
    run_id: str
    task: TaskType
    oof_true: np.ndarray  # shape (n,) or (n, C) for multi*
    oof_pred: np.ndarray  # same shapes / proba for classification
    test_pred: np.ndarray  # same shapes as oof_pred
    fold_scores: List[float]
    cv_mean: float
    cv_std: float
    artifacts_path: Path  # artifacts/models/<run_id>
    manifest: Dict[str, Any]  # meta: lib, params, features, folds_summary
