import sys
from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.models import linear, eval as ME


def main():
    X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=0)
    X_test, _ = make_classification(n_samples=50, n_features=10, n_informative=5, random_state=1)
    folds = []
    indices = np.arange(len(y))
    np.random.seed(0)
    np.random.shuffle(indices)
    split = np.array_split(indices, 3)
    for i in range(3):
        val_idx = split[i]
        train_idx = np.concatenate([split[j] for j in range(3) if j != i])
        folds.append((train_idx, val_idx))

    run = linear.train_cv(
        X,
        y,
        X_test,
        folds,
        task="binary",
        algo="lr",
        params={"max_iter": 100},
        seed=0,
        save=False,
    )
    scorer = ME.get_scorer("binary", "roc_auc")
    print("Fold scores:", run.fold_scores)
    print("CV:", run.cv_mean, run.cv_std)
    print("Artifacts path:", run.artifacts_path)


if __name__ == "__main__":
    main()
