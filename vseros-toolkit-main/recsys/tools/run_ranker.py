"""Train a lightweight ranker over prepared feature matrices.

Example:
    python recsys/tools/run_ranker.py \
        --model configs/recsys/models/lgbm_ranker.yaml \
        --pairs_train recsys/tests/fixtures/tiny_pairs_train.csv \
        --pairs_test recsys/tests/fixtures/tiny_pairs_test.csv \
        --X_train recsys/tests/fixtures/ranker_X_train.csv \
        --X_test recsys/tests/fixtures/ranker_X_test.csv
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from recsys.rankers import artifacts, group as group_utils
from recsys.rankers.trainer import train_ranker

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ranker")
    parser.add_argument("--model", required=True)
    parser.add_argument("--pairs_train", required=True)
    parser.add_argument("--pairs_test", required=False)
    parser.add_argument("--X_train", required=True)
    parser.add_argument("--X_test", required=False)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    return parser.parse_args()


def _load_matrix(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
    model_cfg = yaml.safe_load(Path(args.model).read_text())
    backend = model_cfg.get("model", {}).get("backend", "linear")
    params = model_cfg.get("model", {}).get("params", {})

    pairs_train = pd.read_csv(args.pairs_train)
    y = pairs_train["label"].to_numpy()
    groups = group_utils.build_groups(pairs_train, "query_id")
    X_train = _load_matrix(args.X_train)
    feature_cols = [c for c in X_train.columns if c not in {"query_id", "item_id"}]
    X_train_mat = X_train[feature_cols].to_numpy(dtype=float)

    X_test_mat = None
    if args.X_test:
        X_test = _load_matrix(args.X_test)
        X_test_mat = X_test[feature_cols].to_numpy(dtype=float)

    model_run = train_ranker(
        X_train_mat,
        y,
        groups=groups,
        backend=backend,
        params=params | {"features": feature_cols},
        eval_metric="auc",
        seed=args.seed,
        n_jobs=args.jobs,
        X_test=X_test_mat,
    )
    out_dir = Path(args.out_dir)
    artifacts.save_model_run(model_run, out_dir)
    report = {
        "backend": backend,
        "cv_mean": model_run.cv_mean,
        "cv_std": model_run.cv_std,
        "features": feature_cols,
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    LOGGER.info("Saved model run to %s", out_dir)


if __name__ == "__main__":
    main()
