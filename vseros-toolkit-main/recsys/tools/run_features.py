"""CLI to build time-safe feature matrices for recsys retrieval/ranking.

Example:
    python tools/run_features.py \
        --dataset_id demo \
        --schema recsys/configs/schema.yaml \
        --features recsys/configs/features.yaml \
        --profile recsys/configs/profiles/scout.yaml \
        --pairs_train recsys/tests/fixtures/tiny_pairs_train.csv \
        --pairs_test recsys/tests/fixtures/tiny_pairs_test.csv \
        --interactions recsys/tests/fixtures/tiny_interactions.csv \
        --items recsys/tests/fixtures/tiny_items.csv
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from recsys.dataio.adapters import load_datasets
from recsys.dataio.schema import Schema
from recsys.features.joiner import FeatureJoiner, save_outputs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run feature pipeline")
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--pairs_train", required=True)
    parser.add_argument("--pairs_test", required=True)
    parser.add_argument("--interactions", required=True)
    parser.add_argument("--items")
    parser.add_argument("--cutoff", default="auto")
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", type=int, default=1)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    schema = Schema.from_yaml(args.schema)
    feat_cfg = load_yaml(args.features)
    profile_cfg = load_yaml(args.profile).get("features", {})

    rng = np.random.RandomState(args.seed)

    pairs_train = pd.read_csv(args.pairs_train) if args.pairs_train.endswith(".csv") else pd.read_parquet(args.pairs_train)
    pairs_test = pd.read_csv(args.pairs_test) if args.pairs_test.endswith(".csv") else pd.read_parquet(args.pairs_test)

    cutoff_ts = None
    if args.cutoff != "auto" and "ts" not in pairs_train.columns:
        cutoff_ts = pd.to_datetime(args.cutoff)

    data = load_datasets(
        schema=schema,
        path_interactions=args.interactions,
        path_items=args.items,
        cutoff_ts=cutoff_ts,
    )

    joiner = FeatureJoiner(feat_cfg, profile_cfg, schema=schema, rng=rng)
    joiner.fit(data.interactions, data.items)
    outputs = joiner.transform(pairs_train, pairs_test)

    out_dir = Path(args.out_dir)
    save_outputs(out_dir, outputs, feat_cfg, pairs_train, pairs_test)
    logger.info("Saved features to %s", out_dir)


if __name__ == "__main__":
    main()
