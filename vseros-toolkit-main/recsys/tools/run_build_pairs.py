"""CLI: build train/val/test pairs with negative sampling."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from recsys.dataio.adapters import load_interactions, load_queries
from recsys.dataio.schema import Schema
from recsys.dataio.pairs import build_pairs
from recsys.dataio.queries import build_queries

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pairs for ranking")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--interactions", required=True)
    parser.add_argument("--queries")
    parser.add_argument("--scope", default="session")
    parser.add_argument("--neg_pos_ratio", type=int, default=10)
    parser.add_argument("--neg_strategy", default="pop")
    parser.add_argument("--out_path", required=True)
    args = parser.parse_args()

    schema = Schema.from_yaml(args.schema)
    inter = load_interactions(args.interactions, schema)
    queries_df = load_queries(args.queries, schema)
    if queries_df.empty:
        queries_df = build_queries(inter, scope=args.scope)
    pairs = build_pairs(
        inter,
        queries_df,
        neg_pos_ratio=args.neg_pos_ratio,
        neg_strategy=args.neg_strategy,
        rng=np.random.RandomState(42),
        scope=args.scope,
    )
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(args.out_path, index=False)
    logging.info("Saved pairs to %s", args.out_path)


if __name__ == "__main__":
    main()
