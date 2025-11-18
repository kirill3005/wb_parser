"""CLI: offline evaluation of predictions on pairs with ranking metrics."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from recsys.eval.offline_harness import evaluate_offline
from recsys.dataio.adapters import _read_table

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline evaluation")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--items")
    parser.add_argument("--metrics", default="recall@20,ndcg@20")
    parser.add_argument("--slices")
    parser.add_argument("--bootstrap", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pairs = _read_table(args.pairs)
    preds = _read_table(args.preds)
    items = _read_table(args.items) if args.items else None

    metrics_list = [m.strip() for m in args.metrics.split(",") if m.strip()]
    slice_cols = [s.strip() for s in args.slices.split(",") if args.slices] if args.slices else None

    report = evaluate_offline(
        pairs,
        preds,
        items_df=items,
        metrics_list=metrics_list,
        slices=slice_cols,
        bootstrap_iters=args.bootstrap,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    logging.info("Saved eval report to %s", args.out)


if __name__ == "__main__":
    main()
