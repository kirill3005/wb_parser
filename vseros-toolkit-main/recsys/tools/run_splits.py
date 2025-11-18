"""CLI: assign time-based splits with embargo.

Example:
    python tools/run_splits.py --schema recsys/configs/schema.yaml \
        --interactions data/interactions.csv --train_until 2024-01-01 \
        --val_until 2024-02-01 --out_path artifacts/recsys/dataio/demo/interactions_splits.parquet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from recsys.dataio.schema import Schema
from recsys.dataio.adapters import load_interactions
from recsys.dataio.splits import SplitConfig, assign_time_splits, save_split_report

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign time splits")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--interactions", required=True)
    parser.add_argument("--train_until")
    parser.add_argument("--val_until")
    parser.add_argument("--embargo", default="0D")
    parser.add_argument("--out_path", required=True)
    args = parser.parse_args()

    schema = Schema.from_yaml(args.schema)
    inter = load_interactions(args.interactions, schema)
    cfg = SplitConfig(train_until=args.train_until, val_until=args.val_until, embargo=args.embargo)
    inter_split = assign_time_splits(inter, cfg)
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    inter_split.to_parquet(args.out_path, index=False)
    save_split_report(inter_split, str(Path(args.out_path).with_suffix(".json")))
    logging.info("Finished splits saved to %s", args.out_path)


if __name__ == "__main__":
    main()
