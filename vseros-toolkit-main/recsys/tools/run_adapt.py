"""CLI: adapt raw interactions/items to unified schema and optionally save indexers.

Example:
    python tools/run_adapt.py --dataset_id demo --schema recsys/configs/schema.yaml \
        --interactions data/interactions.csv --items data/items.csv \
        --out_dir artifacts/recsys/dataio/demo --save_indexers 1
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from recsys.dataio.adapters import load_datasets
from recsys.dataio.schema import Schema

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt raw data to standard schema")
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--interactions", required=True)
    parser.add_argument("--items")
    parser.add_argument("--queries")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--save_indexers", type=int, default=1)
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    schema = Schema.from_yaml(args.schema)
    load_datasets(
        schema=schema,
        path_interactions=args.interactions,
        path_items=args.items,
        path_queries=args.queries,
        save_indexers=args.out_dir if args.save_indexers else None,
    )
    logging.info("Data successfully adapted for dataset %s", args.dataset_id)


if __name__ == "__main__":
    main()
