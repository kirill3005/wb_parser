"""CLI: build sessions if session_id is missing using gap-based heuristic."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from recsys.dataio.schema import Schema
from recsys.dataio.adapters import load_interactions
from recsys.dataio.sessionize import build_sessions

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sessionize interactions")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--interactions", required=True)
    parser.add_argument("--session_gap_min", type=int, default=30)
    parser.add_argument("--out_path", required=True)
    args = parser.parse_args()

    schema = Schema.from_yaml(args.schema)
    inter = load_interactions(args.interactions, schema)
    if "session_id" in inter.columns:
        logging.info("session_id already present; skipping sessionization")
        inter.to_parquet(args.out_path, index=False)
        return

    inter_sess = build_sessions(inter, session_gap_min=args.session_gap_min)
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    inter_sess.to_parquet(args.out_path, index=False)
    logging.info("Saved sessionized interactions to %s", args.out_path)


if __name__ == "__main__":
    main()
