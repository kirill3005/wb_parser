from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from recsys.candidates.merge import merge_and_rank

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Merge candidate sources")
    parser.add_argument("--sources", nargs="+", help="Paths to candidate parquet files with column source")
    parser.add_argument("--quotas", required=True, help="JSON dict of quotas")
    parser.add_argument("--weights", required=True, help="JSON dict of weights")
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    quotas = json.loads(args.quotas)
    weights = json.loads(args.weights)
    frames = {}
    for path in args.sources:
        df = pd.read_parquet(path)
        src = df.get("source")
        if src is None or (isinstance(src, pd.Series) and src.nunique() > 1):
            name = Path(path).stem
        else:
            name = df["source"].iloc[0]
        frames[name] = df
    merged = merge_and_rank(frames, quotas=quotas, weights=weights, K=args.K)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.out, index=False)
    logging.info("Merged candidates saved to %s", args.out)


if __name__ == "__main__":
    main()
