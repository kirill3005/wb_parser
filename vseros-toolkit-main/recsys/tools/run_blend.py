"""Blend multiple ModelRun outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recsys.rankers import artifacts
from recsys.rerank.blend import blend_scores


def parse_args():
    p = argparse.ArgumentParser(description="Blend model runs")
    p.add_argument("--runs", required=True, help="Comma-separated model run paths")
    p.add_argument("--mode", default="equal", choices=["equal", "weights"])
    p.add_argument("--weights", default="", help="Comma-separated weights aligned with runs")
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    run_paths = [Path(p) for p in args.runs.split(",") if p]
    runs = [artifacts.load_model_run(p) for p in run_paths]
    mat = np.vstack([r.test_pred for r in runs])
    weights = None
    if args.mode == "weights" and args.weights:
        weights = {f"s{i}": float(w) for i, w in enumerate(args.weights.split(","))}
    df = {}
    for i, run in enumerate(runs):
        df[f"score_s{i}"] = run.test_pred
    import pandas as pd

    df_pd = pd.DataFrame(df)
    blended = blend_scores(df_pd, [f"s{i}" for i in range(len(runs))], mode="weights" if weights else "equal", weights=weights)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "blended_test_pred.npy", blended.to_numpy())
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump({"mode": args.mode, "weights": weights}, f, indent=2)


if __name__ == "__main__":
    main()
