"""Apply reranking (MMR + rules) to candidate lists."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from recsys.rerank import mmr as mmr_fn
from recsys.rerank import rules as rules_fn

LOGGER = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Rerank candidates")
    p.add_argument("--candidates", required=True, help="CSV/parquet with query_id,item_id,score_ranker")
    p.add_argument("--items", required=False)
    p.add_argument("--config", required=False)
    p.add_argument("--out", required=True)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_table(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
    cfg = {}
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())
        cfg = cfg.get("recsys", {}).get("rerank", cfg)
    df = load_table(args.candidates)
    if "score" in df.columns and "score_ranker" not in df.columns:
        df = df.rename(columns={"score": "score_ranker"})
    items = load_table(args.items) if args.items else None
    reranked = mmr_fn.mmr(
        df,
        items,
        K=args.K,
        lambda_div=cfg.get("mmr", {}).get("lambda", 0.7),
        sim_backend=cfg.get("mmr", {}).get("sim", "jaccard"),
        rng=np.random.RandomState(args.seed),
    )
    reranked = rules_fn.apply_rules(
        reranked.merge(df[["query_id", "item_id", "score_ranker"]], on=["query_id", "item_id"], how="left"),
        items,
        K=args.K,
        config=cfg.get("rules", {}),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reranked.to_parquet(out_path, index=False)
    report = {"rows": len(reranked), "queries": reranked["query_id"].nunique()}
    with (out_path.parent / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    LOGGER.info("Saved reranked list to %s", out_path)


if __name__ == "__main__":
    main()
