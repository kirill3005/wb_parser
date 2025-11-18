"""Main driver to build candidate sets for recsys profiles.

Example:
    python tools/run_candidates.py --dataset_id demo \
        --schema recsys/configs/schema.yaml \
        --candidates recsys/configs/candidates.yaml \
        --profile recsys/configs/profiles/scout.yaml \
        --data_interactions data/interactions.parquet \
        --data_items data/items.parquet \
        --data_queries data/queries.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from recsys.candidates.cache import CacheManager
from recsys.candidates.content_image import ContentImageGenerator
from recsys.candidates.content_text import ContentTextGenerator
from recsys.candidates.covis import CoVisGenerator
from recsys.candidates.graph_ppr import GraphPPRGenerator
from recsys.candidates.item2vec import Item2VecGenerator
from recsys.candidates.lightgcn import LightGCNCandidate
from recsys.candidates.merge import merge_and_rank
from recsys.candidates.mf_als import MFALSCandidate
from recsys.candidates.pop import PopularityGenerator
from recsys.candidates.session_ngram import SessionNgramGenerator
from recsys.candidates.twotower import TwoTowerCandidate
from recsys.dataio.adapters import load_datasets
from recsys.dataio.schema import Schema
from recsys.dataio.utils_time import filter_by_cutoff
from recsys.eval.quick_eval import quick_eval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


GENERATOR_REGISTRY = {
    "covis": CoVisGenerator,
    "pop": PopularityGenerator,
    "session_ngram": SessionNgramGenerator,
    "item2vec": Item2VecGenerator,
    "mf_als": MFALSCandidate,
    "lightgcn": LightGCNCandidate,
    "twotower": TwoTowerCandidate,
    "content_text": ContentTextGenerator,
    "content_image": ContentImageGenerator,
    "graph_ppr": GraphPPRGenerator,
}


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run candidate generators")
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--data_interactions", required=True)
    parser.add_argument("--data_items")
    parser.add_argument("--data_queries")
    parser.add_argument("--cutoff", default="auto")
    parser.add_argument("--K", type=int)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", type=int, default=1)
    parser.add_argument("--report_out")
    parser.add_argument("--out_path")
    args = parser.parse_args()

    schema = Schema.from_yaml(args.schema)
    cand_cfg = load_yaml(args.candidates)
    profile = load_yaml(args.profile)

    rng = np.random.RandomState(args.seed)

    cutoff_ts = None
    if args.cutoff != "auto":
        cutoff_ts = pd.to_datetime(args.cutoff)

    data = load_datasets(
        schema=schema,
        path_interactions=args.data_interactions,
        path_items=args.data_items,
        path_queries=args.data_queries,
        cutoff_ts=cutoff_ts,
    )

    if cutoff_ts is None:
        cutoff_ts = data.interactions["ts"].max()

    # instantiate generators
    active_generators = {}
    for name, params in cand_cfg.get("generators", {}).items():
        if not params.get("enabled", True):
            continue
        cls = GENERATOR_REGISTRY.get(name)
        if cls is None:
            continue
        kwargs = {k: v for k, v in params.items() if k != "enabled"}
        gen = cls(**kwargs)
        active_generators[name] = gen

    cache = CacheManager()
    sources = {}
    timing = {}
    for name, gen in active_generators.items():
        logger.info("Fitting %s", name)
        artifact = cache.load(name, gen.params, cutoff_ts) if args.cache else None
        if artifact is not None and hasattr(gen, "__dict__"):
            gen.__dict__.update(artifact)
        else:
            gen.fit(data.interactions, data.items, cutoff_ts=cutoff_ts, schema=schema, rng=rng)
            if args.cache:
                cache.save(name, gen.params, cutoff_ts, gen.__dict__)
        logger.info("Scoring %s", name)
        df = gen.score(data.queries, k=profile.get("quotas", {}).get(name, args.K or profile.get("K", 100)), schema=schema)
        sources[name] = df
        timing[name] = 0.0

    K_final = args.K or profile.get("K", 100)
    merged = merge_and_rank(sources, quotas=profile.get("quotas", {}), weights=profile.get("weights", {}), K=K_final)

    out_dir = Path(args.out_path or f"artifacts/recsys/candidates/{args.dataset_id}/{Path(args.profile).stem}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_path = out_dir / "candidates.parquet"
    merged.to_parquet(cand_path, index=False)

    report = {
        "counts_by_source": {k: len(v) for k, v in sources.items()},
        "coverage": {
            "avg_per_query": float(merged.groupby("query_id").size().mean()) if not merged.empty else 0.0,
            "empty_queries": int((merged.groupby("query_id").size() == 0).sum()) if not merged.empty else 0,
        },
        "timing_sec": timing,
        "params_fingerprint": json.dumps({"profile": profile, "candidates": cand_cfg}),
    }
    quick = quick_eval(merged, data.queries, k=min(20, K_final))
    if quick:
        report["quick_eval"] = quick
    report_path = Path(args.report_out or out_dir / "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved candidates to %s", cand_path)
    logger.info("Report saved to %s", report_path)


if __name__ == "__main__":
    main()
