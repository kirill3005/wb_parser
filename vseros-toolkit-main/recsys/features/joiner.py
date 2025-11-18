from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from recsys.dataio.schema import Schema
from recsys.features.base import build_blocks
from recsys.features.catalog import build_catalog, save_catalog

logger = logging.getLogger(__name__)


def merge_feature_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    result = frames[0]
    for df in frames[1:]:
        result = result.merge(df, on=["query_id", "item_id"], how="left")
    return result


def enforce_dtype(df: pd.DataFrame, cast_float32: bool = True) -> pd.DataFrame:
    for col in df.columns:
        if col in {"query_id", "item_id"}:
            continue
        if cast_float32 and pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(np.float32)
        df[col] = df[col].fillna(0)
    return df


def align_columns(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    all_cols = sorted(set(train.columns) | set(test.columns))
    ordered = [c for c in ["query_id", "item_id"] if c in all_cols] + [c for c in all_cols if c not in {"query_id", "item_id"}]
    for col in ordered:
        if col not in train.columns:
            train[col] = 0
        if col not in test.columns:
            test[col] = 0
    return train[ordered], test[ordered], ordered


class FeatureJoiner:
    def __init__(self, cfg: Dict, profile: Dict, *, schema: Schema, rng: np.random.RandomState):
        self.cfg = cfg
        self.profile = profile
        self.schema = schema
        self.rng = rng
        self.blocks = build_blocks(cfg, profile)
        logger.info("Active feature blocks: %s", list(self.blocks))

    def fit(self, interactions: pd.DataFrame, items: pd.DataFrame | None) -> None:
        for name, block in self.blocks.items():
            start = time.time()
            block.fit(interactions, items, schema=self.schema, profile=self.profile, rng=self.rng)
            logger.info("Fitted %s in %.3fs", name, time.time() - start)

    def transform(self, pairs_train: pd.DataFrame, pairs_test: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        frames_train = []
        frames_test = []
        timings: Dict[str, float] = {}
        block_columns: Dict[str, List[str]] = {}
        for name, block in self.blocks.items():
            start = time.time()
            ftrain = block.transform(pairs_train, schema=self.schema, profile=self.profile)
            ftest = block.transform(pairs_test, schema=self.schema, profile=self.profile)
            timings[name] = time.time() - start
            frames_train.append(ftrain)
            frames_test.append(ftest)
            block_columns[name] = [c for c in ftrain.columns if c not in {"query_id", "item_id"}]
            logger.info("Block %s produced %d features", name, len(block_columns[name]))

        merged_train = merge_feature_frames(frames_train)
        merged_test = merge_feature_frames(frames_test)
        merged_train = enforce_dtype(merged_train, cast_float32=self.cfg.get("io", {}).get("cast_float32", True))
        merged_test = enforce_dtype(merged_test, cast_float32=self.cfg.get("io", {}).get("cast_float32", True))
        merged_train, merged_test, ordered = align_columns(merged_train, merged_test)

        return {
            "train": merged_train,
            "test": merged_test,
            "columns": ordered,
            "catalog": build_catalog(block_columns, timings),
            "timings": timings,
        }


def save_outputs(out_dir: Path, outputs: Dict, cfg: Dict, pairs_train: pd.DataFrame, pairs_test: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dense = cfg.get("io", {}).get("dense", True)
    if dense:
        outputs["train"].to_parquet(out_dir / "X_train.parquet", index=False)
        outputs["test"].to_parquet(out_dir / "X_test.parquet", index=False)
    else:
        import scipy.sparse as sp

        sp.save_npz(out_dir / "X_train.npz", sp.csr_matrix(outputs["train"].drop(columns=["query_id", "item_id"])) )
        sp.save_npz(out_dir / "X_test.npz", sp.csr_matrix(outputs["test"].drop(columns=["query_id", "item_id"])) )
    with open(out_dir / "columns.json", "w", encoding="utf-8") as f:
        json.dump(outputs["columns"], f, indent=2)
    save_catalog(out_dir / "catalog.json", outputs["catalog"])
    pairs_train.to_parquet(out_dir / "pairs_train.parquet", index=False)
    pairs_test.to_parquet(out_dir / "pairs_test.parquet", index=False)
    report = {
        "num_train": len(outputs["train"]),
        "num_test": len(outputs["test"]),
        "num_features": len(outputs["columns"]) - 2,
        "timings": outputs.get("timings", {}),
    }
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
