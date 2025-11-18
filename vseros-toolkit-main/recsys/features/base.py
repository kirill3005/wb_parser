from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Protocol

import numpy as np
import pandas as pd

from recsys.dataio.schema import Schema

logger = logging.getLogger(__name__)


class FeatureBlock(Protocol):
    """Interface for all feature blocks.

    Blocks must be time-safe: any aggregation in :meth:`transform` must only use
    interactions with timestamps ``<= ts_query`` for each pair.
    """

    name: str

    def fit(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None,
        *,
        schema: Schema,
        profile: Dict,
        rng: np.random.RandomState,
    ) -> "FeatureBlock":
        ...

    def transform(
        self, pairs: pd.DataFrame, *, schema: Schema, profile: Dict
    ) -> pd.DataFrame:
        ...


@dataclass
class BlockSpec:
    name: str
    params: Dict
    enabled: bool = True


# registry is populated in __init__.py of the module
REGISTRY: Dict[str, type] = {}


def register(name: str, cls: type) -> None:
    REGISTRY[name] = cls
    logger.debug("Registered feature block %s -> %s", name, cls.__name__)


def build_blocks(cfg: Dict, profile: Dict) -> Dict[str, FeatureBlock]:
    """Instantiate blocks from configuration merged with profile overrides."""

    blocks_cfg: Dict[str, Dict] = cfg.get("blocks", {})
    blocks: Dict[str, FeatureBlock] = {}
    for name, params in blocks_cfg.items():
        enabled = params.get("enabled", True)
        profile_params = profile.get("blocks", {}).get(name, {})
        merged = {**params, **profile_params}
        if not merged.get("enabled", enabled):
            continue
        cls = REGISTRY.get(name)
        if cls is None:
            logger.warning("Unknown feature block %s", name)
            continue
        block_params = {k: v for k, v in merged.items() if k != "enabled"}
        blocks[name] = cls(**block_params)
    return blocks
