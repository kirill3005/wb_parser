from __future__ import annotations

import dataclasses
from typing import Any, Dict, Iterable

import pandas as pd
import yaml


SCHEMA_VERSION = "0.1"


@dataclasses.dataclass
class Schema:
    """Container for column mapping and query scope with validation helpers.

    Parameters
    ----------
    interactions : Dict[str, str]
        Mapping from canonical interaction names to dataset columns.
    items : Dict[str, str]
        Mapping from canonical item names to dataset columns.
    queries : Dict[str, Any]
        Query settings, including ``scope`` and ``id_col``.
    version : str
        Schema version string for cache keys.
    """

    interactions: Dict[str, str]
    items: Dict[str, str]
    queries: Dict[str, Any]
    version: str = SCHEMA_VERSION

    @classmethod
    def from_yaml(cls, path: str) -> "Schema":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(
            interactions=cfg.get("interactions", {}),
            items=cfg.get("items", {}),
            queries=cfg.get("queries", {}),
            version=cfg.get("version", SCHEMA_VERSION),
        )

    @property
    def query_id_col(self) -> str:
        return self.queries.get("id_col", "query_id")

    @property
    def query_scope(self) -> str:
        return self.queries.get("scope", "session")

    def canonical_to_source(self, section: str, name: str) -> str:
        mapping = getattr(self, section)
        return mapping.get(name, name)

    def source_to_canonical(self, section: str, name: str) -> str:
        mapping = getattr(self, section)
        inv = {v: k for k, v in mapping.items()}
        return inv.get(name, name)

    def ensure_timestamp_utc(self, df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
        """Ensure timestamp columns are timezone-aware and UTC.

        Missing columns are ignored to keep the adapter permissive.
        """

        out = df.copy()
        for col in cols:
            if col not in out.columns:
                continue
            out[col] = pd.to_datetime(out[col], utc=True)
        return out

    def validate_required(self, df: pd.DataFrame, *, section: str, required: Iterable[str]) -> None:
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns for {section}: {missing}")


def ensure_dtype(series: pd.Series, dtype: str) -> pd.Series:
    """Cast a series to the requested dtype if possible."""

    try:
        return series.astype(dtype)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(f"Failed to cast series to {dtype}: {exc}") from exc
