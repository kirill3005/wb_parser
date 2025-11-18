from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


def _normalize_paths(paths: Iterable[str | Path], max_per_id: Optional[int], exts: tuple[str, ...]) -> List[str]:
    normed: List[str] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        if path.is_dir():
            continue
        if path.suffix.lower() not in {e.lower() for e in exts}:
            continue
        normed.append(path.resolve().as_posix())
    normed.sort()
    if max_per_id is not None:
        normed = normed[:max_per_id]
    return normed


def build_from_dir(
    images_root: str | Path,
    ids: Iterable,
    pattern: str = "{id}/**/*",
    exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
    max_per_id: int | None = None,
) -> Dict[str, List[str]]:
    """
    Строит индекс путей, подставляя ``{id}`` в ``pattern``.

    Ищет рекурсивно, фильтрует по расширениям, нормализует пути
    и уважает ``max_per_id``.
    """

    base = Path(images_root)
    result: Dict[str, List[str]] = {}
    for raw_id in ids:
        sid = str(raw_id)
        glob_pattern = str(base / pattern.format(id=sid))
        found = glob.glob(glob_pattern, recursive=True)
        result[sid] = _normalize_paths(found, max_per_id, exts)
    return result


def build_from_csv(
    csv_path: str | Path,
    id_col: str = "id",
    path_col: str = "path",
    max_per_id: int | None = None,
) -> Dict[str, List[str]]:
    """Группирует пути по ``id`` из CSV (пути могут быть относительными)."""

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if id_col not in df.columns or path_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{id_col}' and '{path_col}'")

    base_dir = csv_path.parent
    result: Dict[str, List[str]] = {}
    for sid, group in df.groupby(id_col):
        paths = []
        for p in group[path_col].tolist():
            path = Path(p)
            if not path.is_absolute():
                path = base_dir / path
            paths.append(path)
        result[str(sid)] = _normalize_paths(paths, max_per_id, exts=(".jpg", ".jpeg", ".png", ".webp"))
    return result
