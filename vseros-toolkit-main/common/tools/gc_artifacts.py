
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tools/gc_artifacts.py

Умная очистка каталога artifacts/:
- Инвентаризация: models, sets, features-cache, submissions, validation/adversarial/stability.
- Dry-run по умолчанию: формирует план, ничего не удаляет без --apply.
- Защиты: pinned (PIN/pin.json), referenced (сабмиты/индексы), свежие (--protect-days),
  маски --protect (regex). Активные .lock/.lck — ничего не трогаем.
- Политики удержания: TTL (--ttl-days), keep-last (глобально и per-tag/per-block),
  keep-best-per-tag по метрике, drop-broken, budget (--target-free-gb).
- Корзина (--trash): перемещает в artifacts/.trash/<ts>/… вместо удаления.
- Отчёты: inventory.csv, plan.csv, kept.csv, deleted.csv, summary.json, errors.json, report.html.
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ------------------------- Matplotlib (опц., для html) -------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ------------------------- CLI -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Garbage collector for artifacts/ with dry-run plan and retention policies")

    p.add_argument("--apply", action="store_true", help="Применить план (по умолчанию только dry-run)")
    p.add_argument("--trash", action="store_true", help="Перемещать в artifacts/.trash/<ts>/ вместо удаления")
    p.add_argument("--rm-empty-dirs", action="store_true", help="После удаления — зачистить пустые директории")

    p.add_argument("--ttl-days", type=int, default=None, help="Удалять старше N дней (если не защищены)")
    p.add_argument("--protect-days", type=int, default=3, help="Всегда защищать объекты моложе N дней")

    p.add_argument("--keep-last", type=int, default=None, help="Глобально оставить последние N объектов (по возрасту)")
    p.add_argument("--keep-last-per-tag", type=int, default=None, help="Оставлять последние N для каждого tag (models/sets)")
    p.add_argument("--keep-best-per-tag", type=int, default=None, help="Оставлять лучшие K по метрике (models)")
    p.add_argument("--keep-last-per-block", type=int, default=None, help="Оставлять последние N кэшей на каждый block (features)")

    p.add_argument("--only", type=str, default=None, help="Ограничить типы: 'models,features,sets,submissions,validation,adversarial,stability'")
    p.add_argument("--exclude", type=str, default=None, help="Исключить типы (тот же список)")
    p.add_argument("--protect", type=str, default=None, help="Список regex для путей, которые нельзя трогать (через запятую)")

    p.add_argument("--drop-broken", action="store_true", help="Удалять битые объекты (неполные, без ключевых файлов)")
    p.add_argument("--target-free-gb", type=float, default=None, help="Довести свободное место до X ГБ, удаляя самое 'холодное'")

    p.add_argument("--gc-submissions", action="store_true", help="Разрешить удаление submissions (иначе защищены)")
    p.add_argument("--gc-validation", action="store_true", help="Разрешить удаление validation (иначе защищены)")
    p.add_argument("--gc-adversarial", action="store_true", help="Разрешить удаление adversarial (иначе защищены)")
    p.add_argument("--gc-stability", action="store_true", help="Разрешить удаление stability (иначе защищены)")

    p.add_argument("--name", type=str, default="run", help="Имя подпапки отчёта artifacts/gc/<timestamp>-<name>/")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

# ------------------------- FS utils -------------------------

ROOT = Path("artifacts").resolve()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def human_bytes(n: float) -> str:
    units = ["B","KB","MB","GB","TB","PB"]
    i = 0
    while n >= 1024 and i < len(units)-1:
        n /= 1024.0
        i += 1
    return f"{n:.2f} {units[i]}"

def path_age_days(p: Path) -> float:
    try:
        m = p.stat().st_mtime
        c = p.stat().st_ctime
        ts = max(m, c)
    except FileNotFoundError:
        ts = time.time()
    return (time.time() - ts) / 86400.0

def dir_size_bytes(path: Path) -> int:
    """Рекурсивный подсчёт размера. Пропускаем симлинки и системное."""
    total = 0
    if not path.exists():
        return 0
    try:
        for root, dirs, files in os.walk(path, followlinks=False):
            # пропуск скрытых системных
            files = [f for f in files if f not in (".DS_Store",)]
            for f in files:
                fp = Path(root) / f
                try:
                    if not fp.is_symlink():
                        total += fp.stat().st_size
                except FileNotFoundError:
                    pass
    except Exception:
        pass
    return total

def has_active_locks(root: Path) -> bool:
    """Если в дереве есть *.lock|*.lck — считаем, что идут процессы."""
    for p in root.rglob("*"):
        name = p.name.lower()
        if name.endswith(".lock") or name.endswith(".lck"):
            return True
    return False

def rm_tree(path: Path, trash_dir: Optional[Path], apply: bool, verbose: bool, errors: List[str]) -> bool:
    if not path.exists():
        return True
    if not apply:
        if verbose:
            print("[dry-run] remove:", path)
        return True
    try:
        if trash_dir is not None:
            ensure_dir(trash_dir)
            target = trash_dir / path.name
            # если занято — добавим суффикс
            i = 1
            while target.exists():
                target = trash_dir / f"{path.name}__{i}"
                i += 1
            shutil.move(str(path), str(target))
        else:
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
            else:
                shutil.rmtree(path, ignore_errors=False)
        return True
    except Exception as e:
        errors.append(f"rm_tree error for {path}: {e}")
        return False

def remove_empty_dirs(root: Path, verbose: bool):
    removed = 0
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        d = Path(dirpath)
        try:
            if not any(d.iterdir()):
                d.rmdir()
                removed += 1
                if verbose:
                    print("[rm-empty]", d)
        except Exception:
            pass
    return removed

def disk_free_gb(path: Path) -> float:
    try:
        usage = shutil.disk_usage(path)
        return float(usage.free) / (1024**3)
    except Exception:
        return 0.0

# ------------------------- Dataclasses & types -------------------------

@dataclass
class Card:
    type: str                 # models|sets|features|submissions|validation|adversarial|stability
    key: str                  # run_id | tag | block/key | path-name
    path: str                 # absolute path
    size_bytes: int
    age_days: float
    pinned: bool
    referenced: bool
    fresh_protected: bool
    broken: bool
    meta: Dict[str, Any]

# ------------------------- Read indices & manifests -------------------------

def load_json_safe(p: Path) -> Optional[dict]:
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None

def scan_submissions_manifests() -> Tuple[set, set]:
    """Возвращает (used_runs, used_tags) из artifacts/submissions/**/manifest.json."""
    used_runs, used_tags = set(), set()
    base = ROOT / "submissions"
    if not base.exists():
        return used_runs, used_tags
    for manifest in base.rglob("manifest.json"):
        obj = load_json_safe(manifest) or {}
        # допускаем разные названия полей
        runs = obj.get("runs") or obj.get("model_runs") or []
        if isinstance(runs, dict):  # иногда dict с именами
            runs = list(runs.values())
        for r in runs:
            if isinstance(r, str):
                used_runs.add(r)
        tag = obj.get("set_tag") or obj.get("tag")
        if isinstance(tag, str):
            used_tags.add(tag)
        # иногда массив тегов
        tags = obj.get("tags") or []
        for t in tags:
            if isinstance(t, str):
                used_tags.add(t)
    return used_runs, used_tags

def parse_models_index() -> set:
    """Пытаемся извлечь run_id из artifacts/models/index.json (если есть)."""
    runs = set()
    p = ROOT / "models" / "index.json"
    idx = load_json_safe(p) or {}
    # индекс может быть произвольной структуры — соберём все строковые значения, похожие на run_id
    # run_id обычно — имя директории в models/
    models_dir = ROOT / "models"
    known = set([d.name for d in models_dir.glob("*") if d.is_dir()]) if models_dir.exists() else set()
    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            base = Path(x).name
            if base in known:
                runs.add(base)
    walk(idx)
    return runs

def parse_sets_index() -> set:
    """Пытаемся извлечь теги из artifacts/sets/index.json (если есть)."""
    tags = set()
    p = ROOT / "sets" / "index.json"
    idx = load_json_safe(p) or {}
    # ключи вида validate:<tag>:name
    for k in idx.keys():
        if isinstance(k, str) and k.startswith("validate:"):
            parts = k.split(":")
            if len(parts) >= 2:
                tags.add(parts[1])
    # значения могут хранить path
    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            # если путь вида artifacts/sets/<tag>/...
            m = re.search(r"artifacts/sets/([^/]+)/", x)
            if m:
                tags.add(m.group(1))
    walk(idx)
    return tags

# ------------------------- Pinned/Lock helpers -------------------------

def is_pinned_dir(d: Path) -> bool:
    if (d / "PIN").exists():
        return True
    pjson = d / "pin.json"
    obj = load_json_safe(pjson)
    if obj and obj.get("pinned", False):
        return True
    return False

# ------------------------- Inventory builders -------------------------

def read_run_meta(run_dir: Path) -> Dict[str, Any]:
    meta = {}
    m1 = load_json_safe(run_dir / "metrics.json") or {}
    m2 = load_json_safe(run_dir / "meta.json") or {}
    meta.update(m1); meta.update({f"meta_{k}": v for k, v in m2.items()})
    # удобства
    cv = meta.get("cv_mean") or meta.get("cv") or m2.get("cv_mean")
    tag = meta.get("tag") or m2.get("tag") or meta.get("set_tag") or m2.get("set_tag")
    created = meta.get("created_at") or m2.get("created_at")
    meta["cv_mean"] = cv
    meta["tag"] = tag
    meta["created_at"] = created
    return meta

def model_is_broken(run_dir: Path) -> bool:
    # нет test_pred/test_post и нет oof — считаем неполным
    has_pred = any((run_dir / f).exists() for f in ["test_pred.npy", "test_post.npy"])
    has_oof = any(bool(list(run_dir.glob("oof*.npy"))))
    return not (has_pred or has_oof)

def invent_models() -> List[Card]:
    out: List[Card] = []
    base = ROOT / "models"
    if not base.exists():
        return out
    for d in base.glob("*"):
        if not d.is_dir():
            continue
        size = dir_size_bytes(d)
        age = path_age_days(d)
        pinned = is_pinned_dir(d)
        meta = read_run_meta(d)
        card = Card(
            type="models",
            key=d.name,
            path=str(d),
            size_bytes=size,
            age_days=age,
            pinned=pinned,
            referenced=False,
            fresh_protected=False,
            broken=model_is_broken(d),
            meta=meta
        )
        out.append(card)
    return out

def read_set_meta(tag_dir: Path) -> Dict[str, Any]:
    meta = load_json_safe(tag_dir / "meta.json") or {}
    return {"created_at": meta.get("created_at"), "submit": meta.get("submit"), **meta}

def set_is_broken(tag_dir: Path) -> bool:
    # нет ни dense, ни sparse; или нет y/ids
    files = [p.name for p in tag_dir.glob("*")]
    has_dense = ("X_dense_train.parquet" in files) and ("X_dense_test.parquet" in files)
    has_sparse = ("X_sparse_train.npz" in files) and ("X_sparse_test.npz" in files)
    has_y = ("y_train.parquet" in files)
    has_ids = ("ids_test.parquet" in files)
    return not ((has_dense or has_sparse) and has_y and has_ids)

def invent_sets() -> List[Card]:
    out: List[Card] = []
    base = ROOT / "sets"
    if not base.exists():
        return out
    for d in base.glob("*"):
        if not d.is_dir():
            continue
        size = dir_size_bytes(d)
        age = path_age_days(d)
        pinned = is_pinned_dir(d)
        meta = read_set_meta(d)
        card = Card(
            type="sets",
            key=d.name,
            path=str(d),
            size_bytes=size,
            age_days=age,
            pinned=pinned,
            referenced=False,
            fresh_protected=False,
            broken=set_is_broken(d),
            meta=meta
        )
        out.append(card)
    return out

def features_key_is_broken(key_dir: Path) -> bool:
    # пустой каталог — считаем битым
    try:
        for _ in key_dir.rglob("*"):
            return False
        return True
    except Exception:
        return True

def invent_features() -> List[Card]:
    out: List[Card] = []
    base = ROOT / "features"
    if not base.exists():
        return out
    for block in base.glob("*"):
        if not block.is_dir():
            continue
        for key_dir in block.glob("*"):
            if not key_dir.is_dir():
                continue
            size = dir_size_bytes(key_dir)
            age = path_age_days(key_dir)
            pinned = is_pinned_dir(key_dir)
            card = Card(
                type="features",
                key=f"{block.name}/{key_dir.name}",
                path=str(key_dir),
                size_bytes=size,
                age_days=age,
                pinned=pinned,
                referenced=False,
                fresh_protected=False,
                broken=features_key_is_broken(key_dir),
                meta={"block": block.name, "key": key_dir.name}
            )
            out.append(card)
    return out

def invent_simple_tree(kind: str) -> List[Card]:
    """submissions|validation|adversarial|stability — по директориям второго уровня."""
    out: List[Card] = []
    base = ROOT / kind
    if not base.exists():
        return out
    for d in base.rglob("*"):
        if not d.is_dir():
            continue
        # считаем объектом только листовые папки или папки, где есть какие-то полезные файлы
        # ограничим глубину (необязательно), но проще брать только 1й уровень внутри kind
        if d.parent == base or base in d.parents:
            size = dir_size_bytes(d)
            if size == 0:
                continue
            age = path_age_days(d)
            pinned = is_pinned_dir(d)
            broken = False
            if kind == "submissions":
                files = [p.name for p in d.glob("*")]
                if ("submission.csv" not in files) and ("manifest.json" not in files):
                    broken = True
            card = Card(
                type=kind,
                key=d.name,
                path=str(d),
                size_bytes=size,
                age_days=age,
                pinned=pinned,
                referenced=False,
                fresh_protected=False,
                broken=broken,
                meta={}
            )
            out.append(card)
    return out

# ------------------------- Reference graph & protections -------------------------

def build_references() -> Tuple[set, set]:
    used_runs1, used_tags1 = scan_submissions_manifests()
    used_runs2 = parse_models_index()
    used_tags2 = parse_sets_index()
    used_runs = used_runs1 | used_runs2
    used_tags = used_tags1 | used_tags2
    return used_runs, used_tags

def match_any_regex(path: str, regexes: List[re.Pattern]) -> bool:
    return any(r.search(path) for r in regexes)

def apply_protections(cards: List[Card],
                      used_runs: set,
                      used_tags: set,
                      protect_days: Optional[int],
                      protect_regex: Optional[List[re.Pattern]]):
    for c in cards:
        # referenced
        if c.type == "models":
            if c.key in used_runs:
                c.referenced = True
        if c.type == "sets":
            tag = c.key
            if tag in used_tags:
                c.referenced = True
        # freshness
        if protect_days is not None and c.age_days < protect_days:
            c.fresh_protected = True
        # regex
        if protect_regex and match_any_regex(c.path, protect_regex):
            c.pinned = True  # трактуем как принудительный pin

# ------------------------- Filters by scope -------------------------

ALL_TYPES = {"models","sets","features","submissions","validation","adversarial","stability"}

def filter_types(cards: List[Card],
                 only: Optional[set],
                 exclude: Optional[set],
                 allow_groups: Dict[str, bool]) -> List[Card]:
    out = []
    for c in cards:
        if only and c.type not in only:
            continue
        if exclude and c.type in exclude:
            continue
        # safety toggles
        if c.type == "submissions" and not allow_groups.get("submissions", False):
            continue
        if c.type == "validation" and not allow_groups.get("validation", False):
            continue
        if c.type == "adversarial" and not allow_groups.get("adversarial", False):
            continue
        if c.type == "stability" and not allow_groups.get("stability", False):
            continue
        out.append(c)
    return out

# ------------------------- Retention policies -------------------------

def group_models_by_tag(cards: List[Card]) -> Dict[str, List[Card]]:
    by = {}
    for c in cards:
        if c.type != "models":
            continue
        tag = c.meta.get("tag") or "unknown"
        by.setdefault(tag, []).append(c)
    return by

def group_sets_by_tag(cards: List[Card]) -> Dict[str, List[Card]]:
    by = {}
    for c in cards:
        if c.type != "sets":
            continue
        tag = c.key
        by.setdefault(tag, []).append(c)
    return by

def group_features_by_block(cards: List[Card]) -> Dict[str, List[Card]]:
    by = {}
    for c in cards:
        if c.type != "features":
            continue
        block = c.meta.get("block", "unknown")
        by.setdefault(block, []).append(c)
    return by

def importance_score(card: Card) -> float:
    """Чем ВЫШЕ — тем приоритетнее удалять."""
    size_gb = card.size_bytes / (1024**3)
    age = card.age_days
    broken_boost = 100.0 if card.broken else 0.0
    ref_penalty = -1e6 if card.referenced else 0.0
    pin_penalty = -1e6 if card.pinned else 0.0
    # веса: размер и возраст важны
    return broken_boost + size_gb * 10.0 + age - (ref_penalty + pin_penalty)

def apply_retention(cards: List[Card],
                    ttl_days: Optional[int],
                    keep_last_global: Optional[int],
                    keep_last_per_tag: Optional[int],
                    keep_best_per_tag: Optional[int],
                    keep_last_per_block: Optional[int],
                    drop_broken: bool) -> Tuple[List[Card], List[Card], Dict[str, str]]:
    """
    Возвращает (keep, candidates, reason_by_key)
    """
    keep: List[Card] = []
    candidates: List[Card] = []
    reasons: Dict[str, str] = {}

    # 1) Разбиваем по типам
    models = [c for c in cards if c.type == "models"]
    sets_ = [c for c in cards if c.type == "sets"]
    feats = [c for c in cards if c.type == "features"]
    others = [c for c in cards if c.type not in {"models","sets","features"}]

    # Helper: базовая защита
    def protect_or_candidate(c: Card, default_reason: str):
        if c.pinned or c.referenced or c.fresh_protected:
            keep.append(c)
        else:
            candidates.append(c); reasons[c.key] = default_reason

    # 2) models — по tag: keep-best-per-tag, keep-last-per-tag
    if models:
        by_tag = group_models_by_tag(models)
        for tag, arr in by_tag.items():
            arr_sorted_age = sorted(arr, key=lambda x: x.age_days)  # моложе — раньше
            arr_sorted_age_desc = sorted(arr, key=lambda x: -x.age_days)
            # best by metric
            if keep_best_per_tag:
                # чем выше cv_mean — тем лучше; если метрики нет — -inf
                arr_by_cv = sorted(arr, key=lambda x: float(x.meta.get("cv_mean") or -1e9), reverse=True)
                best = []
                for c in arr_by_cv:
                    if len(best) >= keep_best_per_tag:
                        break
                    if c not in keep:
                        keep.append(c); reasons[c.key] = "keep-best-per-tag"
                        best.append(c)
            # last per tag
            if keep_last_per_tag:
                last = []
                for c in arr_sorted_age[:keep_last_per_tag]:
                    if c not in keep:
                        keep.append(c); reasons[c.key] = "keep-last-per-tag"
                        last.append(c)
            # остальное — кандидаты/защита
            for c in arr:
                if c in keep:
                    continue
                protect_or_candidate(c, default_reason="model-default")

    # 3) sets — по tag: keep-last-per-tag
    if sets_:
        by_tag = group_sets_by_tag(sets_)
        for tag, arr in by_tag.items():
            arr_sorted_age = sorted(arr, key=lambda x: x.age_days)  # моложе — раньше
            if keep_last_per_tag:
                kept = 0
                for c in arr_sorted_age[:keep_last_per_tag]:
                    if c not in keep:
                        keep.append(c); reasons[c.key] = "keep-last-per-tag"
                        kept += 1
            for c in arr:
                if c in keep:
                    continue
                protect_or_candidate(c, default_reason="set-default")

    # 4) features — per block: keep-last-per-block
    if feats:
        by_block = group_features_by_block(feats)
        for block, arr in by_block.items():
            arr_sorted_age = sorted(arr, key=lambda x: x.age_days)  # моложе — раньше
            if keep_last_per_block:
                for c in arr_sorted_age[:keep_last_per_block]:
                    if c not in keep:
                        keep.append(c); reasons[c.key] = "keep-last-per-block"
            for c in arr:
                if c in keep:
                    continue
                protect_or_candidate(c, default_reason="feature-default")

    # 5) others (submissions/validation/adversarial/stability)
    for c in others:
        protect_or_candidate(c, default_reason=f"{c.type}-default")

    # 6) TTL — удалять старше ttl_days (если не защищены)
    if ttl_days is not None:
        for c in list(candidates):
            if c.age_days <= ttl_days:
                # слишком свежие — убрать из кандидатов
                candidates.remove(c)
                keep.append(c)
                reasons[c.key] = "ttl-protect"
            else:
                reasons[c.key] = reasons.get(c.key, "") + ("" if reasons.get(c.key) else "") + " ttl-expired"

    # 7) drop-broken — добавить битые объекты в кандидаты (если не защищены)
    if drop_broken:
        for c in cards:
            if c.broken and (c not in keep) and (c not in candidates) and not (c.pinned or c.referenced or c.fresh_protected):
                candidates.append(c); reasons[c.key] = "broken"

    # 8) keep-last (глобально) — поверх всего
    if keep_last_global:
        # Оставим глобально N самых молодых объектов (по возрасту), добавив их в keep
        all_sorted_young = sorted(cards, key=lambda x: x.age_days)
        extra = 0
        for c in all_sorted_young[:keep_last_global]:
            if c not in keep:
                keep.append(c); reasons[c.key] = "keep-last-global"
                extra += 1
            if c in candidates:
                candidates.remove(c)

    # удалить из candidates защищённые (на случай пересечений)
    candidates = [c for c in candidates if c not in keep and not (c.pinned or c.referenced or c.fresh_protected)]
    return keep, candidates, reasons

# ------------------------- Planning -------------------------

def make_plan(inventory: List[Card],
              args,
              used_runs: set,
              used_tags: set,
              protect_regex: Optional[List[re.Pattern]]) -> Dict[str, Any]:
    # фильтрация типов
    only = set([s.strip() for s in args.only.split(",") if s.strip()]) if args.only else None
    exclude = set([s.strip() for s in args.exclude.split(",") if s.strip()]) if args.exclude else None
    allow_groups = {
        "submissions": args.gc_submissions,
        "validation": args.gc_validation,
        "adversarial": args.gc_adversarial,
        "stability": args.gc_stability
    }

    inv_filtered = filter_types(inventory, only, exclude, allow_groups)
    # protections
    apply_protections(inv_filtered, used_runs, used_tags, args.protect_days,
                      [re.compile(r) for r in (args.protect.split(",") if args.protect else [])] if args.protect else None)

    # retention
    keep, candidates, reasons = apply_retention(inv_filtered,
                                                ttl_days=args.ttl_days,
                                                keep_last_global=args.keep_last,
                                                keep_last_per_tag=args.keep_last_per_tag,
                                                keep_best_per_tag=args.keep_best_per_tag,
                                                keep_last_per_block=args.keep_last_per_block,
                                                drop_broken=args.drop_broken)

    # бюджет свободного места
    selected_for_delete = list(candidates)
    if args.target_free_gb is not None:
        current_free = disk_free_gb(ROOT)
        need = args.target_free_gb - current_free
        if need > 0:
            # сортируем кандидатов по importance убыв.
            ordering = sorted(candidates, key=importance_score, reverse=True)
            acc = 0.0
            chosen = []
            for c in ordering:
                sz_gb = c.size_bytes / (1024**3)
                chosen.append(c)
                acc += sz_gb
                if acc >= need:
                    break
            selected_for_delete = chosen

    # помечаем причины
    reason_by_key = {**reasons}
    for c in selected_for_delete:
        if c.key not in reason_by_key:
            reason_by_key[c.key] = "candidate"
        # уточнение причин
        if c.broken:
            reason_by_key[c.key] += "|broken"
        if args.ttl_days is not None and c.age_days > args.ttl_days:
            reason_by_key[c.key] += "|ttl"

    plan = {
        "keep": keep,
        "candidates": candidates,
        "delete": selected_for_delete,
        "reason_by_key": reason_by_key,
    }
    return plan

def summarize_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    del_list: List[Card] = plan["delete"]
    keep_list: List[Card] = plan["keep"]
    total_del_bytes = sum(c.size_bytes for c in del_list)
    by_type = {}
    for c in del_list:
        by_type[c.type] = by_type.get(c.type, 0) + c.size_bytes
    return {
        "n_delete": len(del_list),
        "n_keep": len(keep_list),
        "bytes_delete": int(total_del_bytes),
        "gb_delete": float(total_del_bytes / (1024**3)),
        "by_type_bytes": {k: int(v) for k, v in by_type.items()},
    }

def cards_to_dataframe(cards: List[Card]) -> pd.DataFrame:
    rows = []
    for c in cards:
        rows.append({
            "type": c.type,
            "key": c.key,
            "path": c.path,
            "size_bytes": c.size_bytes,
            "size_h": human_bytes(c.size_bytes),
            "age_days": round(c.age_days, 3),
            "pinned": c.pinned,
            "referenced": c.referenced,
            "fresh_protected": c.fresh_protected,
            "broken": c.broken,
            **{f"meta.{k}": v for k, v in (c.meta or {}).items()}
        })
    return pd.DataFrame(rows)

# ------------------------- Execute -------------------------

def execute_plan(plan: Dict[str, Any], out_dir: Path, args) -> Tuple[List[str], int]:
    errors: List[str] = []
    trash_dir = None
    if args.apply and args.trash:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        trash_dir = ROOT / ".trash" / ts
        ensure_dir(trash_dir)

    # safety: если есть активные lock — не удалять вообще
    if has_active_locks(ROOT):
        errors.append("Detected active .lock/.lck files under artifacts/. Aborting apply.")
        return errors, 0

    deleted = 0
    for c in plan["delete"]:
        ok = rm_tree(Path(c.path), trash_dir, apply=args.apply, verbose=args.verbose, errors=errors)
        if ok:
            deleted += 1

    # зачистка пустых папок
    if args.apply and args.rm_empty_dirs:
        try:
            removed = remove_empty_dirs(ROOT, verbose=args.verbose)
            if args.verbose:
                print("[rm-empty-dirs] removed:", removed)
        except Exception as e:
            errors.append(f"rm-empty-dirs error: {e}")

    return errors, deleted

# ------------------------- HTML report (короткий) -------------------------

def build_html_report(out_dir: Path, inv_df: pd.DataFrame, keep_df: pd.DataFrame,
                      del_df: pd.DataFrame, summary: Dict[str, Any]):
    if plt is None:
        # простая HTML без графиков
        html = io.StringIO()
        html.write("<html><head><meta charset='utf-8'><title>GC Report</title></head><body>")
        html.write("<h1>Artifacts GC Report</h1>")
        html.write("<h2>Summary</h2><pre>")
        html.write(json.dumps(summary, ensure_ascii=False, indent=2))
        html.write("</pre>")
        html.write("<h2>Delete (head)</h2>")
        html.write(del_df.head(50).to_html(index=False))
        html.write("<h2>Keep (head)</h2>")
        html.write(keep_df.head(50).to_html(index=False))
        html.write("</body></html>")
        (out_dir / "report.html").write_text(html.getvalue(), encoding="utf-8")
        return

    # с графиками
    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)

    # распределение по типам (байты)
    by_type = del_df.groupby("type")["size_bytes"].sum().sort_values(ascending=False)
    fig1 = plt.figure()
    by_type.plot(kind="bar")
    plt.title("Bytes to delete by type")
    fig1.savefig(plots_dir / "by_type_delete.png", bbox_inches="tight")
    plt.close(fig1)

    # топ-20 самых тяжёлых к удалению
    top20 = del_df.sort_values("size_bytes", ascending=False).head(20)
    fig2 = plt.figure()
    plt.barh([f"{r['type']}:{r['key']}" for _, r in top20.iterrows()], top20["size_bytes"].values)
    plt.title("Top-20 heavy deletions")
    plt.gca().invert_yaxis()
    fig2.savefig(plots_dir / "heavy_delete.png", bbox_inches="tight")
    plt.close(fig2)

    html = io.StringIO()
    html.write("<html><head><meta charset='utf-8'><title>GC Report</title></head><body>")
    html.write("<h1>Artifacts GC Report</h1>")
    html.write("<h2>Summary</h2><pre>")
    html.write(json.dumps(summary, ensure_ascii=False, indent=2))
    html.write("</pre>")

    html.write("<h2>Delete (head)</h2>")
    html.write(del_df.head(50).to_html(index=False))
    html.write("<h2>Keep (head)</h2>")
    html.write(keep_df.head(50).to_html(index=False))

    html.write("<h2>Plots</h2>")
    html.write("<div><img src='plots/by_type_delete.png' width='640'></div>")
    html.write("<div><img src='plots/heavy_delete.png' width='640'></div>")

    html.write("</body></html>")
    (out_dir / "report.html").write_text(html.getvalue(), encoding="utf-8")

# ------------------------- Main -------------------------

def main():
    args = parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "gc" / f"{ts}-{args.name}"
    ensure_dir(out_dir)

    # область сканирования
    types_allowed = set([s.strip() for s in args.only.split(",") if s.strip()]) if args.only else None
    if types_allowed:
        for t in types_allowed:
            if t not in ALL_TYPES:
                print(f"[warn] unknown type in --only: {t}", file=sys.stderr)

    if args.verbose:
        print("[root]", ROOT)

    # Инвентаризация
    inventory: List[Card] = []
    inventory.extend(invent_models())
    inventory.extend(invent_sets())
    inventory.extend(invent_features())
    inventory.extend(invent_simple_tree("submissions"))
    inventory.extend(invent_simple_tree("validation"))
    inventory.extend(invent_simple_tree("adversarial"))
    inventory.extend(invent_simple_tree("stability"))

    if args.verbose:
        print(f"[inventory] total objects: {len(inventory)}")

    # Ссылочная защита
    used_runs, used_tags = build_references()
    if args.verbose:
        print(f"[refs] used_runs: {len(used_runs)}; used_tags: {len(used_tags)}")

    # План
    protect_regex = [re.compile(r) for r in (args.protect.split(",") if args.protect else [])] if args.protect else None
    plan = make_plan(inventory, args, used_runs, used_tags, protect_regex)
    plan_delete: List[Card] = plan["delete"]
    plan_keep: List[Card] = plan["keep"]

    # Отчёты (таблицы)
    inv_df = cards_to_dataframe(inventory)
    del_df = cards_to_dataframe(plan_delete)
    keep_df = cards_to_dataframe(plan_keep)

    inv_df.to_csv(out_dir / "inventory.csv", index=False)
    del_df.to_csv(out_dir / "plan.csv", index=False)
    keep_df.to_csv(out_dir / "kept.csv", index=False)

    # Сводка
    summary = summarize_plan(plan)
    summary["created_at"] = datetime.now().isoformat()
    summary["apply"] = bool(args.apply)
    summary["trash"] = bool(args.trash)
    summary["ttl_days"] = args.ttl_days
    summary["protect_days"] = args.protect_days
    summary["keep_last"] = args.keep_last
    summary["keep_last_per_tag"] = args.keep_last_per_tag
    summary["keep_best_per_tag"] = args.keep_best_per_tag
    summary["keep_last_per_block"] = args.keep_last_per_block
    summary["drop_broken"] = args.drop_broken
    summary["target_free_gb"] = args.target_free_gb
    summary["disk_free_gb_before"] = disk_free_gb(ROOT)
    summary["objects_total"] = len(inventory)

    # HTML
    try:
        build_html_report(out_dir, inv_df, keep_df, del_df, summary)
    except Exception as e:
        (out_dir / "errors.json").write_text(json.dumps({"html_error": str(e)}, ensure_ascii=False, indent=2), encoding="utf-8")

    # Применение
    errors, deleted = execute_plan(plan, out_dir, args)

    summary["deleted"] = int(deleted)
    summary["disk_free_gb_after"] = disk_free_gb(ROOT)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if errors:
        (out_dir / "errors.json").write_text(json.dumps({"errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")

    # Deleted.csv для удобства
    if args.apply:
        # перечитаем plan.csv как фактически удалённые — те, что ещё существуют, не удалось удалить
        deleted_rows = []
        for _, r in del_df.iterrows():
            if not Path(str(r["path"])).exists():
                deleted_rows.append(r)
        pd.DataFrame(deleted_rows).to_csv(out_dir / "deleted.csv", index=False)

    # Итог
    print("=== GC COMPLETED ===")
    print("report:", out_dir.as_posix())
    print("delete count:", summary["n_delete"], "planned; actually removed:", summary["deleted"])
    print("space to delete (planned):", human_bytes(summary["bytes_delete"]))
    print("free space before/after (GB):", f"{summary['disk_free_gb_before']:.2f} -> {summary['disk_free_gb_after']:.2f}")
    if errors:
        print("errors:", len(errors))
        for e in errors[:5]:
            print("  -", e)

if __name__ == "__main__":
    main()
