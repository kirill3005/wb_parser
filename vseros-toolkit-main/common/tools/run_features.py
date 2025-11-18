#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tools/run_features.py

Единый скрипт построения фичей:
train/test -> сплиты -> блоки (с кэшем) -> сборка dense/sparse -> сохранение артефактов.

Запуск (минимум):
  python tools/run_features.py --profile tour --tag s5e11_run1 --id-col id --target-col loan_paid_back
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import math
import pickle
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# --- опционально: SciPy только для sparse сохранения
try:
    from scipy import sparse as sp
except Exception:
    sp = None  # скрипт отработает без sparse-блоков

# --- наши модули репозитория (как в ноутбуке)
try:
    from common.features import store, assemble
    from common.features import (
        num_basic, cat_freq, cat_te_oof, text_tfidf,
        geo_grid, geo_neighbors, time_agg,
        crosses, img_index, img_stats, img_embed
    )
    try:
        from common.cache import make_key as _make_key
    except Exception:
        _make_key = None
except Exception as e:
    print("[fatal] Не удалось импортировать модули из common.features.*. Проверь PYTHONPATH и структуру репо.", file=sys.stderr)
    raise

# --- YAML (без внешних зависимостей ок, но PyYAML очень желателен)
try:
    import yaml
except Exception:
    yaml = None

# -----------------------------
# Утилиты
# -----------------------------
def read_yaml(path: Optional[Union[str, Path]]) -> Dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML-файл не найден: {p}")
    if yaml is None:
        raise RuntimeError("PyYAML не установлен, а профиль указан. Установи pyyaml или не передавай --profile / --blocks-yaml.")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def setup_logging(log_level: str = "INFO", log_file: Optional[Union[str, Path]] = None) -> logging.Logger:
    lvl = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger("run_features")
    logger.setLevel(lvl)
    logger.handlers[:] = []

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(lvl)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(lvl)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # Утихомирить болтливые либы
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    return logger

def get_git_commit(base: Union[str, Path]) -> Optional[str]:
    try:
        res = subprocess.run(
            ["git", "-C", str(base), "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True
        )
        return res.stdout.strip()
    except Exception:
        return None

def env_info() -> Dict[str, Any]:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": _try_import_version("numpy"),
        "pandas": _try_import_version("pandas"),
        "scipy": _try_import_version("scipy"),
        "lightgbm": _try_import_version("lightgbm"),
        "xgboost": _try_import_version("xgboost"),
        "catboost": _try_import_version("catboost"),
    }
    return info

def _try_import_version(pkg: str) -> Optional[str]:
    try:
        mod = __import__(pkg)
        return getattr(mod, "__version__", None)
    except Exception:
        return None

def mem_gb_df(df: pd.DataFrame) -> float:
    try:
        return float(df.memory_usage(deep=True).sum()) / (1024 ** 3)
    except Exception:
        return 0.0

def auto_detect_columns(train: pd.DataFrame, exclude: set) -> Tuple[List[str], List[str], List[str]]:
    num_cols, cat_cols, text_cols = [], [], []
    for c in train.columns:
        if c in exclude:
            continue
        s = train[c]
        if pd.api.types.is_numeric_dtype(s):
            num_cols.append(c)
        elif pd.api.types.is_string_dtype(s):
            # эвристика "длинного текста"
            is_text_ratio = s.map(lambda x: isinstance(x, str) and len(x) > 30).mean()
            if is_text_ratio > 0.3:
                text_cols.append(c)
            else:
                cat_cols.append(c)
        else:
            # fallback
            cat_cols.append(c)
    return num_cols, cat_cols, text_cols

def check_finite_dense(df: pd.DataFrame, safe: bool, logger: logging.Logger) -> None:
    bad = ~np.isfinite(df.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all()
    if bad:
        msg = "В dense-матрице обнаружены NaN/inf. Проверь трансформации."
        if safe:
            logger.error(msg)
            raise RuntimeError(msg)
        else:
            logger.warning(msg)

def subsample_indices(n: int, frac: float, seed: int, stratify: Optional[np.ndarray] = None) -> np.ndarray:
    if frac >= 0.9999:
        return np.arange(n, dtype=int)
    rs = np.random.RandomState(seed)
    if stratify is not None:
        # стратифицированная подвыборка по бинарному таргету
        uniq, inv = np.unique(stratify, return_inverse=True)
        idx_list = []
        for k in range(len(uniq)):
            group_idx = np.where(inv == k)[0]
            m = max(1, int(round(len(group_idx) * frac)))
            choose = rs.choice(group_idx, size=m, replace=False)
            idx_list.append(choose)
        idx = np.concatenate(idx_list)
        idx.sort()
        return idx
    else:
        m = max(1, int(round(n * frac)))
        idx = np.arange(n)
        rs.shuffle(idx)
        return np.sort(idx[:m])

def make_folds(
    train: pd.DataFrame,
    target: Optional[pd.Series],
    split_kind: str,
    n_splits: int,
    seed: int,
    date_col: Optional[str],
    group_col: Optional[str],
    time_embargo: Optional[str],
    logger: logging.Logger,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    n = len(train)
    idx_all = np.arange(n, dtype=int)

    if split_kind == "time":
        if not date_col:
            raise RuntimeError("TIME split выбран, но date_col не задан.")
        df = train[[date_col]].copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if df[date_col].isna().any():
            raise RuntimeError("В date_col есть NaT; проверь формат дат.")
        df = df.sort_values(date_col).reset_index()
        fold_sizes = np.full(n_splits, len(df) // n_splits, dtype=int)
        fold_sizes[:len(df) % n_splits] += 1
        folds: List[Tuple[np.ndarray, np.ndarray]] = []
        cur = 0
        embargo = None
        if time_embargo:
            try:
                embargo = pd.to_timedelta(time_embargo)
            except Exception:
                embargo = None
        for k in range(n_splits):
            fs = fold_sizes[k]
            val_pos = np.arange(cur, cur + fs)
            val_idx = df.loc[val_pos, "index"].to_numpy()
            tr_mask = np.ones(len(df), dtype=bool)
            tr_mask[val_pos] = False
            if embargo is not None:
                min_val_time = df.loc[val_pos, date_col].min()
                # исключим train-объекты, попадающие в "эмбарго" перед минимумом валидации
                tr_mask &= (df[date_col] <= (min_val_time - embargo))
            train_idx = df.loc[tr_mask, "index"].to_numpy()
            folds.append((train_idx, val_idx))
            cur += fs
        logger.info(f"time split готов: {n_splits} фолдов с эмбарго={time_embargo}")
        return folds

    elif split_kind == "group":
        if not group_col:
            raise RuntimeError("GROUP split выбран, но group_col не задан.")
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=n_splits)
        groups = train[group_col].to_numpy()
        folds = [(tr, va) for tr, va in gkf.split(idx_all, groups=groups)]
        logger.info("group split готов")
        return folds

    elif split_kind == "stratified":
        if target is None:
            raise RuntimeError("STRATIFIED split требует target.")
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = [(tr, va) for tr, va in skf.split(idx_all, target.to_numpy())]
        logger.info("stratified split готов")
        return folds

    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = [(tr, va) for tr, va in kf.split(idx_all)]
        logger.info("kfold split готов")
        return folds

def make_cache_key(block_name: str, params: Dict[str, Any], seed: int, git_hash: Optional[str]) -> str:
    payload = {
        "block": block_name,
        "params": params,
        "seed": seed,
        "git": git_hash or "unknown"
    }
    if _make_key is not None:
        try:
            return _make_key(payload)
        except Exception:
            pass
    # простой резервный ключ
    return str(abs(hash(json.dumps(payload, sort_keys=True, ensure_ascii=False))) % (10**12))

def ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

# -----------------------------
# Билдеры блоков
# -----------------------------
class SafeError(RuntimeError):
    pass

def run_block(
    FS: store.FeatureStore,
    name: str,
    fn,
    cache_key: str,
    logger: logging.Logger,
    safe: bool,
    use_cache: bool,
    *args, **kwargs
):
    if use_cache:
        try:
            pkg = FS.load_cached(name, cache_key)
            if pkg is not None:
                FS.add(pkg)
                logger.info(f"[CACHE] {name} | cols=+{getattr(pkg.train, 'shape', ['?','?'])[1] if hasattr(pkg, 'train') else '?'}")
                return pkg, True
        except Exception as e:
            logger.warning(f"[cache miss error] {name}: {e}")

    t0 = time.time()
    try:
        pkg = fn(*args, **kwargs)
    except Exception as e:
        logger.exception(f"[block failed] {name}: {e}")
        if safe:
            raise SafeError(f"Блок {name} упал в safe-режиме.") from e
        else:
            logger.warning(f"[skip] {name} из-за ошибки")
            return None, False

    try:
        FS.add(pkg)
        if use_cache:
            FS.save_cached(name, cache_key, pkg)
    except Exception as e:
        logger.warning(f"[cache save warn] {name}: {e}")

    dt = time.time() - t0
    ncols = getattr(pkg.train, "shape", [None, None])[1] if hasattr(pkg, "train") else None
    logger.info(f"[OK] {name} in {dt:.1f}s | +{ncols} cols | kind={getattr(pkg, 'kind','?')}")
    return pkg, False

# -----------------------------
# Основной скрипт
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build features set (dense/sparse) with caching & OOF safety.")
    # основные
    p.add_argument("--profile", type=str, default=None, help="Имя профиля в profiles/ или путь к YAML.")
    p.add_argument("--tag", type=str, required=True, help="Имя набора фич (папка artifacts/sets/<tag>).")
    p.add_argument("--base", type=str, default=".", help="Корень репозитория.")
    p.add_argument("--data-dir", type=str, default=None, help="Папка с данными. По умолчанию <base>/data")
    p.add_argument("--train", type=str, default=None, help="Путь к train.csv (перекрывает data-dir).")
    p.add_argument("--test", type=str, default=None, help="Путь к test.csv (перекрывает data-dir).")

    # колонки
    p.add_argument("--id-col", type=str, required=True)
    p.add_argument("--target-col", type=str, default=None)
    p.add_argument("--date-col", type=str, default=None)
    p.add_argument("--group-col", type=str, default=None)
    p.add_argument("--lat-col", type=str, default=None)
    p.add_argument("--lon-col", type=str, default=None)

    # режимы
    p.add_argument("--split-kind", type=str, default=None, choices=["kfold","stratified","group","time"])
    p.add_argument("--n-splits", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--frac", type=float, default=None, help="Доля данных: 1.0, 0.4 (gate), 0.15 (scout)")
    p.add_argument("--fast", action="store_true", help="Ужать тяжёлые параметры блоков")
    p.add_argument("--safe", action="store_true", help="Жёсткие анти-утечки и падение на ошибках")
    p.add_argument("--use-cache", action="store_true", help="Использовать кэш пакетов")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--time-embargo", type=str, default=None, help='Напр. "2D", "3h" для time split')

    # блоки
    p.add_argument("--blocks-yaml", type=str, default=None, help="YAML со списком блоков и параметрами.")
    p.add_argument("--only", type=str, default=None, help="Список блоков через запятую, чтобы оставить только их.")
    p.add_argument("--exclude", type=str, default=None, help="Список блоков через запятую, чтобы исключить.")

    # поведение
    p.add_argument("--save-set", action="store_true", help="Сохранить X/y/folds/ids/каталоги.")
    p.add_argument("--overwrite", action="store_true", help="Перезаписать папку набора, иначе резюм.")
    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--log-file", type=str, default=None)
    p.add_argument("--dry-run", action="store_true", help="Показать план и выйти.")
    return p.parse_args()

def merge_cfg(cli: argparse.Namespace, profile: Dict[str, Any]) -> Dict[str, Any]:
    """CLI > profile > defaults."""
    cfg: Dict[str, Any] = {}

    # общие пути/режимы
    cfg["base"] = Path(cli.base).resolve()
    cfg["data_dir"] = Path(cli.data_dir).resolve() if cli.data_dir else (cfg["base"] / "data")
    if cli.train: cfg["train_path"] = Path(cli.train)
    else: cfg["train_path"] = cfg["data_dir"] / "train.csv"
    if cli.test: cfg["test_path"] = Path(cli.test)
    else: cfg["test_path"] = cfg["data_dir"] / "test.csv"

    cfg["out_dir"] = Path("artifacts/sets") / cli.tag
    cfg["tag"] = cli.tag

    # профили/дефолты
    cfg["seed"]   = cli.seed if cli.seed is not None else profile.get("seed", 42)
    cfg["frac"]   = cli.frac if cli.frac is not None else profile.get("frac", 1.0)
    cfg["n_splits"]   = cli.n_splits if cli.n_splits is not None else profile.get("n_splits", 5)
    cfg["fast"]   = True if cli.fast else bool(profile.get("fast", True))
    cfg["safe"]   = True if cli.safe else bool(profile.get("safe", True))
    cfg["use_cache"] = True if cli.use_cache else bool(profile.get("use_cache", True))
    cfg["threads"] = cli.threads if cli.threads is not None else profile.get("threads", -1)
    cfg["time_embargo"] = cli.time_embargo if cli.time_embargo is not None else profile.get("time_embargo", None)

    # split-kind: по умолчанию stratified, если есть target
    if cli.split_kind is not None:
        cfg["split_kind"] = cli.split_kind
    else:
        cfg["split_kind"] = profile.get("split_kind", "stratified" if (cli.target_col or profile.get("target_col")) else "kfold")

    # колонки
    cfg["id_col"]     = cli.id_col
    cfg["target_col"] = cli.target_col if cli.target_col is not None else profile.get("target_col", None)
    cfg["date_col"]   = cli.date_col if cli.date_col is not None else profile.get("date_col", None)
    cfg["group_col"]  = cli.group_col if cli.group_col is not None else profile.get("group_col", None)
    cfg["lat_col"]    = cli.lat_col if cli.lat_col is not None else profile.get("lat_col", None)
    cfg["lon_col"]    = cli.lon_col if cli.lon_col is not None else profile.get("lon_col", None)

    # блоки
    blocks_from_profile = (profile.get("blocks") or {})
    blocks_from_yaml = {}
    if cli.blocks_yaml:
        blocks_from_yaml = read_yaml(cli.blocks_yaml) or {}
        # допускаем форматы:
        # {block: {enabled: true, ...}} ИЛИ {blocks: {...}}
        if "blocks" in blocks_from_yaml:
            blocks_from_yaml = blocks_from_yaml["blocks"] or {}

    # склеим профили блоков
    blocks: Dict[str, Dict[str, Any]] = {}
    for src in [blocks_from_profile, blocks_from_yaml]:
        for k, v in (src or {}).items():
            if isinstance(v, bool):
                v = {"enabled": bool(v)}
            if k not in blocks:
                blocks[k] = {}
            blocks[k].update(v or {})
    cfg["blocks"] = blocks

    # only/exclude
    cfg["only"] = [s.strip() for s in cli.only.split(",")] if cli.only else None
    cfg["exclude"] = [s.strip() for s in cli.exclude.split(",")] if cli.exclude else []

    # прочее
    cfg["log_level"] = cli.log_level
    cfg["log_file"] = cli.log_file or (cfg["out_dir"] / "run_features.log")
    cfg["save_set"] = bool(cli.save_set or profile.get("save_set", True))
    cfg["overwrite"] = bool(cli.overwrite)
    return cfg

def plan_blocks(cfg: Dict[str, Any]) -> List[str]:
    # известные блоки в рекомендуемом порядке
    known = [
        "num_basic",
        "cat_freq",
        "cat_te_oof",
        "text_tfidf",
        "geo_grid",
        "geo_neighbors",
        "time_agg",
        "crosses",
        "img_stats",
        "img_embed",
    ]
    blocks_cfg: Dict[str, Dict[str, Any]] = cfg["blocks"] or {}
    selected = [b for b in known if blocks_cfg.get(b, {}).get("enabled", False)]
    # only/exclude
    if cfg["only"] is not None:
        selected = [b for b in selected if b in cfg["only"]]
    if cfg["exclude"]:
        selected = [b for b in selected if b not in cfg["exclude"]]
    return selected

def main():
    args = parse_args()
    # профиль
    profile_yaml = None
    if args.profile:
        # допускаем "tour" -> profiles/tour.yaml
        p = Path(args.profile)
        if not p.exists():
            p = Path("profiles") / f"{args.profile}.yaml"
        profile_yaml = read_yaml(p)
    else:
        profile_yaml = {}

    cfg = merge_cfg(args, profile_yaml)
    logger = setup_logging(cfg["log_level"], cfg["log_file"])
    logger.info("=== run_features: старт ===")
    logger.info(json.dumps({
        "tag": cfg["tag"],
        "split_kind": cfg["split_kind"],
        "frac": cfg["frac"],
        "n_splits": cfg["n_splits"],
        "fast": cfg["fast"],
        "safe": cfg["safe"],
        "use_cache": cfg["use_cache"],
        "threads": cfg["threads"],
    }, ensure_ascii=False))

    git_hash = get_git_commit(cfg["base"])
    if git_hash:
        logger.info(f"git commit: {git_hash}")

    # подготовка папки
    out_dir = ensure_dir(cfg["out_dir"])
    if cfg["overwrite"]:
        # мягко: не удаляем, просто помечаем
        logger.warning("--overwrite: артефакты будут перезаписаны при совпадении имён.")

    # --- читаем данные
    train_path = Path(cfg["train_path"])
    test_path  = Path(cfg["test_path"])
    if not train_path.exists() or not test_path.exists():
        logger.error(f"Не найден train/test: {train_path} | {test_path}")
        sys.exit(2)

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    logger.info(f"Train: {train.shape} | Test: {test.shape}")

    id_col = cfg["id_col"]
    if id_col not in train.columns or id_col not in test.columns:
        logger.error(f"ID_COL '{id_col}' отсутствует в train/test.")
        sys.exit(2)

    target_col = cfg["target_col"]
    target = None
    if target_col:
        if target_col not in train.columns:
            logger.error(f"TARGET_COL '{target_col}' отсутствует в train.")
            sys.exit(2)
        target = train[target_col].copy()

    # авто-детект колонок (для блоков, если они их не переопределяют)
    ex = {id_col}
    if target_col: ex.add(target_col)
    num_cols, cat_cols, text_cols = auto_detect_columns(train, ex)
    logger.info(f"AUTO NUM: {len(num_cols)}, CAT: {len(cat_cols)}, TEXT: {len(text_cols)}")

    # --- подвыборка по frac
    stratify_vec = target.to_numpy() if (target is not None and cfg["split_kind"] == "stratified") else None
    subs_idx = subsample_indices(len(train), cfg["frac"], cfg["seed"], stratify=stratify_vec)
    if len(subs_idx) < len(train):
        logger.info(f"Используем подвыборку {len(subs_idx)}/{len(train)} (~{cfg['frac']*100:.1f}%)")
        train = train.iloc[subs_idx].reset_index(drop=True)
        if target is not None:
            target = target.iloc[subs_idx].reset_index(drop=True)

    # --- сплиты
    folds = make_folds(
        train=train,
        target=target,
        split_kind=cfg["split_kind"],
        n_splits=cfg["n_splits"],
        seed=cfg["seed"],
        date_col=cfg["date_col"],
        group_col=cfg["group_col"],
        time_embargo=cfg["time_embargo"],
        logger=logger
    )
    logger.info(f"Folds: {len(folds)} | val sizes: {[len(v) for _, v in folds]}")

    # --- план блоков
    selected_blocks = plan_blocks(cfg)
    if args.dry_run:
        logger.info("[dry-run] Активные блоки: " + ", ".join(selected_blocks))
        logger.info("[dry-run] Выход.")
        sys.exit(0)
    if not selected_blocks:
        logger.warning("Нет активных блоков. Завершаем без сборки матриц.")
        # всё равно сохраним базовые артефакты y/ids/folds/meta
        save_ids_y_folds(train, test, id_col, target_col, target, folds, out_dir, logger)
        save_meta_catalog(cfg, out_dir, logger, built=[], catalog_dense=None, catalog_sparse=None, git_hash=git_hash)
        sys.exit(0)

    # --- FeatureStore
    FS = store.FeatureStore()
    built_order: List[str] = []

    # унифицированный реестр параметров по блокам (дефолты + профили)
    blocks_cfg: Dict[str, Dict[str, Any]] = cfg["blocks"] or {}

    # --- пробег по блокам
    for bname in selected_blocks:
        bcfg = blocks_cfg.get(bname, {}) or {}
        enabled = bcfg.get("enabled", True)
        if not enabled:
            logger.info(f"[skip] {bname} (disabled)")
            continue

        # общие параметры для ключа кэша
        base_params = {
            "fast": cfg["fast"],
            "safe": cfg["safe"],
            "frac": cfg["frac"],
        }
        base_params.update({k: v for k, v in bcfg.items() if k != "enabled"})

        # построение параметров и вызов билдера
        if bname == "num_basic":
            params = dict(
                prefix="num",
                num_cols=bcfg.get("num_cols", num_cols),
                log_cols=bcfg.get("log_cols", None),
                clip_quant=tuple(bcfg.get("clip_quant", (0.01, 0.99))),
                impute=bcfg.get("impute", "median"),
                scale=bcfg.get("scale", None),
                use_cache=cfg["use_cache"],
            )
            key = make_cache_key(bname, {**base_params, **params}, cfg["seed"], git_hash)
            pkg, from_cache = run_block(FS, bname, num_basic.build, key, logger, cfg["safe"], cfg["use_cache"],
                                        train, test, **params)
            if pkg: built_order.append(pkg.name)

        elif bname == "cat_freq":
            params = dict(
                prefix="catf",
                cat_cols=bcfg.get("cat_cols", cat_cols),
                rare_threshold=float(bcfg.get("rare_threshold", 0.01 if cfg["fast"] else 0.005)),
                use_cache=cfg["use_cache"],
            )
            key = make_cache_key(bname, {**base_params, **params}, cfg["seed"], git_hash)
            pkg, _ = run_block(FS, bname, cat_freq.build, key, logger, cfg["safe"], cfg["use_cache"],
                               train, test, **params)
            if pkg: built_order.append(pkg.name)

        elif bname == "cat_te_oof":
            if target is None:
                msg = "cat_te_oof требует target; пропуск."
                if cfg["safe"]:
                    raise SafeError(msg)
                logger.warning(msg)
                continue
            # ограничим top-K по кардинальности в fast-режиме
            te_cols = bcfg.get("cat_cols", cat_cols)
            if te_cols and cfg["fast"]:
                # возьмём топ-3–5 по числу уникальных
                uniq_counts = sorted([(c, int(train[c].nunique())) for c in te_cols], key=lambda x: -x[1])
                k = int(bcfg.get("top_k", 3))
                te_cols = [c for c, _ in uniq_counts[:k]]
                logger.info(f"TE fast: выбрано top-{k} категориальных: {te_cols}")

            params = dict(
                prefix="te",
                cat_cols=te_cols,
                method=bcfg.get("method", "target"),
                smoothing=bcfg.get("smoothing", "m-estimate"),
                use_cache=cfg["use_cache"],
            )
            key = make_cache_key(bname, {**base_params, **params, "folds_seed": cfg["seed"]}, cfg["seed"], git_hash)
            # строго OOF: передаём FOLDS
            pkg, _ = run_block(FS, bname, cat_te_oof.build, key, logger, cfg["safe"], cfg["use_cache"],
                               train, target, test, folds, **params)
            if pkg: built_order.append(pkg.name)

        elif bname == "text_tfidf":
            # берём первый текстовый столбец по умолчанию
            txt_cols = bcfg.get("text_cols", text_cols) or []
            if not txt_cols:
                logger.warning("text_tfidf: нет текстовых колонок — пропуск.")
                continue
            text_col = txt_cols[0]
            params = dict(
                text_col=text_col,
                min_df=int(bcfg.get("min_df", 5 if cfg["fast"] else 2)),
                ngram_range=tuple(bcfg.get("ngram_range", (1, 2))),
                use_char=bool(bcfg.get("use_char", False)),
                svd_k=(None if cfg["fast"] else int(bcfg.get("svd_k", 256))),
                prefix=bcfg.get("prefix", "tfidf"),
                use_cache=cfg["use_cache"],
            )
            key = make_cache_key(bname, {**base_params, **params}, cfg["seed"], git_hash)
            pkg, _ = run_block(FS, bname, text_tfidf.build, key, logger, cfg["safe"], cfg["use_cache"],
                               train, test, **params)
            if pkg: built_order.append(pkg.name)

        elif bname == "geo_grid":
            lat_col = cfg["lat_col"]; lon_col = cfg["lon_col"]
            if not (lat_col and lon_col and lat_col in train.columns and lon_col in train.columns):
                logger.warning("geo_grid: нет lat/lon — пропуск.")
                continue
            params = dict(
                lat_col=lat_col, lon_col=lon_col,
                steps_m=tuple(bcfg.get("steps_m", (1000,) if cfg["fast"] else (300, 1000))),
                prefix=bcfg.get("prefix", "geo"),
                use_cache=cfg["use_cache"],
            )
            key = make_cache_key(bname, {**base_params, **params}, cfg["seed"], git_hash)
            pkg, _ = run_block(FS, bname, geo_grid.build, key, logger, cfg["safe"], cfg["use_cache"],
                               train, test, **params)
            if pkg: built_order.append(pkg.name)

        elif bname == "geo_neighbors":
            lat_col = cfg["lat_col"]; lon_col = cfg["lon_col"]
            if not (lat_col and lon_col and lat_col in train.columns and lon_col in train.columns):
                logger.warning("geo_neighbors: нет lat/lon — пропуск.")
                continue
            params = dict(
                lat_col=lat_col, lon_col=lon_col,
                radii_m=tuple(bcfg.get("radii_m", (1000,) if cfg["fast"] else (300, 1000))),
                prefix=bcfg.get("prefix", "geonb"),
                use_cache=cfg["use_cache"],
            )
            key = make_cache_key(bname, {**base_params, **params}, cfg["seed"], git_hash)
            pkg, _ = run_block(FS, bname, geo_neighbors.build, key, logger, cfg["safe"], cfg["use_cache"],
                               train, test, **params)
            if pkg: built_order.append(pkg.name)

        elif bname == "time_agg":
            date_col = cfg["date_col"]
            if not date_col or date_col not in train.columns:
                logger.warning("time_agg: нет date_col — пропуск.")
                continue
            params = dict(
                date_col=date_col,
                group_cols=bcfg.get("group_cols", [cfg["id_col"]]),
                lags=tuple(bcfg.get("lags", (1, 7))),
                rollings=tuple(bcfg.get("rollings", (7, 30))),
                folds=folds if cfg["safe"] else None,  # анти-утечки «только прошлое»
                prefix=bcfg.get("prefix", "time"),
                use_cache=cfg["use_cache"],
            )
            key = make_cache_key(bname, {**base_params, **params, "folds_seed": cfg["seed"]}, cfg["seed"], git_hash)
            pkg, _ = run_block(FS, bname, time_agg.build, key, logger, cfg["safe"], cfg["use_cache"],
                               train, **params)
            if pkg: built_order.append(pkg.name)

        elif bname == "crosses":
            # безопасно: только при наличии whitelist
            white = bcfg.get("whitelist", None)
            if not white:
                logger.warning("crosses: нет whitelist — пропуск по умолчанию.")
                continue
            params = dict(
                whitelist_num_pairs=white.get("num_pairs", None),
                whitelist_num_cat=white.get("num_cat", None),
                prefix=bcfg.get("prefix", "x"),
                use_cache=cfg["use_cache"],
            )
            key = make_cache_key(bname, {**base_params, **params}, cfg["seed"], git_hash)
            pkg, _ = run_block(FS, bname, crosses.build, key, logger, cfg["safe"], cfg["use_cache"],
                               train, test, **params)
            if pkg: built_order.append(pkg.name)

        elif bname == "img_stats":
            try:
                all_ids = pd.concat([train[[id_col]], test[[id_col]]]).astype(str)[id_col].unique()
                img_dir = Path(cfg["data_dir"]) / "images"
                id2 = img_index.build_from_dir(img_dir, all_ids, pattern="{id}/*.jpg", max_per_id=int(bcfg.get("max_per_id", 4)))
                params = dict(id_col=id_col, id_to_images=id2, prefix=bcfg.get("prefix","imgstats"), use_cache=cfg["use_cache"])
                key = make_cache_key(bname, {**base_params, **params}, cfg["seed"], git_hash)
                pkg, _ = run_block(FS, bname, img_stats.build, key, logger, cfg["safe"], cfg["use_cache"],
                                   train, test, **params)
                if pkg: built_order.append(pkg.name)
            except Exception as e:
                if cfg["safe"]:
                    raise
                logger.warning(f"img_stats: пропуск ({e})")

        elif bname == "img_embed":
            try:
                all_ids = pd.concat([train[[id_col]], test[[id_col]]]).astype(str)[id_col].unique()
                img_dir = Path(cfg["data_dir"]) / "images"
                id2 = img_index.build_from_dir(img_dir, all_ids, pattern="{id}/*.jpg", max_per_id=int(bcfg.get("max_per_id", 4)))
                params = dict(
                    id_col=id_col, id_to_images=id2,
                    backbone=bcfg.get("backbone","resnet50"),
                    image_size=int(bcfg.get("image_size", 224)),
                    agg=bcfg.get("agg","mean"), pool=bcfg.get("pool","avg"),
                    batch_size=int(bcfg.get("batch_size", 64)),
                    device=bcfg.get("device","auto"), precision=bcfg.get("precision","auto"),
                    dtype=bcfg.get("dtype","float16"),
                    prefix=bcfg.get("prefix","img"),
                    use_cache=cfg["use_cache"]
                )
                key = make_cache_key(bname, {**base_params, **params}, cfg["seed"], git_hash)
                pkg, _ = run_block(FS, bname, img_embed.build, key, logger, cfg["safe"], cfg["use_cache"],
                                   train, test, **params)
                if pkg: built_order.append(pkg.name)
            except Exception as e:
                if cfg["safe"]:
                    raise
                logger.warning(f"img_embed: пропуск ({e})")

        else:
            logger.warning(f"[unknown block] {bname} — пропущен.")

    # -----------------
    # Сборка матриц
    # -----------------
    dense_pkgs = [name for name in FS.list() if getattr(FS.get(name), "kind", None) == "dense"]
    sparse_pkgs = [name for name in FS.list() if getattr(FS.get(name), "kind", None) == "sparse"]

    X_dense_tr = X_dense_te = None
    X_sparse_tr = X_sparse_te = None
    catalog_dense = catalog_sparse = None

    if dense_pkgs:
        X_dense_tr, X_dense_te, catalog_dense = assemble.make_dense(FS, include=dense_pkgs)
        logger.info(f"DENSE: {X_dense_tr.shape} | {X_dense_te.shape} | mem={mem_gb_df(X_dense_tr):.3f} GB")
        check_finite_dense(X_dense_tr, cfg["safe"], logger)

    if sparse_pkgs:
        if sp is None:
            logger.warning("SciPy не установлен — sparse пакеты не будут сохранены.")
        X_sparse_tr, X_sparse_te, catalog_sparse = assemble.make_sparse(FS, include=sparse_pkgs)
        logger.info(f"SPARSE: {X_sparse_tr.shape} | {X_sparse_te.shape}")

    # -----------------
    # Сохранение артефактов
    # -----------------
    save_ids_y_folds(train, test, id_col, target_col, target, folds, out_dir, logger)

    if cfg["save_set"]:
        if X_dense_tr is not None:
            X_dense_tr.to_parquet(out_dir / "X_dense_train.parquet")
            X_dense_te.to_parquet(out_dir / "X_dense_test.parquet")
        if (X_sparse_tr is not None) and (sp is not None):
            sp.save_npz(out_dir / "X_sparse_train.npz", X_sparse_tr)
            sp.save_npz(out_dir / "X_sparse_test.npz", X_sparse_te)
        logger.info(f"Сохранили матрицы в {out_dir}")

    # catalog + meta
    save_meta_catalog(
        cfg=cfg,
        out_dir=out_dir,
        logger=logger,
        built=built_order,
        catalog_dense=catalog_dense,
        catalog_sparse=catalog_sparse,
        git_hash=git_hash
    )

    logger.info("=== run_features: успех ===")


def save_ids_y_folds(
    train: pd.DataFrame,
    test: pd.DataFrame,
    id_col: str,
    target_col: Optional[str],
    target: Optional[pd.Series],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    logger: logging.Logger
):
    ensure_dir(out_dir)
    # ids
    pd.DataFrame({id_col: train[id_col].values}).to_parquet(out_dir / "ids_train.parquet", index=False)
    pd.DataFrame({id_col: test[id_col].values}).to_parquet(out_dir / "ids_test.parquet", index=False)
    # y
    if target_col and target is not None:
        ydf = pd.DataFrame({id_col: train[id_col].values, target_col: target.values})
        ydf.to_parquet(out_dir / "y_train.parquet", index=False)
    # folds
    with open(out_dir / "folds.pkl", "wb") as f:
        pickle.dump(folds, f)
    logger.info("Сохранили ids/y/folds")

def save_meta_catalog(
    cfg: Dict[str, Any],
    out_dir: Path,
    logger: logging.Logger,
    built: List[str],
    catalog_dense: Optional[Dict[str, Any]],
    catalog_sparse: Optional[Dict[str, Any]],
    git_hash: Optional[str]
):
    # объединённый каталог
    catalog: Dict[str, Any] = {}
    if isinstance(catalog_dense, dict):
        catalog.update(catalog_dense)
    if isinstance(catalog_sparse, dict):
        for k, v in catalog_sparse.items():
            if k not in catalog:
                catalog[k] = v
            else:
                # если коллизия имён — префикс kind
                catalog[f"{k}__sparse"] = v
    (out_dir / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "tag": cfg["tag"],
        "seed": cfg["seed"],
        "frac": cfg["frac"],
        "split_kind": cfg["split_kind"],
        "n_splits": cfg["n_splits"],
        "fast": cfg["fast"],
        "safe": cfg["safe"],
        "use_cache": cfg["use_cache"],
        "threads": cfg["threads"],
        "date_col": cfg["date_col"],
        "group_col": cfg["group_col"],
        "lat_col": cfg["lat_col"],
        "lon_col": cfg["lon_col"],
        "built": built,
        "git": git_hash,
        "env": env_info(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Сохранили catalog.json и meta.json")


if __name__ == "__main__":
    try:
        main()
    except SafeError as e:
        print(f"[SAFE FATAL] {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Пользователь прервал работу.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)
