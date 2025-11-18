from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


def save_set(
    run_tag: str,
    Xd_tr=None,
    Xd_te=None,
    Xs_tr=None,
    Xs_te=None,
    y: pd.Series | np.ndarray | None = None,
    folds: List[Tuple[np.ndarray, np.ndarray]] | None = None,
    feature_catalog: Dict | List | None = None,
    train_ids: pd.Series | np.ndarray | None = None,
    test_ids: pd.Series | np.ndarray | None = None,
    id_col: str | None = None,
    target_col: str | None = None,
    overwrite: bool = True,
) -> Path:
    """
    Сохраняет фич-набор и метаинформацию в artifacts/sets/<run_tag>/…
    - dense → parquet
    - sparse → .npz
    - y / ids → parquet
    - folds → .pkl
    - meta (каталог фич, формы, таймстемп, кол-во фолдов) → meta.json

    Функция пытается быть обратно совместимой:
    если y/ids/folds не переданы, берёт их из глобалей ноутбука (train/test/ID_COL/TARGET_COL/FOLDS), если доступны.
    """

    g = globals()
    if y is None and "train" in g and isinstance(g["train"], pd.DataFrame):
        tc = target_col or g.get("TARGET_COL", None)
        if tc and tc in g["train"].columns:
            y = g["train"][tc]
    if train_ids is None and "train" in g and "ID_COL" in g and g["ID_COL"] in g["train"].columns:
        train_ids = g["train"][g["ID_COL"]]
        if id_col is None:
            id_col = g["ID_COL"]
    if test_ids is None and "test" in g and "ID_COL" in g and g["ID_COL"] in g["test"].columns:
        test_ids = g["test"][g["ID_COL"]]
        if id_col is None:
            id_col = g["ID_COL"]
    if folds is None and "FOLDS" in g:
        folds = g["FOLDS"]

    n_tr: Optional[int] = None
    n_te: Optional[int] = None

    def _ensure_df(X, name: str):
        if X is None:
            return None
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, np.ndarray):
            cols = [f"f{i}" for i in range(X.shape[1])]
            return pd.DataFrame(X, columns=cols)
        raise TypeError(f"{name}: ожидаю DataFrame или ndarray")

    Xd_tr = _ensure_df(Xd_tr, "Xd_tr")
    Xd_te = _ensure_df(Xd_te, "Xd_te")

    if Xd_tr is not None:
        n_tr = len(Xd_tr)
    if Xd_te is not None:
        n_te = len(Xd_te)

    if Xs_tr is not None:
        n_tr = Xs_tr.shape[0] if n_tr is None else n_tr
        if n_tr != Xs_tr.shape[0]:
            raise ValueError("Xd_tr/Xs_tr: несовпадающее число строк")
    if Xs_te is not None:
        n_te = Xs_te.shape[0] if n_te is None else n_te
        if n_te != Xs_te.shape[0]:
            raise ValueError("Xd_te/Xs_te: несовпадающее число строк")

    if (Xd_tr is None and Xs_tr is None) or (Xd_te is None and Xs_te is None):
        raise ValueError("Нужно передать хотя бы один из наборов (dense или sparse) для train и test")

    if (Xd_tr is not None) and (Xd_te is not None):
        if list(Xd_tr.columns) != list(Xd_te.columns):
            raise ValueError("Dense train/test имеют разные столбцы/порядок")
        if not np.isfinite(Xd_tr.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all():
            raise ValueError("Xd_tr содержит NaN/inf")
        if not np.isfinite(Xd_te.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all():
            raise ValueError("Xd_te содержит NaN/inf")

    if (Xs_tr is not None) and (Xs_te is not None):
        if Xs_tr.shape[1] != Xs_te.shape[1]:
            raise ValueError("Sparse train/test имеют разное число столбцов")

    base = Path("artifacts/sets") / str(run_tag)
    if base.exists() and not overwrite:
        raise FileExistsError(f"{base} уже существует (overwrite=False)")
    base.mkdir(parents=True, exist_ok=True)

    if Xd_tr is not None:
        Xd_tr.to_parquet(base / "X_dense_train.parquet", index=False)
    if Xd_te is not None:
        Xd_te.to_parquet(base / "X_dense_test.parquet", index=False)

    if Xs_tr is not None:
        sparse.save_npz(base / "X_sparse_train.npz", Xs_tr)
    if Xs_te is not None:
        sparse.save_npz(base / "X_sparse_test.npz", Xs_te)

    if y is not None:
        y_series = pd.Series(y).reset_index(drop=True)
        if train_ids is not None and id_col is not None:
            out = pd.DataFrame({id_col: pd.Series(train_ids).reset_index(drop=True), "target": y_series})
        else:
            out = pd.DataFrame({"target": y_series})
        out.to_parquet(base / "y_train.parquet", index=False)

    if train_ids is not None and id_col is not None:
        pd.DataFrame({id_col: pd.Series(train_ids).reset_index(drop=True)}).to_parquet(base / "ids_train.parquet", index=False)
    if test_ids is not None and id_col is not None:
        pd.DataFrame({id_col: pd.Series(test_ids).reset_index(drop=True)}).to_parquet(base / "ids_test.parquet", index=False)

    if folds is not None:
        with open(base / "folds.pkl", "wb") as f:
            pickle.dump(folds, f)

    catalog_wrapped = feature_catalog
    if feature_catalog is None:
        catalog_wrapped = {}
    elif isinstance(feature_catalog, list):
        catalog_wrapped = {"items": feature_catalog}

    meta = {
        "run_tag": str(run_tag),
        "rows_train": int(n_tr) if n_tr is not None else None,
        "rows_test": int(n_te) if n_te is not None else None,
        "dense_cols": list(Xd_tr.columns) if Xd_tr is not None else None,
        "sparse_n_features": int(Xs_tr.shape[1]) if Xs_tr is not None else (int(Xs_te.shape[1]) if Xs_te is not None else None),
        "has_y": bool(y is not None),
        "has_folds": bool(folds is not None),
        "id_col": id_col,
        "target_col": target_col,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "catalog": catalog_wrapped,
    }
    (base / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"[save_set] Saved to: {base}")
    return base


def load_set(
    run_tag: str,
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[sparse.spmatrix],
    Optional[sparse.spmatrix],
    Optional[pd.Series],
    List[Tuple[np.ndarray, np.ndarray]],
    Dict,
]:
    """Загружает сохранённый набор из artifacts/sets/<run_tag>.

    Возвращает (Xd_tr, Xd_te, Xs_tr, Xs_te, y, folds, meta).
    Если какие-то файлы отсутствуют (dense/sparse/y/meta), соответствующие элементы будут None/{}.
    """

    base = Path("artifacts/sets") / str(run_tag)
    if not base.exists():
        raise FileNotFoundError(f"Не найден набор {base}")

    Xd_tr = Xd_te = Xs_tr = Xs_te = y = None
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    meta: Dict = {}

    dense_tr = base / "X_dense_train.parquet"
    dense_te = base / "X_dense_test.parquet"
    if dense_tr.exists() and dense_te.exists():
        Xd_tr = pd.read_parquet(dense_tr)
        Xd_te = pd.read_parquet(dense_te)

    sparse_tr = base / "X_sparse_train.npz"
    sparse_te = base / "X_sparse_test.npz"
    if sparse_tr.exists() and sparse_te.exists():
        Xs_tr = sparse.load_npz(sparse_tr)
        Xs_te = sparse.load_npz(sparse_te)

    y_path = base / "y_train.parquet"
    if y_path.exists():
        df_y = pd.read_parquet(y_path)
        if "target" in df_y.columns:
            y = df_y["target"]
        elif df_y.shape[1] == 1:
            y = df_y.iloc[:, 0]

    folds_path = base / "folds.pkl"
    if folds_path.exists():
        folds = pickle.loads(folds_path.read_bytes())

    meta_path = base / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    return Xd_tr, Xd_te, Xs_tr, Xs_te, y, folds, meta
