import numpy as np
import pandas as pd


def make_folds(
    train: pd.DataFrame,
    task: str = "binary",
    n_splits: int | None = None,
    split_kind: str | None = None,
    target_col: str | None = None,
    group_col: str | None = None,
    date_col: str | None = None,
    time_embargo: str | None = None,
    stratify: bool = True,
    random_state: int = 42,
    n_reg_bins: int = 10,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Возвращает список (train_idx, valid_idx) по индексам строк исходного train.
    Особенности:
    - KFold по умолчанию стратифицирован для классификации (если задан target_col).
    - GroupKFold/StratifiedGroupKFold при наличии group_col.
    - Time split: сортировка по date_col на N чанков; поддерживает эмбарго вокруг валидации.
    - Все индексы ссылаются на ИСХОДНЫЕ индексы train (никаких reindex).

    Совместимость с прежним кодом: если параметры не переданы, читаются из глобальных:
    SPLIT_KIND, N_SPLITS, TARGET_COL, GROUP_COL, DATE_COL.
    """
    g = globals()
    if n_splits is None:
        n_splits = g.get("N_SPLITS", 5)
    if split_kind is None:
        split_kind = g.get("SPLIT_KIND", "kfold")
    if target_col is None:
        target_col = g.get("TARGET_COL", None)
    if group_col is None:
        group_col = g.get("GROUP_COL", None)
    if date_col is None:
        date_col = g.get("DATE_COL", None)
    if time_embargo is None:
        time_embargo = g.get("TIME_EMBARGO", None)

    idx_all = np.arange(len(train))

    if split_kind == "time":
        if not date_col or date_col not in train.columns:
            raise ValueError("split_kind='time', но date_col не задан/нет в train")
        df = train[[date_col]].copy()
        if not np.issubdtype(df[date_col].dtype, np.datetime64):
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if df[date_col].isna().any():
            raise ValueError("date_col содержит NaT после to_datetime")

        df = df.sort_values(date_col)
        sorted_idx = df.index.to_numpy()

        val_blocks = np.array_split(sorted_idx, n_splits)
        folds = []

        embargo_td = None
        if time_embargo:
            try:
                embargo_td = pd.Timedelta(time_embargo)
            except Exception as e:
                raise ValueError(f"Не удалось распарсить time_embargo='{time_embargo}': {e}") from e

        for vb in val_blocks:
            if len(vb) == 0:
                continue
            val_idx = vb
            tr_pool = np.setdiff1d(sorted_idx, val_idx, assume_unique=False)

            if embargo_td is not None:
                vmin = df.loc[val_idx, date_col].min()
                vmax = df.loc[val_idx, date_col].max()
                left = vmin - embargo_td
                right = vmax + embargo_td
                mask_embargo = (df[date_col] >= left) & (df[date_col] <= right)
                embargo_idx = df.index[mask_embargo].to_numpy()
                tr_idx = np.setdiff1d(tr_pool, embargo_idx, assume_unique=False)
            else:
                tr_idx = tr_pool

            folds.append((tr_idx.astype(int), val_idx.astype(int)))

        if len(folds) == 0:
            raise RuntimeError("time-split не смог сформировать фолды (пустые блоки)")
        return folds

    if split_kind == "group":
        if not group_col or group_col not in train.columns:
            raise ValueError("split_kind='group', но group_col не задан/нет в train")
        groups = train[group_col].to_numpy()
        use_sgk = False
        y = None
        if stratify and target_col and target_col in train.columns and task in ("binary", "multiclass"):
            try:
                from sklearn.model_selection import StratifiedGroupKFold

                y = train[target_col].to_numpy()
                sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                folds = [(tr, va) for tr, va in sgk.split(idx_all, y, groups=groups)]
                use_sgk = True
            except Exception:
                use_sgk = False

        if not use_sgk:
            from sklearn.model_selection import GroupKFold

            gkf = GroupKFold(n_splits=n_splits)
            folds = [(tr, va) for tr, va in gkf.split(idx_all, groups=groups)]
        return folds

    y = None
    if target_col and target_col in train.columns:
        y = train[target_col].to_numpy()

    if stratify and y is not None:
        if task in ("binary", "multiclass"):
            from sklearn.model_selection import StratifiedKFold

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            folds = [(tr, va) for tr, va in skf.split(idx_all, y)]
            return folds
        if task == "regression":
            try:
                bins = pd.qcut(
                    train[target_col], q=min(n_reg_bins, len(train[target_col].unique())), duplicates="drop"
                ).cat.codes.to_numpy()
                from sklearn.model_selection import StratifiedKFold

                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                folds = [(tr, va) for tr, va in skf.split(idx_all, bins)]
                return folds
            except Exception:
                pass

    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = [(tr, va) for tr, va in kf.split(idx_all)]
    return folds
