import hashlib
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from common.cache import load_feature_pkg, make_key, save_feature_pkg
from common.features.types import FeaturePackage


def _fingerprint() -> str:
    text = Path(__file__).read_text(encoding="utf-8")
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _build_vectorizer(
    use_char: bool, min_df: int | float, ngram_range: Tuple[int, int]
) -> TfidfVectorizer:
    if use_char:
        return TfidfVectorizer(
            analyzer="char",
            min_df=min_df,
            ngram_range=ngram_range,
        )
    return TfidfVectorizer(
        analyzer="word",
        min_df=min_df,
        ngram_range=ngram_range,
    )


def _fit_transform(
    train_series: pd.Series,
    test_series: pd.Series,
    vectorizer: TfidfVectorizer,
) -> Tuple[csr_matrix, csr_matrix, Sequence[str]]:
    train_series = train_series.fillna("").astype(str)
    test_series = test_series.fillna("").astype(str)

    Xtr = vectorizer.fit_transform(train_series)
    Xte = vectorizer.transform(test_series)
    feature_names = vectorizer.get_feature_names_out()
    return csr_matrix(Xtr), csr_matrix(Xte), feature_names


def _apply_svd(
    Xtr: csr_matrix, Xte: csr_matrix, n_components: int
) -> Tuple[csr_matrix, csr_matrix, Sequence[str]]:
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    Xtr_reduced = svd.fit_transform(Xtr)
    Xte_reduced = svd.transform(Xte)

    names = [f"svd{i}" for i in range(Xtr_reduced.shape[1])]
    return csr_matrix(Xtr_reduced), csr_matrix(Xte_reduced), names


def build(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
    *,
    min_df: int | float = 5,
    ngram_range: Tuple[int, int] = (1, 2),
    use_char: bool = False,
    svd_k: Optional[int] = None,
    prefix: str = "tfidf",
    use_cache: bool = True,
    cache_key_extra: Optional[Dict] = None,
) -> FeaturePackage:
    """TF-IDF (word/char). Возвращает sparse CSR FeaturePackage(kind='sparse')."""

    params = {
        "text_col": text_col,
        "min_df": min_df,
        "ngram_range": ngram_range,
        "use_char": use_char,
        "svd_k": svd_k,
        "prefix": prefix,
    }
    data_stamp = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }
    if cache_key_extra:
        data_stamp.update(cache_key_extra)

    cache_key = make_key(params, code_fingerprint=_fingerprint(), data_stamp=data_stamp)
    if use_cache:
        cached = load_feature_pkg("text_tfidf", cache_key)
        if cached is not None:
            return cached

    t0 = time.time()
    vectorizer = _build_vectorizer(use_char, min_df, ngram_range)
    Xtr, Xte, feature_names = _fit_transform(train_df[text_col], test_df[text_col], vectorizer)

    cols = [f"{prefix}__{name}" for name in feature_names]

    if svd_k is not None:
        Xtr, Xte, svd_names = _apply_svd(Xtr, Xte, svd_k)
        cols = [f"{prefix}__{name}" for name in svd_names]

    meta = {
        "name": "text_tfidf",
        "params": params,
        "time_sec": round(time.time() - t0, 3),
        "cache_key": cache_key,
        "deps": [],
    }

    pkg = FeaturePackage(
        name="text_tfidf",
        train=Xtr,
        test=Xte,
        kind="sparse",
        cols=cols,
        meta=meta,
    )

    if use_cache:
        save_feature_pkg("text_tfidf", cache_key, pkg)

    return pkg
