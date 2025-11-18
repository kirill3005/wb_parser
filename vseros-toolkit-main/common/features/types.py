from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Union

import pandas as pd
from scipy.sparse import spmatrix

Kind = Literal["dense", "sparse", "mixed"]


@dataclass
class FeaturePackage:
    name: str
    train: Union[pd.DataFrame, spmatrix]
    test: Union[pd.DataFrame, spmatrix]
    kind: Kind
    cols: List[str]
    meta: Dict[str, Any]  # params, time_sec, oof: bool, deps: list[str]
