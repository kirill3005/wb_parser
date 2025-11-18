from __future__ import annotations

from copy import deepcopy

from common.configs.loader import load_config
from common.configs.fingerprint import compute_fingerprint


def test_fingerprint_stable_to_paths_change():
    cfg = load_config("recsys", profile="scout")
    base_fp = cfg.fingerprint
    modified = deepcopy(cfg.resolved)
    modified["paths"]["artifacts"] = "artifacts_alt"
    assert compute_fingerprint(modified) == base_fp


def test_fingerprint_changes_on_param_change():
    cfg = load_config("recsys", profile="scout")
    modified = deepcopy(cfg.resolved)
    modified["recsys"]["candidates"]["K"] = 999
    assert compute_fingerprint(modified) != cfg.fingerprint
