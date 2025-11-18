from __future__ import annotations

from common.configs.loader import load_config


def test_loader_basic_profile_and_dataset():
    cfg = load_config("recsys", dataset_id="s5e11", profile="gate", model_names=["lgbm_ranker"])
    resolved = cfg.resolved
    assert resolved["paths"]["artifacts"] == "artifacts"
    assert resolved["recsys"]["candidates"]["K"] == 100
    assert resolved["recsys"]["schema"]["interactions"]["user_id"] == "uid"
    assert cfg.schema_version.startswith("1.")


def test_loader_profile_override_and_dataset_specific():
    cfg = load_config("recsys", dataset_id="s5e11", profile="gate")
    assert cfg.resolved["recsys"]["candidates"]["time_window_days"] == 21
