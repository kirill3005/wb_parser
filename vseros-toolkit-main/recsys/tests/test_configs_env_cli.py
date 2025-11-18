from __future__ import annotations

from common.configs.loader import load_config


def test_env_override(monkeypatch):
    monkeypatch.setenv("CFG_recsys__candidates__K", "77")
    cfg = load_config("recsys", profile="scout")
    assert cfg.resolved["recsys"]["candidates"]["K"] == 77


def test_cli_override():
    cfg = load_config("recsys", profile="scout", cli_overrides={"recsys": {"features": {"blocks": {"sequence": {"enabled": True}}}}})
    assert cfg.resolved["recsys"]["features"]["blocks"]["sequence"]["enabled"] is True
