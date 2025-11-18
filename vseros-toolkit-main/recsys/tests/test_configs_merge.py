from __future__ import annotations

import pathlib
import tempfile
import yaml

from common.configs.loader import load_config


def test_extend_and_null_removal(tmp_path: pathlib.Path):
    override = {
        "recsys": {
          "features": {
            "blocks": {
              "sequence": {
                "last_k": {"+extend": [20]}
              },
              "similarity": None
            }
          }
        }
    }
    override_path = tmp_path / "override.yaml"
    override_path.write_text(yaml.safe_dump(override))
    cfg = load_config("recsys", profile="gate", overrides_paths=[override_path])
    seq_last_k = cfg.resolved["recsys"]["features"]["blocks"]["sequence"]["last_k"]
    assert seq_last_k[-1] == 20
    assert "similarity" not in cfg.resolved["recsys"]["features"]["blocks"]
