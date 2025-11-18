"""Validate resolved config and report issues."""
from __future__ import annotations

import argparse
import sys

from common.configs.loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Lint configuration")
    parser.add_argument("--subsystem", default="recsys")
    parser.add_argument("--dataset_id", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--model", action="append", dest="models", default=[])
    args = parser.parse_args()
    try:
        cfg = load_config(args.subsystem, dataset_id=args.dataset_id, profile=args.profile, model_names=args.models)
    except Exception as exc:  # noqa: BLE001
        print(f"Validation failed: {exc}")
        sys.exit(1)
    print("Validation OK. Fingerprint:", cfg.fingerprint)


if __name__ == "__main__":
    main()
