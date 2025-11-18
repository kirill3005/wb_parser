"""Compute fingerprint for resolved configuration."""
from __future__ import annotations

import argparse

from common.configs.loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Fingerprint config")
    parser.add_argument("--subsystem", default="recsys")
    parser.add_argument("--dataset_id", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--model", action="append", dest="models", default=[])
    args = parser.parse_args()
    cfg = load_config(args.subsystem, dataset_id=args.dataset_id, profile=args.profile, model_names=args.models)
    print(cfg.fingerprint)


if __name__ == "__main__":
    main()
