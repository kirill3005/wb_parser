"""CLI to display resolved configuration layers for recsys."""
from __future__ import annotations

import argparse
import json
import pathlib

from common.configs.loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Show resolved config")
    parser.add_argument("--subsystem", default="recsys")
    parser.add_argument("--dataset_id", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--section", default=None, help="e.g. recsys.candidates")
    parser.add_argument("--as", dest="fmt", choices=["json", "yaml"], default="yaml")
    parser.add_argument("--model", action="append", dest="models", default=[])
    parser.add_argument("--set", dest="sets", action="append", default=[])
    args = parser.parse_args()

    cli_overrides = {}
    for entry in args.sets:
        if "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        cursor = cli_overrides
        parts = key.split(".")
        for p in parts[:-1]:
            cursor = cursor.setdefault(p, {})  # type: ignore[assignment]
        cursor[parts[-1]] = value

    cfg = load_config(
        subsystem=args.subsystem, dataset_id=args.dataset_id, profile=args.profile, model_names=args.models, cli_overrides=cli_overrides
    )
    resolved = cfg.resolved
    if args.section:
        for part in args.section.split("."):
            resolved = resolved.get(part, {}) if isinstance(resolved, dict) else {}
    if args.fmt == "json":
        print(json.dumps(resolved, indent=2, ensure_ascii=False))
    else:
        import yaml

        print(yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()
