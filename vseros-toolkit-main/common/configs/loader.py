from __future__ import annotations

import json
import os
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Mapping, MutableMapping

import jsonschema
import yaml

from .fingerprint import compute_fingerprint

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs"
SCHEMA_DIR = ROOT / "common" / "configs" / "schemas"


class IncludeLoader(yaml.SafeLoader):
    pass


def _construct_include(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
    rel_path = loader.construct_scalar(node)
    include_path = pathlib.Path(loader.name).parent / rel_path
    with include_path.open("r", encoding="utf-8") as f:
        return yaml.load(f, IncludeLoader)


IncludeLoader.add_constructor("!include", _construct_include)


@dataclass
class ResolvedConfig:
    resolved: Mapping[str, Any]
    sources: list[str]
    fingerprint: str
    schema_version: str


_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _load_yaml(path: pathlib.Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loader = IncludeLoader(f)
        loader.name = str(path)
        data = loader.get_single_data()
    return data or {}


def _merge(base: Any, override: Any) -> Any:
    if override is None:
        return None
    if isinstance(base, Mapping) and isinstance(override, Mapping):
        result: dict[str, Any] = {k: v for k, v in base.items()}
        if set(override.keys()) == {"+extend"} and isinstance(base, list):
            return base + list(override.get("+extend", []))
        for k, v in override.items():
            if v is None:
                result.pop(k, None)
                continue
            if k == "+extend":
                continue
            if k in result:
                merged = _merge(result[k], v)
                if merged is None:
                    result.pop(k, None)
                else:
                    result[k] = merged
            else:
                result[k] = v
        return result
    if isinstance(base, list) and isinstance(override, Mapping) and "+extend" in override:
        return base + list(override.get("+extend", []))
    if isinstance(override, list):
        return list(override)
    return override


def _merge_many(dicts: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    merged: Any = {}
    for d in dicts:
        merged = _merge(merged, d)
    return merged


def _parse_value(val: str) -> Any:
    lowered = val.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return json.loads(val)
    except Exception:
        pass
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return val


def _apply_env_overrides(prefix: str) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix) :].strip("_")
        parts = [p for p in path.split("__") if p]
        cursor: MutableMapping[str, Any] = result
        for p in parts[:-1]:
            cursor = cursor.setdefault(p, {})  # type: ignore[assignment]
        cursor[parts[-1]] = _parse_value(value)
    return result


def _apply_cli_overrides(overrides: Mapping[str, Any]) -> Mapping[str, Any]:
    def cast(obj: Any) -> Any:
        if isinstance(obj, str):
            return _parse_value(obj)
        if isinstance(obj, Mapping):
            return {k: cast(v) for k, v in obj.items()}
        return obj

    return cast(overrides or {})


def _resolve_refs(config: Any, full: Mapping[str, Any]) -> Any:
    if isinstance(config, str):
        def replace(match: re.Match[str]) -> str:
            expr = match.group(1)
            if expr.startswith("ENV:"):
                name_default = expr[4:]
                if "|" in name_default:
                    name, default = name_default.split("|", 1)
                else:
                    name, default = name_default, ""
                return os.environ.get(name, default)
            ref_parts = expr.split(".")
            cursor: Any = full
            for part in ref_parts:
                cursor = cursor.get(part) if isinstance(cursor, Mapping) else None
            return str(cursor) if cursor is not None else ""

        return _ENV_PATTERN.sub(replace, config)
    if isinstance(config, Mapping):
        return {k: _resolve_refs(v, full) for k, v in config.items()}
    if isinstance(config, list):
        return [_resolve_refs(v, full) for v in config]
    return config


def _validate(resolved: Mapping[str, Any]) -> None:
    schema_files = {
        "paths": SCHEMA_DIR / "core_paths.json",
        "logging": SCHEMA_DIR / "core_logging.json",
        "resources": SCHEMA_DIR / "core_resources.json",
        "recsys.schema": SCHEMA_DIR / "recsys_schema.json",
        "recsys.dataio": SCHEMA_DIR / "recsys_dataio.json",
        "recsys.candidates": SCHEMA_DIR / "recsys_candidates.json",
        "recsys.features": SCHEMA_DIR / "recsys_features.json",
        "recsys.eval": SCHEMA_DIR / "recsys_eval.json",
        "recsys.rerank": SCHEMA_DIR / "recsys_rerank.json",
        "recsys.blend": SCHEMA_DIR / "recsys_blend.json",
    }
    for path_key, schema_path in schema_files.items():
        pointer = resolved
        for part in path_key.split("."):
            if not isinstance(pointer, Mapping) or part not in pointer:
                pointer = None
                break
            pointer = pointer[part]
        if pointer is None:
            continue
        with schema_path.open("r", encoding="utf-8") as f:
            schema = json.load(f)
        jsonschema.validate(pointer, schema)


def load_config(
    subsystem: Literal["recsys"],
    dataset_id: str | None = None,
    profile: str | pathlib.Path | None = None,
    model_names: list[str] | None = None,
    overrides_paths: list[pathlib.Path] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    env_prefix: str = "CFG_",
) -> ResolvedConfig:
    overrides_paths = overrides_paths or []
    layers: list[Mapping[str, Any]] = []
    version_info = _load_yaml(CONFIG_ROOT / "core" / "version.yaml")
    schema_version = str(version_info.get("config_schema_version", "unknown"))

    base_files = [
        CONFIG_ROOT / "core" / "defaults.yaml",
        CONFIG_ROOT / "core" / "paths.yaml",
        CONFIG_ROOT / "core" / "logging.yaml",
        CONFIG_ROOT / "core" / "resources.yaml",
    ]
    for path in base_files:
        if path.exists():
            layers.append(_load_yaml(path))
    if subsystem == "recsys":
        recsys_files = [
            CONFIG_ROOT / "core" / "schema.recsys.yaml",
            CONFIG_ROOT / "recsys" / "dataio.yaml",
            CONFIG_ROOT / "recsys" / "candidates.yaml",
            CONFIG_ROOT / "recsys" / "features.yaml",
            CONFIG_ROOT / "recsys" / "eval.yaml",
            CONFIG_ROOT / "recsys" / "rerank.yaml",
            CONFIG_ROOT / "recsys" / "blend.yaml",
        ]
        for path in recsys_files:
            if path.exists():
                layers.append(_load_yaml(path))
        model_names = model_names or []
        for model_name in model_names:
            model_path = CONFIG_ROOT / "recsys" / "models" / f"{model_name}.yaml"
            if model_path.exists():
                layers.append(_load_yaml(model_path))
    if profile:
        profile_path = CONFIG_ROOT / "recsys" / "profiles" / (f"{profile}.yaml" if isinstance(profile, str) else profile.name)
        if isinstance(profile, pathlib.Path):
            profile_path = profile
        if profile_path.exists():
            layers.append(_load_yaml(profile_path))
    if dataset_id:
        dataset_dir = CONFIG_ROOT / "datasets" / dataset_id
        for fname in ["overrides.yaml", "model_overrides.yaml", f"profile_{profile}.yaml" if profile else None]:
            if fname:
                path = dataset_dir / fname
                if path.exists():
                    layers.append(_load_yaml(path))
    local_override = CONFIG_ROOT / "overrides" / "local.yaml"
    if local_override.exists():
        layers.append(_load_yaml(local_override))
    for path in overrides_paths:
        if path.exists():
            layers.append(_load_yaml(path))
    layers.append(_apply_env_overrides(env_prefix))
    layers.append(_apply_cli_overrides(cli_overrides or {}))

    merged = _merge_many(layers)
    resolved = _resolve_refs(merged, merged)
    _validate(resolved)
    fingerprint = compute_fingerprint(resolved)
    return ResolvedConfig(resolved=resolved, sources=[str(p) for p in base_files], fingerprint=fingerprint, schema_version=schema_version)
