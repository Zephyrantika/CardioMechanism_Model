"""Configuration loading with includes and environment expansion."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_PATH_KEY_SUFFIXES = ("_file", "_path", "_dir")


class ConfigError(ValueError):
    """Raised when a configuration cannot be loaded safely."""


def load_config(path: Path) -> dict[str, Any]:
    """Load, recursively merge, validate, and resolve a YAML configuration."""
    config_path = Path(path).expanduser().resolve()
    config = _load_file(config_path, stack=())
    project_root = config.get("project_root")
    if not isinstance(project_root, str) or not project_root.strip():
        raise ConfigError("'project_root' must be a non-empty string")

    root = Path(project_root).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    root = root.resolve()
    config["project_root"] = root

    paths = config.get("paths")
    if paths is not None and not isinstance(paths, Mapping):
        raise ConfigError("'paths' must be a mapping when provided")
    if paths is not None:
        invalid_paths = [key for key, value in paths.items() if not isinstance(value, str)]
        if invalid_paths:
            names = ", ".join(str(key) for key in invalid_paths)
            raise ConfigError(f"Path values must be strings; invalid keys: {names}")
    return _resolve_paths(config, root)


def _load_file(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ConfigError(f"Configuration include cycle detected: {chain}")
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"Configuration root must be a mapping: {path}")

    current = _expand_environment(dict(loaded), path)
    includes = current.pop("includes", [])
    if not isinstance(includes, list) or not all(isinstance(item, str) for item in includes):
        raise ConfigError(f"'includes' must be a list of strings: {path}")

    merged: dict[str, Any] = {}
    for include in includes:
        include_path = (path.parent / include).resolve()
        merged = _deep_merge(merged, _load_file(include_path, (*stack, path)), "")
    return _deep_merge(merged, current, "")


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        location = f"{prefix}.{key}" if prefix else str(key)
        if key in result:
            old_is_mapping = isinstance(result[key], Mapping)
            new_is_mapping = isinstance(value, Mapping)
            if old_is_mapping != new_is_mapping:
                raise ConfigError(
                    f"Incompatible configuration override at '{location}': "
                    "cannot merge a mapping with a non-mapping"
                )
            if old_is_mapping:
                result[key] = _deep_merge(dict(result[key]), value, location)
                continue
        result[key] = value
    return result


def _expand_environment(value: Any, source: Path) -> Any:
    if isinstance(value, Mapping):
        return {key: _expand_environment(item, source) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item, source) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        environment_value = os.environ.get(name)
        if environment_value is not None:
            return environment_value
        if default is not None:
            return default
        raise ConfigError(f"Required environment variable '{name}' is not set ({source})")

    return _ENV_PATTERN.sub(replace, value)


def _resolve_paths(value: Any, root: Path, key: str | None = None) -> Any:
    if isinstance(value, Mapping):
        resolved: dict[str, Any] = {}
        for child_key, child_value in value.items():
            if child_key == "project_root":
                resolved[child_key] = child_value
            else:
                resolved[child_key] = _resolve_paths(child_value, root, str(child_key))
        return resolved
    if isinstance(value, list):
        return [_resolve_paths(item, root) for item in value]
    if isinstance(value, str) and key is not None:
        is_path = key in {"raw", "interim", "processed", "folds", "graphs", "outputs"}
        is_path = is_path or key.endswith(_PATH_KEY_SUFFIXES)
        if is_path:
            path = Path(value).expanduser()
            return (path if path.is_absolute() else root / path).resolve()
    return value
