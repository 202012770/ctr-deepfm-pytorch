"""YAML configuration with project-relative paths."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml


def _project_root() -> Path:
    env_root = os.environ.get("CTR_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "configs").exists():
        return candidate
    return Path.cwd()


PROJECT_ROOT = _project_root()
CONFIG_DIR = PROJECT_ROOT / "configs"


class Config(dict):
    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[key] = value
        return value

    def path(self, key: str) -> Path:
        value: Any = self
        for part in key.split("."):
            value = value[part]
        return (
            value
            if isinstance(value, Path)
            else (value if Path(value).is_absolute() else PROJECT_ROOT / value)
        )


@cache
def load_config(name: str = "default") -> Config:
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return Config(yaml.safe_load(handle) or {})
