from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT_DIR / "configs"
DATASETS_DIR = ROOT_DIR / "datasets"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_env(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


class ConfigLoader:
    def __init__(self, root: Path | None = None):
        self.root = root or ROOT_DIR
        self.configs_dir = self.root / "configs"
        self.datasets_dir = self.root / "datasets"

    def load_models(self) -> dict[str, Any]:
        return _resolve_env(_load_yaml(self.configs_dir / "models.yaml"))

    def load_tiers(self) -> dict[str, Any]:
        return _resolve_env(_load_yaml(self.configs_dir / "tiers.yaml"))

    def load_judge(self) -> dict[str, Any]:
        return _resolve_env(_load_yaml(self.configs_dir / "judge.yaml"))

    def load_profiles(self) -> dict[str, Any]:
        return _resolve_env(_load_yaml(self.configs_dir / "profiles.yaml"))

    def load_weights(self) -> dict[str, Any]:
        return _resolve_env(_load_yaml(self.configs_dir / "weights.yaml"))

    def get_model_config(self, name: str) -> dict[str, Any]:
        models = self.load_models().get("models", {})
        if name not in models:
            available = ", ".join(models.keys())
            raise KeyError(f"Model '{name}' not found. Available: {available}")
        return models[name]

    def get_tier_config(self, tier: str) -> dict[str, Any]:
        tiers = self.load_tiers().get("tiers", {})
        if tier not in tiers:
            raise KeyError(f"Tier '{tier}' not found.")
        return tiers[tier]

    def get_profile(self, name: str) -> dict[str, Any]:
        profiles = self.load_profiles().get("profiles", {})
        if name not in profiles:
            available = ", ".join(profiles.keys())
            raise KeyError(f"Profile '{name}' not found. Available: {available}")
        return profiles[name]

    def dataset_path(self, suite: str, profile: str = "smoke") -> Path:
        return self.datasets_dir / profile / f"{suite}.jsonl"
