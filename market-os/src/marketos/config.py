"""Typed configuration loader. One source of truth, versioned in git."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"


@dataclass
class Config:
    data_lake_root: Path
    universe: list[str] = field(default_factory=list)
    themes: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG) -> "Config":
        path = Path(path)
        data = yaml.safe_load(path.read_text()) if path.exists() else {}
        lake_root = Path(data.get("data_lake_root", REPO_ROOT / "data_lake"))
        if not lake_root.is_absolute():
            lake_root = REPO_ROOT / lake_root
        return cls(
            data_lake_root=lake_root,
            universe=data.get("universe", []),
            themes=data.get("themes", {}),
            raw=data,
        )
