"""配置加载：读取 configs/default.json，支持 CLI flag 覆盖。"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    for section in ("transcribe", "segmentation", "style", "mux"):
        cfg.setdefault(section, {})
    return cfg


def merge_overrides(cfg: dict[str, Any], section: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """返回覆盖后的新配置（不修改原配置）。None 值的 override 被忽略。"""
    merged = deepcopy(cfg)
    for key, value in overrides.items():
        if value is not None:
            merged[section][key] = value
    return merged
