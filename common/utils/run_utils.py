"""Shared run helpers: YAML config + CLI overrides, seeding, stdout+file logging.

Every script uses:
    cfg = load_config(args.config, overrides=args.override, smoke=args.smoke, seed=args.seed)
    out_dir = setup_run(cfg)  # creates dir, saves config_used.yaml, tees stdout to log.txt
"""
from __future__ import annotations

import ast
import os
import random
import sys
from typing import Any

import yaml


def _parse_value(v: str) -> Any:
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _load_yaml_with_base(path: str) -> dict:
    """YAML with optional `_base_: other.yaml` (relative to this file), deep-merged."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base:
        cfg = _deep_update(_load_yaml_with_base(os.path.join(os.path.dirname(path), base)), cfg)
    return cfg


def load_config(path: str | None, overrides: list[str] | None = None, **top_level) -> dict:
    """Load YAML, apply `key.sub=value` overrides, then non-None top-level kwargs."""
    cfg: dict = _load_yaml_with_base(path) if path else {}
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        _set_nested(cfg, k, _parse_value(v))
    for k, v in top_level.items():
        if v is not None:
            cfg[k] = v
    # smoke overrides live under `smoke_overrides:` in YAML
    if cfg.get("smoke") and isinstance(cfg.get("smoke_overrides"), dict):
        for k, v in cfg["smoke_overrides"].items():
            _set_nested(cfg, k, v)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()

    def isatty(self):
        return False


def setup_run(cfg: dict, out_dir_key: str = "out_dir") -> str:
    out_dir = cfg[out_dir_key]
    os.makedirs(out_dir, exist_ok=True)
    log_f = open(os.path.join(out_dir, "log.txt"), "a")
    sys.stdout = _Tee(sys.__stdout__, log_f)
    sys.stderr = _Tee(sys.__stderr__, log_f)
    set_seed(int(cfg.get("seed", 0)))
    with open(os.path.join(out_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print("=" * 60)
    print("config:")
    print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print("=" * 60)
    return out_dir


def add_common_args(parser) -> None:
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--override", "-o", nargs="*", default=[], help="key.sub=value overrides"
    )
