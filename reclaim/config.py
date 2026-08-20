"""Layered configuration.

    defaults  <  reclaim.toml  <  environment  <  command-line flags

Each layer overrides the one before it, and `reclaim config` prints which layer
every value came from -- because the usual way a config system wastes an
afternoon is being unable to answer "where did that setting come from".

SECRETS ARE NOT CONFIGURATION
    API keys are read from the environment only, never from the file. A config
    file lives in a working directory, gets committed by accident, and ends up
    in a screenshot. There is a check for exactly that mistake, and it is loud.

`tomllib` is standard library from Python 3.11, so this costs nothing.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_NAME = "reclaim.toml"

DEFAULTS: dict[str, Any] = {
    "gateway.mode": "simulated",        # simulated | mock | razorpay
    "gateway.base": "",                 # overrides the Razorpay API base
    "gateway.allow_writes": False,
    "service.dry_run": True,
    "service.rate": 0.0,
    "service.limit": 60,
    "policy.proposer": "rules",         # rules | ev
    "policy.strict": False,
    "diagnosis.use_learned_tier": True,
}

# Environment overrides, by config key.
ENV_MAP = {
    "gateway.base": "RECLAIM_API_BASE",
    "log.key": "RECLAIM_LOG_KEY",
}

# Keys that must never appear in a config file.
SECRET_KEYS = ("key_secret", "secret", "password", "token", "api_key",
               "razorpay.key_id", "razorpay.key_secret")


@dataclass
class Config:
    values: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    path: Path | None = None
    warnings: list[str] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def source(self, key: str) -> str:
        return self.sources.get(key, "default")


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def find_config(start: Path | None = None) -> Path | None:
    """Nearest reclaim.toml, walking up from here. Same idea as .git."""
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / CONFIG_NAME
        if candidate.is_file():
            return candidate
    return None


def load(path: Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    cfg = Config(values=dict(DEFAULTS),
                 sources={k: "default" for k in DEFAULTS})

    path = path or find_config()
    if path and path.is_file():
        cfg.path = path
        try:
            parsed = _flatten(tomllib.loads(path.read_text()))
        except tomllib.TOMLDecodeError as exc:
            cfg.warnings.append(f"{path.name} is not valid TOML: {exc}")
            parsed = {}
        for key, value in parsed.items():
            if any(s in key.lower() for s in SECRET_KEYS):
                # Refuse the value outright rather than warning and using it.
                # A warning that still loads the secret teaches people the file
                # is a fine place to keep one.
                cfg.warnings.append(
                    f"ignoring '{key}' from {path.name}: secrets are read from "
                    f"the environment only, never from a file that can be "
                    f"committed or screenshotted")
                continue
            cfg.values[key] = value
            cfg.sources[key] = path.name

    for key, env_var in ENV_MAP.items():
        if os.environ.get(env_var):
            cfg.values[key] = os.environ[env_var]
            cfg.sources[key] = f"${env_var}"

    for key, value in (overrides or {}).items():
        if value is None:
            continue
        cfg.values[key] = value
        cfg.sources[key] = "flag"

    return cfg


def credentials() -> tuple[str, str]:
    """Razorpay credentials. Environment only, by design."""
    return (os.environ.get("RAZORPAY_KEY_ID", ""),
            os.environ.get("RAZORPAY_KEY_SECRET", ""))
