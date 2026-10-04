"""Evaluation configuration.

A run is described by a single YAML file. Configs compose with ``extends:`` so a
dataset, a harness and a scale profile can be recombined, and every value can be
overridden from the environment with ``${NAME}`` / ``${NAME:-default}``.

Nothing in this module is model specific: ``model.name`` and ``model.base_url``
are the only places a checkpoint is named, which is what makes a run
reproducible against a different served model.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


class ConfigError(ValueError):
    """The evaluation config is not usable."""


def expand_env(value: Any) -> Any:
    """Expand ``${VAR}`` and ``${VAR:-default}`` references from the environment."""
    if isinstance(value, str):
        expanded = _ENV_RE.sub(
            lambda match: os.environ.get(match.group(1), match.group(2) or ""),
            value,
        )
        return os.path.expanduser(os.path.expandvars(expanded))
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class ModelConfig:
    """An OpenAI-compatible chat completions endpoint."""

    name: str
    base_url: str
    api_key_env: str = "EVAL_API_KEY"
    api_key: str | None = None
    temperature: float = 0.6
    max_tokens_per_turn: int | None = 8192
    timeout_sec: int = 7200
    max_retries: int = 5
    parallel_tool_calls: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)
    # Context accounting only: which tokenizer approximates prompt size. Resolved
    # offline by default so a run never reaches out to Hugging Face.
    tokenizer: str = "o200k_base"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ModelConfig:
        name = str(value.get("name") or "").strip()
        base_url = str(value.get("base_url") or "").strip().rstrip("/")
        if not name:
            raise ConfigError("model.name is required")
        if not base_url:
            raise ConfigError("model.base_url is required")
        max_tokens = value.get("max_tokens_per_turn", 8192)
        return cls(
            name=name,
            base_url=base_url,
            api_key_env=str(value.get("api_key_env") or "EVAL_API_KEY"),
            api_key=value.get("api_key"),
            temperature=float(value.get("temperature", 0.6)),
            max_tokens_per_turn=(
                None if max_tokens in (None, 0, "0") else int(max_tokens)
            ),
            timeout_sec=int(value.get("timeout_sec", 7200)),
            max_retries=int(value.get("max_retries", 5)),
            parallel_tool_calls=bool(value.get("parallel_tool_calls", True)),
            extra_body=dict(value.get("extra_body") or {}),
            tokenizer=str(value.get("tokenizer") or "o200k_base"),
        )

    def resolve_api_key(self) -> str:
        """Return the API key, or a placeholder for unauthenticated endpoints."""
        if self.api_key:
            return str(self.api_key)
        value = os.environ.get(self.api_key_env, "").strip()
        if value:
            return value
        # LM Studio and friends ignore the key but clients require one to be set.
        return "lmstudio"

    def as_env(self) -> dict[str, str]:
        """Model settings exported to harness subprocesses."""
        return {
            "EVAL_MODEL_NAME": self.name,
            "EVAL_MODEL_BASE_URL": self.base_url,
            "EVAL_MODEL_API_KEY": self.resolve_api_key(),
        }


@dataclass(frozen=True)
class HarnessConfig:
    """Which agent harness drives the model inside the task workspace."""

    name: str = "leaf"
    options: dict[str, Any] = field(default_factory=dict)
    # Wall-clock guard for harnesses that have no native step limit.
    max_steps: int | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> HarnessConfig:
        value = dict(value or {})
        name = str(value.pop("name", "leaf")).strip()
        options = dict(value.pop("options", {}) or {})
        max_steps = value.pop("max_steps", None)
        if name not in {"leaf", "opencode", "pi"}:
            raise ConfigError(
                f"unknown harness {name!r}; available: leaf, opencode, pi"
            )
        unknown = set(value) - {"max_steps"}
        if unknown:
            raise ConfigError(f"unknown harness keys: {sorted(unknown)}")
        return cls(
            name=name,
            options=options,
            max_steps=None if max_steps in (None, 0) else int(max_steps),
        )


@dataclass(frozen=True)
class RuntimeConfig:
    """Sandbox that owns the workspace, runs commands and grades the result."""

    kind: str = "host"
    # Expanded on load; see `RuntimeConfig.from_dict`.
    root: Path = Path("~/.cache/cae/workspaces")
    keep_workspaces: bool = False
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> RuntimeConfig:
        value = dict(value or {})
        kind = str(value.pop("kind", "host")).strip()
        if kind != "host":
            raise ConfigError(
                f"unknown runtime {kind!r}; this project ships the 'host' runtime"
            )
        keep = bool(value.pop("keep_workspaces", False))
        return cls(
            kind=kind,
            root=Path(str(value.pop("root", "~/.cache/cae/workspaces"))).expanduser(),
            keep_workspaces=keep,
            options=dict(value.pop("options", {}) or {}),
        )


@dataclass(frozen=True)
class RunConfig:
    dataset: str
    output_dir: Path
    model: ModelConfig
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    task_ids: tuple[str, ...] = ()
    num_tasks: int | None = None
    seed: int = 42
    seeds_per_task: int = 1
    max_workers: int = 1
    max_attempts: int = 2
    max_steps: int = 100
    max_context_tokens: int = 131072
    max_total_time_sec: int | None = 10800
    resume: bool = True

    cache_dir: Path = Path("~/.cache/frognano/harbor")
    dataset_options: dict[str, Any] = field(default_factory=dict)
    source: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunConfig:
        dataset = str(value.get("dataset") or "").strip()
        if not dataset:
            raise ConfigError("dataset is required")
        config = cls(
            dataset=dataset,
            output_dir=Path(str(value.get("output_dir") or "eval-results")),
            model=ModelConfig.from_dict(dict(value.get("model") or {})),
            harness=HarnessConfig.from_dict(value.get("harness")),
            runtime=RuntimeConfig.from_dict(value.get("runtime")),
            task_ids=tuple(str(item) for item in value.get("task_ids") or ()),
            num_tasks=(
                None
                if value.get("num_tasks") is None
                else int(value.get("num_tasks"))
            ),
            seed=int(value.get("seed", 42)),
            seeds_per_task=int(value.get("seeds_per_task", 1)),
            max_workers=int(value.get("max_workers", 1)),
            max_attempts=int(value.get("max_attempts", 2)),
            max_steps=int(value.get("max_steps", 100)),
            max_context_tokens=int(value.get("max_context_tokens", 131072)),
            max_total_time_sec=(
                None
                if value.get("max_total_time_sec") is None
                else int(value.get("max_total_time_sec"))
            ),
            resume=bool(value.get("resume", True)),
            cache_dir=Path(str(value.get("cache_dir") or "~/.cache/frognano/harbor")),
            dataset_options=dict(value.get("dataset_options") or {}),
            source=value.get("source"),
        )
        for name in ("seeds_per_task", "max_workers", "max_attempts", "max_steps"):
            if getattr(config, name) <= 0:
                raise ConfigError(f"{name} must be positive")
        if config.num_tasks is not None and config.num_tasks < 0:
            raise ConfigError("num_tasks cannot be negative")
        if config.max_total_time_sec is not None and config.max_total_time_sec < 60:
            raise ConfigError("max_total_time_sec must be at least 60")
        return config

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["output_dir"] = str(self.output_dir)
        value["cache_dir"] = str(self.cache_dir)
        value["runtime"]["root"] = str(self.runtime.root)
        return value

    def label(self) -> str:
        return f"{self.dataset}/{self.harness.name}/{self.model.name}"


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_chain(path: Path, seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    if path in seen:
        raise ConfigError(f"config extends cycle at {path}")
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ConfigError(f"config must be a YAML mapping: {path}")
    parent = payload.pop("extends", None)
    if parent is None:
        return payload
    parents = parent if isinstance(parent, list) else [parent]
    resolved: dict[str, Any] = {}
    for entry in parents:
        candidate = Path(str(entry)).expanduser()
        if not candidate.is_absolute():
            candidate = (path.parent / candidate).resolve()
        resolved = _merge(resolved, _load_chain(candidate, seen | {path}))
    return _merge(resolved, payload)


def load_payload(path: str | Path) -> dict[str, Any]:
    """Resolve a config file's `extends` chain and env expansion, without interpreting it.

    The shared front half of every loader here: `load_config` parses the result as a
    `RunConfig`, `load_matrix_config` looks for a `matrix:` block in it.
    """
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise ConfigError(f"config file does not exist: {candidate}")
    payload = _load_chain(candidate.resolve())
    return expand_env(payload)


def load_config(path: str | Path) -> RunConfig:
    """Load a config file, applying ``extends`` inheritance then env expansion."""
    payload = load_payload(path)
    config = RunConfig.from_dict(payload)
    if config.source is None:
        config = replace(config, source=str(Path(path).expanduser()))
    return config


def apply_overrides(config: RunConfig, overrides: list[str]) -> RunConfig:
    """Apply ``--set key.path=value`` overrides on top of a loaded config."""
    payload = config.as_dict()
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override must look like key.path=value: {item!r}")
        key, raw = item.split("=", 1)
        parsed = yaml.safe_load(raw)
        cursor: dict[str, Any] = payload
        parts = key.strip().split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ConfigError(f"cannot descend into {key!r}")
        cursor[parts[-1]] = parsed
    return RunConfig.from_dict(payload)