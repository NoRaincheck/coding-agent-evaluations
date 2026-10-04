"""The config matrix every harness test runs against.

Two things vary between a config and the next: the local model served at
``model.base_url`` and the harness driving it. Both are read from config, so a test
asserting a harness is wired for one checkpoint is the same assertion the runner
makes, not a hand-built approximation of it.

By default the matrix is exactly what a run would use — the model named by the
environment (falling back to the values in ``configs/base.yaml``) crossed with all
three harnesses. Point it elsewhere without editing a test::

    EVAL_MATRIX_MODELS='qwen3.8-27b-splash,frognano-4b-2609' .venv/bin/python -m pytest tests -q
    EVAL_MATRIX_HARNESSES=pi .venv/bin/python -m pytest tests -q
    EVAL_MATRIX_BASE_URL=http://127.0.0.1:1234/v1 .venv/bin/python -m pytest tests -q

Every test here is offline: it resolves config and inspects what a harness *would*
do. No test opens a socket to the endpoint, so the suite stays fast and passes with
nothing served. A run that does reach the model is ``scripts/smoke.sh``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from coding_agent_evaluations.config import (
    HarnessConfig,
    ModelConfig,
    RunConfig,
    load_config,
)
from coding_agent_evaluations.harnesses import HarnessRun, build_harness

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"

#: Harnesses in the matrix, in the order the comparison table uses.
HARNESSES: tuple[str, ...] = ("leaf", "opencode", "pi")


class MatrixCell:
    """One (local model, harness) pair the suite runs against."""

    def __init__(self, model: ModelConfig, harness: str) -> None:
        self.model = model
        self.harness = harness

    @property
    def id(self) -> str:
        # Parametrisation id: model first so `-k <model>` selects across harnesses.
        return f"{self.model.name}-{self.harness}"

    @property
    def config_path(self) -> Path:
        return CONFIG_ROOT / "harness" / f"{self.harness}.yaml"

    def run_config(self) -> RunConfig:
        """The shipped config for this harness, resolved against this cell's model.

        Loads the real file — ``extends`` chain, env expansion and all — then pins the model
        to the cell. A test that reads config this way fails when a config file drifts, which
        is the point.
        """
        config = load_config(self.config_path)
        overrides = [
            f"model.name={self.model.name}",
            f"model.base_url={self.model.base_url}",
            f"model.api_key_env={self.model.api_key_env}",
        ]
        return _overrides(config, overrides)

    def harness_config(self) -> HarnessConfig:
        return self.run_config().harness

    def model_config(self) -> ModelConfig:
        return self.run_config().model

    def build(self):
        """A fresh harness instance, as the runner builds one per rollout."""
        return build_harness(self.harness)()

    @property
    def is_cli(self) -> bool:
        """leaf is driven in-process; the CLIs are subprocesses."""
        return self.harness != "leaf"

    def __repr__(self) -> str:  # pragma: no cover - pytest output only
        return f"MatrixCell({self.model.name!r}, {self.harness!r})"


def _overrides(config: RunConfig, overrides: list[str]) -> RunConfig:
    from coding_agent_evaluations.config import apply_overrides

    return apply_overrides(config, overrides)


def matrix_models() -> list[ModelConfig]:
    """The local models under test, in matrix order.

    ``EVAL_MATRIX_MODELS`` is a comma-separated list of served ids, served from a single
    endpoint (``EVAL_MATRIX_BASE_URL``, else ``EVAL_MODEL_BASE_URL``). The endpoint's own
    ``/v1/models`` is deliberately not queried: this suite must run with nothing served.
    """
    raw = os.environ.get("EVAL_MATRIX_MODELS", "").strip()
    base_url = (
        os.environ.get("EVAL_MATRIX_BASE_URL")
        or os.environ.get("EVAL_MODEL_BASE_URL")
        or ""
    ).strip()
    default = load_config(CONFIG_ROOT / "base.yaml").model
    if not raw:
        # Same fallback chain as a real run: env, then the committed defaults.
        return [
            ModelConfig(
                name=os.environ.get("EVAL_MODEL_NAME", default.name),
                base_url=(base_url or default.base_url).rstrip("/"),
                api_key_env=default.api_key_env,
            )
        ]
    models: list[ModelConfig] = []
    for item in raw.split(","):
        name = item.strip()
        if not name:
            continue
        if "@" in name:
            name, _, cell_url = name.partition("@")
            name = name.strip()
            cell_url = cell_url.strip()
        else:
            cell_url = base_url or default.base_url
        if not name:
            continue
        models.append(
            ModelConfig(
                name=name,
                base_url=cell_url.rstrip("/"),
                api_key_env=default.api_key_env,
            )
        )
    return models or [default]


def matrix_harnesses() -> tuple[str, ...]:
    raw = os.environ.get("EVAL_MATRIX_HARNESSES", "").strip()
    if not raw:
        return HARNESSES
    names = tuple(
        item.strip() for item in raw.split(",") if item.strip() in HARNESSES
    )
    return names or HARNESSES


def matrix() -> list[MatrixCell]:
    """Every (model, harness) cell, models outer so a model's harnesses stay adjacent."""
    return [
        MatrixCell(model=model, harness=harness)
        for model in matrix_models()
        for harness in matrix_harnesses()
    ]


class FakeRuntime:
    """The slice of the runtime contract a CLI harness touches, backed by a tmp dir."""

    def __init__(self, state: Path, instruction: str = "fix the bug") -> None:
        self.paths = type(
            "Paths", (), {"state": state, "workspace": state, "root": state}
        )()
        self._instruction = instruction

    def get_task_instruction(self) -> str:
        return self._instruction


def make_run(
    config_dir: Path,
    *,
    harness: str = "pi",
    model: ModelConfig | None = None,
    instruction: str = "fix the bug",
    **overrides: Any,
) -> HarnessRun:
    """A `HarnessRun` for a CLI harness, from its shipped config where possible."""
    model = model or ModelConfig(name="m", base_url="http://127.0.0.1:1234/v1")
    values: dict[str, Any] = {
        "runtime": FakeRuntime(config_dir, instruction),
        "task": {"instance_id": "demo", "dataset": "swebench_verified"},
        "model": model,
        "config": HarnessConfig(name=harness),
        "instance_id": "demo",
        "seed": 0,
        "max_steps": 10,
        "max_context_tokens": 32768,
        "max_total_time_sec": 600,
    }
    values.update(overrides)
    return HarnessRun(**values)


def run_for_cell(cell: MatrixCell, config_dir: Path, **overrides: Any) -> HarnessRun:
    """A `HarnessRun` built from a matrix cell's own resolved config."""
    config = cell.run_config()
    return make_run(
        config_dir,
        harness=cell.harness,
        model=config.model,
        config=config.harness,
        instance_id="demo",
        **overrides,
    )


def cli(cell: MatrixCell, config_dir: Path) -> tuple[Any, HarnessRun, str]:
    """The harness instance, its run, and the session id a CLI rollout would use.

    The binary is resolved by name rather than by `resolve_binary()`: what a harness *would*
    execute is a property of its config, so these assertions must not depend on whether the
    CLI happens to be installed on the machine running the suite.
    """
    if not cell.is_cli:
        pytest.skip(f"{cell.harness} is driven in-process, it has no CLI command")
    run = run_for_cell(cell, config_dir)
    harness = cell.build()
    harness.binary = cell.harness
    return harness, run, harness.session_id(run)


def cli_command(cell: MatrixCell, config_dir: Path) -> list[str]:
    """The argv the harness would execute."""
    harness, run, session_id = cli(cell, config_dir)
    return harness.build_command(run, config_dir, session_id)


def cli_env(cell: MatrixCell, config_dir: Path) -> dict[str, str]:
    """The environment the harness subprocess would receive."""
    harness, run, session_id = cli(cell, config_dir)
    return harness.build_env(run, config_dir, session_id)


def extension_argv(command: list[str], flag: str) -> list[str]:
    """Every value passed after `flag` in an argv list."""
    return [
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == flag
    ]


def jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL file, skipping anything that is not an object."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] != "{":
            continue
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows
