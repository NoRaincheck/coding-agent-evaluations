"""Harness interface and shared trajectory shape."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..config import HarnessConfig, ModelConfig


@dataclass
class HarnessRun:
    """One agent rollout against one provisioned workspace."""

    runtime: Any
    task: dict[str, Any]
    model: ModelConfig
    config: HarnessConfig
    instance_id: str
    seed: int
    max_steps: int
    max_context_tokens: int
    max_total_time_sec: int | None
    logger: logging.Logger = field(default_factory=logging.getLogger)
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None

    @property
    def workspace(self) -> Path:
        return self.runtime.paths.workspace


class Harness(Protocol):
    """Drives a model inside a workspace and returns a trajectory."""

    name: str
    # Whether `max_steps` is actually enforced. leaf owns its loop; the CLI
    # harnesses own theirs and are bounded only by the time budget.
    enforces_max_steps: bool
    # Sampling settings from the reference protocol the harness cannot express.
    ignored_model_params: tuple[str, ...]

    def run(self, run: HarnessRun) -> dict[str, Any]: ...


def empty_trajectory(run: HarnessRun, *, reason: str) -> dict[str, Any]:
    """The minimum trajectory every harness must return."""
    return {
        "trajectory_format": f"{run.config.name}-trajectory",
        "harness": run.config.name,
        "instance_id": run.instance_id,
        "seed": run.seed,
        "messages": [],
        "steps": [],
        "n_steps": 0,
        "exit_reason": reason,
        "error": None,
        "final_message": "",
        "context_tokens": 0,
        "elapsed_sec": 0.0,
        "output_patch": "",
        "partial": False,
    }


def build_harness(name: str) -> type[Harness]:
    """Look up a harness implementation by name."""
    if name == "leaf":
        from .leaf import LeafHarness

        return LeafHarness
    if name == "opencode":
        from .opencode import OpenCodeHarness

        return OpenCodeHarness
    if name == "pi":
        from .pi import PiHarness

        return PiHarness
    raise ValueError(f"unknown harness {name!r}; available: leaf, opencode, pi")


__all__ = ["Harness", "HarnessRun", "build_harness", "empty_trajectory"]