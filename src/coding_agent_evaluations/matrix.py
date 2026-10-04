"""Matrix configs: one file describing several models across several harnesses.

A single-run config names one model and one harness, which makes a cross-model
comparison a shell loop. A matrix config names a *set* of each::

    extends: [../base.yaml]

    matrix:
      datasets: [swebench_verified]
      harnesses: [leaf, opencode, pi]
      models:
        - name: frognano-4b-2609
        - qwen3.8-27b-splash          # shorthand: name only, inherits base_url
        - name: qwen3.6-35b-a3b-mtp
          base_url: http://127.0.0.1:1234/v1

Every other setting (``num_tasks``, ``seeds_per_task``, ``max_steps``, the
runtime) is inherited from the extended config and applies to every cell, so the
cells differ only in what the matrix says they should: the model and the harness
driving it.

Each cell's results land in ``<output_dir>/<dataset>/<harness>/<model>/``, which
is what makes ``cae compare <output_dir>`` able to line them up afterwards.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .config import (
    ConfigError,
    HarnessConfig,
    ModelConfig,
    RunConfig,
    load_config,
    load_payload,
)
from .harnesses import build_harness

#: Marker key that turns a config file into a matrix config.
MATRIX_KEY = "matrix"


@dataclass(frozen=True)
class MatrixConfig:
    """A set of runs that differ only in model and harness."""

    base: RunConfig
    models: tuple[ModelConfig, ...]
    harnesses: tuple[str, ...]
    datasets: tuple[str, ...] = ()
    #: Keep going when one cell fails, so one broken model does not void the matrix.
    continue_on_error: bool = True
    source: str | None = None

    def with_base(self, base: RunConfig) -> MatrixConfig:
        """The same matrix over a different base config — how `--set` reaches every cell.

        Overriding the base rather than each cell is what makes `--set num_tasks=50` mean
        the same thing in all of them.
        """
        return replace(self, base=base)

    @property
    def cells(self) -> list[RunConfig]:
        """Every cell, as a config the runner accepts unchanged.

        Expansion is deterministic — dataset, then harness, then model — so a rerun writes
        to the same directories and ``resume`` finds the work it already did.
        """
        cells: list[RunConfig] = []
        for dataset in self.datasets or (self.base.dataset,):
            for harness in self.harnesses:
                for model in self.models:
                    cells.append(self._cell(dataset, harness, model))
        return cells

    def _cell(
        self, dataset: str, harness: str, model: ModelConfig
    ) -> RunConfig:
        base = self.base
        return replace(
            base,
            dataset=dataset,
            model=model,
            harness=HarnessConfig(name=harness, options=dict(base.harness.options)),
            output_dir=base.output_dir / dataset / harness / model.name,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "matrix": {
                "datasets": list(self.datasets),
                "harnesses": list(self.harnesses),
                "models": [
                    {"name": model.name, "base_url": model.base_url}
                    for model in self.models
                ],
                "continue_on_error": self.continue_on_error,
            },
            "base": self.base.as_dict(),
            "cells": [
                {
                    "dataset": cell.dataset,
                    "harness": cell.harness.name,
                    "model": cell.model.name,
                    "base_url": cell.model.base_url,
                    "output_dir": str(cell.output_dir),
                }
                for cell in self.cells
            ],
        }


def is_matrix_payload(payload: dict[str, Any]) -> bool:
    """Whether a resolved config file describes a matrix."""
    return isinstance(payload.get(MATRIX_KEY), dict)


def load_matrix_config(path: str | Path) -> MatrixConfig:
    """Load a matrix config: a shared base run config plus the models and harnesses to cross."""
    payload = load_payload(path)
    block = payload.pop(MATRIX_KEY, None)
    if not isinstance(block, dict):
        raise ConfigError(
            f"not a matrix config: no '{MATRIX_KEY}:' block in {Path(path)} "
            f"(a matrix config lists models and harnesses under '{MATRIX_KEY}:')"
        )
    base = RunConfig.from_dict(payload)
    source = base.source or str(Path(path).expanduser())
    base = replace(base, source=source)

    harnesses = _harnesses(block, base)
    datasets = _datasets(block)
    models = _models(block, base)
    if not models:
        raise ConfigError(f"matrix config {source} lists no models")
    return MatrixConfig(
        base=base,
        models=models,
        harnesses=harnesses,
        datasets=datasets,
        continue_on_error=bool(block.get("continue_on_error", True)),
        source=source,
    )


def load_run_or_matrix(path: str | Path) -> RunConfig | MatrixConfig:
    """Load whichever kind of config the file turns out to be."""
    payload = load_payload(path)
    if is_matrix_payload(payload):
        return load_matrix_config(path)
    return load_config(path)


def _harnesses(block: dict[str, Any], base: RunConfig) -> tuple[str, ...]:
    raw = block.get("harnesses")
    if raw is None:
        # Default to the harness the extended config names, so a matrix of models against one
        # harness does not have to be spelled out.
        return (base.harness.name,)
    if isinstance(raw, str):
        raw = [raw]
    names: list[str] = []
    for item in raw:
        name = str(item).strip()
        if not name:
            continue
        try:
            build_harness(name)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        names.append(name)
    if not names:
        raise ConfigError("matrix.harnesses is empty")
    return tuple(dict.fromkeys(names))


def _datasets(block: dict[str, Any]) -> tuple[str, ...]:
    raw = block.get("datasets")
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = [raw]
    names = [str(item).strip() for item in raw if str(item).strip()]
    if not names:
        raise ConfigError("matrix.datasets is empty")
    return tuple(dict.fromkeys(names))


def _models(block: dict[str, Any], base: RunConfig) -> tuple[ModelConfig, ...]:
    raw = block.get("models")
    if raw is None:
        # No models listed: inherit the single model from the base config, which makes the
        # file a matrix over harnesses only.
        return (base.model,)
    if isinstance(raw, (str, dict)):
        raw = [raw]
    models: list[ModelConfig] = []
    for item in raw:
        models.append(_model(item, base))
    seen: set[tuple[str, str]] = set()
    unique: list[ModelConfig] = []
    for model in models:
        # The same checkpoint served twice would write to the same directory, so a repeated
        # (name, url) is a mistake worth reporting rather than silently collapsing.
        key = (model.name, model.base_url)
        if key in seen:
            continue
        seen.add(key)
        unique.append(model)
    return tuple(unique)


def _model(item: Any, base: RunConfig) -> ModelConfig:
    """A matrix model: a bare name, or a full model block overriding the base."""
    if isinstance(item, str):
        item = {"name": item}
    if not isinstance(item, dict):
        raise ConfigError(f"matrix.models entry must be a name or a mapping: {item!r}")
    # An omitted field inherits from the base config, so a matrix entry only states what
    # differs: a name, or a name and the endpoint that serves it.
    values = {
        **asdict(base.model),
        **{key: value for key, value in item.items() if value is not None},
    }
    return ModelConfig.from_dict(values)


__all__ = [
    "MATRIX_KEY",
    "MatrixConfig",
    "is_matrix_payload",
    "load_matrix_config",
    "load_run_or_matrix",
]
