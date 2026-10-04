"""Run a matrix of models across harnesses, one cell at a time, with progress.

Cells run sequentially on purpose. A cell is already a full evaluation — a
workspace per task-seed pair, a model call per step — so overlapping cells would
multiply local disk and endpoint load by the number of cells, and a comparison
that saturates the machine measures the contention rather than the harnesses. Use
``max_workers`` inside a cell to get concurrency, which is where it belongs.

The progress bar counts *job-seed pairs* across the whole matrix, not cells: a
five-cell matrix of ten pairs has fifty things to finish, and reporting "2/5
cells" hides that the cell now running is the one still grinding through its last
forty pairs. The bar's description carries the current cell, so both are visible at
once.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .config import RunConfig
from .datasets import load_tasks
from .matrix import MatrixConfig
from .runner import latest_results, run_evaluation

logger = logging.getLogger(__name__)


@dataclass
class CellReport:
    """What one cell of the matrix produced."""

    config: RunConfig
    summary: dict[str, Any]
    error: str | None = None

    @property
    def label(self) -> str:
        return self.config.label()

    @property
    def failed(self) -> bool:
        return self.error is not None or bool(self.summary.get("jobs_failed"))


def run_matrix(
    matrix: MatrixConfig,
    *,
    on_result: Callable[[RunConfig, dict[str, Any]], None] | None = None,
    stream: TextIO | None = None,
    progress: bool = True,
) -> list[CellReport]:
    """Run every cell, returning one report per cell in expansion order.

    A cell that raises is reported and, unless ``continue_on_error`` is off, the matrix
    carries on: one model refusing connections should not discard the measurements taken
    for the others.
    """
    cells = matrix.cells
    plans = _plan(cells)
    bar = _ProgressBar(
        [plan.config for plan in plans],
        total=sum(plan.remaining for plan in plans),
        stream=stream,
        enabled=progress,
    )
    reports: list[CellReport] = []
    try:
        for plan in plans:
            bar.start_cell(plan)
            try:
                summary = run_evaluation(
                    plan.config,
                    tasks=plan.tasks,
                    on_result=lambda row: bar.advance(1),
                )
            except Exception as exc:  # one cell must not void the matrix
                logger.warning("Cell %s failed: %s", plan.config.label(), exc)
                summary = {}
                report = CellReport(plan.config, summary, error=f"{type(exc).__name__}: {exc}")
                reports.append(report)
                if on_result is not None:
                    on_result(plan.config, summary)
                bar.cell_done(plan, failed=True)
                if not matrix.continue_on_error:
                    bar.close()
                    raise
                continue
            bar.advance(plan.remaining)
            bar.cell_done(plan, failed=bool(summary.get("jobs_failed")))
            reports.append(CellReport(plan.config, summary))
            if on_result is not None:
                on_result(plan.config, summary)
    finally:
        bar.close()
    return reports


@dataclass
class _CellPlan:
    """A cell with its task list resolved once and its remaining work counted."""

    config: RunConfig
    tasks: list[dict[str, Any]]
    remaining: int


def _plan(cells: list[RunConfig]) -> list[_CellPlan]:
    """Resolve each cell's tasks, fetching a dataset once for all cells that share it.

    Remaining work is what is left *after* resume, so a rerun of a finished matrix shows a
    completed bar rather than one that stalls at zero and never fills.
    """
    cache: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    plans: list[_CellPlan] = []
    for config in cells:
        key = (
            config.dataset,
            str(config.cache_dir),
            config.task_ids,
            config.num_tasks,
            config.seed,
        )
        tasks = cache.get(key)
        if tasks is None:
            tasks = load_tasks(
                config.dataset,
                cache_dir=Path(config.cache_dir).expanduser(),
                task_ids=config.task_ids,
                limit=config.num_tasks,
                seed=config.seed,
            )
            cache[key] = tasks
        plans.append(
            _CellPlan(
                config=config,
                tasks=tasks,
                remaining=_remaining(config, len(tasks)),
            )
        )
    return plans


def _remaining(config: RunConfig, task_count: int) -> int:
    """How many task-seed pairs this cell still has to run."""
    total = task_count * config.seeds_per_task
    if not config.resume:
        return total
    results = Path(config.output_dir).expanduser() / "results.jsonl"
    if not results.is_file():
        return total
    return total - sum(
        1
        for row in latest_results(results).values()
        if row.get("status") in {"completed", "provisioned"}
    )


class _ProgressBar:
    """A tqdm progress bar over the matrix's task-seed pairs.

    tqdm is imported lazily so the rest of the CLI works without it; without it the same
    information is printed as plain lines, which is what a log file or a CI job wants.
    """

    def __init__(
        self,
        configs: list[RunConfig],
        *,
        total: int,
        stream: TextIO | None,
        enabled: bool,
    ) -> None:
        self.configs = configs
        self.total = total
        self.stream = stream or sys.stderr
        self.enabled = enabled
        self.done = 0
        self.started = time.monotonic()
        self._bar: Any = None
        if not enabled:
            return
        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover - tqdm ships as a dependency
            print(
                f"matrix: {len(configs)} cell(s), {total} task-seed pair(s)",
                file=self.stream,
                flush=True,
            )
            return
        self._bar = tqdm(
            total=total,
            unit="pair",
            dynamic_ncols=True,
            desc=self._describe(configs[0]) if configs else "matrix",
            file=self.stream,
            leave=True,
        )

    @staticmethod
    def _describe(config: RunConfig) -> str:
        return f"{config.dataset}/{config.harness.name}/{config.model.name}"

    def start_cell(self, plan: _CellPlan) -> None:
        if self._bar is not None:
            self._bar.set_description(self._describe(plan.config), refresh=True)
            return
        if self.enabled:
            print(f"-> {self._describe(plan.config)}", file=self.stream, flush=True)

    def advance(self, count: int = 1) -> None:
        self.done += count
        if self._bar is not None:
            self._bar.update(count)

    def cell_done(self, plan: _CellPlan, *, failed: bool) -> None:
        if self._bar is None and self.enabled:
            status = "failed" if failed else "done"
            print(
                f"   {self._describe(plan.config)}: {status} "
                f"({self.done}/{self.total})",
                file=self.stream,
                flush=True,
            )

    def close(self) -> None:
        if self._bar is not None:
            self._bar.set_description("done", refresh=True)
            self._bar.close()
        elif self.enabled:
            elapsed = time.monotonic() - self.started
            print(
                f"matrix: {self.done}/{self.total} task-seed pair(s) in {elapsed:.0f}s",
                file=self.stream,
                flush=True,
            )


__all__ = ["CellReport", "run_matrix"]
