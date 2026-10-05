#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "coding-agent-evaluations @ git+https://github.com/NoRaincheck/coding-agent-evaluations.git",
#   "pyyaml>=6.0",
# ]
# ///
"""Run one model x harness benchmark through `cae` and print the standard report.

The point of this script is that every benchmark run anywhere, by anyone, is driven
the same way and printed in the same shape, so two reports can be diffed line by
line.

It exists rather than a bare `cae run` because:

  * the published wheel ships the Python package but **not** `configs/*.yaml`, so
    `cae run --config configs/dataset/swebench-verified.yaml` only works from a
    source checkout. This script writes a self-contained matrix config instead.
  * `resume` matches on `(instance_id, seed)` and status only. It does not notice
    that the limits changed, so re-running a cell directory with a different
    `max_steps` silently keeps the old rows. This script puts every run under
    `<output_root>/<scale>/`, where `<scale>` encodes the limits, so two runs that
    disagree about their scale cannot contaminate each other.

Usage is in SKILL.md next to this file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

HARNESSES = ("leaf", "opencode", "pi")

#: Per-benchmark values from the reference protocol (configs/dataset/*.yaml). The
#: two datasets differ only where the benchmark differs.
DATASETS: dict[str, dict[str, Any]] = {
    "swebench_verified": {
        "max_tokens_per_turn": 8192,
        "max_retries": 5,
        "python": "3.9",
    },
    "terminal_bench_2_verified": {
        "max_tokens_per_turn": 32000,
        "max_retries": 2,
        "python": "3.12",
    },
}

#: Not a limit and not model specific, but fixed by the reference protocol, so it
#: is part of the contract rather than something each run re-picks.
SEED = 42
MAX_CONTEXT_TOKENS = 131072
MAX_ATTEMPTS = 2
STEP_TIMEOUT_SEC = 1800

REPORT_VERSION = 1


# --------------------------------------------------------------------------- spec


class Spec:
    """The complete, explicit description of one benchmark run."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.dataset = args.dataset
        self.models = [args.model] + [
            name.strip() for name in (args.models or "").split(",") if name.strip()
        ]
        self.base_url = args.base_url.rstrip("/")
        self.api_key_env = args.api_key_env
        self.harnesses = [
            name.strip() for name in (args.harnesses or "").split(",") if name.strip()
        ] or list(HARNESSES)

        unknown = [name for name in self.harnesses if name not in HARNESSES]
        if unknown:
            raise SystemExit(
                f"unknown harness(es) {unknown}; available: {', '.join(HARNESSES)}"
            )
        if self.dataset not in DATASETS:
            raise SystemExit(
                f"unknown dataset {self.dataset!r}; available: {', '.join(DATASETS)}"
            )

        self.num_tasks: int | None = (
            None if args.num_tasks in (None, "all") else int(args.num_tasks)
        )
        self.seeds = args.seeds
        self.max_steps = args.max_steps
        self.max_time = args.max_time
        self.max_workers = args.max_workers
        self.resume = not args.no_resume
        self.temperature = args.temperature
        self.max_tokens_per_turn = args.max_tokens_per_turn
        self.thinking = not args.no_thinking

        self.output_root = Path(args.output_root).expanduser().resolve()
        self.scale = self._scale()

    @property
    def run_root(self) -> Path:
        """Where every cell of this run lands."""
        return self.output_root / self.scale

    @property
    def config_path(self) -> Path:
        return self.run_root / "benchmark.yaml"

    def _scale(self) -> str:
        """A directory name that changes whenever any limit changes.

        `cae`'s own resume is keyed on `(instance_id, seed)` only, so a run that
        reused a cell directory across two different scales would report the first
        scale's rows as the second's. Encoding the scale in the path is what stops
        that, and it also keeps `cae compare <output_root>` lining runs up.
        """
        tasks = "all" if self.num_tasks is None else str(self.num_tasks)
        return f"n{tasks}s{self.seeds}st{self.max_steps}to{self.max_time}w{self.max_workers}"

    def config(self) -> dict[str, Any]:
        """A self-contained matrix config: one base every cell inherits."""
        preset = DATASETS[self.dataset]
        model: dict[str, Any] = {
            "name": self.models[0],
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "temperature": self.temperature,
            "max_tokens_per_turn": self.max_tokens_per_turn
            if self.max_tokens_per_turn is not None
            else preset["max_tokens_per_turn"],
            "max_retries": preset["max_retries"],
            "tokenizer": "o200k_base",
        }
        if self.thinking:
            model["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
        return {
            "dataset": self.dataset,
            "output_dir": str(self.run_root),
            "cache_dir": "~/.cache/cae/datasets",
            "task_ids": [],
            "num_tasks": self.num_tasks,
            "seed": SEED,
            "seeds_per_task": self.seeds,
            "max_workers": self.max_workers,
            "max_attempts": MAX_ATTEMPTS,
            "max_steps": self.max_steps,
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "max_total_time_sec": self.max_time,
            "resume": self.resume,
            "model": model,
            "runtime": {
                "kind": "host",
                "root": "~/.cache/cae/workspaces",
                "keep_workspaces": False,
                "options": {
                    "step_timeout_sec": STEP_TIMEOUT_SEC,
                    "python": preset["python"],
                },
            },
            "matrix": {
                "datasets": [self.dataset],
                "harnesses": self.harnesses,
                "models": [{"name": name} for name in self.models],
                "continue_on_error": True,
            },
        }


# -------------------------------------------------------------------------- runner


def cae(*args: str, check: bool = True, capture: bool = False) -> tuple[int, str]:
    """Invoke the runner in this script's own environment.

    `python -m` rather than the `cae` console script, so this works whatever PATH
    the script was launched with.

    stderr is always inherited, so run logs and the progress bar stream live and a
    backgrounded run still shows progress in its log file. `capture` takes stdout
    instead — used for `cae run`, whose own table is the same data as the standard
    report, in a second, non-standard shape.
    """
    print(f"$ cae {' '.join(args)}", file=sys.stderr, flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "coding_agent_evaluations", *args],
        stdout=subprocess.PIPE if capture else None,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result.returncode, result.stdout or ""


def write_config(spec: Spec) -> Path:
    spec.run_root.mkdir(parents=True, exist_ok=True)
    spec.config_path.write_text(
        "# Generated by the run-benchmarks skill. Every field is the standard contract;\n"
        "# change it by changing the skill's arguments, not by editing this file.\n"
        + yaml.safe_dump(spec.config(), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return spec.config_path


# --------------------------------------------------------------------------- report


def collect(run_root: Path) -> list[dict[str, Any]]:
    """One row per finished cell, read back from `summary.json`."""
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(run_root.glob("*/*/*/summary.json"))
    ]


COLUMNS = (
    ("dataset", "dataset", str, 0),
    ("harness", "harness", str, 0),
    ("model", "model", str, 0),
    ("tasks", "tasks_selected", int, 0),
    ("jobs", "jobs_total", int, 0),
    ("resolved", "resolved", int, 0),
    ("resolve_rate", "resolve_rate", lambda v: f"{v * 100:.2f}%", 0),
    (
        "graded",
        "resolve_rate_graded",
        lambda v: f"{v * 100:.2f}%" if v is not None else "-",
        0,
    ),
    ("infra", "jobs_infrastructure_failed", int, 0),
    ("failed", "jobs_failed", int, 0),
    ("mean_steps", "mean_steps", lambda v: f"{v:.2f}", 0),
    ("mean_sec", "mean_elapsed_sec", lambda v: f"{v:.1f}", 0),
)


def _cell(value: Any, fmt: Any) -> str:
    if value is None:
        return "-"
    try:
        return str(fmt(value))
    except (TypeError, ValueError):
        return "-"


def render_report(spec: Spec, rows: list[dict[str, Any]], *, runner: str) -> str:
    """The fixed output block. Same inputs, same lines, every time."""
    widths = []
    for header, _, _, _ in COLUMNS:
        widths.append(len(header))
    cells = []
    for row in rows:
        line = [_cell(row.get(key), fmt) for _, key, fmt, _ in COLUMNS]
        cells.append(line)
        for index, value in enumerate(line):
            widths[index] = max(widths[index], len(value))

    def line(values: list[str]) -> str:
        return "  ".join(v.ljust(widths[i]) for i, v in enumerate(values)).rstrip()

    tasks = "all" if spec.num_tasks is None else str(spec.num_tasks)
    out = [
        f"benchmark report v{REPORT_VERSION}",
        f"generated    {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"runner       {runner}",
        f"dataset      {spec.dataset}",
        (
            f"scale        tasks={tasks} seeds={spec.seeds} max_steps={spec.max_steps} "
            f"max_total_time_sec={spec.max_time} max_workers={spec.max_workers}"
        ),
        f"harnesses    {', '.join(spec.harnesses)}",
        f"output       {spec.run_root}",
        "",
        line([header for header, _, _, _ in COLUMNS]),
        line(["-" * width for width in widths]),
    ]
    out.extend(line(values) for values in cells)
    if not cells:
        out.append("(no cells finished)")

    notes = []
    try:
        from coding_agent_evaluations.report import caveats, load_summaries

        paths = sorted(spec.run_root.glob("*/*/*/summary.json"))
        if paths:
            notes = caveats(load_summaries([str(path) for path in paths]))
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        # Notes are supplementary; a report that lost them still beats no report.
        notes = []
    if notes:
        out.append("")
        out.append("notes:")
        out.extend(f"  - {note}" for note in notes)
    return "\n".join(out)


def emit(spec: Spec, *, as_json: bool, runner: str) -> str:
    rows = collect(spec.run_root)
    if as_json:
        payload = {
            "report_version": REPORT_VERSION,
            "dataset": spec.dataset,
            "scale": spec.scale,
            "harnesses": spec.harnesses,
            "models": spec.models,
            "base_url": spec.base_url,
            "run_root": str(spec.run_root),
            "cells": rows,
        }
        text = json.dumps(payload, indent=2, sort_keys=False)
    else:
        text = render_report(spec, rows, runner=runner)
    # A file next to the results, so a report survives the terminal and can be diffed.
    suffix = "json" if as_json else "md"
    try:
        (spec.run_root / f"REPORT.{suffix}").write_text(text + "\n", encoding="utf-8")
    except OSError:
        pass
    print(text)
    return text


def runner_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return f"cae {version('coding-agent-evaluations')}"
    except PackageNotFoundError:
        return "cae (unknown version)"


# ------------------------------------------------------------------------------ cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark.py",
        description="Run one model x harness benchmark and print the standard report.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", required=True, help="served model id")
        p.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint")
        p.add_argument(
            "--models",
            help="comma-separated extra models for the same matrix",
        )
        p.add_argument(
            "--dataset",
            default="swebench_verified",
            help=f"one of: {', '.join(DATASETS)}",
        )
        p.add_argument(
            "--harnesses",
            help=f"comma-separated, from: {', '.join(HARNESSES)} (default: all)",
        )
        p.add_argument("--num-tasks", default="2", help="task count, or 'all'")
        p.add_argument("--seeds", type=int, default=1)
        p.add_argument("--max-steps", type=int, default=10)
        p.add_argument("--max-time", type=int, default=900, help="per-job wall clock")
        p.add_argument("--max-workers", type=int, default=1)
        p.add_argument("--no-resume", action="store_true")
        p.add_argument("--no-thinking", action="store_true")
        p.add_argument("--temperature", type=float, default=0.6)
        p.add_argument("--max-tokens-per-turn", type=int, default=None)
        p.add_argument("--api-key-env", default="EVAL_API_KEY")
        p.add_argument(
            "--output-root",
            default="eval-results/bench",
            help="parent for every run (default: eval-results/bench)",
        )

    run = sub.add_parser("run", help="doctor, run every cell, then report")
    common(run)
    run.add_argument(
        "--dry-run", action="store_true", help="print the cells, run nothing"
    )
    run.add_argument("--skip-doctor", action="store_true")
    run.add_argument("--no-progress", action="store_true")
    run.add_argument("--json", dest="as_json", action="store_true")

    doctor = sub.add_parser("doctor", help="preflight only")
    common(doctor)

    report = sub.add_parser("report", help="reprint a finished run, no evaluation")
    common(report)
    report.add_argument("--json", dest="as_json", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = Spec(args)

    if args.command == "report":
        # Read-only: reprinting a finished run must not create the directory it
        # is asking about.
        emit(spec, as_json=args.as_json, runner=runner_version())
        return 0

    config = write_config(spec)

    if args.command == "doctor":
        return cae("doctor", "--config", str(config))[0]

    if not args.skip_doctor and not args.dry_run:
        cae("doctor", "--config", str(config))

    if args.dry_run:
        cae("run", "--config", str(config), "--cells")
        return 0

    command = ["run", "--config", str(config)]
    if args.no_progress:
        command.append("--no-progress")
    code, stdout = cae(*command, check=False, capture=True)
    if code != 0:
        # A failing cell's own output is the diagnosis, so it is not suppressed.
        print(stdout, file=sys.stderr)
    emit(spec, as_json=args.as_json, runner=runner_version())
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
