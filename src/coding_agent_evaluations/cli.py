"""Command line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from . import __version__
from .config import ConfigError, RunConfig, apply_overrides, load_config
from .datasets import SUPPORTED, describe, load_tasks
from .doctor import format_checks, run_checks
from .matrix import MatrixConfig, load_run_or_matrix
from .matrix_runner import run_matrix
from .report import (
    as_markdown,
    as_table,
    as_token_table,
    caveats,
    load_summaries,
)
from .runner import run_evaluation

_HARNESSES = ("leaf", "opencode", "pi")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cae",
        description=(
            "Evaluate an OpenAI-compatible model on SWE-bench Verified and "
            "Terminal-Bench 2 Verified across agent harnesses."
        ),
    )
    parser.add_argument("--version", action="version", version=f"cae {__version__}")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run an evaluation")
    run.add_argument("--config", required=True, help="path to a config YAML file")
    run.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set num_tasks=2",
    )
    run.add_argument(
        "--harness",
        choices=_HARNESSES,
        help="override the harness",
    )
    run.add_argument(
        "--task-id",
        dest="task_ids",
        action="append",
        default=[],
        help="restrict the run to a task id (repeatable)",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve the config and list the selected tasks, then exit",
    )
    run.add_argument(
        "--no-progress",
        action="store_true",
        help="do not draw the matrix progress bar",
    )
    run.add_argument(
        "--cells",
        action="store_true",
        help="print the matrix cells and their output directories, then exit",
    )
    run.add_argument(
        "--markdown",
        action="store_true",
        help="print the matrix comparison as a markdown table",
    )

    tasks = subparsers.add_parser("tasks", help="list the tasks a config selects")
    tasks.add_argument("--config", required=True)
    tasks.add_argument("--set", dest="overrides", action="append", default=[])
    tasks.add_argument("--limit", type=int, default=20)

    describe_parser = subparsers.add_parser(
        "dataset", help="show a registered dataset source"
    )
    describe_parser.add_argument("name", choices=SUPPORTED)

    doctor = subparsers.add_parser(
        "doctor", help="check tools, disk and the model endpoint"
    )
    doctor.add_argument("--config", help="use the model and paths from a config")
    doctor.add_argument("--harness", choices=_HARNESSES, action="append", default=[])

    config_parser = subparsers.add_parser(
        "config", help="print the fully resolved config as JSON"
    )
    config_parser.add_argument("--config", required=True)
    config_parser.add_argument("--set", dest="overrides", action="append", default=[])

    compare = subparsers.add_parser(
        "compare", help="tabulate summaries from one or more runs"
    )
    compare.add_argument("paths", nargs="+", help="summary.json files or run directories")
    compare.add_argument("--markdown", action="store_true", help="emit a markdown table")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "tasks":
            return _cmd_tasks(args)
        if args.command == "dataset":
            return _cmd_dataset(args)
        if args.command == "doctor":
            return _cmd_doctor(args)
        if args.command == "config":
            return _cmd_config(args)
        if args.command == "compare":
            return _cmd_compare(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1


def _cmd_compare(args: argparse.Namespace) -> int:
    rows = load_summaries(list(args.paths))
    print(as_markdown(rows) if args.markdown else as_table(rows))
    speed = as_token_table(rows)
    if speed:
        print("\ntoken speed:")
        print(speed)
    notes = caveats(rows)
    if notes:
        print("\nnotes:")
        for note in notes:
            print(f"  - {note}")
    return 0


def _apply_overrides(config: RunConfig, args: argparse.Namespace) -> RunConfig:
    overrides = list(getattr(args, "overrides", []) or [])
    if getattr(args, "harness", None):
        overrides.append(f"harness.name={args.harness}")
    if getattr(args, "task_ids", None):
        overrides.append("task_ids=" + json.dumps(list(args.task_ids)))
    if not overrides:
        return config
    return apply_overrides(config, overrides)


def _resolve(args: argparse.Namespace) -> RunConfig:
    return _apply_overrides(load_config(args.config), args)


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_run_or_matrix(args.config)
    if isinstance(config, MatrixConfig):
        return _run_matrix(args, _apply_matrix_overrides(config, args))
    config = _apply_overrides(config, args)
    if args.dry_run:
        return _print_tasks(config, limit=50)
    summary = run_evaluation(config)
    print(json.dumps(_headline(summary), indent=2))
    return 0 if summary["jobs_failed"] == 0 else 1


def _apply_matrix_overrides(matrix: MatrixConfig, args: argparse.Namespace) -> MatrixConfig:
    """Apply `--set` to a matrix's base config, and `--harness` to its harness set.

    `--set` reaches every cell, which is the only useful reading of it for a matrix;
    `--harness` narrows the matrix to the harnesses asked for.
    """
    harness = getattr(args, "harness", None)
    if harness:
        matrix = replace(matrix, harnesses=(harness,))
    overrides = [
        *(getattr(args, "overrides", []) or []),
        *(
            ["task_ids=" + json.dumps(list(args.task_ids))]
            if getattr(args, "task_ids", None)
            else []
        ),
    ]
    if not overrides:
        return matrix
    return matrix.with_base(apply_overrides(matrix.base, overrides))


def _run_matrix(args: argparse.Namespace, matrix: MatrixConfig) -> int:
    """Run every cell, then compare them.

    The comparison table is printed at the end because the point of a matrix is the
    side-by-side row; a caller wanting the raw numbers has them in each cell's summary.json.
    """
    if args.cells or args.dry_run:
        print(json.dumps(matrix.as_dict(), indent=2, default=str))
        return 0
    reports = run_matrix(matrix, progress=not args.no_progress)
    print()
    for report in reports:
        if report.error:
            print(f"{report.label}: failed — {report.error}")
    paths = [str(report.config.output_dir) for report in reports]
    if paths:
        rows = load_summaries(paths)
        print()
        print(as_markdown(rows) if args.markdown else as_table(rows))
        speed = as_token_table(rows)
        if speed:
            print("\ntoken speed:")
            print(speed)
        notes = caveats(rows)
        if notes:
            print("\nnotes:")
            for note in notes:
                print(f"  - {note}")
    return 0 if not any(report.failed for report in reports) else 1


def _cmd_tasks(args: argparse.Namespace) -> int:
    config = _resolve(args)
    return _print_tasks(config, limit=args.limit)


def _print_tasks(config: RunConfig, *, limit: int) -> int:
    tasks = load_tasks(
        config.dataset,
        cache_dir=Path(config.cache_dir).expanduser(),
        task_ids=config.task_ids,
        limit=config.num_tasks,
        seed=config.seed,
    )
    print(
        json.dumps(
            {
                "dataset": config.dataset,
                "harness": config.harness.name,
                "model": config.model.name,
                "tasks_selected": len(tasks),
                "seeds_per_task": config.seeds_per_task,
                "jobs": len(tasks) * config.seeds_per_task,
                "instances": [task["instance_id"] for task in tasks[:limit]],
            },
            indent=2,
        )
    )
    return 0


def _cmd_dataset(args: argparse.Namespace) -> int:
    info = describe(args.name)
    print(json.dumps(info.__dict__, indent=2))
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    cache_dir = None
    workspace_root = None
    model = None
    harnesses: tuple[str, ...] = tuple(args.harness) or _HARNESSES
    if args.config:
        config = load_run_or_matrix(args.config)
        if isinstance(config, MatrixConfig):
            return _doctor_matrix(config, args)
        model = config.model
        cache_dir = Path(config.cache_dir).expanduser()
        workspace_root = Path(config.runtime.root).expanduser()
        harnesses = tuple(dict.fromkeys((config.harness.name, *harnesses)))
    checks = run_checks(
        model=model,
        harnesses=harnesses,
        cache_dir=cache_dir,
        workspace_root=workspace_root,
    )
    print(format_checks(checks))
    return 0 if all(check.ok for check in checks) else 1


def _doctor_matrix(matrix: MatrixConfig, args: argparse.Namespace) -> int:
    """Preflight a matrix: the shared tooling once, then every model's endpoint.

    A matrix fails late by nature — a model that is not served shows up as a wall of failed
    rollouts hours in — so each endpoint is asked for the model the matrix names before any
    of them runs.
    """
    harnesses = tuple(dict.fromkeys((*tuple(args.harness), *matrix.harnesses)))
    checks = run_checks(
        harnesses=harnesses,
        cache_dir=Path(matrix.base.cache_dir).expanduser(),
        workspace_root=Path(matrix.base.runtime.root).expanduser(),
    )
    print(format_checks(checks))
    ok = all(check.ok for check in checks)
    for model in matrix.models:
        print(f"\n== {model.name} @ {model.base_url} ==")
        endpoint = [check for check in run_checks(model=model) if check.name == "endpoint"]
        print(format_checks(endpoint))
        ok = ok and all(check.ok for check in endpoint)
    return 0 if ok else 1


def _cmd_config(args: argparse.Namespace) -> int:
    config = load_run_or_matrix(args.config)
    if isinstance(config, MatrixConfig):
        # A matrix config has no single resolved form; show its base and its cells, which is
        # what a run would actually execute.
        print(json.dumps(_apply_matrix_overrides(config, args).as_dict(), indent=2, default=str))
        return 0
    print(json.dumps(_apply_overrides(config, args).as_dict(), indent=2, default=str))
    return 0


def _headline(summary: dict[str, object]) -> dict[str, object]:
    keys = (
        "dataset",
        "harness",
        "model",
        "tasks_selected",
        "jobs_total",
        "jobs_completed",
        "jobs_failed",
        "resolved",
        "resolve_rate",
        "resolve_rate_graded",
        "jobs_infrastructure_failed",
        "pass_at_k",
        "duration_sec",
    )
    return {key: summary[key] for key in keys if key in summary}


__all__ = ["build_parser", "main"]