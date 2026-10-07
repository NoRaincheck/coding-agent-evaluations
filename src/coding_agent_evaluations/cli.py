"""Command line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import __version__
from .config import ConfigError, RunConfig, apply_overrides, load_config
from .ctxbench import load_ctx_config, run_ctxbench
from .ctxbench.config import SUITES
from .ctxbench.report import as_json as ctx_as_json
from .ctxbench.report import as_markdown as ctx_as_markdown
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

    ctx = subparsers.add_parser(
        "ctxbench",
        help="sweep context sizes: speed, retrieval, pass@1, harness health",
    )
    ctx.add_argument("--config", required=True, help="path to a ctxbench config YAML")
    ctx.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set seed=7",
    )
    ctx.add_argument(
        "--contexts",
        help="comma-separated token ladder, e.g. 16000,32000,64000,128000",
    )
    ctx.add_argument(
        "--harness",
        choices=_HARNESSES,
        help="harness for the pass@1 suite (default: leaf)",
    )
    ctx.add_argument(
        "--suites",
        help=f"comma-separated subset of: {', '.join(SUITES)}",
    )
    ctx.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="print the JSON payload instead of the table",
    )
    ctx.add_argument(
        "--plan",
        action="store_true",
        help="print the resolved config and ladder, then exit",
    )
    ctx.add_argument(
        "--no-progress", action="store_true", help="do not log each context size"
    )
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
        if args.command == "ctxbench":
            return _cmd_ctxbench(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1


def _cmd_ctxbench(args: argparse.Namespace) -> int:
    overrides = list(getattr(args, "overrides", []) or [])
    if args.contexts:
        overrides.append("context_sizes=" + json.dumps(_int_list(args.contexts)))
    if args.harness:
        overrides.append(f"harness={args.harness}")
    if args.suites:
        overrides.append("ctxbench.enabled=" + json.dumps(_name_list(args.suites)))
    config = load_ctx_config(args.config, overrides)

    if args.plan:
        print(json.dumps(config.as_dict(), indent=2))
        return 0

    def on_size(result: Any) -> None:
        if args.no_progress:
            return
        rows = result.as_dict()
        print(
            f"  {result.context_size_tokens:>7,} tokens  "
            f"needle {rows['correctness'].get('needle_in_haystack_pass_rate')}",
            flush=True,
        )

    payload = run_ctxbench(config, on_size=on_size)
    # The table is printed even when suites failed, so a failed run still reports what it
    # managed to measure; the exit code is what says it was incomplete.
    print(ctx_as_json(payload) if args.as_json else ctx_as_markdown(payload))
    errors = sum(len(row.get("errors") or []) for row in payload.get("results", []))
    return 1 if errors else 0


def _int_list(value: str) -> list[int]:
    """Parse a comma-separated token ladder, rejecting anything non-numeric.

    A silently ignored entry would shift every later row off its label, so this raises
    rather than skipping.
    """
    out: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if not item.isdigit():
            raise ConfigError(f"context size must be a positive integer, got {item!r}")
        out.append(int(item))
    if not out:
        raise ConfigError("no context sizes given")
    return out


def _name_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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
        # A ctxbench config is a different shape, and pointing `doctor` at one is the
        # natural thing to do before a sweep. Detected by the `ctxbench:` block rather than
        # by trial and error, so a genuine config error is still reported as one.
        if _is_ctxbench_config(args.config):
            config = load_ctx_config(args.config)
            model = config.model
            cache_dir = config.suites.cache_dir
            workspace_root = config.runtime.root
            harnesses = tuple(dict.fromkeys((config.harness, *harnesses)))
        else:
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


def _is_ctxbench_config(path: str | Path) -> bool:
    """Whether a config file describes a ctxbench sweep rather than a benchmark run.

    Keyed on the `ctxbench:` block alone, which is what the ctxbench loader requires and
    what nothing else writes. Keyed on the *absence* of `dataset:` instead would misread
    the shipped `configs/ctxbench.yaml`, which extends `base.yaml` and therefore resolves
    to a payload that has both.
    """
    from .config import load_payload

    try:
        payload = load_payload(path)
    except ConfigError:
        # A config that will not even parse is not a ctxbench config; let the real loader
        # report why it is broken.
        return False
    return "ctxbench" in payload


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