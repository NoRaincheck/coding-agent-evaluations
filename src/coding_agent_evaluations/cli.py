"""Command line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .config import ConfigError, RunConfig, apply_overrides, load_config
from .datasets import SUPPORTED, describe, load_tasks
from .doctor import format_checks, run_checks
from .report import as_markdown, as_table, caveats, load_summaries
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
    notes = caveats(rows)
    if notes:
        print("\nnotes:")
        for note in notes:
            print(f"  - {note}")
    return 0


def _resolve(args: argparse.Namespace) -> RunConfig:
    config = load_config(args.config)
    overrides = list(getattr(args, "overrides", []) or [])
    if getattr(args, "harness", None):
        overrides.append(f"harness.name={args.harness}")
    if getattr(args, "task_ids", None):
        overrides.append("task_ids=" + json.dumps(list(args.task_ids)))
    if overrides:
        config = apply_overrides(config, overrides)
    return config


def _cmd_run(args: argparse.Namespace) -> int:
    config = _resolve(args)
    if args.dry_run:
        return _print_tasks(config, limit=50)
    summary = run_evaluation(config)
    print(json.dumps(_headline(summary), indent=2))
    return 0 if summary["jobs_failed"] == 0 else 1


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
    model = None
    cache_dir = None
    workspace_root = None
    harnesses: tuple[str, ...] = tuple(args.harness) or _HARNESSES
    if args.config:
        config = load_config(args.config)
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


def _cmd_config(args: argparse.Namespace) -> int:
    print(json.dumps(_resolve(args).as_dict(), indent=2, default=str))
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