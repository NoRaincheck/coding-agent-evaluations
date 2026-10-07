#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "coding-agent-evaluations @ git+https://github.com/NoRaincheck/coding-agent-evaluations.git",
#   "pyyaml>=6.0",
# ]
# ///
"""Run one context-scaling sweep through `cae` and print the standard report.

Same role as the `run-benchmarks` skill's script, for a different question: that one runs
task suites, this one sweeps the context ladder. It exists rather than a bare
`cae ctxbench` because:

  * the published wheel ships the Python package but **not** `configs/*.yaml`, so
    `cae ctxbench --config configs/ctxbench.yaml` only works from a source checkout. This
    script writes a self-contained config instead.
  * a sweep's *scale* is part of its identity. Filler corpora are cached per context size
    under a key that includes the tokenizer and seed, and results are written per scale, so
    two sweeps with different ladders cannot quietly overwrite each other.

Usage is in SKILL.md next to this file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

SUITES = ("prefill", "generation", "needle", "execution", "memory")

DEFAULT_LADDER = (16_000, 32_000, 64_000, 128_000)


class Spec:
    """The complete, explicit description of one context sweep."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.model = args.model
        self.base_url = args.base_url.rstrip("/")
        self.api_key_env = args.api_key_env
        self.contexts = tuple(_int_list(args.contexts))
        self.suites = tuple(_names(args.suites)) if args.suites else SUITES

        unknown = set(self.suites) - set(SUITES)
        if unknown:
            raise SystemExit(
                f"unknown suite(s) {sorted(unknown)}; available: {', '.join(SUITES)}"
            )

        self.repeats = args.repeats
        self.needle_trials = args.needle_trials
        self.harness = args.harness
        self.num_tasks = None if args.num_tasks in (None, "all") else int(args.num_tasks)
        self.memory_iterations = args.memory_iterations
        self.temperature = args.temperature
        self.thinking = not args.no_thinking
        self.tokenizer = args.tokenizer

        self.output_root = Path(args.output_root).expanduser().resolve()
        self.scale = self._scale()

    @property
    def run_root(self) -> Path:
        return self.output_root / self.scale

    @property
    def config_path(self) -> Path:
        return self.run_root / "ctxbench.yaml"

    def _scale(self) -> str:
        """A directory name that changes whenever the measurement changes.

        The ladder and the suite limits *are* the measurement, so they are in the path: two
        sweeps that disagree about scale cannot end up in one directory and be mistaken for
        each other.
        """
        ladder = "-".join(f"{size // 1000}k" for size in self.contexts)
        tasks = "all" if self.num_tasks is None else str(self.num_tasks)
        suites = "+".join(self.suites)
        return f"c{ladder}r{self.repeats}n{self.needle_trials}t{tasks}_{suites}"

    def config(self) -> dict[str, Any]:
        model: dict[str, Any] = {
            "name": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "temperature": self.temperature,
            "tokenizer": self.tokenizer,
        }
        if self.thinking:
            model["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
        return {
            "output_dir": str(self.run_root),
            "context_sizes": list(self.contexts),
            "harness": {"name": self.harness},
            "seed": 42,
            "model": model,
            "runtime": {
                "kind": "host",
                "root": "~/.cache/cae/workspaces",
                "keep_workspaces": False,
                "options": {"step_timeout_sec": 1800},
            },
            "ctxbench": {
                "enabled": list(self.suites),
                "repeats": self.repeats,
                "needle_trials": self.needle_trials,
                "execution_num_tasks": self.num_tasks,
                "memory_iterations": self.memory_iterations,
                "tokenizer": self.tokenizer,
                "corpus_cache": "~/.cache/cae/ctxbench",
                "dataset": "swebench_verified",
                "cache_dir": "~/.cache/cae/datasets",
            },
        }


def _int_list(value: str) -> list[int]:
    """Parse a token ladder, rejecting anything non-numeric.

    A silently skipped entry would shift every later row off its label, and a mislabelled
    row is worse than a rejected run.
    """
    out: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if not item.isdigit():
            raise SystemExit(f"context size must be a positive integer, got {item!r}")
        out.append(int(item))
    if not out:
        raise SystemExit("no context sizes given")
    return sorted(set(out))


def _names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def cae(*args: str, check: bool = True, capture: bool = False) -> tuple[int, str]:
    """Invoke the runner in this script's own environment.

    `python -m` rather than the `cae` console script, so this works whatever PATH the
    script was launched with. stderr is inherited so run logs and progress stream live.
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
        "# Generated by the ctxbench skill. Every field is the standard contract;\n"
        "# change it by changing the skill's arguments, not by editing this file.\n"
        + yaml.safe_dump(spec.config(), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return spec.config_path


def collect(run_root: Path) -> Path | None:
    return (run_root / "ctxbench.json") if (run_root / "ctxbench.json").is_file() else None


def emit(spec: Spec, *, as_json: bool, runner: str) -> str:
    """Print the finished run, and write the same text beside the results."""
    path = collect(spec.run_root)
    if path is None:
        text = (
            f"no ctxbench.json under {spec.run_root}; "
            "nothing to report (was the run interrupted?)"
        )
        print(text)
        return text
    payload = json.loads(path.read_text(encoding="utf-8"))
    if as_json:
        text = json.dumps(payload, indent=2)
    else:
        from coding_agent_evaluations.ctxbench.report import as_markdown

        text = as_markdown({**payload, "harness_version": runner})
    suffix = "json" if as_json else "md"
    try:
        (spec.run_root / f"REPORT.{suffix}").write_text(text + "\n", encoding="utf-8")
    except OSError:
        pass
    print(text)
    return text


def runner_version() -> str:
    try:
        from importlib.metadata import version

        return f"cae {version('coding-agent-evaluations')}"
    except Exception:  # noqa: BLE001 - version is metadata, never worth failing a run
        return "cae (unknown version)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ctxbench.py",
        description="Sweep context sizes against one served model and print the report.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", required=True, help="served model id")
        p.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint")
        p.add_argument(
            "--contexts",
            default=",".join(str(size) for size in DEFAULT_LADDER),
            help="comma-separated token ladder (default: 16000,32000,64000,128000)",
        )
        p.add_argument("--suites", help=f"comma-separated subset of: {', '.join(SUITES)}")
        p.add_argument("--repeats", type=int, default=3)
        p.add_argument("--needle-trials", type=int, default=6)
        p.add_argument("--harness", default="leaf", choices=("leaf", "opencode", "pi"))
        p.add_argument("--num-tasks", default="2", help="pass@1 task count, or 'all'")
        p.add_argument("--memory-iterations", type=int, default=10)
        p.add_argument("--temperature", type=float, default=0.6)
        p.add_argument("--tokenizer", default="o200k_base")
        p.add_argument("--no-thinking", action="store_true")
        p.add_argument("--api-key-env", default="EVAL_API_KEY")
        p.add_argument("--output-root", default="eval-results/ctxbench")

    run = sub.add_parser("run", help="preflight, sweep every context size, then report")
    common(run)
    run.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    run.add_argument("--skip-doctor", action="store_true")
    run.add_argument("--json", dest="as_json", action="store_true")

    doctor = sub.add_parser("doctor", help="preflight only")
    common(doctor)

    report = sub.add_parser("report", help="reprint a finished run, run nothing")
    common(report)
    report.add_argument("--json", dest="as_json", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = Spec(args)

    if args.command == "report":
        emit(spec, as_json=args.as_json, runner=runner_version())
        return 0

    config = write_config(spec)

    if args.command == "doctor":
        return cae("doctor", "--config", str(config))[0]

    if args.dry_run:
        print(yaml.safe_dump(spec.config(), sort_keys=False))
        print(f"\nscale directory: {spec.run_root}")
        print(f"suites: {', '.join(spec.suites)}")
        if "execution" in spec.suites:
            print(
                "note: the execution suite provisions real task workspaces and runs their "
                "verifiers; it dominates the wall clock of a sweep."
            )
        return 0

    if not args.skip_doctor:
        cae("doctor", "--config", str(config))

    code, _ = cae("ctxbench", "--config", str(config), check=False)
    emit(spec, as_json=args.as_json, runner=runner_version())
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None