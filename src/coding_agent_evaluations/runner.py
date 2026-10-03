"""Evaluation runner.

Same accounting as the reference runner: every scheduled task-seed pair appears
in ``results.jsonl``, the resolve rate denominator is all scheduled pairs (not
pass@k), completed runs keep their reward, failed and unstarted pairs are
eligible on resume, and each pair gets a bounded number of attempts.

Two additions matter when comparing harnesses on one machine:

* host-only infrastructure failures (a task that cannot be provisioned or graded)
  are recorded with ``exit_reason`` set to ``provision_error`` /
  ``verifier_error`` and are reported separately, so a resolution rate is not
  quietly diluted by a benchmark detail that only exists without containers.
* every result row carries the harness and model, so one results file can hold
  several harnesses.
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import threading
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from frognano.runtimes.errors import CommandTimeoutError, PodExecutionError

from .config import RunConfig
from .datasets import load_tasks
from .harnesses import HarnessRun, build_harness
from .runtimes import HostTaskRuntime, get_runtime

logger = logging.getLogger(__name__)

# Exit reasons that mean "the sandbox or the harness broke", not "the agent failed".
INFRASTRUCTURE_REASONS = frozenset(
    {"provision_error", "verifier_error", "harness_unavailable", "runtime_error"}
)
RETRYABLE_REASONS = frozenset({"llm_query_error", "tool_error", "unknown"})


@dataclass
class Job:
    task: dict[str, Any]
    seed: int

    @property
    def key(self) -> tuple[str, int]:
        return (str(self.task["instance_id"]), self.seed)


def run_evaluation(config: RunConfig) -> dict[str, Any]:
    """Run one config end to end and write results, summary and trajectories."""
    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    write_config(output_dir / "config.json", config)

    tasks = load_tasks(
        config.dataset,
        cache_dir=Path(config.cache_dir).expanduser(),
        task_ids=config.task_ids,
        limit=config.num_tasks,
        seed=config.seed,
    )
    jobs = [Job(task=task, seed=seed) for task in tasks for seed in range(config.seeds_per_task)]
    expected = {job.key for job in jobs}
    processed: set[tuple[str, int]] = set()
    if config.resume:
        latest = latest_results(results_path)
        processed = {
            key
            for key, row in latest.items()
            if key in expected and row.get("status") in {"completed", "provisioned"}
        }
    jobs = [job for job in jobs if job.key not in processed]
    if config.resume and processed:
        logger.info("Resume: %s of %s pairs already done", len(processed), len(expected))

    stop_event = threading.Event()
    previous_handlers = _install_stop_handlers(stop_event)
    write_lock = threading.Lock()
    started = time.time()
    try:
        with ExitStack() as stack:
            pool = stack.enter_context(
                ThreadPoolExecutor(max_workers=config.max_workers)
            )
            futures = {
                pool.submit(_run_job, config, job, stop_event, output_dir): job
                for job in jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    result = _failed_result(job, "internal_error", f"{type(exc).__name__}: {exc}")
                    logger.debug("Job crashed", exc_info=True)
                with write_lock:
                    append_result(results_path, result)
                    log_result(result)
    finally:
        _restore_stop_handlers(previous_handlers)

    return write_summary(output_dir, config, expected, results_path, time.time() - started)


def _run_job(
    config: RunConfig,
    job: Job,
    stop_event: threading.Event,
    output_dir: Path,
) -> dict[str, Any]:
    instance_id = str(job.task["instance_id"])
    task_dir = Path(config.output_dir).expanduser() / "trajectories" / instance_id
    task_dir.mkdir(parents=True, exist_ok=True)
    budget = config.max_total_time_sec or int(job.task["agent_timeout_sec"])
    errors: list[str] = []
    last_reason = "runtime_error"
    for attempt in range(1, config.max_attempts + 1):
        if stop_event.is_set():
            return _failed_result(job, "cancelled", "evaluation cancelled")
        # The whole pair shares one wall-clock budget, matching the reference
        # runner's per-task `agent_timeout_sec` rather than restarting it on retry.
        deadline = time.monotonic() + budget
        runtime: HostTaskRuntime | None = None
        try:
            remaining = int(max(1, deadline - time.monotonic()))
            run_id = f"s{job.seed}-a{attempt}-{uuid.uuid4().hex[:6]}"
            runtime_cls = get_runtime(config.runtime.kind)
            runtime = runtime_cls(job.task, config.runtime, run_id=run_id)
            if remaining < budget:
                runtime.logger.info(
                    "Retry %s for %s seed %s has %ss of the %ss budget left",
                    attempt,
                    instance_id,
                    job.seed,
                    remaining,
                    budget,
                )
            harness_cls = build_harness(config.harness.name)
            partial_path = task_dir / f"trajectory_seed-{job.seed}.partial.json"
            run = HarnessRun(
                runtime=runtime,
                task=job.task,
                model=config.model,
                config=config.harness,
                instance_id=instance_id,
                seed=job.seed,
                # A harness that owns its own loop is bounded by time only.
                max_steps=(
                    config.max_steps
                    if getattr(harness_cls, "enforces_max_steps", True)
                    else config.harness.max_steps or config.max_steps
                ),
                max_context_tokens=config.max_context_tokens,
                max_total_time_sec=remaining,
                logger=logger,
                checkpoint_callback=lambda value, path=partial_path: write_json(path, value),
            )
            trajectory = harness_cls().run(run)
            trajectory["attempt"] = attempt
            if trajectory.get("exit_reason") == "cancelled":
                return _failed_result(job, "cancelled", "evaluation cancelled")
            if trajectory.get("exit_reason") in RETRYABLE_REASONS:
                last_reason = str(trajectory["exit_reason"])
                raise RuntimeError(
                    f"retryable {last_reason}: {trajectory.get('error')}"
                )
            try:
                reward, test_output = runtime.compute_reward()
            except Exception as exc:  # noqa: BLE001 - any verifier failure is ours, not the model's
                # The agent ran; grading did not. Record it separately instead of
                # charging the model for a verifier we could not execute.
                trajectory["reward"] = None
                trajectory["test_output"] = None
                trajectory["grading_error"] = f"{type(exc).__name__}: {exc}"
                write_json(task_dir / f"trajectory_seed-{job.seed}.json", trajectory)
                partial_path.unlink(missing_ok=True)
                logger.warning(
                    "Verifier failed for %s seed %s: %s", instance_id, job.seed, exc
                )
                return _failed_result(
                    job,
                    "verifier_error",
                    trajectory["grading_error"],
                    attempt=attempt,
                    trajectory=str(task_dir / f"trajectory_seed-{job.seed}.json"),
                    harness=config.harness.name,
                )
            trajectory["reward"] = reward
            trajectory["test_output"] = test_output[-8000:]
            write_json(task_dir / f"trajectory_seed-{job.seed}.json", trajectory)
            partial_path.unlink(missing_ok=True)
            patch = str(trajectory.get("output_patch") or "")
            if patch:
                (task_dir / f"generated_seed-{job.seed}.patch").write_text(
                    patch, encoding="utf-8"
                )
            return {
                "instance_id": instance_id,
                "seed": job.seed,
                "attempt": attempt,
                "status": "completed",
                "reward": reward,
                "exit_reason": trajectory.get("exit_reason"),
                "n_steps": trajectory.get("n_steps"),
                "elapsed_sec": trajectory.get("elapsed_sec"),
                "harness": config.harness.name,
                "trajectory": str(task_dir / f"trajectory_seed-{job.seed}.json"),
                "source": job.task.get("source"),
            }
        except Exception as exc:
            reason = _classify(exc)
            last_reason = reason
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            logger.warning(
                "Attempt %s/%s failed for %s seed %s: %s",
                attempt,
                config.max_attempts,
                instance_id,
                job.seed,
                exc,
            )
            if reason not in INFRASTRUCTURE_REASONS:
                logger.debug("Failure detail", exc_info=True)
            else:
                logger.debug("Infrastructure detail", exc_info=sys.exc_info())
        finally:
            if runtime is not None:
                try:
                    runtime.close()
                except Exception:  # pragma: no cover - cleanup best effort
                    logger.debug("Workspace cleanup failed", exc_info=True)
    return _failed_result(
        job,
        last_reason,
        "\n".join(errors),
        attempt=config.max_attempts,
        harness=config.harness.name,
    )


def _classify(exc: BaseException) -> str:
    """Map an exception onto an exit reason."""
    from .runtimes.provision import ProvisionError

    if isinstance(exc, ProvisionError):
        return "provision_error"
    if isinstance(exc, CommandTimeoutError):
        return "tool_error"
    if isinstance(exc, PodExecutionError):
        return "tool_error"
    if isinstance(exc, FileNotFoundError):
        return "harness_unavailable"
    if isinstance(exc, (TimeoutError,)):
        return "max_time"
    return "runtime_error"


def _failed_result(
    job: Job,
    reason: str,
    error: str,
    *,
    attempt: int = 0,
    harness: str | None = None,
    trajectory: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "instance_id": str(job.task["instance_id"]),
        "seed": job.seed,
        "attempt": attempt,
        "status": "failed",
        "reward": 0.0,
        "exit_reason": reason,
        "error": error,
        "source": job.task.get("source"),
    }
    if harness:
        row["harness"] = harness
    if trajectory:
        row["trajectory"] = trajectory
    return row


def write_summary(
    output_dir: Path,
    config: RunConfig,
    expected: set[tuple[str, int]],
    results_path: Path,
    duration_sec: float,
) -> dict[str, Any]:
    """Summarize results, separating agent outcomes from host-only failures."""
    rows = [row for key, row in latest_results(results_path).items() if key in expected]
    completed = [row for row in rows if row.get("status") == "completed"]
    infrastructure = [
        row for row in rows if row.get("exit_reason") in INFRASTRUCTURE_REASONS
    ]
    resolved = sum(float(row.get("reward") or 0) >= 1 for row in completed)
    jobs_total = len(expected)
    graded = jobs_total - len(infrastructure)
    summary: dict[str, Any] = {
        "dataset": config.dataset,
        "harness": config.harness.name,
        "model": config.model.name,
        "base_url": config.model.base_url,
        "runtime": config.runtime.kind,
        "tasks_selected": len({key[0] for key in expected}),
        "jobs_total": jobs_total,
        "jobs_recorded": len(rows),
        "jobs_completed": len(completed),
        "jobs_failed": jobs_total - len(completed),
        "resolved": resolved,
        "unresolved": len(completed) - resolved,
        # Matches the reference runner: the denominator is every scheduled pair.
        "resolve_rate": resolved / jobs_total if jobs_total else 0.0,
        # Excludes pairs that never got a fair test on this host.
        "resolve_rate_graded": resolved / graded if graded > 0 else 0.0,
        "jobs_infrastructure_failed": len(infrastructure),
        "infrastructure_reasons": _count_by(infrastructure, "exit_reason"),
        "error_rate": (jobs_total - len(completed)) / jobs_total if jobs_total else 0.0,
        "duration_sec": duration_sec,
        "config": config.as_dict(),
    }
    if config.seeds_per_task > 1:
        summary["pass_at_k"] = _pass_at_k(completed, k=config.seeds_per_task)
    summary["exit_reasons"] = _count_by(rows, "exit_reason")
    summary["mean_steps"] = _mean(completed, "n_steps")
    summary["mean_elapsed_sec"] = _mean(completed, "elapsed_sec")
    summary["limits"] = _limits(config)
    write_json(output_dir / "summary.json", summary)
    return summary


def _limits(config: RunConfig) -> dict[str, Any]:
    """Which limits actually bound this harness.

    Comparing harnesses means knowing which constraints were in force: the
    reference loop honours every knob, a CLI harness honours the clock.
    """
    harness = build_harness(config.harness.name)
    return {
        "max_steps": config.max_steps,
        "max_context_tokens": config.max_context_tokens,
        "max_total_time_sec": config.max_total_time_sec,
        "max_steps_enforced": bool(getattr(harness, "enforces_max_steps", True)),
        "model_params_ignored": list(
            getattr(harness, "ignored_model_params", ()) or ()
        ),
    }


def _pass_at_k(rows: Sequence[dict[str, Any]], *, k: int) -> dict[str, Any]:
    """Share of tasks solved by at least one of their ``k`` seeds."""
    tasks: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "completed":
            tasks.setdefault(str(row["instance_id"]), []).append(row)
    if not tasks:
        return {"k": k, "percent": 0.0, "solved_tasks": 0, "total_tasks": 0}
    solved = sum(
        1
        for seeds in tasks.values()
        if any(float(seed.get("reward") or 0) >= 1 for seed in seeds)
    )
    return {
        "k": k,
        "percent": 100.0 * solved / len(tasks),
        "solved_tasks": solved,
        "total_tasks": len(tasks),
    }


def _count_by(rows: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return sum(values) / len(values) if values else None


def latest_results(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.is_file():
        return latest
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        latest[(str(row["instance_id"]), int(row.get("seed", 0)))] = row
    return latest


def append_result(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, default=str) + "\n")


def log_result(row: dict[str, Any]) -> None:
    if row.get("status") == "completed":
        logger.info(
            "resolved=%s %s seed %s steps=%s reason=%s",
            row.get("reward"),
            row.get("instance_id"),
            row.get("seed"),
            row.get("n_steps"),
            row.get("exit_reason"),
        )
    else:
        logger.warning(
            "FAILED %s seed %s (%s): %s",
            row.get("instance_id"),
            row.get("seed"),
            row.get("exit_reason"),
            str(row.get("error") or "")[:300],
        )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_config(path: Path, config: RunConfig) -> None:
    write_json(path, config.as_dict())


def _install_stop_handlers(stop_event: threading.Event) -> dict[signal.Signals, Any]:
    if threading.current_thread() is not threading.main_thread():
        return {}
    previous: dict[signal.Signals, Any] = {}
    signal_count = 0

    def handle(signal_number: int, _frame: Any) -> None:
        nonlocal signal_count
        signal_count += 1
        if signal_count == 1:
            stop_event.set()
            return
        _restore_stop_handlers(previous)
        raise KeyboardInterrupt

    for signal_number in (signal.SIGINT, signal.SIGTERM):
        previous[signal_number] = signal.getsignal(signal_number)
        signal.signal(signal_number, handle)
    return previous


def _restore_stop_handlers(previous: dict[signal.Signals, Any]) -> None:
    for signal_number, handler in previous.items():
        if handler is not None:
            signal.signal(signal_number, handler)


__all__ = ["INFRASTRUCTURE_REASONS", "Job", "run_evaluation", "write_summary"]