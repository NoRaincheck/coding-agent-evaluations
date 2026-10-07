"""Orchestration: one pass per context size, then the standard report.

Each context size runs the enabled suites and produces one row. The suites are
independent on purpose -- a run where prefill times out at 128k should still produce the
needle row for 128k, because "the endpoint could not prefill this and also could not
retrieve from it" is two findings, and collapsing them into one failed cell loses the
second.

The one structural dependency is `execution`: pass@1 comes from real benchmark tasks
graded by their own verifiers, so it runs the existing evaluation runner with
`max_context_tokens` set to the context size under test. A cell that cannot be graded is
recorded as infrastructure-failed and excluded from the rate, exactly as in the
benchmark summary -- the denominator stays honest instead of counting a broken sandbox as
a failed attempt.
"""

from __future__ import annotations

import json
import logging
import random
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import RunConfig
from ..runner import run_evaluation
from .config import CtxBenchConfig
from .endpoint import EndpointProbe, ProbeError, StreamResult, TruncationCheck
from .filler import build_filler, count_tokens, encoding_for
from .needle import extract_answer, make_needle, plant, score, signature_needle
from .rss import MemoryTrace

logger = logging.getLogger(__name__)

REPORT_VERSION = 1

#: The prefill probe asks for one word. If the model replies with an essay, that is a
#: finding worth surfacing rather than a reason to quietly retry.
PREFILL_INSTRUCTION = (
    "Reply with exactly the single word: Acknowledged\n"
    "Do not explain, do not add punctuation, do not add anything else."
)

#: Prefix that makes each repeat after the first a distinct prompt, so an endpoint's
#: prefix cache cannot turn repeats 2..n into cache reads.
CACHE_BUST_MARKER = "# probe-variant "

#: A fixed coding prompt so the generation suite varies only in context, never in task.
#: Two runs of the same size are comparable; two sizes are comparable to each other.
GENERATION_INSTRUCTION = (
    "Write a Python function `parse_duration(text: str) -> int` that parses strings like "
    "'90s', '5m', '2h', '3d' into seconds, raises ValueError on malformed input, and "
    "accepts a space between the number and the unit. Return only the function."
)


@dataclass
class SizeResult:
    """Everything measured at one context size."""

    context_size_tokens: int
    speed: dict[str, Any] = field(default_factory=dict)
    correctness: dict[str, Any] = field(default_factory=dict)
    harness_health: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "context_size_tokens": self.context_size_tokens,
            "speed": self.speed,
            "correctness": self.correctness,
            "harness_health": self.harness_health,
            "errors": self.errors,
        }


def run_ctxbench(
    config: CtxBenchConfig,
    *,
    on_size: Callable[[SizeResult], None] | None = None,
) -> dict[str, Any]:
    """Sweep the context ladder and return the report payload."""
    encoding = encoding_for(config.suites.tokenizer)
    probe = EndpointProbe(
        model=config.model,
        timeout_sec=config.suites.request_timeout_sec,
        retries=config.suites.retries,
    )
    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Wall clock for the id and `time.perf_counter` for the duration: the two answer
    # different questions, and passing a monotonic counter to a date formatter produces an
    # id from an arbitrary epoch rather than from when the run happened.
    wall_start = time.time()
    started = time.perf_counter()

    results: list[SizeResult] = []
    for size in config.context_sizes:
        logger.info("ctxbench: %s tokens", f"{size:,}")
        result = SizeResult(context_size_tokens=size)
        corpus = corpus_for(config, encoding, size)
        # The corpus is exactly `size` tokens, so the sent count is the corpus plus the
        # instruction wrapped around it. Measured locally rather than assumed, because it
        # is the baseline every truncation check compares the server's accounting against.
        sent_tokens = _prompt_tokens(
            PREFILL_INSTRUCTION + corpus.text, encoding
        )
        result.harness_health["prompt_tokens_sent"] = sent_tokens

        if config.suites.runs["prefill"]:
            _run_prefill(config, probe, corpus.text, encoding, result)
        if config.suites.runs["generation"]:
            _run_generation(config, probe, corpus.text, result)
        if config.suites.runs["needle"]:
            _run_needle(config, probe, corpus.text, size, result)
        if config.suites.runs["execution"]:
            _run_execution(config, size, output_dir, result)

        results.append(result)
        if on_size is not None:
            on_size(result)

    # Collected during the sweep rather than folded into a single row: the question is
    # whether repeated large-context handling leaks across the whole run, which no one
    # context size can answer on its own.
    memory: dict[str, Any] = {}
    if config.suites.runs["memory"]:
        memory = _run_memory(config, probe, encoding, probe_stats=probe.stats)

    payload = {
        "report_version": REPORT_VERSION,
        "run_id": _run_id(wall_start),
        "generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": config.model.name,
        "base_url": config.model.base_url,
        "harness": config.harness,
        "harness_version": _harness_version(),
        "tokenizer": config.suites.tokenizer,
        "duration_sec": round(time.perf_counter() - started, 1),
        "config": config.as_dict(),
        "results": [result.as_dict() for result in results],
        "memory": memory,
        "endpoint_stats": dict(probe.stats),
    }
    (output_dir / "ctxbench.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


# ------------------------------------------------------------------ corpus caching


def corpus_for(config: CtxBenchConfig, encoding: Any, size: int) -> Any:
    """A filler corpus of exactly ``size`` tokens, cached on disk.

    Generating 128k tokens of validated filler costs a couple of seconds, and a resumed
    run pays it again for every size. The cache key includes the tokenizer and seed,
    because a corpus built for one tokenizer is not a corpus of the same size for another.
    """
    cache = Path(config.suites.corpus_cache).expanduser()
    key = f"{size}-{config.suites.tokenizer}-{config.seed}.txt"
    path = cache / key
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        if count_tokens(text, encoding) == size:
            return _Cached(text=text)
        # A stale or truncated cache entry would silently change the measured context
        # size, so it is rebuilt rather than trusted.
        logger.warning("Discarding filler cache %s: token count mismatch", path)
    corpus = build_filler(size, encoding, seed=config.seed)
    cache.mkdir(parents=True, exist_ok=True)
    path.write_text(corpus.text, encoding="utf-8")
    return _Cached(text=corpus.text)


@dataclass(frozen=True)
class _Cached:
    text: str


# ------------------------------------------------------------------------- prefill


def _prompt_tokens(content: str, encoding: Any) -> int:
    """Tokens in a single-user-message request, counted the way the endpoint will count it.

    The chat envelope is included rather than just the prompt text: a count that omits the
    role and message scaffolding understates what the server received, and this number is
    what the truncation check compares against.
    """
    return count_tokens(
        json.dumps([{"role": "user", "content": content}], separators=(",", ":")),
        encoding,
    )


def _run_prefill(
    config: CtxBenchConfig,
    probe: EndpointProbe,
    corpus: str,
    encoding: Any,
    result: SizeResult,
) -> None:
    """Time-to-first-token against a padded prompt with a one-word answer.

    Repeated because a single sample at the small sizes is dominated by noise; the median
    is reported and the spread is kept so a bimodal endpoint is visible rather than
    averaged into a plausible-looking number.
    """
    samples: list[StreamResult] = []
    # The cache-busted repeats are marginally longer than the first, so the truncation
    # baseline is the largest prompt actually sent rather than the first.
    sent_tokens = 0
    for index in range(config.suites.repeats):
        # Repeats send byte-identical input, so an endpoint with a prefix cache makes
        # repeats 2..n prefill almost free. Measured on this machine: a 42-second prefill
        # became 75 milliseconds on the second identical request, and the median would
        # have reported a cache read as the cost of a context. Every repeat after the first
        # therefore opens with a unique marker, which invalidates any prefix cache while
        # changing the prompt by a handful of tokens.
        prompt = PREFILL_INSTRUCTION + corpus
        if index:
            prompt = f"{CACHE_BUST_MARKER}{index}\n{prompt}"
        sent_tokens = max(sent_tokens, _prompt_tokens(prompt, encoding))
        try:
            samples.append(
                probe.stream(
                    [{"role": "user", "content": prompt}],
                    max_tokens=config.suites.prefill_max_tokens,
                )
            )
        except ProbeError as exc:
            if exc.context_overflow:
                # The endpoint named the limit it has. Recorded as a fact about the served
                # checkpoint rather than as a failure of the model at this context size.
                result.harness_health["context_truncation"] = TruncationCheck(
                    sent_tokens=sent_tokens,
                    received_tokens=None,
                    status=exc.status,
                    rejected=True,
                    server_context_limit=exc.server_context_limit,
                    server_prompt_tokens=exc.server_prompt_tokens,
                ).as_dict()
                result.speed["prefill"] = {
                    "error": "rejected: request exceeds the server's context window",
                    "server_context_limit": exc.server_context_limit,
                }
                result.errors.append(
                    f"prefill: endpoint context limit "
                    f"{exc.server_context_limit:,} < sent {sent_tokens:,}"
                )
                return
            result.errors.append(f"prefill: {exc}")
            break
    if not samples:
        result.speed["prefill"] = {"error": "no successful prefill probe"}
        return

    ttfts = sorted(item.ttft_sec * 1000 for item in samples)
    last = samples[-1]
    # The smallest count the server reported across the repeats: if any single request lost
    # context, that is the finding, and taking the last sample's count could hide it behind
    # a healthy repeat.
    reported = [item.prompt_tokens for item in samples if item.prompt_tokens is not None]
    received = min(reported) if reported else None
    truncation = TruncationCheck(sent_tokens=sent_tokens, received_tokens=received)
    prefill_rates = [item.prefill_tps for item in samples if item.prefill_tps]
    result.speed["prefill"] = {
        "ttft_ms": round(statistics.median(ttfts), 1),
        "ttft_ms_min": round(ttfts[0], 1),
        "ttft_ms_max": round(ttfts[-1], 1),
        "ttft_ms_stdev": (
            round(statistics.stdev(ttfts), 1) if len(ttfts) > 1 else 0.0
        ),
        "repeats": len(samples),
        # Uncached by construction: see the cache-bust marker above. A reader comparing
        # this against a warm-prefix number should know which one they are looking at.
        "prefix_cache_busted": config.suites.repeats > 1,
        "prompt_tokens_reported": received,
        "prefill_tps": (
            round(statistics.median(prefill_rates), 1) if prefill_rates else None
        ),
        "answer": last.text.strip()[:64],
        "completion_tokens": last.completion_tokens,
        # `None` when the model spent its whole budget reasoning and never reached an
        # answer. Scoring that as a failure would report the token budget as a model
        # behaviour, so an unobservable answer is reported as unobservable.
        "acknowledged": (
            None if last.answer_unobservable else "acknowledged" in last.text.lower()
        ),
    }
    result.harness_health["context_truncation"] = truncation.as_dict()
    result.harness_health["context_truncated"] = truncation.truncated
    result.harness_health["prompt_tokens_sent"] = max(
        sent_tokens, result.harness_health.get("prompt_tokens_sent", 0)
    )


# --------------------------------------------------------------------- generation


def _run_generation(
    config: CtxBenchConfig,
    probe: EndpointProbe,
    corpus: str,
    result: SizeResult,
) -> None:
    """Decode throughput and end-to-end latency on a fixed coding prompt."""
    samples: list[StreamResult] = []
    for _ in range(config.suites.repeats):
        try:
            samples.append(
                probe.stream(
                    [{"role": "user", "content": corpus + "\n\n" + GENERATION_INSTRUCTION}],
                    max_tokens=config.suites.generation_max_tokens,
                    temperature=config.model.temperature,
                )
            )
        except ProbeError as exc:
            result.errors.append(f"generation: {exc}")
            break
    if not samples:
        result.speed["generation"] = {"error": "no successful generation probe"}
        return

    tps = [item.decode_tps for item in samples if item.decode_tps]
    latencies = sorted(item.total_sec * 1000 for item in samples)
    tokens = [item.completion_tokens for item in samples if item.completion_tokens]
    result.speed["generation"] = {
        "tps": round(statistics.median(tps), 2) if tps else None,
        "tps_samples": [round(value, 2) for value in tps],
        "total_latency_ms": round(statistics.median(latencies), 1),
        "total_latency_ms_min": round(latencies[0], 1),
        "total_latency_ms_max": round(latencies[-1], 1),
        "output_tokens_mean": round(statistics.mean(tokens), 1) if tokens else None,
        "finish_reason": samples[-1].finish_reason,
        "stopped_early": samples[-1].finish_reason == "length",
        "repeats": len(samples),
        "sample": samples[-1].text.strip()[:400],
    }


# -------------------------------------------------------------------------- needle


def _run_needle(
    config: CtxBenchConfig,
    probe: EndpointProbe,
    corpus: str,
    size: int,
    result: SizeResult,
) -> None:
    """Retrieval of a definition planted at a known depth.

    Trials are spread across depths rather than sampled uniformly at random, so every
    context size probes the same set of positions and the pass rate is comparable between
    rows. Per-depth results are kept because a model that reads the head of a long context
    and gives up has a pass rate near 1.0 at 16k and near 0.0 at 128k, which an average
    alone would report as a smooth decline.
    """
    trials = config.suites.needle_trials
    depths = [index / max(1, trials - 1) for index in range(trials)] if trials > 1 else [0.5]
    # Evenly spread but never exactly 0.0 or 1.0: a needle in the very first or very last
    # token position is a best case for every retrieval strategy and would flatter the
    # large sizes.
    depths = [0.05 + 0.9 * depth for depth in depths]

    rng = random.Random(config.seed + size)
    passed = 0
    attempts = 0
    #: Which trials produced an actual reply. The pass rate is over these; the rest are
    #: reported separately rather than folded in as failures.
    observed: list[bool] = []
    by_depth: list[dict[str, Any]] = []
    for trial, depth in enumerate(depths):
        # Alternate literal and structural needles so a run exercises both matchers.
        needle = (
            signature_needle(rng, module="", depth=depth)
            if trial % 2
            else make_needle(rng, module="", depth=depth)
        )
        placement = plant(corpus, needle)
        located = replace(needle, module=placement.module)
        messages = [
            {
                "role": "user",
                "content": (
                    "The following is a Python codebase.\n\n"
                    f"{placement.text}\n\n{located.question}"
                ),
            }
        ]
        retried = False
        try:
            reply = probe.stream(
                messages, max_tokens=config.suites.needle_max_tokens
            )
            if reply.answer_unobservable and (
                config.suites.needle_retry_max_tokens
                > config.suites.needle_max_tokens
            ):
                # The model spent the whole budget thinking and never answered, so this
                # trial has not tested retrieval yet. Reasoning effort grows with the
                # prompt, so the budget is raised once rather than writing the trial off
                # -- otherwise the suite measures fewer trials as context grows and the
                # decline looks like a retrieval regression.
                retried = True
                reply = probe.stream(
                    messages, max_tokens=config.suites.needle_retry_max_tokens
                )
        except ProbeError as exc:
            result.errors.append(f"needle@{placement.module}: {exc}")
            by_depth.append(
                {
                    "depth": round(placement.actual_depth, 3),
                    "requested_depth": round(depth, 3),
                    "kind": located.kind,
                    "module": placement.module,
                    "passed": False,
                    "error": str(exc),
                }
            )
            attempts += 1
            observed.append(True)
            continue
        hit = score(located, reply.text)
        # A model that spent its whole answer budget reasoning never reached a reply, and
        # scoring that as a retrieval failure would report the token budget as a model
        # behaviour. Such a trial is recorded but kept out of the pass-rate denominator.
        unobservable = reply.answer_unobservable
        passed += bool(hit)
        attempts += 1
        observed.append(not unobservable)
        by_depth.append(
            {
                "depth": round(placement.actual_depth, 3),
                "requested_depth": round(depth, 3),
                "kind": located.kind,
                "module": placement.module,
                "passed": None if unobservable else bool(hit),
                "unobservable": unobservable,
                "retried_with_larger_budget": retried,
                "reasoning_tokens": reply.reasoning_tokens,
                "expected": located.answer,
                "got": extract_answer(reply.text),
            }
        )
        # Three outcomes, not two. Printing "FAIL" for a trial the model never answered
        # would read as a retrieval failure when it was a budget, which is exactly the
        # misreading the `unobservable` distinction exists to prevent.
        logger.info(
            "  needle depth %.2f %s %s%s",
            placement.actual_depth,
            "NO-ANSWER" if unobservable else ("PASS" if hit else "FAIL"),
            placement.module,
            " (retried)" if retried else "",
        )

    scored = sum(observed)
    unobservable = len(observed) - scored
    result.correctness["needle"] = {
        # Over trials that produced a reply. A trial where the model reasoned through its
        # whole budget never tested retrieval, so including it would measure the budget.
        "pass_rate": round(passed / scored, 4) if scored else None,
        "passed": passed,
        "trials": scored,
        "trials_attempted": attempts,
        "unobservable": unobservable,
        "context_size_tokens": size,
        "by_depth": by_depth,
    }
    result.correctness["needle_in_haystack_pass_rate"] = (
        round(passed / scored, 4) if scored else None
    )


# ----------------------------------------------------------------------- execution


def execution_config(config: CtxBenchConfig, size: int, output_dir: Path) -> RunConfig:
    """The benchmark run for one context size, with the context budget pinned to ``size``.

    Separate from `_run_execution` so the translation is testable without materializing a
    dataset. `max_context_tokens` is what the harness is allowed to carry, so this measures
    the harness under a context constraint and not only the model.
    """
    suites = config.suites
    cell_dir = output_dir / "execution" / f"ctx{size}"
    return RunConfig.from_dict(
        {
            "dataset": suites.dataset,
            "output_dir": str(cell_dir),
            "num_tasks": suites.execution_num_tasks,
            "seeds_per_task": suites.execution_seeds,
            "seed": config.seed,
            "max_steps": suites.execution_max_steps,
            "max_total_time_sec": suites.execution_max_time_sec,
            "max_context_tokens": size,
            "model": {
                "name": config.model.name,
                "base_url": config.model.base_url,
                "api_key_env": config.model.api_key_env,
                "api_key": config.model.api_key,
                "temperature": config.model.temperature,
                "tokenizer": config.suites.tokenizer,
                "extra_body": config.model.extra_body,
            },
            "harness": {"name": config.harness},
            "runtime": {
                "kind": config.runtime.kind,
                "root": str(config.runtime.root),
                "keep_workspaces": config.runtime.keep_workspaces,
                "options": config.runtime.options,
            },
            "cache_dir": str(suites.cache_dir),
        }
    )


def _run_execution(
    config: CtxBenchConfig,
    size: int,
    output_dir: Path,
    result: SizeResult,
) -> None:
    """pass@1 from real benchmark tasks at one context size.

    Delegated to the existing runner so grading is the benchmark's own verifier rather
    than a reimplementation of it.
    """
    try:
        run_config = execution_config(config, size, output_dir)
    except Exception as exc:  # noqa: BLE001 - a bad cell config is ours, not the model's
        logger.warning("Execution suite config invalid at %s: %s", size, exc)
        result.errors.append(f"execution config: {type(exc).__name__}: {exc}")
        result.correctness["code_execution_pass_at_1"] = None
        return
    try:
        summary = run_evaluation(run_config)
    except Exception as exc:  # noqa: BLE001 - a cell failure must not lose the other suites
        logger.warning("Execution suite failed at %s: %s", size, exc)
        result.errors.append(f"execution: {type(exc).__name__}: {exc}")
        result.correctness["code_execution_pass_at_1"] = None
        return

    # `resolve_rate_graded` excludes pairs the host could not provision or grade, so a
    # broken sandbox is not reported as a failed attempt.
    result.correctness["code_execution_pass_at_1"] = summary.get("resolve_rate_graded")
    result.correctness["execution"] = {
        "dataset": summary.get("dataset"),
        "harness": summary.get("harness"),
        "tasks": summary.get("tasks_selected"),
        "jobs_total": summary.get("jobs_total"),
        "resolved": summary.get("resolved"),
        "resolve_rate": summary.get("resolve_rate"),
        "resolve_rate_graded": summary.get("resolve_rate_graded"),
        "infrastructure_failed": summary.get("jobs_infrastructure_failed"),
        "infrastructure_reasons": summary.get("infrastructure_reasons"),
        "max_context_tokens": size,
        "exit_reasons": summary.get("exit_reasons"),
    }


# -------------------------------------------------------------------------- memory


def _run_memory(
    config: CtxBenchConfig,
    probe: EndpointProbe,
    encoding: Any,
    *,
    probe_stats: dict[str, int],
) -> dict[str, Any]:
    """Repeat the largest request and watch RSS.

    Run once for the run rather than per context size: the question is whether repeated
    large-context handling leaks, not whether 128k specifically leaks more than 64k. The
    iteration count is reported next to the growth, since a growth figure from three
    samples is not a trend.
    """
    size = config.suites.memory_context_tokens or max(config.context_sizes)
    corpus = corpus_for(config, encoding, size)
    trace = MemoryTrace()
    latencies: list[float] = []
    timeouts = 0
    for index in range(config.suites.memory_iterations):
        trace.sample()
        trace.iterations += 1
        try:
            sample = probe.stream(
                [{"role": "user", "content": PREFILL_INSTRUCTION + corpus.text}],
                max_tokens=config.suites.prefill_max_tokens,
            )
            latencies.append(sample.total_sec * 1000)
        except ProbeError as exc:
            if exc.context_overflow:
                # The requested memory size is larger than the endpoint can hold, so there
                # is no repeated large request to leak over. Recorded as the limit it is,
                # rather than as a trace of failures.
                logger.info(
                    "Memory suite skipped: endpoint context limit %s < requested %s",
                    exc.server_context_limit,
                    size,
                )
                return {
                    "context_size_tokens": size,
                    "iterations": 0,
                    "peak_mb": None,
                    "growth_mb": None,
                    "monotonic_growth": False,
                    "samples": len(trace.samples_mb),
                    "samples_mb": trace.samples_mb,
                    "skipped": True,
                    "server_context_limit": exc.server_context_limit,
                    "note": (
                        "the requested size exceeds the endpoint's context window, so no "
                        "leak trace could be taken at this size"
                    ),
                }
            timeouts += 1
            logger.warning("Memory iteration %s failed: %s", index + 1, exc)
            break
    trace.sample()

    payload = trace.as_dict()
    payload.update(
        {
            "context_size_tokens": size,
            "latency_ms_median": (
                round(statistics.median(latencies), 1) if latencies else None
            ),
            "latency_ms_first": round(latencies[0], 1) if latencies else None,
            "latency_ms_last": round(latencies[-1], 1) if latencies else None,
            "timeouts": timeouts,
            "endpoint_stats": dict(probe_stats),
            "note": (
                "ru_maxrss is a high-water mark for this process tree: it covers the "
                "runner and any child it reaps, so a CLI harness that runs as a "
                "subprocess is included but a daemon it spawned and left running is not"
            ),
        }
    )
    return payload


# ------------------------------------------------------------------------- helpers


def _run_id(wall_start: float) -> str:
    """A sortable, human-readable id: `ctx_<UTC timestamp>`.

    Seconds rather than microseconds so two runs started in the same minute are still
    distinguishable in a directory listing.
    """
    return "ctx_" + datetime.fromtimestamp(wall_start, UTC).strftime("%Y%m%d_%H%M%S")


def _harness_version() -> str:
    try:
        from importlib.metadata import version

        return f"cae {version('coding-agent-evaluations')}"
    except Exception:  # noqa: BLE001 - version is metadata, never worth failing a run
        return "cae (unknown version)"


__all__ = ["REPORT_VERSION", "SizeResult", "run_ctxbench"]