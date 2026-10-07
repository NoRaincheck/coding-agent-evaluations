"""Configuration for a context-scaling run.

Extends the existing config vocabulary rather than replacing it. `ctxbench` reuses
`ModelConfig`, `RuntimeConfig` and the dataset loader, so a context-scaling run is
described in the same terms as a benchmark run and the two reports are comparable: same
`model:`, same `runtime:`, same `${ENV}` expansion.

Only the genuinely new knobs live here:

* `context_sizes` — the ladder to sweep, in tokens.
* `suites` — which of prefill / generation / needle / execution / memory to run.
* per-suite limits, so a smoke run is one flag rather than a second config file.

The token ladder is validated to be strictly increasing. An unsorted or duplicated ladder
produces a curve whose x-axis is not the context size, and a curve like that is
indistinguishable from a real finding until someone checks the config by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from ..config import ConfigError, ModelConfig, RuntimeConfig, load_payload

SUITES = ("prefill", "generation", "needle", "execution", "memory")

#: The ladder the report is built around. Ordered smallest first so the printed table and
#: the JSON `results` array both read left to right as "context grows, things degrade".
DEFAULT_CONTEXT_SIZES: tuple[int, ...] = (16_000, 32_000, 64_000, 128_000)


@dataclass(frozen=True)
class CtxSuiteConfig:
    """Limits for the suites that do not need a workspace."""

    enabled: tuple[str, ...] = SUITES
    # Repeats per measurement. Speed is noisy enough that a single sample is not a number,
    # it is an anecdote; the median of a few is.
    repeats: int = 3
    # Output tokens for the prefill probe. One word is the whole point: prefill cost must
    # not be mixed with decode cost -- and TTFT is unaffected by this number, so it is room
    # for the sanity check rather than for generation. Generous enough that a reasoning
    # model can think and still answer; too small and the probe measures the token budget.
    prefill_max_tokens: int = 64
    generation_max_tokens: int = 512
    # Needle probes per context size, spread across depths.
    needle_trials: int = 6
    # The answer is a value or a signature, so this is only a ceiling. Sized for the same
    # reason as above: a reasoning model that cannot finish thinking inside the budget
    # produces no reply, and the trial is then recorded as unobservable rather than as a
    # retrieval failure.
    needle_max_tokens: int = 512
    # Retry budget for a trial that produced no reply. Reasoning effort grows with the
    # prompt, so a budget that is ample at 16k can be exhausted at 64k; without the retry
    # the suite silently measures fewer trials as the context grows, which reads as a
    # retrieval regression when it is a budget.
    #
    # Generous on purpose, and expensive: a rescued trial costs one full-budget request,
    # which on a reasoning model is minutes rather than seconds. Lower it if a sweep is
    # running long, accepting that some trials become unobservable and leave the
    # pass-rate denominator.
    needle_retry_max_tokens: int = 4096
    # Execution suite delegates to the existing runner, so it borrows its knobs.
    execution_num_tasks: int | None = 2
    execution_seeds: int = 1
    execution_max_steps: int = 50
    execution_max_time_sec: int = 1800
    # Memory suite: how many repeats of the largest request to watch RSS across.
    memory_iterations: int = 10
    memory_context_tokens: int | None = None
    # Where filler corpora are cached, so a resumed run does not regenerate 128k tokens.
    corpus_cache: Path = Path("~/.cache/cae/ctxbench")
    tokenizer: str = "o200k_base"
    request_timeout_sec: int = 900
    retries: int = 1
    dataset: str = "swebench_verified"
    cache_dir: Path = Path("~/.cache/cae/datasets")

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> CtxSuiteConfig:
        value = dict(value or {})
        enabled = value.get("enabled", SUITES)
        if isinstance(enabled, str):
            enabled = (enabled,)
        enabled = tuple(str(item) for item in enabled)
        unknown = set(enabled) - set(SUITES)
        if unknown:
            raise ConfigError(f"unknown suite(s): {sorted(unknown)}; available: {SUITES}")
        repeats = int(value.get("repeats", 3))
        if repeats <= 0:
            raise ConfigError("ctxbench.repeats must be positive")
        memory_iterations = int(value.get("memory_iterations", 10))
        if memory_iterations <= 0:
            raise ConfigError("ctxbench.memory_iterations must be positive")
        needle_trials = int(value.get("needle_trials", 6))
        if needle_trials <= 0:
            raise ConfigError("ctxbench.needle_trials must be positive")
        sizes = value.get("execution_num_tasks", 2)
        return cls(
            enabled=enabled,
            repeats=repeats,
            prefill_max_tokens=int(value.get("prefill_max_tokens", 64)),
            generation_max_tokens=int(value.get("generation_max_tokens", 512)),
            needle_trials=needle_trials,
            needle_max_tokens=int(value.get("needle_max_tokens", 512)),
            needle_retry_max_tokens=int(value.get("needle_retry_max_tokens", 4096)),
            execution_num_tasks=None if sizes in (None, 0, "0") else int(sizes),
            execution_seeds=int(value.get("execution_seeds", 1)),
            execution_max_steps=int(value.get("execution_max_steps", 50)),
            execution_max_time_sec=int(value.get("execution_max_time_sec", 1800)),
            memory_iterations=memory_iterations,
            memory_context_tokens=(
                None
                if value.get("memory_context_tokens") in (None, 0, "0")
                else int(value["memory_context_tokens"])
            ),
            corpus_cache=Path(
                str(value.get("corpus_cache", "~/.cache/cae/ctxbench"))
            ).expanduser(),
            tokenizer=str(value.get("tokenizer") or "o200k_base"),
            request_timeout_sec=int(value.get("request_timeout_sec", 900)),
            retries=int(value.get("retries", 1)),
            dataset=str(value.get("dataset") or "swebench_verified"),
            cache_dir=Path(str(value.get("cache_dir", "~/.cache/cae/datasets"))).expanduser(),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": list(self.enabled),
            "repeats": self.repeats,
            "prefill_max_tokens": self.prefill_max_tokens,
            "generation_max_tokens": self.generation_max_tokens,
            "needle_trials": self.needle_trials,
            "needle_max_tokens": self.needle_max_tokens,
            "needle_retry_max_tokens": self.needle_retry_max_tokens,
            "execution_num_tasks": self.execution_num_tasks,
            "execution_seeds": self.execution_seeds,
            "execution_max_steps": self.execution_max_steps,
            "execution_max_time_sec": self.execution_max_time_sec,
            "memory_iterations": self.memory_iterations,
            "memory_context_tokens": self.memory_context_tokens,
            "corpus_cache": str(self.corpus_cache),
            "tokenizer": self.tokenizer,
            "request_timeout_sec": self.request_timeout_sec,
            "retries": self.retries,
            "dataset": self.dataset,
            "cache_dir": str(self.cache_dir),
        }

    @property
    def runs(self) -> dict[str, bool]:
        return {name: name in self.enabled for name in SUITES}


@dataclass(frozen=True)
class CtxBenchConfig:
    """A complete context-scaling run."""

    model: ModelConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    suites: CtxSuiteConfig = field(default_factory=CtxSuiteConfig)
    output_dir: Path = Path("eval-results/ctxbench")
    context_sizes: tuple[int, ...] = DEFAULT_CONTEXT_SIZES
    # Which agent harness the execution suite drives. The speed and needle suites talk to
    # the endpoint directly, so they measure the endpoint; this is what carries through to
    # the pass@1 column.
    harness: str = "leaf"
    seed: int = 42
    source: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CtxBenchConfig:
        model = value.get("model")
        if not isinstance(model, dict):
            raise ConfigError("model is required")
        sizes = value.get("context_sizes", list(DEFAULT_CONTEXT_SIZES))
        if not isinstance(sizes, (list, tuple)) or not sizes:
            raise ConfigError("ctxbench.context_sizes must be a non-empty list")
        parsed = tuple(sorted({int(size) for size in sizes}))
        if any(size <= 0 for size in parsed):
            raise ConfigError("context_sizes must be positive")
        # `harness:` is a name here but a mapping in base.yaml, so both shapes are read.
        harness = value.get("harness")
        if isinstance(harness, dict):
            harness = harness.get("name")
        config = cls(
            model=ModelConfig.from_dict(dict(model)),
            runtime=RuntimeConfig.from_dict(value.get("runtime")),
            suites=CtxSuiteConfig.from_dict(value.get("ctxbench")),
            output_dir=Path(str(value.get("output_dir", "eval-results/ctxbench"))),
            context_sizes=parsed,
            harness=str(harness or "leaf"),
            seed=int(value.get("seed", 42)),
        )
        if config.harness not in {"leaf", "opencode", "pi"}:
            raise ConfigError(
                f"unknown harness {config.harness!r}; available: leaf, opencode, pi"
            )
        return config

    def as_dict(self) -> dict[str, Any]:
        runtime = {
            "kind": self.runtime.kind,
            "root": str(self.runtime.root),
            "keep_workspaces": self.runtime.keep_workspaces,
            "options": dict(self.runtime.options),
        }
        from dataclasses import asdict

        return {
            "output_dir": str(self.output_dir),
            "context_sizes": list(self.context_sizes),
            "harness": self.harness,
            "seed": self.seed,
            "model": asdict(self.model),
            "runtime": runtime,
            "ctxbench": self.suites.as_dict(),
        }

    def label(self) -> str:
        return f"ctxbench/{self.harness}/{self.model.name}"


def load_ctx_config(path: str | Path, overrides: list[str] | None = None) -> CtxBenchConfig:
    """Load a ctxbench config, then apply ``--set key.path=value`` overrides.

    Reuses the shared loader for `extends:` chains and `${ENV}` expansion, so a ctxbench
    config can extend the same `configs/base.yaml` a benchmark run uses and inherit the
    same `model:` block.
    """
    payload = load_payload(path)
    config = CtxBenchConfig.from_dict(payload)
    if overrides:
        config = apply_ctx_overrides(config, overrides)
    if config.source is None:
        # Recorded so a report can name the config that produced it.
        config = replace(config, source=str(Path(path).expanduser()))
    return config


def apply_ctx_overrides(config: CtxBenchConfig, overrides: list[str]) -> CtxBenchConfig:
    """Apply overrides, reusing the shared `--set` semantics."""
    payload = config.as_dict()
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override must look like key.path=value: {item!r}")
        key, raw = item.split("=", 1)
        cursor: dict[str, Any] = payload
        parts = key.strip().split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ConfigError(f"cannot descend into {key!r}")
        cursor[parts[-1]] = yaml.safe_load(raw)
    return CtxBenchConfig.from_dict(payload)


__all__ = [
    "DEFAULT_CONTEXT_SIZES",
    "SUITES",
    "CtxBenchConfig",
    "CtxSuiteConfig",
    "apply_ctx_overrides",
    "load_ctx_config",
]