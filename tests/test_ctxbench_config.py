"""Tests for ctxbench config loading and the invariants the report depends on."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coding_agent_evaluations.config import ConfigError
from coding_agent_evaluations.ctxbench.config import (
    DEFAULT_CONTEXT_SIZES,
    SUITES,
    CtxBenchConfig,
    load_ctx_config,
)

MODEL = {"name": "m", "base_url": "http://h/v1"}


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_are_the_standard_ladder() -> None:
    config = CtxBenchConfig.from_dict({"model": MODEL})
    assert config.context_sizes == DEFAULT_CONTEXT_SIZES
    assert config.context_sizes == (16_000, 32_000, 64_000, 128_000)
    assert config.suites.enabled == SUITES


def test_ladder_is_sorted_and_deduplicated() -> None:
    """A row labelled N must be a measurement of N. An unsorted ladder would print an
    x-axis that is not the context size, and a duplicated size would produce two rows
    that look like separate measurements of the same thing."""
    config = CtxBenchConfig.from_dict(
        {"model": MODEL, "context_sizes": [64000, 16000, 64000, 32000]}
    )
    assert config.context_sizes == (16_000, 32_000, 64_000)


def test_empty_ladder_is_rejected() -> None:
    with pytest.raises(ConfigError, match="context_sizes"):
        CtxBenchConfig.from_dict({"model": MODEL, "context_sizes": []})


def test_non_positive_size_is_rejected() -> None:
    with pytest.raises(ConfigError, match="positive"):
        CtxBenchConfig.from_dict({"model": MODEL, "context_sizes": [0]})


def test_model_is_required() -> None:
    with pytest.raises(ConfigError):
        CtxBenchConfig.from_dict({"context_sizes": [8000]})


def test_unknown_suite_is_rejected() -> None:
    """A typo'd suite name would otherwise run nothing and report an empty column as if
    it had been measured."""
    with pytest.raises(ConfigError, match="unknown suite"):
        CtxBenchConfig.from_dict(
            {"model": MODEL, "ctxbench": {"enabled": ["prefll"]}}
        )


def test_unknown_harness_is_rejected() -> None:
    with pytest.raises(ConfigError, match="harness"):
        CtxBenchConfig.from_dict({"model": MODEL, "harness": "cline"})


def test_harness_accepts_the_run_config_shape() -> None:
    """base.yaml writes `harness:` as a mapping; ctxbench reads it either way so a
    ctxbench config can extend base.yaml unchanged."""
    assert CtxBenchConfig.from_dict(
        {"model": MODEL, "harness": {"name": "opencode"}}
    ).harness == "opencode"


@pytest.mark.parametrize("key", ("repeats", "memory_iterations", "needle_trials"))
def test_non_positive_limits_are_rejected(key: str) -> None:
    with pytest.raises(ConfigError, match=key):
        CtxBenchConfig.from_dict({"model": MODEL, "ctxbench": {key: 0}})


def test_runs_flags_reflect_enabled_suites() -> None:
    config = CtxBenchConfig.from_dict(
        {"model": MODEL, "ctxbench": {"enabled": ["prefill", "needle"]}}
    )
    runs = config.suites.runs
    assert runs["prefill"] and runs["needle"]
    assert not runs["execution"] and not runs["memory"]


def test_extends_and_env_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CTX_TEST_MODEL", "served-model")
    write(
        tmp_path / "base.yaml",
        "model:\n  name: ${CTX_TEST_MODEL}\n  base_url: http://h/v1\nseed: 7\n",
    )
    child = write(
        tmp_path / "child.yaml",
        "extends: [./base.yaml]\ncontext_sizes: [8000]\n",
    )
    config = load_ctx_config(child)
    assert config.model.name == "served-model"
    assert config.seed == 7
    assert config.context_sizes == (8000,)
    assert config.source == str(child)


def test_overrides_apply(tmp_path: Path) -> None:
    path = write(tmp_path / "c.yaml", "model:\n  name: m\n  base_url: u\n")
    config = load_ctx_config(
        path,
        ["context_sizes=[8000]", "harness=pi", "ctxbench.repeats=5"],
    )
    assert config.context_sizes == (8000,)
    assert config.harness == "pi"
    assert config.suites.repeats == 5


def test_override_rejects_a_bare_value(tmp_path: Path) -> None:
    path = write(tmp_path / "c.yaml", "model:\n  name: m\n  base_url: u\n")
    with pytest.raises(ConfigError, match="key.path=value"):
        load_ctx_config(path, ["nonsense"])


def test_as_dict_round_trips(tmp_path: Path) -> None:
    """The resolved config is written into the report, so it has to survive being read
    back -- otherwise the record of what ran cannot be replayed."""
    config = CtxBenchConfig.from_dict(
        {"model": MODEL, "context_sizes": [8000, 16000], "ctxbench": {"repeats": 2}}
    )
    again = CtxBenchConfig.from_dict(json.loads(json.dumps(config.as_dict())))
    assert again.context_sizes == config.context_sizes
    assert again.suites.repeats == config.suites.repeats
    assert again.model.name == config.model.name


def test_shipped_config_loads() -> None:
    """The config in the repo must load, not just the ones written in tests."""
    path = Path(__file__).resolve().parents[1] / "configs" / "ctxbench.yaml"
    config = load_ctx_config(path)
    assert config.context_sizes == DEFAULT_CONTEXT_SIZES
    assert set(config.suites.enabled) == set(SUITES)
    assert config.model.name