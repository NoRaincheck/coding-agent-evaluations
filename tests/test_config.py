from __future__ import annotations

import json
from pathlib import Path

import pytest

from coding_agent_evaluations.config import (
    ConfigError,
    ModelConfig,
    apply_overrides,
    expand_env,
    load_config,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_expand_env_uses_defaults_and_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAE_TEST_MODEL", "frognano-4b-2609")
    assert expand_env("${CAE_TEST_MODEL}") == "frognano-4b-2609"
    assert expand_env("${CAE_MISSING:-fallback}") == "fallback"
    assert expand_env("${CAE_MISSING:-}") == ""
    monkeypatch.delenv("CAE_TEST_MODEL")
    assert expand_env({"a": ["${CAE_TEST_MODEL:-none}"]}) == {"a": ["none"]}


def test_model_config_requires_name_and_base_url() -> None:
    with pytest.raises(ConfigError):
        ModelConfig.from_dict({"base_url": "http://localhost:1234/v1"})
    with pytest.raises(ConfigError):
        ModelConfig.from_dict({"name": "m"})
    model = ModelConfig.from_dict({"name": "m", "base_url": "http://h/v1/"})
    assert model.base_url == "http://h/v1"
    assert model.max_tokens_per_turn == 8192
    assert ModelConfig.from_dict(
        {"name": "m", "base_url": "u", "max_tokens_per_turn": 0}
    ).max_tokens_per_turn is None


def test_model_api_key_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_KEY", "from-env")
    model = ModelConfig(name="m", base_url="u", api_key_env="SOME_KEY")
    assert model.resolve_api_key() == "from-env"
    assert model.resolve_api_key.__self__ is model
    monkeypatch.delenv("SOME_KEY")
    # An unauthenticated local endpoint still needs a non-empty placeholder.
    assert model.resolve_api_key() == "lmstudio"
    explicit = ModelConfig(name="m", base_url="u", api_key="literal")
    assert explicit.resolve_api_key() == "literal"


def test_load_config_applies_extends_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAE_TEST_DATASET", "terminal_bench_2_verified")
    write(
        tmp_path / "base.yaml",
        "dataset: swebench_verified\nmax_steps: 150\nmodel:\n  name: base-model\n"
        "  base_url: http://base/v1\n",
    )
    write(
        tmp_path / "child.yaml",
        "extends: base.yaml\ndataset: ${CAE_TEST_DATASET}\n"
        "model:\n  name: child-model\n",
    )
    config = load_config(tmp_path / "child.yaml")
    assert config.dataset == "terminal_bench_2_verified"
    assert config.model.name == "child-model"
    # Sibling keys survive the merge.
    assert config.model.base_url == "http://base/v1"
    assert config.max_steps == 150
    assert config.source is not None


def test_extends_accepts_a_list_and_last_parent_wins(tmp_path: Path) -> None:
    write(tmp_path / "a.yaml", "max_steps: 1\nseeds_per_task: 1\n")
    write(tmp_path / "b.yaml", "max_steps: 2\n")
    write(tmp_path / "c.yaml", "extends: [a.yaml, b.yaml]\ndataset: swebench_verified\n"
          "model:\n  name: m\n  base_url: u\n")
    assert load_config(tmp_path / "c.yaml").max_steps == 2


def test_extends_cycle_is_reported(tmp_path: Path) -> None:
    write(tmp_path / "a.yaml", "extends: b.yaml\n")
    write(tmp_path / "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ConfigError):
        load_config(tmp_path / "a.yaml")


def test_unknown_harness_and_runtime_are_rejected() -> None:
    from coding_agent_evaluations.config import HarnessConfig, RuntimeConfig

    with pytest.raises(ConfigError):
        HarnessConfig.from_dict({"name": "aider"})
    with pytest.raises(ConfigError):
        RuntimeConfig.from_dict({"kind": "kubernetes"})


def test_run_config_validation() -> None:
    from coding_agent_evaluations.config import RunConfig

    base = {"dataset": "swebench_verified", "model": {"name": "m", "base_url": "u"}}
    for override in ({"max_steps": 0}, {"seeds_per_task": -1}, {"max_total_time_sec": 5}):
        with pytest.raises(ConfigError):
            RunConfig.from_dict({**base, **override})


def test_apply_overrides_supports_nested_paths(tmp_path: Path) -> None:
    write(
        tmp_path / "c.yaml",
        "dataset: swebench_verified\nmodel:\n  name: m\n  base_url: u\n",
    )
    config = load_config(tmp_path / "c.yaml")
    updated = apply_overrides(
        config,
        ["num_tasks=3", "harness.name=pi", "model.temperature=0.1", "task_ids=[\"a\"]"],
    )
    assert updated.num_tasks == 3
    assert updated.harness.name == "pi"
    assert updated.model.temperature == 0.1
    assert updated.task_ids == ("a",)
    with pytest.raises(ConfigError):
        apply_overrides(config, ["nonsense"])


@pytest.mark.parametrize(
    "name",
    [
        "base.yaml",
        "dataset/swebench-verified.yaml",
        "dataset/terminal-bench-2-verified.yaml",
        "harness/leaf.yaml",
        "harness/opencode.yaml",
        "harness/pi.yaml",
        "profile/smoke.yaml",
    ],
)
def test_shipped_configs_resolve(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_MODEL_NAME", "frognano-4b-2609")
    monkeypatch.setenv("EVAL_MODEL_BASE_URL", "http://127.0.0.1:1234/v1")
    config = load_config(CONFIG_DIR / name)
    assert config.model.name == "frognano-4b-2609"
    assert config.model.base_url == "http://127.0.0.1:1234/v1"
    assert config.runtime.kind == "host"
    assert config.harness.name in {"leaf", "opencode", "pi"}
    # Every config must serialize cleanly for config.json.
    json.dumps(config.as_dict())


def test_paper_protocol_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_MODEL_NAME", "m")
    monkeypatch.setenv("EVAL_MODEL_BASE_URL", "u")
    for dataset, expected_tokens in (
        ("swebench-verified", 8192),
        ("terminal-bench-2-verified", 32000),
    ):
        config = load_config(CONFIG_DIR / "dataset" / f"{dataset}.yaml")
        assert config.seeds_per_task == 3
        assert config.max_steps == 150
        assert config.max_context_tokens == 131072
        assert config.max_total_time_sec == 10800
        assert config.model.max_tokens_per_turn == expected_tokens


def test_smoke_profile_is_smaller_than_the_paper_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_MODEL_NAME", "m")
    monkeypatch.setenv("EVAL_MODEL_BASE_URL", "u")
    paper = load_config(CONFIG_DIR / "dataset" / "swebench-verified.yaml")
    smoke = load_config(CONFIG_DIR / "profile" / "smoke.yaml")
    assert (smoke.num_tasks or 0) < (paper.num_tasks or 500)
    assert smoke.seeds_per_task < paper.seeds_per_task
    assert smoke.max_steps < paper.max_steps
    assert smoke.max_total_time_sec < paper.max_total_time_sec


def test_as_dict_is_json_serializable_and_typed() -> None:
    payload = ModelConfig(name="m", base_url="u").as_env()
    assert payload["EVAL_MODEL_NAME"] == "m"
    assert payload["EVAL_MODEL_BASE_URL"] == "u"
    assert payload["EVAL_MODEL_API_KEY"]