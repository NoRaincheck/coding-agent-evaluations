from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from coding_agent_evaluations.harnesses import build_harness
from coding_agent_evaluations.harnesses.base import HarnessRun, empty_trajectory
from coding_agent_evaluations.harnesses.cli import hermetic_env, parse_jsonl
from coding_agent_evaluations.harnesses.opencode import _opencode_config
from coding_agent_evaluations.harnesses.pi import _models_config, _text_of
from coding_agent_evaluations.report import as_table, caveats, load_summaries


class FakeRuntime:
    def __init__(self, state: Path, instruction: str = "fix the bug") -> None:
        self.paths = type("Paths", (), {"state": state, "workspace": state})()
        self._instruction = instruction

    def get_task_instruction(self) -> str:
        return self._instruction


def make_run(tmp_path: Path, **overrides: Any) -> HarnessRun:
    from coding_agent_evaluations.config import HarnessConfig, ModelConfig

    values: dict[str, Any] = {
        "runtime": FakeRuntime(tmp_path),
        "task": {"instance_id": "demo", "dataset": "swebench_verified"},
        "model": ModelConfig(name="m", base_url="http://127.0.0.1:1234/v1"),
        "config": HarnessConfig(name="leaf"),
        "instance_id": "demo",
        "seed": 0,
        "max_steps": 10,
        "max_context_tokens": 32768,
        "max_total_time_sec": 600,
    }
    values.update(overrides)
    return HarnessRun(**values)


def test_build_harness_covers_the_three_targets() -> None:
    assert {build_harness(name).name for name in ("leaf", "opencode", "pi")} == {
        "leaf",
        "opencode",
        "pi",
    }
    with pytest.raises(ValueError):
        build_harness("aider")


def test_harnesses_declare_which_limits_they_enforce() -> None:
    leaf = build_harness("leaf")
    opencode = build_harness("opencode")
    pi = build_harness("pi")
    assert leaf.enforces_max_steps is True
    # The CLI harnesses own their loops, so a step limit would be a lie.
    assert opencode.enforces_max_steps is False
    assert pi.enforces_max_steps is False
    assert "temperature" in pi.ignored_model_params


def test_empty_trajectory_has_the_common_shape(tmp_path: Path) -> None:
    trajectory = empty_trajectory(make_run(tmp_path), reason="harness_unavailable")
    for key in (
        "trajectory_format",
        "instance_id",
        "seed",
        "steps",
        "n_steps",
        "exit_reason",
        "context_tokens",
        "output_patch",
    ):
        assert key in trajectory
    assert trajectory["exit_reason"] == "harness_unavailable"


def test_parse_jsonl_ignores_noise() -> None:
    text = '{"type":"a"}\nnot json\n\n{"type":"b","x":1}\n[1,2]\n'
    assert [event["type"] for event in parse_jsonl(text)] == ["a", "b"]


def test_hermetic_env_isolates_state_and_stays_offline(tmp_path: Path) -> None:
    environment = hermetic_env({"PATH": "/usr/bin", "KEEP": "1"}, tmp_path)
    assert environment["KEEP"] == "1"
    assert environment["XDG_CONFIG_HOME"].startswith(str(tmp_path))
    assert environment["HOME"].startswith(str(tmp_path))
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["PI_OFFLINE"] == "1"
    assert Path(environment["HOME"]).is_dir()


def test_opencode_config_points_at_the_endpoint(tmp_path: Path) -> None:
    config = _opencode_config(make_run(tmp_path))
    provider = config["provider"]["cae-eval"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == "http://127.0.0.1:1234/v1"
    assert provider["options"]["apiKey"]
    assert "m" in provider["models"]
    # A custom model declaring a `limit` block is rejected by opencode.
    assert "limit" not in provider["models"]["m"]
    assert config["permission"]["bash"] == "allow"
    assert config["autoupdate"] is False
    # The config must stay JSON-serializable.
    json.dumps(config)


def test_pi_models_config_declares_the_endpoint(tmp_path: Path) -> None:
    provider = _models_config(make_run(tmp_path))["providers"]["cae-eval"]
    assert provider["baseUrl"] == "http://127.0.0.1:1234/v1"
    assert provider["api"] == "openai-completions"
    model = provider["models"][0]
    assert model["id"] == "m"
    assert model["contextWindow"] == 32768
    assert model["maxTokens"] == 8192


def test_text_of_handles_pi_content_shapes() -> None:
    assert _text_of([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
    assert _text_of("plain") == "plain"
    assert _text_of(None) == ""
    assert _text_of({"content": [{"text": "c"}]}) == "c"


def write_summary(root: Path, harness: str, **overrides: Any) -> Path:
    payload: dict[str, Any] = {
        "dataset": "swebench_verified",
        "harness": harness,
        "model": "m",
        "tasks_selected": 4,
        "jobs_total": 8,
        "jobs_completed": 8,
        "jobs_infrastructure_failed": 0,
        "resolved": 2,
        "resolve_rate": 0.25,
        "resolve_rate_graded": 0.25,
        "mean_steps": 12.5,
        "mean_elapsed_sec": 300.0,
        "duration_sec": 600.0,
        "limits": {
            "max_steps": 150,
            "max_steps_enforced": harness == "leaf",
            "model_params_ignored": [] if harness == "leaf" else ["temperature"],
        },
    }
    payload.update(overrides)
    directory = root / harness
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory / "summary.json"


def test_compare_tabulates_and_flags_differences(tmp_path: Path) -> None:
    write_summary(tmp_path, "leaf")
    write_summary(tmp_path, "opencode", resolved=1)
    rows = load_summaries([str(tmp_path)])
    assert len(rows) == 2
    table = as_table(rows)
    assert "harness" in table and "resolve_rate" in table
    assert "25.0%" in table
    notes = caveats(rows)
    assert any("max_steps" in note and "NOT enforced" in note for note in notes)
    assert any("temperature" in note for note in notes)


def test_compare_surfaces_infrastructure_failures(tmp_path: Path) -> None:
    write_summary(
        tmp_path,
        "pi",
        jobs_infrastructure_failed=2,
        infrastructure_reasons={"provision_error": 2},
    )
    notes = caveats(load_summaries([str(tmp_path)]))
    assert any("failed before the agent could be graded" in note for note in notes)


def test_load_summaries_reports_a_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_summaries([str(tmp_path / "nope")])