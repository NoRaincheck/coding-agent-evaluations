from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from coding_agent_evaluations.config import HarnessConfig, ModelConfig
from coding_agent_evaluations.harnesses import build_harness
from coding_agent_evaluations.harnesses.base import HarnessRun, empty_trajectory
from coding_agent_evaluations.harnesses.cli import CliHarness, hermetic_env, parse_jsonl
from coding_agent_evaluations.harnesses.opencode import _opencode_config
from coding_agent_evaluations.harnesses.pi import _models_config, _text_of
from coding_agent_evaluations.report import (
    as_table,
    as_token_table,
    caveats,
    load_summaries,
)


class FakeRuntime:
    def __init__(self, state: Path, instruction: str = "fix the bug") -> None:
        self.paths = type(
            "Paths", (), {"state": state, "workspace": state, "root": state}
        )()
        self._instruction = instruction

    def get_task_instruction(self) -> str:
        return self._instruction


def make_run(tmp_path: Path, **overrides: Any) -> HarnessRun:
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


def test_cli_harness_pins_pwd_to_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A stale `PWD` inherited from the launching shell makes a harness resolve the
    # wrong project directory and edit the wrong tree.
    monkeypatch.setenv("PWD", "/somewhere/else")

    class Probe(CliHarness):
        name = "probe"
        enforces_max_steps = False
        ignored_model_params: tuple[str, ...] = ()

        def resolve_binary(self) -> str:
            return "true"

        def session_id(self, run: HarnessRun) -> str:
            return "ses_probe"

        def build_command(
            self, run: HarnessRun, config_dir: Path, session_id: str
        ) -> list[str]:
            return ["true"]

        def build_env(
            self, run: HarnessRun, config_dir: Path, session_id: str
        ) -> dict[str, str]:
            return {"PATH": os.environ.get("PATH", "")}

        def parse(self, run: HarnessRun, stdout: str, exit_code: int) -> dict[str, Any]:
            return {"steps": [], "n_steps": 0, "exit_reason": "agent"}

    recorded: dict[str, Any] = {}
    original = subprocess.Popen

    def spy(command: list[str], **kwargs: Any) -> Any:
        recorded["env"] = kwargs.get("env") or {}
        recorded["cwd"] = kwargs.get("cwd")
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", spy)
    run = make_run(tmp_path)
    Probe().run(run)
    assert recorded["env"]["PWD"] == str(run.workspace)
    assert "OLDPWD" not in recorded["env"]
    assert recorded["cwd"] == str(run.workspace)


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


def test_pi_extension_is_loaded_and_nothing_else(tmp_path: Path) -> None:
    # Discovery is off (`--no-extensions`), so an operator's own pi extensions cannot
    # silently change what the harness measured. The one explicit path is the observer.
    run = make_run(tmp_path, config=HarnessConfig(name="pi"))
    harness = build_harness("pi")()
    harness.binary = "pi"
    command = harness.build_command(run, tmp_path / "cfg", harness.session_id(run))
    assert "--no-extensions" in command
    extensions = [
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == "--extension"
    ]
    assert len(extensions) == 1
    assert Path(extensions[0]).is_file()
    assert extensions[0].endswith("pi_token_stats/index.ts")


def test_pi_can_be_run_without_the_extension(tmp_path: Path) -> None:
    run = make_run(
        tmp_path, config=HarnessConfig(name="pi", options={"token_stats": False})
    )
    harness = build_harness("pi")()
    harness.binary = "pi"
    config_dir = tmp_path / "cfg"
    command = harness.build_command(run, config_dir, harness.session_id(run))
    assert "--extension" not in command
    environment = harness.build_env(run, config_dir, harness.session_id(run))
    assert "PI_TOKEN_SPEED_STATS_DIR" not in environment


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


def test_opencode_parse_reads_part_scoped_events(tmp_path: Path) -> None:
    # opencode nests the payload in `part`; reading the top level silently yields
    # zero steps and zero tokens.
    stdout = "\n".join(
        [
            json.dumps({"type": "session", "sessionID": "ses_x", "part": {}}),
            json.dumps(
                {
                    "type": "tool_use",
                    "part": {
                        "tool": "bash",
                        "state": {
                            "status": "completed",
                            "input": {"command": "ls"},
                            "output": "a.txt",
                        },
                    },
                }
            ),
            json.dumps(
                {
                    "type": "step_finish",
                    "tokens": {"input": 100, "output": 20, "cache": {"read": 0}},
                }
            ),
            json.dumps({"type": "text", "part": {"text": "all done"}}),
        ]
    )
    parsed = build_harness("opencode")().parse(make_run(tmp_path), stdout, 0)
    assert parsed["n_steps"] == 1
    assert parsed["steps"][0]["tool"] == "bash"
    assert parsed["steps"][0]["input"] == {"command": "ls"}
    assert parsed["steps"][0]["observation"] == "a.txt"
    assert parsed["final_message"] == "all done"
    assert parsed["usage"]["input_tokens"] == 100
    assert parsed["context_tokens"] == 100
    assert parsed["exit_reason"] == "agent"


def test_opencode_parse_reports_a_crash(tmp_path: Path) -> None:
    parsed = build_harness("opencode")().parse(make_run(tmp_path), "", 1)
    assert parsed["exit_reason"] == "harness_error"
    assert parsed["error"]


def test_pi_parse_pairs_tool_start_and_end(tmp_path: Path) -> None:
    # pi reports a call twice; counting both would double every step.
    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "tool_execution_start",
                    "toolCallId": "c1",
                    "toolName": "bash",
                    "args": {"command": "ls"},
                }
            ),
            json.dumps(
                {
                    "type": "tool_execution_end",
                    "toolCallId": "c1",
                    "toolName": "bash",
                    "isError": False,
                    "result": {"content": [{"type": "text", "text": "a.txt"}]},
                }
            ),
            json.dumps(
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "hmm"},
                            {"type": "text", "text": "done"},
                        ],
                        "usage": {"input": 50, "output": 10, "reasoning": 5},
                        "stopReason": "stop",
                    },
                }
            ),
        ]
    )
    parsed = build_harness("pi")().parse(make_run(tmp_path), stdout, 0)
    assert parsed["n_steps"] == 1
    assert parsed["steps"][0]["tool"] == "bash"
    assert parsed["steps"][0]["input"] == {"command": "ls"}
    assert parsed["steps"][0]["observation"] == "a.txt"
    assert parsed["steps"][0]["is_error"] is False
    # Thinking blocks must not leak into the reported final message.
    assert parsed["final_message"] == "done"
    assert parsed["usage"]["input_tokens"] == 50


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


TOKEN_STATS = {
    "requests": 20,
    "turns": 4,
    "input_tokens": 5_000,
    "output_tokens": 1_000,
    "ttft_ms_mean": 250.0,
    "ttft_ms_p50": 200.0,
    "ttft_ms_p90": 600.0,
    "decode_tps_mean": 40.0,
    "decode_tps_total": 38.0,
    "gen_tps_mean": 42.0,
    "prefill_tps_mean": 900.0,
    "generation_sec": 25.0,
    "cost_usd": 0.02,
    "models": ["m"],
    "provider_id": "cae-eval",
}


def test_token_speed_gets_its_own_table(tmp_path: Path) -> None:
    # A separate table, not more columns: it is measured by some harnesses only, and it
    # answers a different question from whether the task was solved.
    write_summary(tmp_path, "pi", token_stats=TOKEN_STATS)
    table = as_token_table(load_summaries([str(tmp_path)]))
    assert "ttft_p50_ms" in table and "decode_tps" in table
    assert "250.0" in table or "200.0" in table
    assert "harness" in table


def test_token_speed_table_is_empty_when_nothing_measured(tmp_path: Path) -> None:
    write_summary(tmp_path, "leaf")
    assert as_token_table(load_summaries([str(tmp_path)])) == ""


def test_compare_flags_rows_without_token_speed(tmp_path: Path) -> None:
    # Comparing an endpoint against a harness would look like a speed difference.
    write_summary(tmp_path, "pi", token_stats=TOKEN_STATS)
    write_summary(tmp_path, "leaf")
    notes = caveats(load_summaries([str(tmp_path)]))
    assert any("token speed measured only for pi" in note for note in notes)
    assert any("not comparable on speed" in note for note in notes)