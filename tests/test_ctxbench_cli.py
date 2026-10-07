"""CLI wiring for `cae ctxbench`.

Tests the argument surface and the exit code, not the measurement -- `run_ctxbench` is
covered directly. The exit code matters most: a run whose suites failed has to be
distinguishable from a clean one by a script, or a pipeline will happily publish a
half-measured sweep.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from coding_agent_evaluations.cli import main
from coding_agent_evaluations.config import ConfigError

CONFIG = """
model:
  name: m
  base_url: http://h/v1
output_dir: {output}
context_sizes: [16000]
ctxbench:
  enabled: [prefill]
  repeats: 1
"""


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "ctx.yaml"
    path.write_text(CONFIG.format(output=tmp_path / "out"), encoding="utf-8")
    return path


def test_plan_prints_the_resolved_config(config_path: Path, capsys) -> None:
    assert main(["ctxbench", "--config", str(config_path), "--plan"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["context_sizes"] == [16_000]
    assert payload["ctxbench"]["enabled"] == ["prefill"]


def test_contexts_flag_replaces_the_ladder(config_path: Path, capsys) -> None:
    code = main(
        [
            "ctxbench",
            "--config",
            str(config_path),
            "--plan",
            "--contexts",
            "8000, 16000,128000",
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["context_sizes"] == [
        8_000,
        16_000,
        128_000,
    ]


def test_harness_flag_is_applied(config_path: Path, capsys) -> None:
    code = main(["ctxbench", "--config", str(config_path), "--plan", "--harness", "pi"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["harness"] == "pi"


def test_suites_flag_is_applied(config_path: Path, capsys) -> None:
    code = main(
        [
            "ctxbench",
            "--config",
            str(config_path),
            "--plan",
            "--suites",
            "prefill,needle",
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["ctxbench"]["enabled"] == [
        "prefill",
        "needle",
    ]


def test_bad_context_size_is_rejected(config_path: Path, capsys) -> None:
    """A silently skipped entry would shift every later row off its label."""
    code = main(["ctxbench", "--config", str(config_path), "--contexts", "16000,oops"])
    assert code == 2
    assert "positive integer" in capsys.readouterr().err


def test_unknown_suite_is_rejected(config_path: Path, capsys) -> None:
    code = main(["ctxbench", "--config", str(config_path), "--suites", "prefll"])
    assert code == 2
    assert "unknown suite" in capsys.readouterr().err


def test_set_override_is_applied(config_path: Path, capsys) -> None:
    code = main(
        ["ctxbench", "--config", str(config_path), "--plan", "--set", "seed=99"]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["seed"] == 99


def test_missing_config_is_reported(tmp_path: Path, capsys) -> None:
    code = main(["ctxbench", "--config", str(tmp_path / "nope.yaml"), "--plan"])
    assert code == 2
    assert "config error" in capsys.readouterr().err


def test_json_flag_emits_the_record(
    config_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    payload = {
        "report_version": 1,
        "results": [],
        "config": {"context_sizes": [16_000], "ctxbench": {"enabled": ["prefill"]}},
    }
    monkeypatch.setattr(
        "coding_agent_evaluations.cli.run_ctxbench", lambda config, on_size=None: payload
    )
    assert main(["ctxbench", "--config", str(config_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["report_version"] == 1


def test_suite_errors_set_a_nonzero_exit(
    config_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """A pipeline has to be able to tell a clean sweep from a half-measured one."""
    payload = {
        "report_version": 1,
        "results": [{"errors": ["prefill: HTTP 400"]}],
        "config": {"context_sizes": [16_000], "ctxbench": {"enabled": ["prefill"]}},
    }
    monkeypatch.setattr(
        "coding_agent_evaluations.cli.run_ctxbench", lambda config, on_size=None: payload
    )
    assert main(["ctxbench", "--config", str(config_path)]) == 1
    assert "ctxbench report v1" in capsys.readouterr().out


def test_doctor_accepts_a_ctxbench_config(
    config_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Pointing `doctor` at a sweep's config before running the sweep is the natural
    thing to do, and a ctxbench config is a different shape from a benchmark one."""
    seen: dict[str, Any] = {}

    def fake_checks(**kwargs: Any) -> list[Any]:
        from coding_agent_evaluations.doctor import Check

        seen.update(kwargs)
        return [Check("endpoint", True, "reachable")]

    monkeypatch.setattr("coding_agent_evaluations.cli.run_checks", fake_checks)
    assert main(["doctor", "--config", str(config_path)]) == 0
    assert seen["model"].name == "m"
    assert "ok" in capsys.readouterr().out


def test_doctor_recognises_a_config_that_extends_a_benchmark_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """`configs/ctxbench.yaml` extends `base.yaml`, so its resolved payload has a
    `dataset:` too. Detection keyed on the absence of `dataset:` would misread it and
    report the wrong cache and workspace paths."""
    (tmp_path / "base.yaml").write_text(
        "dataset: swebench_verified\nmodel:\n  name: from-base\n  base_url: u\n"
        "cache_dir: /tmp/base-cache\n",
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        "extends: [./base.yaml]\nctxbench:\n  repeats: 2\n"
        "runtime:\n  root: /tmp/ctx-workspaces\n",
        encoding="utf-8",
    )
    seen: dict[str, Any] = {}

    def fake_checks(**kwargs: Any) -> list[Any]:
        from coding_agent_evaluations.doctor import Check

        seen.update(kwargs)
        return [Check("endpoint", True, "reachable")]

    monkeypatch.setattr("coding_agent_evaluations.cli.run_checks", fake_checks)
    assert main(["doctor", "--config", str(child)]) == 0
    # The ctxbench loader's own paths, not the benchmark base's.
    assert str(seen["cache_dir"]) == str(Path("~/.cache/cae/datasets").expanduser())
    assert str(seen["workspace_root"]) == "/tmp/ctx-workspaces"


def test_doctor_still_reports_a_broken_benchmark_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Detection must not swallow a genuine config error by claiming it was a ctxbench
    config."""
    broken = tmp_path / "broken.yaml"
    broken.write_text("model:\n  name: m\n", encoding="utf-8")  # no dataset, no ctxbench
    monkeypatch.setattr(
        "coding_agent_evaluations.cli.run_checks", lambda **kwargs: []
    )
    assert main(["doctor", "--config", str(broken)]) == 2
    assert "config error" in capsys.readouterr().err


def test_int_list_helper() -> None:
    from coding_agent_evaluations.cli import _int_list

    assert _int_list(" 8000 , 16000 ") == [8_000, 16_000]
    with pytest.raises(ConfigError, match="no context sizes"):
        _int_list(" , ")
    with pytest.raises(ConfigError, match="positive integer"):
        _int_list("-1")