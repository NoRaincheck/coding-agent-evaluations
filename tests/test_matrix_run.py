"""Matrix configs and matrix runs.

A matrix config is the entry point for "which model, on which harness" comparisons, so what
matters is that the cells are exactly the cross product asked for, land in their own
directories, and that the run reports progress over the *pairs* rather than the cells.

Everything here is offline: the task list and the evaluation itself are stubbed, because what
is being tested is the expansion, the ordering, the failure isolation and the progress
accounting — none of which need a model.
"""

from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from coding_agent_evaluations import matrix_runner
from coding_agent_evaluations.cli import main
from coding_agent_evaluations.config import ConfigError, ModelConfig, load_config
from coding_agent_evaluations.matrix import (
    MatrixConfig,
    load_matrix_config,
    load_run_or_matrix,
)

BASE = Path(__file__).resolve().parents[1] / "configs/base.yaml"
SHIPPED_MATRIX = (
    Path(__file__).resolve().parents[1] / "configs/matrix/local-models.yaml"
)


def write_matrix(path: Path, body: str) -> Path:
    """A matrix config extending the shipped base, with `body` as its `matrix:` block."""
    path.write_text(f"extends: [{BASE}]\n{body}", encoding="utf-8")
    return path


def base_config(tmp_path: Path) -> Any:
    return replace(
        load_config(BASE),
        output_dir=tmp_path / "out",
        num_tasks=2,
        seeds_per_task=1,
    )


def test_the_shipped_matrix_resolves() -> None:
    matrix = load_matrix_config(SHIPPED_MATRIX)
    assert matrix.datasets == ("swebench_verified",)
    assert matrix.harnesses == ("leaf", "opencode", "pi")
    assert len(matrix.models) == 3
    # A matrix is a comparison; it must not be a week of compute by default.
    assert matrix.base.num_tasks is not None and matrix.base.num_tasks <= 5
    assert matrix.base.max_steps == 10
    assert matrix.continue_on_error is True


def test_cells_are_the_cross_product() -> None:
    matrix = load_matrix_config(SHIPPED_MATRIX)
    cells = matrix.cells
    assert len(cells) == 9
    assert {(cell.dataset, cell.harness.name, cell.model.name) for cell in cells} == {
        (dataset, harness, model.name)
        for dataset in matrix.datasets
        for harness in matrix.harnesses
        for model in matrix.models
    }


def test_every_cell_gets_its_own_output_dir(tmp_path: Path) -> None:
    # Cells that shared a directory would overwrite each other's results.jsonl and
    # summary.json, and the comparison would show the last cell three times.
    matrix = MatrixConfig(
        base=base_config(tmp_path),
        models=(ModelConfig(name="a", base_url="http://127.0.0.1:1/v1"),),
        harnesses=("leaf", "pi"),
    )
    directories = [str(cell.output_dir) for cell in matrix.cells]
    assert len(set(directories)) == len(directories)
    for cell in matrix.cells:
        assert cell.output_dir.is_relative_to(tmp_path / "out")


def test_cells_differ_only_in_model_and_harness(tmp_path: Path) -> None:
    # Everything else identical, or the comparison is measuring something else.
    matrix = MatrixConfig(
        base=base_config(tmp_path),
        models=(
            ModelConfig(name="a", base_url="http://127.0.0.1:1/v1"),
            ModelConfig(name="b", base_url="http://127.0.0.1:2/v1"),
        ),
        harnesses=("leaf", "opencode", "pi"),
    )
    cells = matrix.cells
    # Expansion is harness-outer, model-inner, so cells two apart differ only in harness.
    first, same_model_other_harness, other_model = cells[0], cells[2], cells[1]
    assert first.harness.name != same_model_other_harness.harness.name
    assert first.model.name == same_model_other_harness.model.name
    assert other_model.model.name != first.model.name
    for field in ("num_tasks", "seeds_per_task", "max_steps", "max_total_time_sec", "seed"):
        assert getattr(first, field) == getattr(same_model_other_harness, field), field
    for field in ("max_context_tokens", "resume", "cache_dir", "runtime"):
        assert getattr(first, field) == getattr(same_model_other_harness, field), field


def test_a_model_name_inherits_the_base_url(tmp_path: Path) -> None:
    path = write_matrix(
        tmp_path / "matrix.yaml",
        "matrix:\n  harnesses: [pi]\n  models: [served-model]\n",
    )
    matrix = load_matrix_config(path)
    assert matrix.models[0].name == "served-model"
    assert matrix.models[0].base_url == matrix.base.model.base_url


def test_a_model_mapping_overrides_only_what_it_names(tmp_path: Path) -> None:
    path = write_matrix(
        tmp_path / "matrix.yaml",
        "matrix:\n"
        "  harnesses: [pi]\n"
        "  models:\n"
        "    - name: big\n"
        "      base_url: http://127.0.0.1:9999/v1\n",
    )
    model = load_matrix_config(path).models[0]
    assert model.name == "big"
    assert model.base_url == "http://127.0.0.1:9999/v1"
    # Sampling settings still come from the base config, so cells are comparable.
    assert model.temperature == load_config(BASE).model.temperature


def test_omitted_harnesses_falls_back_to_the_base(tmp_path: Path) -> None:
    path = write_matrix(tmp_path / "matrix.yaml", "matrix:\n  models: [a, b]\n")
    matrix = load_matrix_config(path)
    assert matrix.harnesses == (load_config(BASE).harness.name,)
    assert len(matrix.cells) == 2


def test_omitted_models_falls_back_to_the_base(tmp_path: Path) -> None:
    path = write_matrix(
        tmp_path / "matrix.yaml", "matrix:\n  harnesses: [leaf, pi]\n"
    )
    matrix = load_matrix_config(path)
    assert [cell.model.name for cell in matrix.cells] == [matrix.base.model.name] * 2


def test_a_duplicate_model_is_collapsed(tmp_path: Path) -> None:
    # The same checkpoint listed twice would run twice into the same directory.
    path = write_matrix(
        tmp_path / "matrix.yaml", "matrix:\n  harnesses: [pi]\n  models: [a, a, b]\n"
    )
    assert [model.name for model in load_matrix_config(path).models] == ["a", "b"]


def test_a_matrix_must_list_something(tmp_path: Path) -> None:
    path = write_matrix(tmp_path / "matrix.yaml", "matrix:\n  models: []\n")
    with pytest.raises(ConfigError):
        load_matrix_config(path)


def test_an_unknown_harness_is_rejected(tmp_path: Path) -> None:
    path = write_matrix(
        tmp_path / "matrix.yaml", "matrix:\n  harnesses: [aider]\n  models: [a]\n"
    )
    with pytest.raises(ConfigError):
        load_matrix_config(path)


def test_a_run_config_is_not_a_matrix(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not a matrix config"):
        load_matrix_config(BASE)


def test_load_run_or_matrix_dispatches(tmp_path: Path) -> None:
    assert isinstance(load_run_or_matrix(BASE), type(load_config(BASE)))
    assert isinstance(load_run_or_matrix(SHIPPED_MATRIX), MatrixConfig)


def test_cli_cells_lists_every_cell(capsys: pytest.CaptureFixture[str]) -> None:
    # `cae run --config <matrix> --cells` must show what would run without running it.
    assert main(["run", "--config", str(SHIPPED_MATRIX), "--cells"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["cells"]) == 9
    for cell in payload["cells"]:
        assert cell["output_dir"].endswith(
            f"{cell['dataset']}/{cell['harness']}/{cell['model']}"
        )


def test_cli_set_applies_to_every_cell(capsys: pytest.CaptureFixture[str]) -> None:
    # `--set num_tasks=50` has to mean the same thing in all nine cells, otherwise scaling a
    # matrix up silently scales some of it.
    assert (
        main(
            [
                "run",
                "--config",
                str(SHIPPED_MATRIX),
                "--cells",
                "--set",
                "num_tasks=50",
                "--set",
                "max_workers=4",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["base"]["num_tasks"] == 50
    assert payload["base"]["max_workers"] == 4
    assert len(payload["cells"]) == 9


def test_cli_harness_narrows_a_matrix(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--config", str(SHIPPED_MATRIX), "--cells", "--harness", "pi"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {cell["harness"] for cell in payload["cells"]} == {"pi"}
    assert len(payload["cells"]) == 3


def test_cli_config_prints_a_matrix(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config", "--config", str(SHIPPED_MATRIX)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["matrix"]["harnesses"] == ["leaf", "opencode", "pi"]
    assert payload["base"]["source"] == str(SHIPPED_MATRIX)


# -- the run itself ---------------------------------------------------------


def stub_matrix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **kwargs: Any):
    """A matrix whose tasks and evaluations are stubbed, so nothing is fetched or run."""
    monkeypatch.setattr(
        matrix_runner,
        "load_tasks",
        lambda *args, **kw: [{"instance_id": f"task-{index}"} for index in range(2)],
    )
    calls: list[dict[str, Any]] = []

    def fake_run(config, *, tasks=None, on_result=None):
        calls.append(
            {
                "label": config.label(),
                "output_dir": str(config.output_dir),
                "tasks": len(tasks or []),
            }
        )
        for index in range(2):
            row = {"instance_id": f"task-{index}", "status": "completed", "reward": 1.0}
            if on_result is not None:
                on_result(row)
        summary = {
            "dataset": config.dataset,
            "harness": config.harness.name,
            "model": config.model.name,
            "jobs_total": 2,
            "jobs_failed": 0,
            "resolved": 2,
            "resolve_rate": 1.0,
        }
        Path(config.output_dir).expanduser().mkdir(parents=True, exist_ok=True)
        (Path(config.output_dir).expanduser() / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        return summary

    monkeypatch.setattr(matrix_runner, "run_evaluation", fake_run)
    matrix = MatrixConfig(
        base=base_config(tmp_path),
        models=(
            ModelConfig(name="small", base_url="http://127.0.0.1:1/v1"),
            ModelConfig(name="big", base_url="http://127.0.0.1:1/v1"),
        ),
        harnesses=("leaf", "pi"),
        **kwargs,
    )
    return matrix, calls


def test_every_cell_runs_in_expansion_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    matrix, calls = stub_matrix(monkeypatch, tmp_path)
    reports = matrix_runner.run_matrix(matrix, progress=False)
    assert [report.label for report in reports] == [
        f"swebench_verified/{harness}/{model}"
        for harness in matrix.harnesses
        for model in ("small", "big")
    ]
    assert [call["label"] for call in calls] == [r.label for r in reports]
    # Each cell is handed its own task list rather than sharing one list.
    assert all(call["tasks"] == 2 for call in calls)


def test_progress_counts_every_pair_not_every_cell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Two cells of two pairs each: a bar over cells would stop at 2/4 having done no work
    # in the second cell, which is exactly the point at which a long run looks stalled.
    matrix, _ = stub_matrix(monkeypatch, tmp_path)
    stream = io.StringIO()
    matrix_runner.run_matrix(matrix, stream=stream, progress=True)
    output = stream.getvalue()
    assert "8" in output  # total task-seed pairs
    assert "pair" in output or "4/4" in output
    for report in matrix.cells:
        assert f"{report.dataset}/{report.harness.name}/{report.model.name}" in output


def test_progress_reports_done_and_remaining(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    matrix, _ = stub_matrix(monkeypatch, tmp_path)
    stream = io.StringIO()
    matrix_runner.run_matrix(matrix, stream=stream, progress=True)
    output = stream.getvalue()
    assert "4/4" in output or "100%" in output


def test_no_progress_prints_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    matrix, _ = stub_matrix(monkeypatch, tmp_path)
    stream = io.StringIO()
    matrix_runner.run_matrix(matrix, stream=stream, progress=False)
    assert stream.getvalue() == ""


def test_one_failing_cell_does_not_void_the_matrix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    matrix, calls = stub_matrix(monkeypatch, tmp_path)
    real = matrix_runner.run_evaluation

    def explode_on_big(config, *, tasks=None, on_result=None):
        if config.model.name == "big" and config.harness.name == "pi":
            raise RuntimeError("connection refused")
        return real(config, tasks=tasks, on_result=on_result)

    monkeypatch.setattr(matrix_runner, "run_evaluation", explode_on_big)
    reports = matrix_runner.run_matrix(matrix, progress=False)
    assert len(reports) == 4, "every cell is still attempted"
    failed = [report for report in reports if report.error]
    assert len(failed) == 1
    assert "connection refused" in failed[0].error
    assert len(calls) == 3


def test_continue_on_error_false_stops_at_the_first_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    matrix, calls = stub_matrix(monkeypatch, tmp_path, continue_on_error=False)

    def always_explode(config, *, tasks=None, on_result=None):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(matrix_runner, "run_evaluation", always_explode)
    with pytest.raises(RuntimeError):
        matrix_runner.run_matrix(matrix, progress=False)
    assert len(calls) == 0


def test_a_resumed_cell_is_not_counted_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The bar's total is what is left, so a rerun of finished work shows as complete rather
    # than as a bar that never fills.
    matrix, calls = stub_matrix(monkeypatch, tmp_path)
    matrix_runner.run_matrix(matrix, progress=False)
    rows = Path(calls[0]["output_dir"]) / "results.jsonl"
    rows.write_text(
        json.dumps({"instance_id": "task-0", "seed": 0, "status": "completed"}) + "\n"
        + json.dumps({"instance_id": "task-1", "seed": 0, "status": "completed"}) + "\n",
        encoding="utf-8",
    )
    plans = matrix_runner._plan(matrix.cells)
    assert plans[0].remaining == 0
    assert plans[-1].remaining == 2
