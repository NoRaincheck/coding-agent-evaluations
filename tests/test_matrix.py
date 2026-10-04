"""Every matrix cell must be runnable as configured.

The matrix is (local model, harness) pairs read from config. These tests assert the things
that break a run *before* it starts — a config that no longer names the harness under test, a
model that never reaches the subprocess, a pi rollout that does not load the token-stats
extension — because each of those fails hours later, mid-run, with no clear cause.

They are the cheapest place to notice a config drift: no endpoint, no workspace, no model.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from matrix import (
    HARNESSES,
    cli_command,
    cli_env,
    extension_argv,
    matrix,
)

from coding_agent_evaluations.harnesses import build_harness
from coding_agent_evaluations.harnesses.pi import PiHarness
from coding_agent_evaluations.harnesses.pi_stats import VENDORED_COMMIT, extension_path


def test_the_matrix_is_not_empty() -> None:
    # A silently empty matrix would make every test in this file vacuously pass. Which
    # harnesses appear depends on EVAL_MATRIX_HARNESSES, so only the cells are checked.
    cells = matrix()
    assert cells, "no matrix cells; check EVAL_MATRIX_MODELS and EVAL_MATRIX_HARNESSES"
    for cell in cells:
        assert cell.model.name and cell.model.base_url
        assert cell.harness in HARNESSES
    assert len({cell.id for cell in cells}) == len(cells)


def test_every_matrix_cell_names_its_own_harness(cell) -> None:
    config = cell.run_config()
    assert config.harness.name == cell.harness
    assert build_harness(cell.harness).name == cell.harness
    assert config.model.name == cell.model.name
    assert config.model.base_url == cell.model.base_url


def test_every_matrix_cell_comes_from_a_shipped_config(cell) -> None:
    # The harness configs are the ones `cae run --harness <name>` composes with; if one is
    # missing, the matrix is testing a configuration nothing can actually launch.
    assert cell.config_path.is_file(), f"missing {cell.config_path}"
    assert cell.config_path.parent.name == "harness"


@pytest.mark.parametrize("harness", HARNESSES)
def test_every_harness_in_the_matrix_builds(harness: str) -> None:
    assert build_harness(harness).name == harness


def test_cli_harness_reaches_its_endpoint(cell, tmp_path: Path) -> None:
    # The harness may only name one model, and it has to be the cell's.
    if not cell.is_cli:
        pytest.skip(f"{cell.harness} is driven in-process")
    config_dir = tmp_path / "config"
    command = cli_command(cell, config_dir)
    environment = cli_env(cell, config_dir)
    assert command[-1] == "fix the bug", "the instruction must be the final argument"
    assert "--model" in command
    assert command[command.index("--model") + 1].endswith(cell.model.name)
    assert environment["EVAL_MODEL_NAME"] == cell.model.name
    assert environment["EVAL_MODEL_BASE_URL"] == cell.model.base_url


def test_cli_harness_stays_hermetic(cell, tmp_path: Path) -> None:
    # A leaked global config turns a measurement into an anecdote: pi's own extensions and
    # credentials, or the operator's pi calibration state, would be part of the rollout.
    if not cell.is_cli:
        pytest.skip(f"{cell.harness} is driven in-process")
    config_dir = tmp_path / "config"
    environment = cli_env(cell, config_dir)
    # State the CLI writes must land inside this rollout's own directory, never the
    # operator's: pi's agent dir, the calibration file, opencode's config and XDG tree.
    assert environment["HOME"].startswith(str(config_dir))
    assert environment["XDG_CONFIG_HOME"].startswith(str(config_dir))
    assert environment["PI_OFFLINE"] == "1"
    if cell.harness == "pi":
        assert environment["PI_CODING_AGENT_DIR"].startswith(str(config_dir))
        assert environment["PI_TOKEN_SPEED_STATS_DIR"] == environment["PI_CODING_AGENT_DIR"]
    else:
        assert environment["OPENCODE_CONFIG"].startswith(str(config_dir))


def test_pi_cells_load_the_token_stats_extension(cell, tmp_path: Path) -> None:
    if cell.harness != "pi":
        pytest.skip(f"cell is {cell.harness}")
    command = cli_command(cell, tmp_path / "config")
    loaded = extension_argv(command, "--extension")
    assert len(loaded) == 1, "exactly one extension: the pinned observer, nothing else"
    assert loaded[0].endswith("index.ts")
    assert Path(loaded[0]).is_file(), "the vendored extension must ship with the package"
    # Discovery stays off: pi's global extension directory must not enter the measurement.
    assert "--no-extensions" in command


def test_token_stats_are_on_by_default_and_can_be_switched_off() -> None:
    # Opting out has to be possible, and it has to actually drop the extension — otherwise
    # "off" would still measure, and the flag would be a lie.
    pi_config = next(cell for cell in matrix() if cell.harness == "pi")
    default = pi_config.harness_config()
    assert extension_path(default) is not None, "on by default"
    assert extension_path(default).is_file()

    disabled = replace(default, options={**default.options, "token_stats": False})
    assert extension_path(disabled) is None


def test_the_extension_override_is_honoured(tmp_path: Path) -> None:
    # The escape hatch for evaluating a newer upstream revision without editing the tree.
    alternate = tmp_path / "index.ts"
    alternate.write_text("// newer upstream\n", encoding="utf-8")
    default = next(cell for cell in matrix() if cell.harness == "pi").harness_config()
    overridden = replace(
        default, options={**default.options, "token_stats_extension": str(alternate)}
    )
    assert extension_path(overridden) == alternate


def test_the_recorded_commit_is_the_vendored_one() -> None:
    # A summary says which implementation produced its numbers; that has to be the pinned
    # commit the vendored file actually came from.
    provenance = (
        Path(__file__).resolve().parents[1]
        / "src/coding_agent_evaluations/_vendor/pi_token_stats/PROVENANCE.md"
    )
    assert VENDORED_COMMIT in provenance.read_text(encoding="utf-8")


def test_pi_configs_are_loadable_as_written() -> None:
    # The shipped pi config is the one users copy; a typo in it should fail here, not at run.
    from coding_agent_evaluations.config import load_config

    config = load_config(Path(__file__).resolve().parents[1] / "configs/harness/pi.yaml")
    assert config.harness.name == "pi"
    assert config.harness.options["token_stats"] is True
    # Everything the harness reads must survive a round trip through JSON.
    json.dumps(config.as_dict(), default=str)
    assert PiHarness.name == "pi"
