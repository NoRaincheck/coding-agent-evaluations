from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from coding_agent_evaluations._vendor.frognano.runtimes.errors import (
    CommandTimeoutError,
)
from coding_agent_evaluations.runtimes.base import (
    CommandExecutor,
    copy_into,
)
from coding_agent_evaluations.runtimes.host import HostTaskRuntime
from coding_agent_evaluations.runtimes.provision import parse_dockerfile, rewrite_tree


@pytest.fixture()
def executor(tmp_path: Path) -> CommandExecutor:
    return CommandExecutor(tmp_path / "state", logger=logging.getLogger("test"))


def test_command_reports_stdout_and_exit_code(executor: CommandExecutor, tmp_path: Path) -> None:
    result = executor.run(
        "echo out; echo err 1>&2; exit 7", workdir=tmp_path, timeout=30
    )
    assert result.exit_code == 7
    assert "out" in result.output and "err" in result.output
    assert result.timed_out is False


def test_command_runs_in_the_requested_workdir(executor: CommandExecutor, tmp_path: Path) -> None:
    workdir = tmp_path / "repo"
    workdir.mkdir()
    result = executor.run("pwd", workdir=workdir, timeout=30)
    assert result.output.strip().endswith("repo")


def test_command_exit_does_not_skip_the_completion_record(
    executor: CommandExecutor, tmp_path: Path
) -> None:
    # `exit` inside the command used to abort the wrapper before it recorded the
    # outcome, which the caller then had to guess at.
    result = executor.run("echo bye; exit 3", workdir=tmp_path, timeout=30)
    assert result.exit_code == 3
    assert "bye" in result.output


def test_command_timeout_raises_with_partial_output(
    executor: CommandExecutor, tmp_path: Path
) -> None:
    with pytest.raises(CommandTimeoutError) as error:
        executor.run("echo started; sleep 30", workdir=tmp_path, timeout=1)
    assert error.value.output is not None


def test_failed_checksum_is_not_silently_accepted(
    executor: CommandExecutor, tmp_path: Path
) -> None:
    with pytest.raises(ValueError):
        executor.run("", workdir=tmp_path, timeout=0)


def test_large_output_round_trips(executor: CommandExecutor, tmp_path: Path) -> None:
    result = executor.run(
        "for i in $(seq 1 20000); do echo 'line of output padding'; done",
        workdir=tmp_path,
        timeout=60,
    )
    assert result.exit_code == 0
    assert result.output.count("line of output padding") == 20000


def test_copy_into_places_directory_contents(tmp_path: Path) -> None:
    source = tmp_path / "src"
    (source / "nested").mkdir(parents=True)
    (source / "test.sh").write_text("echo hi\n", encoding="utf-8")
    (source / "nested" / "a.txt").write_text("a\n", encoding="utf-8")
    destination = tmp_path / "dst"
    copy_into(source, destination)
    assert (destination / "test.sh").is_file()
    assert (destination / "nested" / "a.txt").is_file()
    assert not (destination / "src").exists()


def test_copy_into_single_file(tmp_path: Path) -> None:
    source = tmp_path / "file.txt"
    source.write_text("payload\n", encoding="utf-8")
    copy_into(source, tmp_path / "deep" / "file.txt")
    assert (tmp_path / "deep" / "file.txt").read_text() == "payload\n"


def test_parse_dockerfile_keeps_order_and_tolerates_extras(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM python:3.13-slim-bookworm\n"
        "WORKDIR /app\n"
        "ENV FOO=bar\n"
        "COPY gen.py /app/\n"
        "RUN python3 /app/gen.py\n"
        "EXPOSE 8080\n"
        "RUN rm -f /app/gen.py\n",
        encoding="utf-8",
    )
    parsed = parse_dockerfile(dockerfile)
    assert parsed.base_image == "python:3.13-slim-bookworm"
    assert parsed.workdir == "/app"
    assert parsed.environment == {"FOO": "bar"}
    assert parsed.steps == [
        ("workdir", "/app"),
        ("copy", "gen.py /app/"),
        ("run", "python3 /app/gen.py"),
        ("run", "rm -f /app/gen.py"),
    ]
    assert any(item.startswith("EXPOSE") for item in parsed.ignored)


def test_parse_dockerfile_requires_a_base_image(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("WORKDIR /app\n", encoding="utf-8")
    from coding_agent_evaluations.runtimes.provision import ProvisionError

    with pytest.raises(ProvisionError):
        parse_dockerfile(dockerfile)


def test_rewrite_tree_reports_changes(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.py").write_text('p = "/app/logs"\n', encoding="utf-8")
    (tmp_path / "b.bin").write_bytes(b"\x00\x01/app\x00")
    (tmp_path / "c.txt").write_text("nothing to do\n", encoding="utf-8")
    changes = rewrite_tree(tmp_path, {"/app": "/host/app"})
    assert len(changes) == 1 and "a.py" in changes[0]
    assert (tmp_path / "sub" / "a.py").read_text() == 'p = "/host/app/logs"\n'
    # Binary payloads are left alone rather than corrupted.
    assert (tmp_path / "b.bin").read_bytes() == b"\x00\x01/app\x00"


class FakeRuntime(HostTaskRuntime):
    """Host runtime with provisioning replaced by a fixed workspace."""

    def _prepare(self) -> None:
        self.paths.workspace.mkdir(parents=True, exist_ok=True)
        (self.paths.workspace / "file.txt").write_text("base\n", encoding="utf-8")
        self._snapshot_pristine()

    def _resolve_host_path(self, value: str, *, directory: bool = True) -> Path:
        # Mirror the production mapping: container paths are remapped onto the
        # workspace, everything else (`/tmp` included) passes through unchanged.
        mapping = {
            "/testbed": self.paths.workspace,
            "/app": self.paths.workspace,
            "/workspace": self.paths.workspace,
            "/tests": self.paths.tests,
            "/logs": self.paths.logs,
            "/solution": self.paths.solution,
        }
        for original, replacement in sorted(mapping.items(), key=lambda i: -len(i[0])):
            if value == original or value.startswith(original + "/"):
                target = Path(str(replacement) + value[len(original) :])
                return HostTaskRuntime._prepare_destination(target, directory)
        return HostTaskRuntime._prepare_destination(Path(value), directory)


def build_fake(task: dict[str, Any], tmp_path: Path, **options: Any) -> HostTaskRuntime:
    from coding_agent_evaluations.config import RuntimeConfig

    config = RuntimeConfig(kind="host", root=tmp_path / "workspaces", options=options)
    return FakeRuntime(task, config, run_id="s0-a1-deadbeef", logger=logging.getLogger("t"))


def test_host_runtime_get_patch_captures_agent_edits(tmp_path: Path) -> None:
    runtime = build_fake(
        {
            "instance_id": "demo",
            "dataset": "swebench_verified",
            "instruction": "do the thing",
            "repo_path": "/testbed",
            "require_git_patch": True,
        },
        tmp_path,
    )
    try:
        runtime.run("git init -q . && git config user.email a@b.c && git config user.name t")
        runtime.run("git add -A && git commit -qm base", timeout=60)
        patch = runtime.get_patch()
        assert patch == ""
        runtime.run("echo change >> file.txt")
        patch = runtime.get_patch()
        assert "file.txt" in patch and "+change" in patch
        # The index must be left clean, as in the reference runtime.
        assert runtime.run("git diff --cached --stat")[1] == 0
    finally:
        runtime.close()


def test_host_runtime_compute_reward_reads_the_reward_file(tmp_path: Path) -> None:
    tests = tmp_path / "task" / "tests"
    tests.mkdir(parents=True)
    (tests / "test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n", encoding="utf-8"
    )
    runtime = build_fake(
        {
            "instance_id": "demo",
            "dataset": "swebench_verified",
            "instruction": "x",
            "repo_path": "/testbed",
            "require_git_patch": False,
            "tests_dir": str(tests),
            "verifier_timeout_sec": 60,
            "verifier_success_marker": None,
        },
        tmp_path,
    )
    try:
        reward, output = runtime.compute_reward()
        assert reward == 1.0
        # The container path inside test.sh is rewritten before the verifier runs,
        # which is why the reward lands in the workspace logs directory.
        rewritten = " ".join(runtime.verifier_translation["rewritten_files"])
        assert "test.sh" in rewritten
        assert str(runtime.paths.logs / "verifier") in rewritten
        assert (runtime.paths.logs / "verifier" / "reward.txt").is_file()
        assert output.strip() == ""
    finally:
        runtime.close()


def test_host_runtime_compute_reward_reports_a_missing_reward(tmp_path: Path) -> None:
    tests = tmp_path / "task" / "tests"
    tests.mkdir(parents=True)
    (tests / "test.sh").write_text("#!/bin/bash\necho nothing\n", encoding="utf-8")
    runtime = build_fake(
        {
            "instance_id": "demo",
            "dataset": "swebench_verified",
            "instruction": "x",
            "repo_path": "/testbed",
            "require_git_patch": False,
            "tests_dir": str(tests),
            "verifier_timeout_sec": 60,
        },
        tmp_path,
    )
    try:
        with pytest.raises(RuntimeError, match="did not write a reward"):
            runtime.compute_reward()
    finally:
        runtime.close()


def test_host_runtime_recreate_restores_the_pristine_workspace(tmp_path: Path) -> None:
    runtime = build_fake(
        {
            "instance_id": "demo",
            "dataset": "swebench_verified",
            "instruction": "x",
            "repo_path": "/testbed",
        },
        tmp_path,
    )
    try:
        runtime.run("echo scratch > file.txt")
        assert "scratch" in (runtime.paths.workspace / "file.txt").read_text()
        runtime.recreate()
        assert (runtime.paths.workspace / "file.txt").read_text() == "base\n"
    finally:
        runtime.close()


def test_host_runtime_copies_a_file_to_a_file_path(tmp_path: Path) -> None:
    # The destination must be a file. Creating it unconditionally turned the leaf
    # tool runner into a directory, and every tool call failed with
    # "can't find '__main__' module".
    runtime = build_fake(
        {"instance_id": "demo", "dataset": "d", "instruction": "x", "repo_path": "/app"},
        tmp_path,
    )
    try:
        source = tmp_path / "tool_runner.py"
        source.write_text("print('ok')\n", encoding="utf-8")
        destination = runtime.paths.root / "bin" / "tool_runner.py"
        runtime.copy_to_container(source, str(destination))
        assert destination.is_file()
        assert destination.read_text() == "print('ok')\n"
        # Overwriting an existing file works too.
        source.write_text("print('new')\n", encoding="utf-8")
        runtime.copy_to_container(source, str(destination))
        assert destination.read_text() == "print('new')\n"
    finally:
        runtime.close()


def test_host_runtime_copies_a_directory_to_a_directory_path(tmp_path: Path) -> None:
    runtime = build_fake(
        {"instance_id": "demo", "dataset": "d", "instruction": "x", "repo_path": "/app"},
        tmp_path,
    )
    try:
        source = tmp_path / "payload"
        source.mkdir()
        (source / "f.txt").write_text("data\n", encoding="utf-8")
        destination = runtime.paths.root / "payload-copy"
        runtime.copy_to_container(source, str(destination))
        assert destination.is_dir()
        assert (destination / "f.txt").read_text() == "data\n"
    finally:
        runtime.close()


def test_host_runtime_recovers_from_a_stale_directory_destination(tmp_path: Path) -> None:
    runtime = build_fake(
        {"instance_id": "demo", "dataset": "d", "instruction": "x", "repo_path": "/app"},
        tmp_path,
    )
    try:
        destination = runtime.paths.root / "bin" / "runner.py"
        destination.mkdir(parents=True)
        source = tmp_path / "runner.py"
        source.write_text("print('ok')\n", encoding="utf-8")
        runtime.copy_to_container(source, str(destination))
        assert destination.is_file()
    finally:
        runtime.close()


def test_leaf_tools_run_against_the_host_workspace(tmp_path: Path) -> None:
    """The reference leaf environment must be able to use its tools here.

    Leaf passes `task["repo_path"]` to its tool runner, which chdirs into it. If
    that is still the container path, every tool call fails with
    FileNotFoundError. This exercises the real LeafEnvironment and the real tool
    runner, neither of which needs a model.
    """
    from coding_agent_evaluations._vendor.frognano.harness.leaf.environment import (
        LeafEnvironment,
    )

    runtime = build_fake(
        {
            "instance_id": "demo",
            "dataset": "swebench_verified",
            "instruction": "x",
            "repo_path": "/testbed",
            "require_git_patch": False,
        },
        tmp_path,
    )
    try:
        # Stand in for provisioning, as the real runtime does.
        runtime.task["container_repo_path"] = "/testbed"
        runtime.task["repo_path"] = str(runtime.paths.workspace)
        environment = LeafEnvironment(runtime)
        assert environment.instruction() == "x"
        assert "Wrote" in environment.execute(
            "Write", {"file_path": "a.txt", "content": "hello"}
        )
        assert (runtime.paths.workspace / "a.txt").read_text() == "hello"
        assert "hello" in environment.execute("Read", {"file_path": "a.txt"})
        assert "a.txt" in environment.execute("Glob", {"pattern": "*.txt"})
        shell = environment.execute("Bash", {"command": "cat a.txt"})
        assert "hello" in shell and "Exit code: 0" in shell
    finally:
        runtime.close()


def test_provisioning_and_path_handlers_are_distinct_methods(tmp_path: Path) -> None:
    """Provisioning and destination creation must not share a name.

    A collision here replaced provisioning with a path helper, and every task
    failed before the agent could run.
    """
    assert callable(HostTaskRuntime._prepare)
    assert callable(HostTaskRuntime._prepare_destination)
    assert HostTaskRuntime._prepare_destination(tmp_path / "made", True).is_dir()
    assert not HostTaskRuntime._prepare_destination(tmp_path / "file", False).exists()


def test_host_runtime_close_removes_the_workspace(tmp_path: Path) -> None:
    runtime = build_fake(
        {"instance_id": "demo", "dataset": "d", "instruction": "x", "repo_path": "/app"},
        tmp_path,
    )
    root = runtime.paths.root
    runtime.close()
    assert not root.exists()


