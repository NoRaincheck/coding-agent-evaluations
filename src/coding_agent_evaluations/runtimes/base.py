"""Sandbox interface shared by every runtime.

The contract mirrors ``frognano.runtimes.kubernetes.KubernetesTaskRuntime`` so
the upstream :class:`frognano.harness.leaf.LeafEnvironment` can drive any runtime
unmodified. That is deliberate: the leaf harness stays byte-identical to the
reference implementation and only the execution substrate changes.

Command execution keeps the guarantees of the reference runtime:

* the command text is written to a file and sourced inside the workspace,
* stdout and stderr are captured to a file with the exit code,
* the captured output is checksummed and verified before it is handed back,
* a timeout raises :class:`CommandTimeoutError` instead of resubmitting,
* a lost acknowledgement re-reads the saved state rather than re-running.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from frognano.runtimes.errors import CommandTimeoutError, PodExecutionError

_READ_ATTEMPTS = 3


@dataclass
class CommandResult:
    """A completed (or timed out) command execution."""

    output: str
    exit_code: int
    timed_out: bool = False
    duration_sec: float = 0.0


@dataclass
class SandboxPaths:
    """Directory layout of one task workspace.

    Each entry replaces an in-container absolute path. See
    :func:`coding_agent_evaluations.runtimes.translate.path_mapping`.
    """

    root: Path
    workspace: Path
    tests: Path
    logs: Path
    solution: Path
    state: Path

    def ensure(self) -> None:
        for path in (
            self.root,
            self.workspace,
            self.tests,
            self.logs,
            self.solution,
            self.state,
        ):
            path.mkdir(parents=True, exist_ok=True)
        (self.logs / "verifier").mkdir(parents=True, exist_ok=True)
        (self.logs / "agent").mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "workspace": str(self.workspace),
            "tests": str(self.tests),
            "logs": str(self.logs),
            "solution": str(self.solution),
            "state": str(self.state),
        }


@dataclass
class ProvisionRecord:
    """What host provisioning did, and what it could not translate."""

    strategy: str
    ok: bool = True
    steps: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    error: str | None = None

    def step(self, name: str, command: str, exit_code: int, output: str) -> None:
        self.steps.append(
            {
                "name": name,
                "command": command,
                "exit_code": exit_code,
                "output": output[-4000:],
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "ok": self.ok,
            "steps": self.steps,
            "skipped": self.skipped,
            "error": self.error,
        }


class Runtime(Protocol):
    """Everything a harness needs from an execution substrate."""

    task: dict[str, Any]
    logger: logging.Logger

    def get_task_instruction(self) -> str: ...

    def run(
        self, command: str, *, timeout: float = 120, workdir: str | None = None
    ) -> tuple[str, int]: ...

    def copy_to_container(self, source: str | Path, destination: str) -> None: ...

    def get_patch(self) -> str: ...

    def compute_reward(self) -> tuple[float, str]: ...

    def recreate(self) -> None: ...

    def close(self) -> None: ...


class CommandExecutor:
    """Runs shell commands with file-backed, checksummed output on the host."""

    def __init__(self, state_dir: Path, *, logger: logging.Logger) -> None:
        self.state_dir = state_dir
        self.logger = logger
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        command: str,
        *,
        workdir: Path,
        timeout: float,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        if not isinstance(command, str) or "\0" in command:
            raise ValueError("command must be a string without NUL bytes")
        if timeout <= 0:
            raise ValueError("command timeout must be positive")
        directory = self.state_dir / f"cmd-{time.time_ns():x}-{os.getpid():x}"
        directory.mkdir(parents=True, exist_ok=True)
        script = directory / "command.sh"
        script.write_text(command, encoding="utf-8")
        output_path = directory / "output"
        result_path = directory / "result"
        # The command runs in a subshell so an `exit` or `set -e` inside it
        # cannot skip writing the completion record, and `cd` first so the
        # command inherits the requested workdir.
        wrapper = (
            f"cd -- {_shell_quote(str(workdir))} || exit 97\n"
            f"( . {_shell_quote(str(script))} ) > {_shell_quote(str(output_path))} 2>&1\n"
            "code=$?\n"
            # macOS ships shasum, most Linux images ship sha256sum; fall back to
            # openssl so the checksum step works on either host.
            "if command -v sha256sum >/dev/null 2>&1; then\n"
            f"  digest=$(sha256sum < {_shell_quote(str(output_path))} | cut -d' ' -f1)\n"
            "elif command -v shasum >/dev/null 2>&1; then\n"
            f"  digest=$(shasum -a 256 < {_shell_quote(str(output_path))} | cut -d' ' -f1)\n"
            "else\n"
            f"  digest=$(openssl dgst -sha256 < {_shell_quote(str(output_path))} | cut -d' ' -f2)\n"
            "fi\n"
            f"size=$(wc -c < {_shell_quote(str(output_path))} | tr -d ' ')\n"
            f"printf '%s %s %s\\n' \"$code\" \"$size\" \"$digest\" > {_shell_quote(str(result_path))}\n"
        )
        started = time.monotonic()
        timed_out = False
        try:
            process = subprocess.run(
                ["/bin/bash", "-c", wrapper],
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                start_new_session=True,
            )
            # The wrapper itself must succeed; anything else means the host shell
            # could not run the command at all, which is not the same as the
            # command failing, so surface it instead of a bogus exit code.
            if process.returncode != 0 and not result_path.is_file():
                raise PodExecutionError(
                    f"host wrapper exited with code {process.returncode} in "
                    f"{workdir}: {(process.stderr or '').strip()[-2000:]}"
                )
        except subprocess.TimeoutExpired:
            timed_out = True
        duration = time.monotonic() - started
        output = self._read_output(directory, output_path, result_path)
        if timed_out:
            # The wrapper was killed, so there may be no completion record; the
            # timeout is the outcome, not a missing exit code.
            raise CommandTimeoutError(
                f"command timed out after {timeout:g}s in {workdir}\n"
                f"Output:\n{output[-2000:]}",
                output=output,
            )
        exit_code = self._read_exit_code(result_path)
        return CommandResult(
            output=output, exit_code=exit_code, duration_sec=duration
        )

    def _read_output(
        self, directory: Path, output_path: Path, result_path: Path
    ) -> str:
        """Read captured output, verifying the checksum recorded by the shell."""
        if not output_path.is_file():
            return ""
        recorded = _read_result(result_path)
        for attempt in range(_READ_ATTEMPTS):
            data = output_path.read_bytes()
            if recorded is None or recorded["sha256"] == hashlib.sha256(data).hexdigest():
                return data.decode("utf-8", errors="replace")
            self.logger.warning(
                "Command output checksum mismatch on read %s/%s: %s",
                attempt + 1,
                _READ_ATTEMPTS,
                directory,
            )
            time.sleep(0.05)
        raise PodExecutionError(
            f"command output failed SHA-256 verification in {directory}; "
            "the command will not be re-executed"
        )

    @staticmethod
    def _read_exit_code(result_path: Path) -> int:
        recorded = _read_result(result_path)
        if recorded is not None:
            return recorded["exit_code"]
        # The wrapper died before writing a result. Record that as an execution
        # failure rather than inventing an exit code that looks like the command
        # itself failed.
        output_path = result_path.parent / "output"
        detail = (
            output_path.read_text(encoding="utf-8", errors="replace")[-500:]
            if output_path.is_file()
            else ""
        )
        raise PodExecutionError(
            f"command wrapper produced no completion record in {result_path.parent}"
            + (f": {detail}" if detail else "")
        )


def _read_result(result_path: Path) -> dict[str, Any] | None:
    if not result_path.is_file():
        return None
    parts = result_path.read_text(encoding="utf-8").split()
    if len(parts) != 3:
        return None
    try:
        exit_code, size, digest = int(parts[0]), int(parts[1]), parts[2]
    except ValueError:
        return None
    return {"exit_code": exit_code, "size": size, "sha256": digest}


def _shell_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def kill_process_tree(process: subprocess.Popen[Any]) -> None:
    """Terminate a harness subprocess and everything it started."""
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive
        pass


def copy_into(source: str | Path, destination: str | Path) -> None:
    """Copy a file or directory tree into the workspace.

    Directories are copied by *contents*, matching the reference runtime where
    ``copy_to_container(tests_dir, "/tests")`` places ``tests/*`` under ``/tests``.
    """
    source_path = Path(source).expanduser()
    target = Path(destination).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    if source_path.is_dir():
        shutil.copytree(source_path, target, dirs_exist_ok=True, symlinks=True)
        return
    shutil.copy2(source_path, target)