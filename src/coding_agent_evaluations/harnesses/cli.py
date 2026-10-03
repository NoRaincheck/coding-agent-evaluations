"""Run a harness CLI as a subprocess in the task workspace.

opencode and pi are full agent CLIs: they own their own system prompt, tool set,
compaction and context management. Driving them means running the real CLI, not
reimplementing their loop, so the measurement stays honest about what the
harness does.

Two rules keep a run reproducible:

* a hermetic config directory per rollout, so the machine's global agent
  configuration cannot leak into the measurement,
* the harness is killed on the run's time budget, and its output is parsed for
  steps, tokens and the final message.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from frognano.runtimes.errors import CommandTimeoutError

from .base import HarnessRun, empty_trajectory

# Grace period between the wall-clock budget expiring and SIGTERM, then SIGKILL.
_TERM_GRACE_SEC = 20


class CliHarness:
    """Shared plumbing for harnesses that are separate CLIs."""

    name = "cli"

    def __init__(self) -> None:
        self.binary: str | None = None

    # -- subclass contract --------------------------------------------------

    def build_command(
        self, run: HarnessRun, config_dir: Path, session_id: str
    ) -> list[str]:
        raise NotImplementedError

    def build_env(
        self, run: HarnessRun, config_dir: Path, session_id: str
    ) -> dict[str, str]:
        raise NotImplementedError

    def parse(self, run: HarnessRun, stdout: str, exit_code: int) -> dict[str, Any]:
        raise NotImplementedError

    def resolve_binary(self) -> str:
        raise NotImplementedError

    def session_id(self, run: HarnessRun) -> str:
        """Stable per-rollout session id."""
        raise NotImplementedError

    # -- shared execution ---------------------------------------------------

    def run(self, run: HarnessRun) -> dict[str, Any]:
        try:
            self.binary = self.resolve_binary()
        except FileNotFoundError as exc:
            trajectory = empty_trajectory(run, reason="harness_unavailable")
            trajectory["error"] = str(exc)
            return trajectory

        run.workspace.mkdir(parents=True, exist_ok=True)
        config_dir = run.runtime.paths.state / f"harness-{self.name}"
        config_dir.mkdir(parents=True, exist_ok=True)
        session_id = self.session_id(run)
        command = self.build_command(run, config_dir, session_id)
        environment = self.build_env(run, config_dir, session_id)
        timeout = run.max_total_time_sec
        started = time.monotonic()
        exit_code, stdout, stderr, timed_out = self._execute(
            command, run.workspace, environment, timeout
        )
        trajectory = self.parse(run, stdout, exit_code)
        trajectory.setdefault("harness", self.name)
        trajectory["trajectory_format"] = f"{self.name}-cli"
        trajectory["instance_id"] = run.instance_id
        trajectory["seed"] = run.seed
        trajectory["elapsed_sec"] = time.monotonic() - started
        trajectory["partial"] = False
        trajectory["cli_command"] = command
        if stderr.strip():
            trajectory["stderr_tail"] = stderr[-4000:]
        if timed_out:
            # Reaching the budget is an outcome, not a crash: report it the way
            # leaf reports its own limit so the two are comparable.
            trajectory["exit_reason"] = "max_time"
            trajectory["error"] = None
            trajectory["timed_out"] = True
        elif exit_code != 0:
            trajectory["timed_out"] = False
        (config_dir / "stdout.log").write_text(stdout, encoding="utf-8")
        (config_dir / "stderr.log").write_text(stderr, encoding="utf-8")
        return trajectory

    def _execute(
        self,
        command: list[str],
        cwd: Path,
        environment: dict[str, str],
        timeout: int | None,
    ) -> tuple[int, str, str, bool]:
        """Run the CLI, streaming its output, and enforce the time budget."""
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        chunks: list[str] = []
        error_chunks: list[str] = []
        timed_out = False

        def drain(stream: Any, sink: list[str]) -> None:
            assert stream is not None
            sink.extend(iter(stream.readline, ""))

        readers = [
            threading.Thread(target=drain, args=(process.stdout, chunks), daemon=True),
            threading.Thread(
                target=drain, args=(process.stderr, error_chunks), daemon=True
            ),
        ]
        for reader in readers:
            reader.start()
        while process.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                self._terminate(process)
                break
            time.sleep(0.2)
        for reader in readers:
            reader.join(timeout=5)
        exit_code = process.poll()
        if exit_code is None:
            exit_code = process.wait(timeout=30)
        return (
            int(exit_code),
            "".join(chunks),
            "".join(error_chunks),
            timed_out,
        )

    @staticmethod
    def _terminate(process: subprocess.Popen[Any]) -> None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            process.terminate()
        try:
            process.wait(timeout=_TERM_GRACE_SEC)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            pass


def parse_jsonl(text: str) -> list[dict[str, Any]]:
    """Parse a JSONL event stream, ignoring anything that is not an object."""
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] != "{":
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def hermetic_env(base: dict[str, str], config_dir: Path) -> dict[str, str]:
    """Environment with caches and state redirected into the workspace.

    Also pins offline tokenizers: a run must never reach out to Hugging Face.
    """
    environment = dict(base)
    for name, value in (
        ("HF_HUB_OFFLINE", "1"),
        ("TRANSFORMERS_OFFLINE", "1"),
        ("PI_OFFLINE", "1"),
        ("PI_SKIP_VERSION_CHECK", "1"),
        ("PI_TELEMETRY", "0"),
        ("DO_NOT_TRACK", "1"),
        ("NO_COLOR", "1"),
        ("CI", "1"),
        ("TERM", "dumb"),
    ):
        environment.setdefault(name, value)
    environment["XDG_CACHE_HOME"] = str(config_dir / "xdg-cache")
    environment["XDG_CONFIG_HOME"] = str(config_dir / "xdg-config")
    environment["XDG_DATA_HOME"] = str(config_dir / "xdg-data")
    environment["XDG_STATE_HOME"] = str(config_dir / "xdg-state")
    environment["HOME"] = str(config_dir / "home")
    Path(environment["HOME"]).mkdir(parents=True, exist_ok=True)
    return environment


__all__ = ["CliHarness", "CommandTimeoutError", "hermetic_env", "parse_jsonl"]