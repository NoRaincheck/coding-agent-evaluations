"""Host task runtime.

One task plus one seed becomes one workspace on the machine running the
evaluation. It provides the same surface as
``frognano.runtimes.kubernetes.KubernetesTaskRuntime`` so the upstream leaf
harness, its tool runner and its patch capture work unchanged; only pod creation
is replaced by directory provisioning and only ``kubectl exec`` is replaced by a
local shell.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shlex
import shutil
import uuid
from pathlib import Path
from typing import Any

from frognano.runtimes.errors import CommandTimeoutError

from .base import CommandExecutor, ProvisionRecord, SandboxPaths, copy_into
from .provision import (
    ProvisionError,
    build_provisioner,
    default_options,
    rewrite_tree,
)
from .translate import env_overrides, iter_supported, translate_verifier


class HostTaskRuntime:
    """Runs one benchmark task in a local workspace directory."""

    def __init__(
        self,
        task: dict[str, Any],
        config: Any,
        *,
        run_id: str,
        root: Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.task = task
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.run_id = run_id
        base = Path(root or config.root).expanduser()
        run_dir = base / self._slug(str(task["instance_id"])) / _run_dir(run_id)
        self.paths = SandboxPaths(
            root=run_dir,
            workspace=run_dir / "workspace",
            tests=run_dir / "tests",
            logs=run_dir / "logs",
            solution=run_dir / "solution",
            state=run_dir / "state",
        )
        self.paths.ensure()
        self.executor = CommandExecutor(self.paths.state, logger=self.logger)
        options = {**default_options(), **dict(config.options or {})}
        self._provision_options = options
        self._venv_bin: Path | None = None
        self.provision_record: ProvisionRecord = ProvisionRecord(strategy="unknown")
        self.verifier_translation: dict[str, Any] = {}
        self.unsupported: list[str] = list(iter_supported(task))
        self._prepare()

    # -- lifecycle ----------------------------------------------------------

    def _prepare(self) -> None:
        provisioner = build_provisioner(
            self.paths, self.task, self._provision_options
        )
        try:
            self.provision_record = provisioner.provision()
            self._venv_bin = provisioner.venv_bin
        except ProvisionError as exc:
            self.provision_record.ok = False
            self.provision_record.error = str(exc)
            provisioner._write_record()
            raise
        # Keep a pristine copy so pod-style recovery can restore and replay.
        self._snapshot_pristine()
        self.logger.info(
            "Workspace ready at %s (strategy=%s)",
            self.paths.workspace,
            self.provision_record.strategy,
        )

    def _snapshot_pristine(self) -> None:
        pristine = self.paths.root / "pristine"
        shutil.rmtree(pristine, ignore_errors=True)
        shutil.copytree(self.paths.workspace, pristine, symlinks=True)

    def recreate(self) -> None:
        """Restore the pristine workspace so mutating tools can be replayed.

        The reference runtime recreates a pod and replays the agent's mutating
        tool calls. On the host the equivalent is a fresh copy of the
        provisioned workspace.
        """
        pristine = self.paths.root / "pristine"
        workspace = self.paths.workspace
        if not pristine.is_dir():
            self.logger.warning("No pristine snapshot for %s", workspace)
            return
        self.logger.warning("Restoring pristine workspace at %s", workspace)
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.copytree(pristine, workspace, symlinks=True)

    def close(self) -> None:
        if getattr(self.config, "keep_workspaces", False):
            self.logger.info("Keeping workspace %s", self.paths.root)
            return
        shutil.rmtree(self.paths.root, ignore_errors=True)

    # -- execution ----------------------------------------------------------

    def get_task_instruction(self) -> str:
        return str(self.task["instruction"])

    def run(
        self,
        command: str,
        *,
        timeout: float = 120,
        workdir: str | None = None,
    ) -> tuple[str, int]:
        if not isinstance(command, str) or "\0" in command:
            raise ValueError("command must be a string without NUL bytes")
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("command timeout must be positive and finite")
        directory = workdir or str(self.task["repo_path"])
        host_workdir = self._resolve_host_path(directory)
        environment = self._environment()
        try:
            result = self.executor.run(
                command, workdir=host_workdir, timeout=timeout, env=environment
            )
        except CommandTimeoutError as exc:
            raise CommandTimeoutError(
                f"{exc}\nworkspace: {self.paths.root}",
                output=exc.output,
            ) from exc
        return result.output, result.exit_code

    def copy_to_container(
        self, source: str | Path, destination: str, *, timeout: int = 300
    ) -> None:
        """Copy a file or tree into the workspace (upstream-compatible name)."""
        target = self._resolve_host_path(destination)
        copy_into(source, target)

    def read_file(self, relative: str) -> str:
        path = self.paths.root / relative
        return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""

    def exists(self, relative: str) -> bool:
        return (self.paths.root / relative).is_file()

    # -- grading ------------------------------------------------------------

    def get_patch(self) -> str:
        if not self._has_git_checkout():
            return ""
        repo = str(self.paths.workspace)
        output, exit_code = self.run(
            f"git -C {shlex.quote(repo)} add -A 2>/dev/null && "
            f"git -C {shlex.quote(repo)} diff --cached --binary 2>/dev/null; "
            "rc=$?; git -C " + shlex.quote(repo) + " reset --mixed HEAD >/dev/null 2>&1; "
            "exit $rc",
            workdir="/",
        )
        if exit_code != 0:
            raise RuntimeError(f"failed to capture patch: {output[-2000:]}")
        return output

    def _has_git_checkout(self) -> bool:
        if self.task.get("require_git_patch", True):
            return True
        output, exit_code = self.run(
            "command -v git >/dev/null 2>&1 && "
            f"git -C {shlex.quote(str(self.paths.workspace))} "
            "rev-parse --is-inside-work-tree",
            timeout=30,
            workdir="/",
        )
        if exit_code == 0 and output.strip() == "true":
            return True
        self.logger.info(
            "Skipping optional Git artifact (probe exit %s): %s",
            exit_code,
            output[-1000:],
        )
        return False

    def compute_reward(self) -> tuple[float, str]:
        """Run the benchmark verifier in the workspace and read its reward.

        Mirrors the reference runtime: reset the verifier directories, stage the
        task's ``tests`` tree (translated for host paths), keep agent-created
        files by staging them first, run ``tests/test.sh``, then read
        ``logs/verifier/reward.{json,txt}``.
        """
        tests_source = self.task.get("tests_dir")
        if not tests_source:
            raise RuntimeError("task has no verifier to run")
        shutil.rmtree(self.paths.tests, ignore_errors=True)
        shutil.rmtree(self.paths.logs / "verifier", ignore_errors=True)
        (self.paths.logs / "verifier").mkdir(parents=True, exist_ok=True)
        if self._has_git_checkout():
            output, exit_code = self.run("git add -A", timeout=120)
            if exit_code != 0:
                raise RuntimeError(f"could not stage agent changes: {output[-2000:]}")
        copy_into(tests_source, self.paths.tests)
        # Verifier files name container paths absolutely, exactly as they do in
        # the image; without this a correct patch could never pass.
        rewritten_files = rewrite_tree(
            self.paths.tests, self._container_path_mapping()
        )
        script = self.paths.tests / "test.sh"
        translation = translate_verifier(script, self.paths, self.task)
        script.write_text(translation.text, encoding="utf-8")
        (self.paths.tests / "test.sh.host-translation").write_text(
            json.dumps(translation.as_dict(), indent=2) + "\n", encoding="utf-8"
        )
        self.verifier_translation = translation.as_dict()
        self.verifier_translation["rewritten_files"] = rewritten_files
        self.logger.info(
            "Running verifier: %s substitution(s), %s image-only command(s) neutralized",
            len(translation.substitutions),
            len(translation.skipped),
        )
        # `test.sh` cds to the repo and then refers to the tests directory
        # absolutely, so run it from the tests directory with an explicit path
        # rather than assuming a container-relative layout.
        output, _ = self.run(
            f"bash {shlex.quote(str(script))}",
            timeout=int(self.task.get("verifier_timeout_sec", 600)),
            workdir=str(self.paths.workspace),
        )
        (self.paths.logs / "verifier" / "output.log").write_text(
            output, encoding="utf-8"
        )
        marker = self.task.get("verifier_success_marker")
        if marker and marker not in output:
            raise RuntimeError(
                "verifier exited before producing its completion marker; see "
                f"{self.paths.logs / 'verifier' / 'output.log'}"
            )
        return self._read_reward(output)

    def _read_reward(self, verifier_output: str) -> tuple[float, str]:
        for name in ("reward.json", "reward.txt"):
            path = self.paths.logs / "verifier" / name
            if not path.is_file():
                continue
            raw = path.read_text(encoding="utf-8", errors="replace").strip()
            if not raw:
                continue
            if name.endswith(".json"):
                payload = json.loads(raw)
                value = payload.get("reward") if isinstance(payload, dict) else payload
                if value is None and isinstance(payload, dict) and len(payload) == 1:
                    value = next(iter(payload.values()))
            else:
                value = raw
            reward = float(value)
            if reward < 0:
                raise RuntimeError(f"verifier returned negative reward {reward}")
            return reward, verifier_output
        raise RuntimeError(
            "verifier did not write a reward; see "
            f"{self.paths.logs / 'verifier' / 'output.log'}"
        )

    # -- helpers ------------------------------------------------------------

    def _environment(self) -> dict[str, str]:
        environment = {**os.environ}
        environment.update(
            {
                "CAE_INSTANCE_ID": str(self.task.get("instance_id") or ""),
                "CAE_DATASET": str(self.task.get("dataset") or ""),
                "CAE_WORKSPACE": str(self.paths.workspace),
                "CAE_TESTS": str(self.paths.tests),
                "CAE_LOGS": str(self.paths.logs),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        environment.update(env_overrides(self.paths, self.task, venv_bin=self._venv_bin))
        return environment

    def _container_path_mapping(self) -> dict[str, str]:
        """Container-absolute prefixes that resolve inside the workspace.

        ``/tmp`` is deliberately absent: the reference leaf environment hardcodes
        ``/tmp/frognano_leaf_tool_runner.py`` in both the copy destination and
        the command it then runs, so the two must agree. The host ``/tmp`` is
        writable and those paths are uuid-scoped.
        """
        return {
            str(self.task.get("repo_path") or "/app"): str(self.paths.workspace),
            "/testbed": str(self.paths.workspace),
            "/app": str(self.paths.workspace),
            "/workspace": str(self.paths.workspace),
            "/src": str(self.paths.workspace),
            "/tests": str(self.paths.tests),
            "/logs/verifier": str(self.paths.logs / "verifier"),
            "/logs": str(self.paths.logs),
            "/solution": str(self.paths.solution),
        }

    def _resolve_host_path(self, value: str) -> Path:
        """Map a container-absolute path used by a task onto the workspace."""
        mapping = self._container_path_mapping()
        for original, replacement in sorted(mapping.items(), key=lambda i: -len(i[0])):
            if value == original or value.startswith(original + "/"):
                candidate = Path(replacement + value[len(original) :])
                candidate.mkdir(parents=True, exist_ok=True)
                return candidate
        candidate = Path(value)
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    @staticmethod
    def _slug(instance_id: str) -> str:
        safe = "".join(
            character if character.isalnum() or character in "-_." else "-"
            for character in instance_id
        )
        return f"{safe[:80]}-{uuid.uuid5(uuid.NAMESPACE_URL, instance_id).hex[:8]}"


def _run_dir(run_id: str) -> str:
    """Stable workspace directory name for a run id (e.g. ``0-1-ab12cd``)."""
    return "".join(
        character if character.isalnum() or character in "-." else "-"
        for character in run_id
    )


__all__ = ["HostTaskRuntime"]