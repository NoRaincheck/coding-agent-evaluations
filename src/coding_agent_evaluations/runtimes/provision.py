"""Provision a task workspace on the host.

The benchmarks assume a prebuilt image that already contains the repository at a
pinned commit, its dependencies and a pinned interpreter. Without a container
runtime that state has to be reconstructed per task. This module does the two
things both benchmarks need:

* **SWE-bench Verified** — clone ``repo`` at ``base_commit`` (the metadata is in
  ``tests/config.json``) and install the repo plus its test extra into a
  workspace virtual environment.
* **Terminal-Bench 2 Verified** — translate the task ``Dockerfile`` into host
  steps: pick the interpreter named by the base image, then replay the
  ``pip``/``uv``/filesystem instructions and log whatever could not be
  translated (notably ``apt-get``).

Anything that cannot be reproduced is recorded, and a task that cannot be
provisioned at all fails fast with ``provision_error`` instead of silently
scoring zero for the agent.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import ProvisionRecord, SandboxPaths
from .translate import rewrite_paths

_UV = shutil.which("uv") or "uv"

# System package management has no host equivalent here.
_DROPPED_COMMAND_RE = re.compile(
    r"(^|[;&|]\s*)(apt-get|apt|apk|yum|dnf|microdnf|dpkg)\b"
)

# Only small text payloads are scanned for container paths.
_MAX_REWRITTEN_FILE_BYTES = 1 << 20

_PIP_INSTALL_RE = re.compile(
    r"^(?:python3?\s+-m\s+)?pip3?\s+(?P<args>(?:install|uninstall)\b.*)$"
)


def _quote(value: Any) -> str:
    return shlex.quote(str(value))

# Kept out of generated patches.
_PROVISION_IGNORE = "\n# added by coding-agent-evaluations provisioning\n/.venv/\n/.cae/\n"
_BASE_PYTHON_RE = re.compile(r"(?:^|/)python:(\d+\.\d+)")
_UBUNTU_PYTHON = {"24.04": "3.12", "22.04": "3.11", "20.04": "3.10"}
_DEBIAN_PYTHON = {"trixie": "3.13", "forky": "3.14", "bookworm": "3.11", "bullseye": "3.10"}


class ProvisionError(RuntimeError):
    """The workspace could not be prepared on the host."""


@dataclass
class ParsedDockerfile:
    """The parts of a task Dockerfile that can be replayed on the host."""

    base_image: str
    workdir: str
    environment: dict[str, str]
    steps: list[tuple[str, str]]
    ignored: list[str]


def parse_dockerfile(path: Path) -> ParsedDockerfile:
    """Parse a task Dockerfile tolerantly, keeping instruction order.

    Unlike the strict parser in the reference implementation this one keeps
    going on instructions a host workspace cannot use (``EXPOSE``, ``USER``) and
    reports them instead of failing, because a task is often still gradable
    without them. ``COPY`` is replayed against the Dockerfile's own directory,
    which is the task's environment payload.
    """
    base_image = ""
    workdir = "/app"
    environment: dict[str, str] = {}
    steps: list[tuple[str, str]] = []
    ignored: list[str] = []
    for instruction, value in _dockerfile_instructions(path.read_text(encoding="utf-8")):
        if instruction == "FROM" and not base_image:
            parts = value.split()
            if len(parts) >= 4 and parts[-2] == "as":
                ignored.append(f"multi-stage FROM ignored: {value}")
            base_image = parts[0]
        elif instruction == "WORKDIR":
            workdir = value.strip()
            steps.append(("workdir", workdir))
        elif instruction == "ENV":
            key, separator, item = value.partition("=")
            if separator:
                environment[key.strip()] = item.strip()
        elif instruction in {"COPY", "ADD"}:
            steps.append(("copy", value))
        elif instruction == "RUN":
            steps.append(("run", value))
        else:
            ignored.append(f"{instruction} ignored: {value[:80]}")
    if not base_image:
        raise ProvisionError(f"task Dockerfile has no FROM image: {path}")
    return ParsedDockerfile(
        base_image=base_image,
        workdir=workdir,
        environment=environment,
        steps=steps,
        ignored=ignored,
    )


def _dockerfile_instructions(text: str) -> list[tuple[str, str]]:
    logical: list[str] = []
    current = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        current = f"{current} {line}".strip()
        if current.endswith("\\"):
            current = current[:-1].rstrip()
            continue
        logical.append(current)
        current = ""
    instructions: list[tuple[str, str]] = []
    for line in logical:
        parts = line.split(None, 1)
        if len(parts) == 2:
            instructions.append((parts[0].upper(), parts[1]))
    return instructions


@dataclass
class Provisioner:
    """Builds a workspace for one task on the host."""

    paths: SandboxPaths
    task: dict[str, Any]
    options: dict[str, Any]
    record: ProvisionRecord

    # -- public API ---------------------------------------------------------

    def provision(self) -> ProvisionRecord:
        dataset = str(self.task.get("dataset") or "")
        if dataset.startswith("swebench"):
            self._provision_swebench()
        elif dataset.startswith("terminal_bench"):
            self._provision_terminal_bench()
        else:
            raise ProvisionError(
                f"no host provisioner for dataset {dataset!r}; "
                "this runtime covers swebench_verified and terminal_bench_2_verified"
            )
        self._write_record()
        return self.record

    # -- SWE-bench ----------------------------------------------------------

    def _provision_swebench(self) -> None:
        tests_dir = self.task.get("tests_dir")
        if not tests_dir:
            raise ProvisionError("SWE-bench task has no tests directory")
        config_path = Path(str(tests_dir)) / "config.json"
        if not config_path.is_file():
            raise ProvisionError(f"missing SWE-bench metadata: {config_path}")
        metadata = json.loads(config_path.read_text(encoding="utf-8"))
        repo = str(metadata.get("repo") or "")
        base_commit = str(metadata.get("base_commit") or "")
        if not repo or not base_commit:
            raise ProvisionError("SWE-bench metadata lacks repo/base_commit")
        self.task["repo"] = repo
        self.task["base_commit"] = base_commit
        self._clone(repo, base_commit)
        self._create_venv(str(self.options.get("python") or self._swebench_python()))
        extras = self.options.get("swebench_install_args")
        if extras is None:
            extras = ".[test]"
        self._pip_install(f"-e {extras}" if extras else ".")
        self.record.step(
            "metadata",
            json.dumps({"repo": repo, "base_commit": base_commit}),
            0,
            "",
        )

    def _clone(self, repo: str, base_commit: str) -> None:
        workspace = self.paths.workspace
        workspace.mkdir(parents=True, exist_ok=True)
        url = self.options.get("git_url_template")
        remote = (
            str(url).format(repo=repo) if url else f"https://github.com/{repo}.git"
        )
        commands = [
            ["git", "init", "--quiet"],
            ["git", "remote", "add", "origin", remote],
            ["git", "fetch", "--quiet", "--depth", "1", "origin", base_commit],
            ["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"],
        ]
        for command in commands:
            output, exit_code = self._sh_retry(command, workdir=workspace)
            if exit_code != 0:
                raise ProvisionError(
                    f"{' '.join(command)} failed with {exit_code}: {output[-1500:]}"
                )
        # Provisioning artifacts live outside the repository, but a stray file in
        # the checkout would still land in the generated patch.
        exclude = workspace / ".git" / "info" / "exclude"
        try:
            existing = exclude.read_text(encoding="utf-8") if exclude.is_file() else ""
            exclude.write_text(
                existing + _PROVISION_IGNORE, encoding="utf-8"
            )
        except OSError:  # pragma: no cover - defensive
            pass

    def _sh_retry(
        self, command: list[str], *, workdir: Path
    ) -> tuple[str, int]:
        """Retry a provisioning command; clones are network-bound and flaky."""
        attempts = int(self.options.get("network_retries", 3))
        output, exit_code = self._sh(command, workdir=workdir)
        for attempt in range(2, attempts + 1):
            if exit_code == 0:
                return output, exit_code
            self.record.step(
                "retry",
                f"{' '.join(command)} (attempt {attempt})",
                exit_code,
                output,
            )
            time.sleep(min(2 ** (attempt - 1) * 2, 15))
            output, exit_code = self._sh(command, workdir=workdir)
        return output, exit_code

    def _swebench_python(self) -> str:
        """SWE-bench images pin the interpreter per repository revision."""
        overrides = dict(self.options.get("python_by_repo") or {})
        repo = str(self.task.get("repo") or "")
        if repo in overrides:
            return str(overrides[repo])
        return str(self.options.get("python") or "3.9")

    # -- Terminal-Bench -----------------------------------------------------

    def _provision_terminal_bench(self) -> None:
        dockerfile = Path(str(self.task.get("tests_dir") or "")).parent
        dockerfile = dockerfile / "environment" / "Dockerfile"
        if not dockerfile.is_file():
            raise ProvisionError(f"missing Terminal-Bench Dockerfile: {dockerfile}")
        parsed = parse_dockerfile(dockerfile)
        self.record.strategy = "dockerfile-translation"
        self.task["host_base_image"] = parsed.base_image
        protected = sorted(dockerfile.parent.glob("protected*"))
        if protected:
            raise ProvisionError(
                "task ships an encrypted environment payload "
                f"({protected[0].name}); it can only be materialized by the "
                "benchmark image"
            )
        python = self._terminal_bench_python(parsed.base_image)
        self.task["host_python"] = python
        self.task["repo_path"] = parsed.workdir or str(self.paths.workspace)
        self._create_venv(python)
        for kind, value in parsed.steps:
            if kind == "copy":
                self._replay_copy(value, source_root=dockerfile.parent)
                continue
            if kind == "workdir":
                continue
            translated = self._translate_setup_command(value)
            if translated is None:
                self.record.skipped.append(value)
                continue
            output, exit_code = self._sh(
                translated, workdir=self.paths.workspace, env=self._venv_env()
            )
            self.record.step("setup", translated, exit_code, output)
            if exit_code != 0 and _is_required(translated):
                raise ProvisionError(
                    f"setup command failed ({exit_code}): {translated}\n"
                    f"{output[-1500:]}"
                )
        self.record.skipped.extend(parsed.ignored)

    def _replay_copy(self, value: str, *, source_root: Path) -> None:
        """Replay a Dockerfile ``COPY`` from the task's environment directory."""
        tokens = [
            token
            for token in re.split(r"\s+", value.strip().strip('"').strip("'"))
            if token
        ]
        if len(tokens) < 2:
            self.record.skipped.append(f"COPY {value}")
            return
        *sources, destination = tokens
        mapping = self._container_paths()
        rewritten_sources = [
            rewrite_paths(_strip_copy_flag(source), mapping)[0] for source in sources
        ]
        target = rewrite_paths(_strip_copy_flag(destination), mapping)[0]
        for source in rewritten_sources:
            origin = Path(source)
            if not origin.is_absolute():
                origin = source_root / source
            if not origin.exists():
                self.record.skipped.append(f"COPY {value} (missing {source})")
                continue
            destination_path = self._copy_destination(
                origin, target, sources=rewritten_sources
            )
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            if origin.is_dir():
                shutil.copytree(origin, destination_path, dirs_exist_ok=True)
                for copied in sorted(destination_path.rglob("*")):
                    if copied.is_file():
                        self._rewrite_paths_in_place(copied, mapping)
            else:
                shutil.copy2(origin, destination_path)
                self._rewrite_paths_in_place(destination_path, mapping)
            self.record.step(
                "copy", f"{origin} -> {destination_path}", 0, "copied"
            )

    @staticmethod
    def _copy_destination(
        origin: Path, target: str, *, sources: list[str]
    ) -> Path:
        """Resolve where a COPY lands, following Docker's directory rules."""
        destination = Path(target)
        multiple = len(sources) > 1
        if multiple or origin.is_dir() or target.endswith("/"):
            destination.mkdir(parents=True, exist_ok=True)
            return destination / origin.name
        return destination

    def _rewrite_paths_in_place(self, path: Path, mapping: dict[str, str]) -> None:
        change = rewrite_file(path, mapping)
        if change:
            self.record.step("rewrite-paths", str(path), 0, change)

    def _container_paths(self) -> dict[str, str]:
        """Container path prefixes that map onto the workspace."""
        mapping = {
            str(self.task.get("repo_path") or "/app"): str(self.paths.workspace),
        }
        for alias in ("/app", "/workspace", "/testbed", "/src", "/opt/app"):
            mapping.setdefault(alias, str(self.paths.workspace))
        return mapping

    def _terminal_bench_python(self, base_image: str) -> str:
        overrides = dict(self.options.get("python_by_base_image") or {})
        if base_image in overrides:
            return str(overrides[base_image])
        match = _BASE_PYTHON_RE.search(base_image)
        if match:
            return match.group(1)
        for tag, version in _UBUNTU_PYTHON.items():
            if base_image.startswith(f"ubuntu:{tag}"):
                return version
        for tag, version in _DEBIAN_PYTHON.items():
            if base_image.startswith(f"debian:{tag}"):
                return version
        return str(self.options.get("python") or "3.12")

    def _translate_setup_command(self, command: str) -> str | None:
        """Translate one Dockerfile ``RUN`` into a host command.

        Two kinds of instruction cannot be replayed and are dropped with a log
        entry: system package managers (there is no ``apt-get`` on the host) and
        instructions that only make sense inside a build stage.
        """
        normalized = command.strip()
        if not normalized:
            return None
        if _DROPPED_COMMAND_RE.search(normalized):
            return None
        if normalized.startswith(("ENV ", "ARG ", "COPY ", "ADD ", "EXPOSE ", "USER ")):
            return None
        # Container paths become workspace paths, so `python3 /app/x.py` finds
        # the file the image would have had in its WORKDIR.
        rewritten, _ = rewrite_paths(normalized, self._container_paths())
        return self._translate_pip(rewritten)

    def _translate_pip(self, command: str) -> str:
        """Route ``pip install`` into the workspace environment.

        ``uv venv`` does not install ``pip``, and a bare ``pip install`` on the
        host would target the system interpreter instead of the task's. The image
        means "install into the task environment", so the request is forwarded to
        the interpreter the task actually runs on.
        """
        match = _PIP_INSTALL_RE.match(command)
        if match is None:
            return command
        arguments = match.group("args").strip()
        if not arguments:
            return command
        interpreter = self.paths.root / "venv" / "bin" / "python"
        return f"{_UV} pip install --python {_quote(interpreter)} {arguments}"

    # -- shared -------------------------------------------------------------

    def _create_venv(self, python: str) -> Path:
        venv = self.paths.root / "venv"
        command = [_UV, "venv", "--python", python, str(venv)]
        output, exit_code = self._sh(command, workdir=self.paths.workspace)
        self.record.step(
            "venv", " ".join(shlex.quote(part) for part in command), exit_code, output
        )
        if exit_code != 0:
            raise ProvisionError(
                f"could not create a Python {python} environment: {output[-1500:]}"
            )
        # Shim directory outside the repository: provisioning artifacts must never
        # show up in a generated patch.
        shim = self.paths.root / "bin"
        shim.mkdir(parents=True, exist_ok=True)
        for tool in ("python", "python3", "pip", "pytest"):
            target = venv / "bin" / tool
            if target.exists():
                link = shim / tool
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(target)
        return venv

    def _pip_install(self, target: str) -> None:
        venv = self.paths.root / "venv"
        command = [
            _UV,
            "pip",
            "install",
            "--python",
            str(venv / "bin" / "python"),
            target,
        ]
        output, exit_code = self._sh(command, workdir=self.paths.workspace)
        self.record.step(
            "install", " ".join(shlex.quote(part) for part in command), exit_code, output
        )
        if exit_code != 0:
            raise ProvisionError(
                f"dependency installation failed ({target}): {output[-1500:]}"
            )

    @property
    def venv_bin(self) -> Path | None:
        candidate = self.paths.root / "venv" / "bin"
        return candidate if candidate.is_dir() else None

    def _sh(
        self, command: str | list[str], *, workdir: Path, env: dict[str, str] | None = None
    ) -> tuple[str, int]:
        """Run a provisioning command, either argv-style or as a shell string."""
        shell = isinstance(command, str)
        try:
            completed = subprocess.run(
                command,
                cwd=str(workdir),
                env=env,
                shell=shell,
                executable="/bin/bash" if shell else None,
                capture_output=True,
                text=True,
                timeout=float(self.options.get("step_timeout_sec", 1800)),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"timed out after {command}", 124
        return (completed.stdout or "") + (completed.stderr or ""), completed.returncode

    def _venv_env(self) -> dict[str, str]:
        """Environment that puts the workspace interpreter first on PATH."""
        environment = dict(os.environ)
        venv_bin = self.paths.root / "venv" / "bin"
        if venv_bin.is_dir():
            inherited = environment.get("PATH", "/usr/bin:/bin")
            environment["PATH"] = os.pathsep.join(
                dict.fromkeys([str(venv_bin), str(self.paths.root / "bin"), inherited])
            )
            environment["VIRTUAL_ENV"] = str(self.paths.root / "venv")
        return environment

    def _write_record(self) -> None:
        path = self.paths.root / "provision.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.record.as_dict(), indent=2, default=str) + "\n",
            encoding="utf-8",
        )


def _is_required(command: str) -> bool:
    """Setup steps that the task cannot be graded without."""
    return command.startswith(("uv ", "pip", "python", "curl", "chmod", "mkdir"))


def rewrite_file(path: Path, mapping: dict[str, str]) -> str | None:
    """Rewrite container paths inside one small text file.

    Returns a description of what changed, or ``None`` when nothing did.
    """
    try:
        if path.is_symlink() or path.stat().st_size > _MAX_REWRITTEN_FILE_BYTES:
            return None
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None
    rewritten, substitutions = rewrite_paths(text, mapping)
    if not substitutions:
        return None
    try:
        path.write_text(rewritten, encoding="utf-8")
    except OSError:
        return None
    return "; ".join(substitutions)


def _strip_copy_flag(token: str) -> str:
    """Drop the JSON-array syntax some task Dockerfiles use in COPY."""
    token = token.strip()
    if token.startswith("[") and token.endswith("]"):
        token = token[1:-1].strip().strip('"')
    if token.startswith("--"):
        return ""
    return token


def rewrite_tree(root: Path, mapping: dict[str, str]) -> list[str]:
    """Point container-absolute paths in a copied tree at the workspace.

    Task payloads and verifiers name their own directories absolutely, exactly
    as they do inside the image, so the same substitution is applied to both.
    Only small text files are scanned. Returns one line per rewritten file.
    """
    changes: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        change = rewrite_file(path, mapping)
        if change:
            changes.append(f"{path.name}: {change}")
    return changes


def build_provisioner(
    paths: SandboxPaths,
    task: dict[str, Any],
    options: dict[str, Any],
) -> Provisioner:
    """Choose the provisioning strategy from the dataset name."""
    strategy = (
        "swebench-checkout"
        if str(task.get("dataset", "")).startswith("swebench")
        else "dockerfile-translation"
    )
    return Provisioner(
        paths=paths,
        task=task,
        options=options,
        record=ProvisionRecord(strategy=strategy),
    )


def default_options() -> dict[str, Any]:
    return {
        "step_timeout_sec": 1800,
        "network_retries": 3,
        "python": "3.9",
        "python_by_repo": {},
        "python_by_base_image": {},
    }


__all__ = [
    "ParsedDockerfile",
    "ProvisionError",
    "Provisioner",
    "build_provisioner",
    "default_options",
    "parse_dockerfile",
]