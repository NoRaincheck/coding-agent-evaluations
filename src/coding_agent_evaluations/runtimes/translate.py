"""Translate image-scoped benchmark verifiers into host-runnable scripts.

Both benchmarks ship a ``tests/test.sh`` written for a container: it installs
system packages with ``apt-get``, activates a conda environment under
``/opt/miniconda3``, and reads fixtures from absolute paths such as
``/tests/config.json`` and writes rewards to ``/logs/verifier/reward.txt``.

On the host those absolute paths do not exist and cannot be created. This module
rewrites the verifier to the workspace layout and records every substitution so a
run can be audited. The original script is always preserved next to the
translated one, and the substitution list travels into the trajectory and the
run summary.

Translation is deliberately conservative:

* absolute benchmark paths are remapped to workspace paths,
* image-only package managers are turned into logged no-ops,
* ``conda activate`` is replaced by the workspace virtual environment,
* everything else is left byte-identical.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import SandboxPaths

# Lines that only make sense inside the benchmark image. Matched at the start of a
# logical command; each becomes a logged no-op instead of a hard failure.
_IMAGE_ONLY_PREFIXES = (
    "apt-get",
    "apt ",
    "apk ",
    "apk add",
    "yum ",
    "dnf ",
    "microdnf ",
    "conda install",
    "conda activate",
    "source /opt/miniconda3/bin/activate",
    "source /opt/conda/bin/activate",
    "source $CONDA_PREFIX/bin/activate",
    ". /opt/miniconda3/bin/activate",
    "pip3 install",
    "pip install",
    "python3 -m pip install",
    "python -m pip install",
    "update-alternatives",
    "dpkg --configure",
)


@dataclass
class Translation:
    """The rewritten verifier plus an audit trail."""

    text: str
    substitutions: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "substitutions": self.substitutions,
            "skipped_image_only_commands": self.skipped,
        }


_PATH_SEPARATOR = ":"


def path_mapping(paths: SandboxPaths, task: dict[str, Any]) -> dict[str, str]:
    """Map container-absolute benchmark paths onto workspace paths.

    Longest keys are rewritten first so ``/logs/verifier`` wins over ``/logs``.
    """
    repo_path = str(task.get("repo_path") or "/app")
    mapping = {
        "/logs/verifier": str(paths.logs / "verifier"),
        "/logs/agent": str(paths.logs / "agent"),
        "/solution": str(paths.solution),
        "/tests": str(paths.tests),
        "/logs": str(paths.logs),
        repo_path: str(paths.workspace),
    }
    for alias in ("/testbed", "/app", "/workspace"):
        mapping.setdefault(alias, str(paths.workspace))
    # Only keep mappings that cannot be substrings of each other in a damaging way.
    return dict(sorted(mapping.items(), key=lambda item: -len(item[0])))


def rewrite_paths(text: str, mapping: dict[str, str]) -> tuple[str, list[str]]:
    """Replace container-absolute paths with host paths in one pass.

    A single alternation (longest key first) matters twice over: it prevents
    ``/logs`` from clobbering ``/logs/verifier``, and it means already-substituted
    host paths are never rescanned for another match. A path only matches at a
    token boundary, so a longer word or URL containing it is left alone.
    """
    if not mapping:
        return text, []
    counts: dict[str, int] = {}
    keys = sorted(mapping, key=len, reverse=True)
    pattern = re.compile(
        r"(?<![\w./-])(?P<path>"
        + "|".join(re.escape(key) for key in keys)
        + r")(?=$|[\s\"'`);:&|,\]/])"
    )

    def substitute(match: re.Match[str]) -> str:
        original = match.group("path")
        counts[original] = counts.get(original, 0) + 1
        return mapping[original]

    result = pattern.sub(substitute, text)
    substitutions = [
        f"{original} -> {mapping[original]} ({count} occurrence(s))"
        for original, count in sorted(counts.items())
    ]
    return result, substitutions


def neutralize_image_only_commands(text: str) -> tuple[str, list[str]]:
    """Replace image-only commands with logged no-ops."""
    lines = text.split("\n")
    skipped: list[str] = []
    output: list[str] = []
    heredoc: str | None = None
    for line in lines:
        if heredoc is not None:
            output.append(line)
            if heredoc in line:
                heredoc = None
            continue
        stripped = line.strip()
        heredoc = _heredoc_marker(stripped)
        if stripped.startswith("#") or not stripped:
            output.append(line)
            continue
        if _is_image_only(stripped):
            skipped.append(stripped)
            indent = line[: len(line) - len(line.lstrip())]
            output.append(
                f"{indent}# cae: image-only command neutralized: {stripped}"
            )
            continue
        output.append(line)
    return "\n".join(output), skipped


def _heredoc_marker(stripped: str) -> str | None:
    match = re.search(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", stripped)
    return match.group(1) if match else None


def _is_image_only(stripped: str) -> bool:
    normalized = stripped.lstrip("&").strip()
    if normalized.startswith(("if ", "then", "fi", "else", "export DEBIAN")):
        return False
    # `python -m pip install -e .[test]` still works inside the workspace
    # virtual environment, which the host runtime activates.
    if "pip install" in normalized and not normalized.startswith("pip3"):
        return False
    return any(
        normalized == prefix.rstrip() or normalized.startswith(prefix)
        for prefix in _IMAGE_ONLY_PREFIXES
    )


def translate_verifier(
    script_path: Path, paths: SandboxPaths, task: dict[str, Any]
) -> Translation:
    """Rewrite a benchmark verifier script for host execution."""
    original = script_path.read_text(encoding="utf-8")
    mapping = path_mapping(paths, task)
    rewritten, substitutions = rewrite_paths(original, mapping)
    rewritten, skipped = neutralize_image_only_commands(rewritten)
    return Translation(text=rewritten, substitutions=substitutions, skipped=skipped)


def env_overrides(
    paths: SandboxPaths, task: dict[str, Any], *, venv_bin: Path | None
) -> dict[str, str]:
    """Environment for workspace commands.

    The workspace virtual environment is prepended to ``PATH`` so ``python`` and
    ``pytest`` resolve to the provisioned interpreter instead of the host's.
    The inherited ``PATH`` is preserved: a task's own ``apt-get``/``curl``
    fallbacks still need ``sh``, ``rm``, ``git`` and friends.
    """
    env: dict[str, str] = {}
    if venv_bin is not None and venv_bin.is_dir():
        shim = paths.workspace / ".cae" / "bin"
        inherited = os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
        entries = [str(venv_bin), str(shim)]
        entries.extend(
            part for part in inherited.split(_PATH_SEPARATOR) if part not in entries
        )
        env["PATH"] = _PATH_SEPARATOR.join(entries)
        env["VIRTUAL_ENV"] = str(venv_bin.parent)
        env.pop("PYTHONHOME", None)
    return env


def iter_supported(task: dict[str, Any]) -> Iterable[str]:
    """Report the container features this host runtime does not reproduce."""
    if task.get("verifier_protocol") == "patch_eval_verified":
        yield "verifier_protocol=patch_eval_verified requires the benchmark image"
    if task.get("setup_commands"):
        yield "image setup_commands are not replayed verbatim"
    network = task.get("verifier_network_mode")
    if network == "no-network":
        yield "network isolation is not enforced on the host"