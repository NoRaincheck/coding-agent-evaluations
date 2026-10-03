"""Execution runtimes.

``host`` is the only runtime in this project: it needs no container runtime and
no Kubernetes cluster. It is selected with ``runtime.kind: host``.
"""

from __future__ import annotations

from .base import CommandExecutor, ProvisionRecord, Runtime, SandboxPaths, copy_into
from .host import HostTaskRuntime
from .provision import ProvisionError, parse_dockerfile

_RUNTIMES = {"host": HostTaskRuntime}


def get_runtime(kind: str) -> type[HostTaskRuntime]:
    """Look up a runtime by ``runtime.kind``."""
    try:
        return _RUNTIMES[kind]
    except KeyError as exc:
        raise ValueError(
            f"unknown runtime {kind!r}; available: {sorted(_RUNTIMES)}"
        ) from exc


__all__ = [
    "CommandExecutor",
    "HostTaskRuntime",
    "ProvisionError",
    "ProvisionRecord",
    "Runtime",
    "SandboxPaths",
    "copy_into",
    "get_runtime",
    "parse_dockerfile",
]