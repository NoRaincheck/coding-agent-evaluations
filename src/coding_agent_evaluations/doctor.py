"""Preflight checks.

A harness evaluation fails in slow, confusing ways when something basic is
missing: no ``uv``, a harness CLI that is not installed, an endpoint that does
not answer, or a dataset that has never been fetched. ``cae doctor`` checks all
of that in a few seconds so a run does not discover it mid-flight.
"""

from __future__ import annotations

import json
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ModelConfig
from .datasets import SUPPORTED


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    hint: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "detail": self.detail,
            "hint": self.hint,
            **({"data": self.data} if self.data else {}),
        }


def run_checks(
    *,
    model: ModelConfig | None = None,
    harnesses: tuple[str, ...] = ("leaf", "opencode", "pi"),
    cache_dir: Path | None = None,
    workspace_root: Path | None = None,
) -> list[Check]:
    checks: list[Check] = [
        _python(),
        _uv(),
        _harnesses(harnesses),
        _disk(workspace_root or Path("~/.cache/cae/workspaces").expanduser()),
        _datasets(cache_dir or Path("~/.cache/frognano/harbor").expanduser()),
    ]
    if model is not None:
        checks.append(_endpoint(model))
    return checks


def _python() -> Check:
    version = ".".join(str(part) for part in sys.version_info[:3])
    ok = sys.version_info >= (3, 12)
    return Check(
        "python",
        ok,
        f"{version} ({sys.executable})",
        hint="" if ok else "requires-python is >=3.12",
    )


def _uv() -> Check:
    binary = shutil.which("uv")
    if binary is None:
        return Check(
            "uv",
            False,
            "not found on PATH",
            hint="install with `brew install uv`; provisioning and some verifiers use it",
        )
    return Check("uv", True, binary)


def _harnesses(names: tuple[str, ...]) -> Check:
    rows: list[str] = []
    missing: list[str] = []
    for name in names:
        if name == "leaf":
            rows.append("leaf: reference harness (python)")
            continue
        binary = shutil.which(name)
        if binary is None:
            missing.append(name)
            rows.append(f"{name}: MISSING")
        else:
            rows.append(f"{name}: {binary}")
    return Check(
        "harnesses",
        not missing,
        "; ".join(rows),
        hint=(
            f"install the missing CLI(s): {', '.join(missing)}"
            if missing
            else ""
        ),
    )


def _disk(path: Path) -> Check:
    target = path
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as exc:  # pragma: no cover - platform dependent
        return Check("disk", False, f"could not stat {target}: {exc}")
    free_gib = usage.free / 1024**3
    ok = free_gib >= 5
    return Check(
        "disk",
        ok,
        f"{free_gib:.1f} GiB free at {probe}",
        hint="" if ok else "workspaces and dataset caches need several GiB",
    )


def _datasets(cache_dir: Path) -> Check:
    roots = sorted(p for p in cache_dir.glob("*/*") if p.is_dir()) if cache_dir.is_dir() else []
    return Check(
        "datasets",
        True,
        f"cache {cache_dir}: {len(roots)} materialized source(s)",
        data={"cached": [str(p) for p in roots[:10]]},
    )


def _endpoint(model: ModelConfig) -> Check:
    url = f"{model.base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model.name,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 8,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {model.resolve_api_key()}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return Check(
            "endpoint",
            False,
            f"{url} returned HTTP {exc.code}",
            hint=exc.read()[:300].decode(errors="replace"),
        )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return Check(
            "endpoint",
            False,
            f"{url} unreachable: {exc}",
            hint="start the server, or set model.base_url in the config",
        )
    served = str(body.get("model") or "")
    ok = served == model.name or not served
    return Check(
        "endpoint",
        ok,
        f"{url} reachable, served model {served!r}",
        hint="" if ok else f"endpoint serves {served!r}, config asks for {model.name!r}",
    )


def format_checks(checks: list[Check]) -> str:
    lines = []
    for check in checks:
        mark = "ok  " if check.ok else "FAIL"
        line = f"[{mark}] {check.name}: {check.detail}"
        if check.hint:
            line += f"\n        -> {check.hint}"
        lines.append(line)
    return "\n".join(lines)


def datasets_hint() -> str:
    return ", ".join(SUPPORTED)


__all__ = ["Check", "datasets_hint", "format_checks", "run_checks"]