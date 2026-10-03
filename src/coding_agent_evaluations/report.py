"""Cross-harness comparison.

Reads ``summary.json`` files written by one or more runs and puts them side by
side. This is the point of the project: the same tasks and model measured
through leaf, opencode and pi, with the limits that were in force shown next to
the numbers so two rows are not mistaken for equivalent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_COLUMNS = (
    ("dataset", "dataset"),
    ("harness", "harness"),
    ("model", "model"),
    ("tasks", "tasks_selected"),
    ("jobs", "jobs_total"),
    ("completed", "jobs_completed"),
    ("infra_fail", "jobs_infrastructure_failed"),
    ("resolved", "resolved"),
    ("resolve_rate", "resolve_rate"),
    ("resolve_rate_graded", "resolve_rate_graded"),
    ("mean_steps", "mean_steps"),
    ("mean_min", "mean_elapsed_min"),
    ("duration_min", "duration_min"),
)


@dataclass
class Row:
    summary: dict[str, Any]
    path: Path

    def get(self, key: str) -> Any:
        return self.summary.get(key)


def load_summaries(paths: list[str]) -> list[Row]:
    """Load summaries from files or from directories containing them."""
    rows: list[Row] = []
    for raw in paths:
        path = Path(raw).expanduser()
        candidates = [path] if path.is_file() else sorted(path.glob("**/summary.json"))
        if not candidates:
            raise FileNotFoundError(f"no summary.json found under {path}")
        for candidate in candidates:
            rows.append(
                Row(summary=json.loads(candidate.read_text()), path=candidate)
            )
    rows.sort(key=lambda row: (str(row.get("dataset")), str(row.get("harness"))))
    return rows


def as_table(rows: list[Row]) -> str:
    """Render summaries as a fixed-width table."""
    headers = [header for header, _ in _COLUMNS]
    columns = [_prepare(rows, key) for _, key in _COLUMNS]
    widths = [
        max([len(headers[index])] + [len(value) for value in column])
        for index, column in enumerate(columns)
    ]
    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    for position in range(len(rows)):
        lines.append(
            "  ".join(
                column[position].ljust(widths[index])
                for index, column in enumerate(columns)
            )
        )
    return "\n".join(lines)


def _prepare(rows: list[Row], key: str) -> list[str]:
    """Format one column: one string per row."""
    prepared: list[str] = []
    for row in rows:
        value = row.get(key)
        if key in {"mean_elapsed_min", "duration_min"}:
            prepared.append(_minutes(row.get(key.replace("_min", "_sec"))))
        elif key == "resolve_rate":
            prepared.append(f"{float(value or 0) * 100:.1f}%" if value is not None else "-")
        elif isinstance(value, float):
            prepared.append(f"{value:.2f}")
        else:
            prepared.append(str(value if value is not None else "-"))
    return prepared


def _minutes(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)):
        return "-"
    return f"{seconds / 60:.1f}"


def as_markdown(rows: list[Row]) -> str:
    """Render summaries as a markdown table."""
    headers = [header for header, _ in _COLUMNS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_prepare([row], key)[0] for _, key in _COLUMNS) + " |")
    return "\n".join(lines)


def caveats(rows: list[Row]) -> list[str]:
    """Surface the limits that differ between runs, so a comparison stays honest."""
    notes: list[str] = []
    for row in rows:
        limits = row.get("limits") or {}
        harness = row.get("harness")
        if limits and not limits.get("max_steps_enforced", True):
            notes.append(
                f"{harness}: max_steps={limits.get('max_steps')} is NOT enforced "
                f"(the harness owns its loop); bounded by "
                f"max_total_time_sec={limits.get('max_total_time_sec')} only"
            )
        ignored = limits.get("model_params_ignored") or []
        if ignored:
            notes.append(
                f"{harness}: model settings not applied: {', '.join(ignored)}"
            )
        if row.get("jobs_infrastructure_failed"):
            notes.append(
                f"{harness}: {row.get('jobs_infrastructure_failed')} task-seed pair(s) "
                "failed before the agent could be graded "
                f"({row.get('infrastructure_reasons')})"
            )
    return notes


__all__ = ["Row", "as_markdown", "as_table", "caveats", "load_summaries"]