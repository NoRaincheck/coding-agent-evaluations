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

# Token speed, shown as a second table rather than more columns: it is only measured by the
# harnesses that load the pi-token-stats extension, and it answers a different question
# (how fast the endpoint was) from the first table (did the task get solved).
_TOKEN_COLUMNS = (
    ("harness", "harness"),
    ("model", "model"),
    ("requests", "requests"),
    ("turns", "turns"),
    ("ttft_p50_ms", "ttft_ms_p50"),
    ("ttft_p90_ms", "ttft_ms_p90"),
    ("decode_tps", "decode_tps_mean"),
    ("prefill_tps", "prefill_tps_mean"),
    ("input_tokens", "input_tokens"),
    ("output_tokens", "output_tokens"),
    ("generation_min", "generation_min"),
)

# Keys of `_TOKEN_COLUMNS` that live inside a summary's `token_stats` block. `harness`,
# `model` and `generation_min` are handled separately above.
_TOKEN_KEYS = frozenset(
    key for _, key in _TOKEN_COLUMNS if key not in {"harness", "model", "generation_min"}
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
    return _render(rows, _COLUMNS)


def as_token_table(rows: list[Row]) -> str:
    """Render the token-speed table, or a note when no run measured it."""
    measured = [row for row in rows if isinstance(row.get("token_stats"), dict)]
    if not measured:
        return ""
    return _render(measured, _TOKEN_COLUMNS)


def _render(rows: list[Row], columns: tuple[tuple[str, str], ...]) -> str:
    headers = [header for header, _ in columns]
    prepared = [_prepare(rows, key) for _, key in columns]
    widths = [
        max([len(headers[index])] + [len(value) for value in column])
        for index, column in enumerate(prepared)
    ]
    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    for position in range(len(rows)):
        lines.append(
            "  ".join(
                column[position].ljust(widths[index])
                for index, column in enumerate(prepared)
            )
        )
    return "\n".join(lines)


def _prepare(rows: list[Row], key: str) -> list[str]:
    """Format one column: one string per row."""
    prepared: list[str] = []
    for row in rows:
        value = row.get(key)
        if key == "resolve_rate":
            prepared.append(f"{float(value or 0) * 100:.1f}%" if value is not None else "-")
        elif key in {"mean_elapsed_min", "duration_min"}:
            prepared.append(_minutes(row.get(key.replace("_min", "_sec"))))
        elif key == "generation_min":
            prepared.append(
                _minutes((row.get("token_stats") or {}).get("generation_sec"))
            )
        elif key in _TOKEN_KEYS:
            # Token-speed keys live inside `token_stats`, not at the top of the summary.
            value = (row.get("token_stats") or {}).get(key)
            prepared.append(
                f"{value:.1f}" if isinstance(value, float) else str(value if value is not None else "-")
            )
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
    measured = sorted(
        {str(row.get("harness")) for row in rows if isinstance(row.get("token_stats"), dict)}
    )
    if measured and len(measured) < len({str(row.get("harness")) for row in rows}):
        unmeasured = sorted(
            {
                str(row.get("harness"))
                for row in rows
                if not isinstance(row.get("token_stats"), dict)
            }
        )
        # Comparing token speed across a table where only some rows measured it would be
        # comparing an endpoint against a harness, so say which rows are missing.
        notes.append(
            f"token speed measured only for {', '.join(measured)}; "
            f"{', '.join(unmeasured)} report none, so those rows are not comparable on speed"
        )
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
    # A matrix has one row per (model, harness), so the same limit note would repeat across
    # every cell that shares it. Report each once: nine identical lines is noise, not caution.
    return list(dict.fromkeys(notes))


__all__ = [
    "Row",
    "as_markdown",
    "as_table",
    "as_token_table",
    "caveats",
    "load_summaries",
]