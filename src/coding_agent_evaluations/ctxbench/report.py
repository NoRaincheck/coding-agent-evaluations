"""Rendering the ctxbench payload: one table, one Markdown summary.

The JSON is the record; this is for reading. Three choices are worth stating, because each
one exists to stop a reader drawing a conclusion the data does not support.

**Absent is shown as `-`, never as `0`.** A suite that did not run, and a suite that ran
and measured zero, are different facts. Collapsing them makes a skipped suite look like a
finding.

**Pass rates carry their trial count.** `0.50` from 6 trials and `0.50` from 2 trials are
not the same claim, and a rate without its denominator invites the wrong reading.

**Health flags are spelled out, not inferred.** "Truncated" appears only when the server
reported fewer prompt tokens than were sent. A suite that errored out is reported as
errored, not as truncated, because an absent measurement is not a negative one.
"""

from __future__ import annotations

import json
from typing import Any

from .runner import REPORT_VERSION

#: (header, path into a result row, formatter). `None` renders as "-".
_COLUMNS: tuple[tuple[str, tuple[str, ...], Any], ...] = (
    ("tokens", ("context_size_tokens",), lambda v: f"{int(v):,}"),
    (
        "ttft_ms",
        ("speed", "prefill", "ttft_ms"),
        lambda v: f"{float(v):,.0f}",
    ),
    (
        "prefill_tps",
        ("speed", "prefill", "prefill_tps"),
        lambda v: f"{float(v):,.0f}",
    ),
    (
        "decode_tps",
        ("speed", "generation", "tps"),
        lambda v: f"{float(v):.1f}",
    ),
    (
        "e2e_ms",
        ("speed", "generation", "total_latency_ms"),
        lambda v: f"{float(v):,.0f}",
    ),
    (
        "needle",
        ("correctness", "needle_in_haystack_pass_rate"),
        lambda v: f"{float(v) * 100:.0f}%",
    ),
    (
        "niah_n",
        ("correctness", "needle", "trials"),
        lambda v: str(int(v)),
    ),
    (
        "pass@1",
        ("correctness", "code_execution_pass_at_1"),
        lambda v: f"{float(v) * 100:.0f}%",
    ),
    (
        "truncated",
        ("harness_health", "context_truncated"),
        lambda v: "YES" if v else "no",
    ),
    (
        "sent",
        ("harness_health", "prompt_tokens_sent"),
        lambda v: f"{int(v):,}",
    ),
    (
        "received",
        ("harness_health", "context_truncation", "received_tokens"),
        lambda v: f"{int(v):,}" if v is not None else "-",
    ),
    ("errors", ("errors",), lambda v: str(len(v))),
)


def _dig(row: dict[str, Any], path: tuple[str, ...]) -> Any:
    cursor: Any = row
    for key in path:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    return cursor


def _cell(row: dict[str, Any], path: tuple[str, ...], fmt: Any) -> str:
    value = _dig(row, path)
    if value is None:
        return "-"
    try:
        return str(fmt(value))
    except (TypeError, ValueError):
        return "-"


def as_table(payload: dict[str, Any]) -> str:
    """The fixed-width table: one row per context size."""
    headers = [header for header, _, _ in _COLUMNS]
    rows = [
        [_cell(result, path, fmt) for _, path, fmt in _COLUMNS]
        for result in payload.get("results", [])
    ]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render(values: list[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(values)).rstrip()

    lines = [render(headers), render(["-" * width for width in widths])]
    lines.extend(render(row) for row in rows)
    if not rows:
        lines.append("(no context sizes measured)")
    return "\n".join(lines)


def caveats(payload: dict[str, Any]) -> list[str]:
    """Notes that must travel with the numbers.

    Printed under the table every time, because each one is a limit on what the numbers
    can be read as claiming -- and a limit nobody sees is not a limit that was honoured.
    """
    notes: list[str] = []
    suites = payload.get("config", {}).get("ctxbench", {})
    enabled = suites.get("enabled") or []
    for name in ("execution", "memory"):
        if name not in enabled:
            notes.append(
                f"{name} suite did not run; its column is absent rather than zero"
            )
    if suites.get("repeats", 1) < 3:
        notes.append(
            f"repeats={suites.get('repeats')}: speed figures are single- or few-sample "
            "medians and the spread is the number to read"
        )
    elif any(
        row.get("speed", {}).get("prefill", {}).get("prefix_cache_busted")
        for row in payload.get("results", [])
    ):
        # Worth stating: the TTFT column is uncached prefill, which is the cost of a fresh
        # context. It is not comparable to a warm-prefix number from a long session.
        notes.append(
            "ttft is measured with repeats 2..n prefix-cache-busted, so it is the cost of "
            "a fresh context rather than of a warm one"
        )
    # The locally measured `sent` count uses `model.tokenizer`, which is not the model's
    # own vocabulary. Compared against 1.0 rather than across rows: the question is
    # whether local and server counts describe the same context, and one row answers that.
    ratios = [
        row.get("harness_health", {}).get("context_truncation", {}).get("received_over_sent")
        for row in payload.get("results", [])
    ]
    ratios = [float(ratio) for ratio in ratios if isinstance(ratio, (int, float))]
    if ratios:
        low, high = min(ratios), max(ratios)
        # One-sided: a server that reports *more* tokens than were sent is a vocabulary
        # difference, not truncation, and `context_truncated` already says so.
        if abs(low - 1.0) > 0.05 or abs(high - 1.0) > 0.05:
            notes.append(
                f"context sizes are labeled by local {payload.get('tokenizer')} counts; the "
                f"endpoint reported {low:.2f}x-{high:.2f}x that (model tokenizer differs)"
            )
        else:
            notes.append(
                f"context sizes are labeled by local {payload.get('tokenizer')} counts; "
                f"the endpoint agreed within {(high - low) * 100:.1f}%"
            )

    # A row the endpoint refused for exceeding its window is not a measurement of the model
    # at that context size, and reading it as one is the easiest way to draw a wrong
    # conclusion from this table.
    limits = {
        row.get("harness_health", {})
        .get("context_truncation", {})
        .get("server_context_limit")
        for row in payload.get("results", [])
    }
    limits.discard(None)
    if limits:
        smallest = min(limits)
        blocked = [
            row["context_size_tokens"]
            for row in payload.get("results", [])
            if row.get("harness_health", {})
            .get("context_truncation", {})
            .get("exceeds_server_limit")
        ]
        if blocked:
            notes.append(
                f"the endpoint reports a {smallest:,}-token context window; rows at "
                f"{', '.join(f'{size:,}' for size in blocked)} were rejected as oversized "
                "and carry no model measurement"
            )

    # A reasoning model can burn its whole answer budget thinking. Those trials never tested
    # retrieval, and a pass rate that counted them as failures would be reporting the token
    # budget as a model property.
    unobservable = sum(
        row.get("correctness", {}).get("needle", {}).get("unobservable", 0) or 0
        for row in payload.get("results", [])
    )
    if unobservable:
        notes.append(
            f"{unobservable} needle trial(s) produced no answer because the model spent the "
            "whole output budget reasoning; they are excluded from the pass rate, which is "
            "therefore over fewer trials than were attempted"
        )

    memory = payload.get("memory") or {}
    if "memory" not in enabled:
        # Already covered by the "did not run" note above; a "0 samples" note on top of it
        # reads as a broken measurement rather than a suite that was never asked for.
        pass
    elif memory.get("skipped"):
        notes.append(
            f"memory suite did not run: {memory.get('note', 'no size available')}"
        )
    elif memory.get("iterations", 0) < 5:
        notes.append(
            f"memory trace has {memory.get('iterations', 0)} samples; growth from a short "
            "trace is not evidence of a leak either way"
        )
    elif memory.get("monotonic_growth"):
        notes.append(
            f"peak RSS rose monotonically across {memory.get('iterations')} iterations "
            f"(+{memory.get('growth_mb')} MB); treat as a leak candidate and re-run to confirm"
        )
    if memory and not memory.get("iterations") and not memory.get("skipped"):
        # The suite ran but produced no samples, which means the platform would not report
        # RSS. A skipped suite is a different fact and is already noted above.
        notes.append("RSS was not measurable on this platform; memory figures are absent")
    return notes


def as_markdown(payload: dict[str, Any]) -> str:
    """The standard report block, identical in shape on every run."""
    config = payload.get("config", {})
    suites = config.get("ctxbench", {})
    lines = [
        f"ctxbench report v{payload.get('report_version', REPORT_VERSION)}",
        f"generated    {payload.get('generated')}",
        f"run_id       {payload.get('run_id')}",
        f"runner       {payload.get('harness_version')}",
        f"model        {payload.get('model')}",
        f"endpoint     {payload.get('base_url')}",
        f"harness      {payload.get('harness')}",
        f"tokenizer    {payload.get('tokenizer')}",
        f"contexts     {', '.join(f'{int(s):,}' for s in config.get('context_sizes', []))}",
        (
            f"suites       {', '.join(suites.get('enabled') or [])} "
            f"(repeats={suites.get('repeats')}, needle_trials={suites.get('needle_trials')})"
        ),
        f"duration     {payload.get('duration_sec')}s",
        "",
        as_table(payload),
    ]
    notes = caveats(payload)
    if notes:
        lines.append("")
        lines.append("notes:")
        lines.extend(f"  - {note}" for note in notes)
    return "\n".join(lines)


def as_json(payload: dict[str, Any]) -> str:
    """The machine-readable form: exactly what was written to `ctxbench.json`."""
    return json.dumps(payload, indent=2)


__all__ = ["as_json", "as_markdown", "as_table", "caveats"]