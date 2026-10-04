"""Token-speed statistics for pi rollouts.

The ``pi-token-stats`` extension (vendored under ``_vendor/pi_token_stats``, see the
``PROVENANCE.md`` there) is an observer: it hooks pi's message lifecycle and appends one
``custom`` entry per assistant message plus one aggregate entry per turn into pi's session
JSONL. Nothing about the agent's behaviour changes — no prompt, no tool, no sampling knob.

pi writes its session JSONL, not stdout, so the harness reads the file back after the CLI
exits and folds the entries into the trajectory. Two shapes come out:

* ``messages`` / ``turns`` — the raw upstream entries, unchanged, so a result can be
  re-aggregated differently later without re-running anything.
* ``aggregate`` — the per-rollout numbers worth tracking across a run: time to first token,
  decode and prefill throughput, tokens, and generation wall time.

The aggregate uses snake_case keys because it is this project's summary, not the extension's
schema; the raw entries keep the extension's camelCase so upstream's ``jq`` recipes work
against them verbatim.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..config import HarnessConfig

#: Entry types the vendored extension appends to the session JSONL.
ENTRY_TYPE = "token_speed_stats"
SUMMARY_ENTRY_TYPE = f"{ENTRY_TYPE}_turn_summary"

#: Pinned upstream commit of the vendored extension, recorded in every result so a summary
#: says which implementation produced its numbers.
VENDORED_COMMIT = "fb1abd2fe7649f3fb9315511cc11a84460238bca"

_VENDORED = Path(__file__).resolve().parents[1] / "_vendor" / "pi_token_stats" / "index.ts"

# Fields summed per rollout. `None` upstream (cache reads on a local endpoint) counts as zero.
_TOKEN_FIELDS = {
    "input_tokens": "inputTokens",
    "output_tokens": "outputTokens",
    "cache_read_tokens": "cacheReadTokens",
    "total_tokens": "totalTokens",
}


def extension_path(config: HarnessConfig) -> Path | None:
    """Resolve the extension to load, or ``None`` when token stats are off.

    ``harness.options.token_stats_extension`` overrides the vendored copy, which is the escape
    hatch for evaluating a newer upstream revision without editing the tree.
    """
    if not _enabled(config):
        return None
    override = config.options.get("token_stats_extension")
    if override:
        return Path(str(override)).expanduser()
    return _VENDORED


def token_stats_enabled(config: HarnessConfig) -> bool:
    """Whether this rollout should record token statistics."""
    return _enabled(config)


def _enabled(config: HarnessConfig) -> bool:
    # On by default: the extension is an observer, and a run that does not record token speed
    # is a run whose throughput was measured by guesswork later.
    return bool(config.options.get("token_stats", True))


def read_token_stats(session_dir: Path, session_id: str) -> dict[str, Any]:
    """Read the stats the extension left in pi's session JSONL.

    pi names a session file ``<timestamp>_<session-id>.jsonl``, so the rollout's own file is
    found by suffix. Every other JSONL in the directory is only read as a fallback: a session
    that was resumed or forked under a different id still has to yield its stats.
    """
    files = sorted(session_dir.glob(f"*_{session_id}.jsonl")) or sorted(
        session_dir.glob("*.jsonl")
    )
    messages: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    for path in files:
        for entry in _read_entries(path):
            kind = str(entry.get("customType") or "")
            data = entry.get("data")
            if not isinstance(data, dict):
                continue
            if kind == ENTRY_TYPE:
                messages.append(data)
            elif kind == SUMMARY_ENTRY_TYPE:
                turns.append(data)
    messages.sort(key=lambda item: _number(item.get("timestamp")) or 0)
    turns.sort(key=lambda item: _number(item.get("timestamp")) or 0)
    return {
        "source": "pi-token-stats",
        "commit": VENDORED_COMMIT,
        "session_files": [str(path) for path in files],
        "messages": messages,
        "turns": turns,
        "aggregate": aggregate(messages, turns),
    }


def aggregate(
    messages: list[dict[str, Any]], turns: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Fold per-message entries into the numbers worth tracking.

    Latency is reported as a distribution, because a mean TTFT hides exactly the tail that
    makes an evaluation slow. Throughput is reported both as a per-request mean and as a
    ratio of totals: the mean rewards runs with many tiny requests, the ratio does not.
    """
    requests = len(messages)
    summary: dict[str, Any] = {
        "requests": requests,
        "turns": len(turns or []),
    }
    for name, key in _TOKEN_FIELDS.items():
        summary[name] = int(sum(_number(item.get(key)) or 0 for item in messages))

    ttft = _positive(messages, "ttftMs")
    summary["ttft_ms_mean"] = _mean(ttft)
    summary["ttft_ms_p50"] = _percentile(ttft, 0.50)
    summary["ttft_ms_p90"] = _percentile(ttft, 0.90)
    summary["ttft_ms_min"] = min(ttft, default=None)
    summary["ttft_ms_max"] = max(ttft, default=None)

    decode_seconds = [v for v in (_number(m.get("decodeSeconds")) for m in messages) if v and v > 0]
    decode_tokens = [v for v in (_number(m.get("decodeTokens")) for m in messages) if v and v > 0]
    summary["generation_sec"] = _round(sum(decode_seconds))
    summary["decode_tps_mean"] = _mean(_positive(messages, "decodeTps"))
    summary["decode_tps_total"] = _round(_ratio(decode_tokens, decode_seconds))
    summary["gen_tps_mean"] = _mean(_positive(messages, "genTps"))
    summary["prefill_tps_mean"] = _mean(_positive(messages, "prefillTps"))

    # Cache-aware prefill exists only when the provider reported cache reads.
    cache_aware = [v for v in (_number(m.get("cacheAwarePrefillTokens")) for m in messages) if v]
    summary["cache_aware_prefill_tokens"] = int(sum(cache_aware)) if cache_aware else None

    cost = [v for v in (_number(m.get("costTotal")) for m in messages) if v is not None]
    summary["cost_usd"] = _round(sum(cost), 6) if cost else None

    starts = [_number(m.get("requestStartAt")) for m in messages]
    ends = [_number(m.get("lastChunkAt")) for m in messages]
    starts = [v for v in starts if v]
    ends = [v for v in ends if v]
    summary["span_sec"] = (
        _round((max(ends) - min(starts)) / 1000, 3) if starts and ends else None
    )

    models = sorted({str(item.get("model")) for item in messages if item.get("model")})
    summary["models"] = models
    summary["provider_id"] = next(
        (str(item.get("providerId")) for item in messages if item.get("providerId")), ""
    )
    return summary


def merge_aggregates(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Combine per-rollout aggregates into one run-level view.

    Rates are weighted by request count, so a rollout with 200 requests counts for more than
    one with 3 — an unweighted mean would let the quiet rollouts set the headline number.
    Totals are summed. Latency distributions cannot be recovered from per-rollout means, so
    the run-level figure is the request-weighted mean of the per-rollout means and is labelled
    as such rather than passed off as a pooled percentile.
    """
    rows = [row for row in rows if isinstance(row, dict) and row.get("requests")]
    if not rows:
        return None
    total_requests = sum(int(row.get("requests") or 0) for row in rows)
    merged: dict[str, Any] = {
        "source": "pi-token-stats",
        "rollouts": len(rows),
        "requests": total_requests,
        "turns": sum(int(row.get("turns") or 0) for row in rows),
    }
    for name in _TOKEN_FIELDS:
        merged[name] = sum(int(row.get(name) or 0) for row in rows)
    for name in ("generation_sec", "span_sec"):
        merged[name] = _round(_weighted(_weighted_pairs(rows, name)))
    for name in (
        "ttft_ms_mean",
        "ttft_ms_p90",
        "decode_tps_mean",
        "decode_tps_total",
        "gen_tps_mean",
        "prefill_tps_mean",
    ):
        merged[name] = _round(_weighted(_weighted_pairs(rows, name)), 2)
    costs = [row.get("cost_usd") for row in rows if row.get("cost_usd") is not None]
    merged["cost_usd"] = _round(sum(costs), 6) if costs else None
    models = sorted({str(name) for row in rows for name in row.get("models") or []})
    merged["models"] = models
    return merged


def _read_entries(path: Path) -> list[dict[str, Any]]:
    """Stream one session JSONL, keeping only `custom` entries.

    Read line by line: a 150-step session JSONL carries every message and tool result, and
    this only needs the handful of lines the extension appended.
    """
    entries: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                stripped = line.strip()
                if not stripped or stripped[0] != "{":
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and payload.get("type") == "custom":
                    entries.append(payload)
    except OSError:
        # A run killed mid-write leaves a truncated file; whatever was flushed is still data.
        pass
    return entries


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _positive(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [
        value
        for value in (_number(row.get(key)) for row in rows)
        if value is not None and value > 0
    ]


def _mean(values: list[float]) -> float | None:
    return _round(statistics.fmean(values)) if values else None


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return _round(ordered[0])
    position = quantile * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return _round(ordered[low] * (1 - weight) + ordered[high] * weight)


def _ratio(numerators: list[float], denominators: list[float]) -> float | None:
    total = sum(denominators)
    return sum(numerators) / total if total > 0 else None


def _weighted_pairs(
    rows: list[dict[str, Any]], key: str
) -> Iterator[tuple[Any, Any]]:
    """Pair each row's value with its weight (its request count)."""
    for row in rows:
        yield row.get(key), row.get("requests")


def _weighted(pairs: Iterator[tuple[Any, Any]]) -> float | None:
    total = 0.0
    weight_total = 0.0
    for value, weight in pairs:
        number = _number(value)
        factor = _number(weight) or 0.0
        if number is None or factor <= 0:
            continue
        total += number * factor
        weight_total += factor
    return total / weight_total if weight_total > 0 else None


def _round(value: float | None, decimals: int = 3) -> float | None:
    return None if value is None else round(value, decimals)


__all__ = [
    "ENTRY_TYPE",
    "SUMMARY_ENTRY_TYPE",
    "VENDORED_COMMIT",
    "aggregate",
    "extension_path",
    "merge_aggregates",
    "read_token_stats",
    "token_stats_enabled",
]
