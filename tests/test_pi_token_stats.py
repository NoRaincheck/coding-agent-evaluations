"""Token-speed statistics read back out of a pi session.

The numbers come from the extension, but what this project owns is the reading of them and
the reporting of them, so that is what is tested: that entries are found at all, that a
statistic which is silently zero is distinguishable from one that was never recorded, and
that a run-level rollup does not let one rollout set the headline.

Fixtures are shaped like real pi output, including the fields a local endpoint never sends
(`cacheReadTokens`, `costTotal`) — those have to read as absent rather than as zero, because
a zero cache read means "the provider reported none" and a zero TTFT means "never recorded".
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from matrix import CONFIG_ROOT, jsonl, make_run

from coding_agent_evaluations.config import HarnessConfig, load_config
from coding_agent_evaluations.harnesses.pi import PiHarness
from coding_agent_evaluations.harnesses.pi_stats import (
    ENTRY_TYPE,
    SUMMARY_ENTRY_TYPE,
    aggregate,
    merge_aggregates,
    read_token_stats,
)
from coding_agent_evaluations.report import as_token_table, load_summaries
from coding_agent_evaluations.runner import _token_stats

SESSION_ID = "cae-0-abcdef012345"


def message_entry(
    *,
    timestamp: int = 1_000,
    ttft_ms: float | None = 100.0,
    decode_seconds: float | None = 2.0,
    decode_tokens: int | None = 200,
    gen_tps: float | None = 100.0,
    decode_tps: float | None = 100.0,
    prefill_tps: float | None = 1_000.0,
    input_tokens: int | None = 500,
    output_tokens: int | None = 200,
    cache_read: int | None = None,
    cost: float | None = None,
    model: str = "m",
) -> dict[str, Any]:
    """One `token_speed_stats` entry as the extension writes it."""
    data: dict[str, Any] = {
        "providerId": "cae-eval",
        "api": "openai-completions",
        "model": model,
        "timestamp": timestamp,
        "requestStartAt": timestamp - 500,
        "messageStartAt": timestamp - 400,
        "lastChunkAt": timestamp,
    }
    optional = {
        "ttftMs": ttft_ms,
        "decodeSeconds": decode_seconds,
        "decodeTokens": decode_tokens,
        "genTps": gen_tps,
        "decodeTps": decode_tps,
        "prefillTps": prefill_tps,
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
    }
    data.update({key: value for key, value in optional.items() if value is not None})
    if cache_read is not None:
        data["cacheReadTokens"] = cache_read
        data["cacheAwarePrefillTokens"] = max(0, (input_tokens or 0) - cache_read)
    if cost is not None:
        data["costTotal"] = cost
    return {"type": "custom", "customType": ENTRY_TYPE, "data": data}


def turn_summary_entry(
    *, timestamp: int = 2_000, message_count: int = 1, **kwargs: Any
) -> dict[str, Any]:
    """One `token_speed_stats_turn_summary` entry."""
    return {
        "type": "custom",
        "customType": SUMMARY_ENTRY_TYPE,
        "data": {
            "timestamp": timestamp,
            "messageCount": message_count,
            "avgTTFTMs": 100,
            "providerId": "cae-eval",
            "models": ["m"],
            **kwargs,
        },
    }


def write_session(directory: Path, *entries: dict[str, Any], name: str | None = None) -> Path:
    """Write a session JSONL the way pi names it: `<timestamp>_<session-id>.jsonl`."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name or f"2026-01-01T00-00-00-000Z_{SESSION_ID}.jsonl")
    with path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"type": "session", "id": SESSION_ID}) + "\n")
        for entry in entries:
            stream.write(json.dumps(entry) + "\n")
    return path


def test_reads_both_entry_types(tmp_path: Path) -> None:
    write_session(
        tmp_path,
        message_entry(timestamp=1_000),
        turn_summary_entry(timestamp=2_000),
        message_entry(timestamp=3_000),
    )
    stats = read_token_stats(tmp_path, SESSION_ID)
    assert len(stats["messages"]) == 2
    assert len(stats["turns"]) == 1
    assert stats["aggregate"]["requests"] == 2
    assert stats["source"] == "pi-token-stats"


def test_entries_are_ordered_by_timestamp(tmp_path: Path) -> None:
    # pi appends in order, but a resumed session is appended to by more than one process.
    write_session(
        tmp_path,
        message_entry(timestamp=3_000),
        message_entry(timestamp=1_000),
        message_entry(timestamp=2_000),
    )
    stats = read_token_stats(tmp_path, SESSION_ID)
    assert [entry["timestamp"] for entry in stats["messages"]] == [
        1_000,
        2_000,
        3_000,
    ]


def test_raw_entries_are_preserved_verbatim(tmp_path: Path) -> None:
    # The extension's own schema is kept so a result can be re-aggregated later, or read
    # with upstream's `jq` recipes, without re-running anything.
    entry = message_entry(cache_read=100, cost=0.25)
    write_session(tmp_path, entry)
    stats = read_token_stats(tmp_path, SESSION_ID)
    assert stats["messages"] == [entry["data"]]
    assert stats["aggregate"]["cache_aware_prefill_tokens"] == 400


def test_only_this_rollout_session_is_read(tmp_path: Path) -> None:
    # Two rollouts can share a parent directory; reading both would double every count.
    write_session(tmp_path, message_entry(), name=f"2026-01-01T00-00-00-000Z_{SESSION_ID}.jsonl")
    write_session(tmp_path, message_entry(), name="2026-01-01T00-00-01-000Z_other.jsonl")
    stats = read_token_stats(tmp_path, SESSION_ID)
    assert stats["aggregate"]["requests"] == 1
    assert stats["session_files"] == [
        str(tmp_path / f"2026-01-01T00-00-00-000Z_{SESSION_ID}.jsonl")
    ]


def test_falls_back_to_any_session_when_the_id_does_not_match(tmp_path: Path) -> None:
    # pi renames the file on fork/resume. Losing the stats to a filename would be a silent
    # gap in the run, so the whole directory is read rather than reporting nothing.
    write_session(tmp_path, message_entry(), name="2026-01-01T00-00-00-000Z_forked.jsonl")
    stats = read_token_stats(tmp_path, SESSION_ID)
    assert stats["aggregate"]["requests"] == 1


def test_a_missing_session_reports_nothing_rather_than_failing(tmp_path: Path) -> None:
    # A harness killed before pi's first write leaves no file. That is a run to report, not
    # an exception to raise.
    stats = read_token_stats(tmp_path, SESSION_ID)
    assert stats["aggregate"]["requests"] == 0
    assert stats["session_files"] == []
    assert stats["aggregate"]["ttft_ms_mean"] is None


def test_a_truncated_session_keeps_what_was_flushed(tmp_path: Path) -> None:
    # pi is killed on the time budget; the last line can be half-written.
    write_session(tmp_path, message_entry())
    path = tmp_path / f"2026-01-01T00-00-00-000Z_{SESSION_ID}.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"type":"custom","customType":"token_spe')
    stats = read_token_stats(tmp_path, SESSION_ID)
    assert stats["aggregate"]["requests"] == 1


def test_non_custom_entries_are_ignored(tmp_path: Path) -> None:
    write_session(
        tmp_path,
        {"type": "message", "message": {"role": "assistant", "content": []}},
        message_entry(),
        {"type": "custom", "customType": "some_other_extension", "data": {"x": 1}},
    )
    stats = read_token_stats(tmp_path, SESSION_ID)
    assert stats["aggregate"]["requests"] == 1


def test_token_totals_are_summed_not_maxed(tmp_path: Path) -> None:
    # Each assistant message is a separate request; the rollout's cost is all of them.
    write_session(
        tmp_path,
        message_entry(input_tokens=500, output_tokens=100),
        message_entry(input_tokens=700, output_tokens=150),
    )
    aggregate_ = read_token_stats(tmp_path, SESSION_ID)["aggregate"]
    assert aggregate_["input_tokens"] == 1_200
    assert aggregate_["output_tokens"] == 250


def test_missing_cache_reads_read_as_zero_not_none(tmp_path: Path) -> None:
    # A local OpenAI-compatible endpoint reports no cache reads. `cache_read_tokens` should
    # read as a real zero (nothing was cached) while `cache_aware_prefill_tokens` stays None,
    # because there is no cache-aware figure to report.
    write_session(tmp_path, message_entry(input_tokens=500))
    aggregate_ = read_token_stats(tmp_path, SESSION_ID)["aggregate"]
    assert aggregate_["cache_read_tokens"] == 0
    assert aggregate_["cache_aware_prefill_tokens"] is None


def test_absent_metrics_are_none_not_zero(tmp_path: Path) -> None:
    # The distinction that matters: zero means "measured, it was zero"; None means "not
    # measured". A TTFT of 0 ms is not a real measurement.
    write_session(tmp_path, message_entry(ttft_ms=None, decode_tps=None, cost=None))
    aggregate_ = read_token_stats(tmp_path, SESSION_ID)["aggregate"]
    assert aggregate_["ttft_ms_mean"] is None
    assert aggregate_["ttft_ms_p90"] is None
    assert aggregate_["decode_tps_mean"] is None
    assert aggregate_["cost_usd"] is None


def test_latency_is_reported_as_a_distribution(tmp_path: Path) -> None:
    # A mean hides the tail, and the tail is what makes an evaluation slow.
    write_session(
        tmp_path,
        *[message_entry(timestamp=index * 1_000, ttft_ms=value) for index, value in enumerate([10, 20, 30, 4_000])],
    )
    aggregate_ = read_token_stats(tmp_path, SESSION_ID)["aggregate"]
    assert aggregate_["ttft_ms_min"] == 10
    assert aggregate_["ttft_ms_max"] == 4_000
    assert aggregate_["ttft_ms_p50"] == pytest.approx(25)
    assert aggregate_["ttft_ms_p90"] > aggregate_["ttft_ms_p50"]
    assert aggregate_["ttft_ms_mean"] == pytest.approx(1_015)


def test_throughput_is_reported_as_a_mean_and_as_a_ratio(tmp_path: Path) -> None:
    # The mean rewards a run made of many tiny requests; the ratio of totals does not.
    write_session(
        tmp_path,
        message_entry(decode_tps=1_000.0, decode_tokens=1_000, decode_seconds=1.0),
        message_entry(decode_tps=10.0, decode_tokens=100, decode_seconds=10.0),
    )
    aggregate_ = read_token_stats(tmp_path, SESSION_ID)["aggregate"]
    assert aggregate_["decode_tps_mean"] == pytest.approx(505.0)
    assert aggregate_["decode_tps_total"] == pytest.approx(100.0)
    assert aggregate_["generation_sec"] == pytest.approx(11.0)


def test_generation_span_is_measured_end_to_end(tmp_path: Path) -> None:
    write_session(
        tmp_path,
        message_entry(timestamp=10_000),
        message_entry(timestamp=20_000),
    )
    aggregate_ = read_token_stats(tmp_path, SESSION_ID)["aggregate"]
    # First request start to last chunk, so the number includes the time the harness spent
    # running tools between requests.
    assert aggregate_["span_sec"] == pytest.approx((20_000 - (10_000 - 500)) / 1000)


def test_models_seen_are_reported(tmp_path: Path) -> None:
    # pi can switch model mid-session (a compaction fallback); the run should say so.
    write_session(
        tmp_path,
        message_entry(model="small"),
        message_entry(model="big"),
        message_entry(model="small"),
    )
    aggregate_ = read_token_stats(tmp_path, SESSION_ID)["aggregate"]
    assert aggregate_["models"] == ["big", "small"]
    assert aggregate_["provider_id"] == "cae-eval"


def test_aggregate_of_nothing_is_empty_not_broken() -> None:
    aggregate_ = aggregate([], [])
    assert aggregate_["requests"] == 0
    assert aggregate_["input_tokens"] == 0
    assert aggregate_["models"] == []
    assert aggregate_["ttft_ms_mean"] is None


def test_merging_rollouts_weights_rates_by_request_count() -> None:
    # An unweighted mean would let a 3-request rollout set the run's headline throughput.
    merged = merge_aggregates(
        [
            {
                "requests": 100,
                "turns": 10,
                "input_tokens": 1_000,
                "output_tokens": 500,
                "ttft_ms_mean": 1_000.0,
                "decode_tps_mean": 100.0,
                "decode_tps_total": 90.0,
                "gen_tps_mean": 110.0,
                "prefill_tps_mean": 10.0,
                "generation_sec": 100.0,
                "span_sec": 200.0,
                "cost_usd": 0.5,
                "models": ["big"],
            },
            {
                "requests": 1,
                "turns": 1,
                "input_tokens": 10,
                "output_tokens": 5,
                "ttft_ms_mean": 10.0,
                "decode_tps_mean": 10.0,
                "decode_tps_total": 10.0,
                "gen_tps_mean": 10.0,
                "prefill_tps_mean": 10.0,
                "generation_sec": 1.0,
                "span_sec": 1.0,
                "cost_usd": 0.01,
                "models": ["small"],
            },
        ]
    )
    assert merged is not None
    assert merged["rollouts"] == 2
    assert merged["requests"] == 101
    # The weighted mean must sit far closer to the 100-request rollout than a plain mean of
    # 55.0 would, which is the whole point of weighting.
    assert merged["ttft_ms_mean"] == pytest.approx((100 * 1_000 + 10) / 101, abs=0.01)
    assert merged["decode_tps_mean"] == pytest.approx((100 * 100 + 10) / 101, abs=0.01)
    assert merged["input_tokens"] == 1_010
    assert merged["cost_usd"] == pytest.approx(0.51)
    assert merged["models"] == ["big", "small"]


def test_merging_nothing_reports_nothing() -> None:
    # A run with no measured rollouts must not claim a speed of zero.
    assert merge_aggregates([]) is None
    assert merge_aggregates([{"requests": 0}]) is None
    assert merge_aggregates([None, {}]) is None


def test_trajectory_carries_the_stats(tmp_path: Path) -> None:
    # End to end through the harness: build the command and env, write the session pi would
    # have written, then parse. This is the path a real rollout takes.
    run = make_run(tmp_path)
    harness = PiHarness()
    config_dir = tmp_path / "harness-pi"
    session_id = harness.session_id(run)
    harness.build_command(run, config_dir, session_id)
    write_session(
        harness.session_dir(config_dir),
        message_entry(),
        turn_summary_entry(),
        name=f"2026-01-01T00-00-00-000Z_{session_id}.jsonl",
    )
    parsed = harness.parse(run, "", 0)
    assert parsed["token_stats"]["aggregate"]["requests"] == 1
    assert len(parsed["token_stats"]["turns"]) == 1
    assert parsed["token_stats"]["messages"][0]["model"] == "m"


def test_trajectory_omits_stats_when_switched_off(tmp_path: Path) -> None:
    # No key at all, so "not measured" is never mistaken for "measured zero".
    run = make_run(tmp_path)
    config = type(run.config)(name="pi", options={"token_stats": False})
    run = type(run)(**{**run.__dict__, "config": config})
    assert "token_stats" not in PiHarness().parse(run, "", 0)


def test_an_empty_stats_reading_is_still_reported(tmp_path: Path) -> None:
    # If the extension failed to load, the trajectory should show zero requests rather than
    # silently omitting the key — the omission would read as "switched off".
    run = make_run(tmp_path)
    harness = PiHarness()
    config_dir = tmp_path / "harness-pi"
    harness.build_command(run, config_dir, harness.session_id(run))
    parsed = harness.parse(run, "", 0)
    assert parsed["token_stats"]["aggregate"]["requests"] == 0


def test_the_session_file_survives_into_the_harness_logs(tmp_path: Path) -> None:
    # The workspace is deleted when a run ends; the raw session is the only way to re-derive
    # a number this project did not think to compute.
    run = make_run(tmp_path)
    harness = PiHarness()
    config_dir = tmp_path / "harness-pi"
    session_id = harness.session_id(run)
    harness.build_command(run, config_dir, session_id)
    path = write_session(
        harness.session_dir(config_dir),
        message_entry(),
        name=f"2026-01-01T00-00-00-000Z_{session_id}.jsonl",
    )
    parsed = harness.parse(run, "", 0)
    # Parsing reads the session without consuming or rewriting it.
    assert parsed["token_stats"]["aggregate"]["requests"] == 1
    assert path.is_file()
    assert len(jsonl(path)) == 2


def test_parse_still_works_without_any_session(tmp_path: Path) -> None:
    # A pi crash must not also lose the trajectory.
    run = make_run(tmp_path)
    parsed = PiHarness().parse(run, '{"type":"agent_start"}\n', 1)
    assert parsed["exit_reason"] == "harness_error"
    assert parsed["token_stats"]["aggregate"]["requests"] == 0


def test_the_summary_rolls_the_rollouts_up(tmp_path: Path) -> None:
    # The bridge from a harness trajectory to `summary.json`: the per-rollout aggregate goes
    # into results.jsonl, and the run-level view is what the comparison table reads.
    from dataclasses import replace as dataclass_replace

    from coding_agent_evaluations.config import ModelConfig
    from coding_agent_evaluations.runner import append_result, write_summary

    config = dataclass_replace(
        load_config(CONFIG_ROOT / "base.yaml"),
        output_dir=tmp_path / "out",
        harness=HarnessConfig(name="pi"),
        model=ModelConfig(name="m", base_url="http://127.0.0.1:1234/v1"),
    )
    results_path = tmp_path / "out" / "results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    expected = set()
    for index in range(2):
        instance_id = f"task-{index}"
        expected.add((instance_id, 0))
        trajectory = {
            "token_stats": {
                "aggregate": {
                    "requests": 10,
                    "input_tokens": 1_000,
                    "output_tokens": 200,
                    "ttft_ms_mean": 100.0,
                    "decode_tps_mean": 50.0,
                }
            }
        }
        append_result(
            results_path,
            {
                "instance_id": instance_id,
                "seed": 0,
                "status": "completed",
                "reward": 1.0,
                "exit_reason": "agent",
                "n_steps": 3,
                "elapsed_sec": 10.0,
                "token_stats": _token_stats(trajectory),
            },
        )
    summary = write_summary(tmp_path / "out", config, expected, results_path, 20.0)
    assert summary["token_stats"]["rollouts"] == 2
    assert summary["token_stats"]["requests"] == 20
    assert summary["token_stats"]["input_tokens"] == 2_000
    assert summary["token_stats"]["ttft_ms_mean"] == pytest.approx(100.0)
    # And it reaches the comparison table the run prints.
    assert "ttft_p50_ms" in as_token_table(load_summaries([str(tmp_path / "out")]))


def test_a_finished_rollout_carries_its_token_stats_into_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end through the runner, with the harness and the runtime stubbed.

    This is the seam the unit tests above skip: a harness trajectory becoming a results row
    and a summary. A rollout's token speed is only useful if it survives that trip, and no
    benchmark should be needed to check it.
    """
    from dataclasses import replace as dataclass_replace

    from coding_agent_evaluations import runner
    from coding_agent_evaluations.config import ModelConfig

    config = dataclass_replace(
        load_config(CONFIG_ROOT / "base.yaml"),
        output_dir=tmp_path / "out",
        harness=HarnessConfig(name="pi"),
        model=ModelConfig(name="m", base_url="http://127.0.0.1:1234/v1"),
        num_tasks=1,
        seeds_per_task=1,
        resume=False,
    )
    monkeypatch.setattr(
        runner, "load_tasks", lambda *args, **kw: [{"instance_id": "task-0"}]
    )

    class StubRuntime:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.task = {"instance_id": "task-0"}
            self.logger = logging.getLogger("stub")

        def compute_reward(self) -> tuple[float, str]:
            return 1.0, "passed"

        def close(self) -> None:
            pass

    class StubHarness:
        name = "pi"
        enforces_max_steps = False
        ignored_model_params = ()

        def run(self, run: Any) -> dict[str, Any]:
            return {
                "token_stats": {"aggregate": {"requests": 4, "input_tokens": 900}},
                "n_steps": 2,
                "exit_reason": "agent",
                "output_patch": "",
            }

    monkeypatch.setattr(runner, "get_runtime", lambda kind: StubRuntime)
    monkeypatch.setattr(runner, "build_harness", lambda name: StubHarness)

    summary = runner.run_evaluation(config)
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["token_stats"] == {"requests": 4, "input_tokens": 900}
    assert summary["token_stats"]["requests"] == 4
    assert summary["token_stats"]["input_tokens"] == 900

    trajectory = json.loads(
        (tmp_path / "out" / "trajectories/task-0/trajectory_seed-0.json").read_text(
            encoding="utf-8"
        )
    )
    assert trajectory["token_stats"]["aggregate"]["requests"] == 4


def test_a_run_without_token_stats_says_so(tmp_path: Path) -> None:
    # leaf and opencode record none: the key must be absent rather than zero-filled, or the
    # comparison would show a speed of 0 tokens/s for a harness that never measured one.
    from dataclasses import replace as dataclass_replace

    from coding_agent_evaluations.config import ModelConfig
    from coding_agent_evaluations.runner import append_result, write_summary

    config = dataclass_replace(
        load_config(CONFIG_ROOT / "base.yaml"),
        output_dir=tmp_path / "out",
        harness=HarnessConfig(name="leaf"),
        model=ModelConfig(name="m", base_url="http://127.0.0.1:1234/v1"),
    )
    results_path = tmp_path / "out" / "results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    append_result(
        results_path,
        {
            "instance_id": "task-0",
            "seed": 0,
            "status": "completed",
            "reward": 1.0,
            "exit_reason": "agent",
            "n_steps": 3,
            "elapsed_sec": 10.0,
            "token_stats": None,
        },
    )
    summary = write_summary(
        tmp_path / "out", config, {("task-0", 0)}, results_path, 20.0
    )
    assert "token_stats" not in summary
    assert as_token_table(load_summaries([str(tmp_path / "out")])) == ""
