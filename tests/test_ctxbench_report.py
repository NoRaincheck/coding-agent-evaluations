"""Tests for the ctxbench report.

The report is where a reader decides what a run means, so the cases here are all ways a
number can be shown in a way that misleads: a suite that did not run rendered as a zero,
a rate without its denominator, a truncation flag raised by a tokenizer difference rather
than by dropped context.
"""

from __future__ import annotations

from typing import Any

from coding_agent_evaluations.ctxbench.report import as_markdown, as_table, caveats


def payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "report_version": 1,
        "run_id": "ctx_20261007_1430",
        "generated": "2026-10-07T14:30:00Z",
        "model": "m",
        "base_url": "http://h/v1",
        "harness": "leaf",
        "harness_version": "cae 0.1.0",
        "tokenizer": "o200k_base",
        "duration_sec": 12.0,
        "config": {
            "context_sizes": [16_000],
            "ctxbench": {
                "enabled": ["prefill", "generation", "needle", "execution", "memory"],
                "repeats": 3,
                "needle_trials": 6,
            },
        },
        "results": [
            {
                "context_size_tokens": 16_000,
                "speed": {
                    "prefill": {"ttft_ms": 450.0, "prefill_tps": 35.0},
                    "generation": {"tps": 65.2, "total_latency_ms": 3200.0},
                },
                "correctness": {
                    "needle_in_haystack_pass_rate": 1.0,
                    "needle": {"trials": 6},
                    "code_execution_pass_at_1": 0.85,
                },
                "harness_health": {
                    "prompt_tokens_sent": 16_000,
                    "context_truncated": False,
                    "context_truncation": {"received_tokens": 20_297, "received_over_sent": 1.27},
                },
                "errors": [],
            }
        ],
        "memory": {"iterations": 10, "growth_mb": 12.0, "monotonic_growth": False},
    }
    base.update(overrides)
    return base


def test_table_has_a_row_per_context_size() -> None:
    data = payload()
    data["config"]["context_sizes"] = [16_000, 128_000]
    data["results"].append(dict(data["results"][0], context_size_tokens=128_000))
    table = as_table(data)
    assert "16,000" in table
    assert "128,000" in table
    assert table.count("\n") == 3


def test_absent_suite_renders_as_dash_not_zero() -> None:
    """A skipped suite and a suite that measured zero are different facts."""
    data = payload()
    data["config"]["ctxbench"]["enabled"] = ["prefill"]
    data["results"][0]["correctness"].pop("code_execution_pass_at_1")
    cells = as_table(data).splitlines()[-1].split()
    # pass@1 is the column after the needle rate and its trial count.
    assert cells[cells.index("6") + 1] == "-"


def test_measured_zero_renders_as_zero_percent() -> None:
    """The counterpart: a suite that ran and measured zero is a real finding and must be
    shown as one."""
    data = payload()
    data["results"][0]["correctness"]["code_execution_pass_at_1"] = 0.0
    cells = as_table(data).splitlines()[-1].split()
    assert cells[cells.index("6") + 1] == "0%"


def test_needle_rate_is_shown_with_its_denominator() -> None:
    row = as_table(payload()).splitlines()[-1]
    assert "100%" in row
    assert " 6" in row


def test_truncation_flag_is_shown_yes_or_no() -> None:
    data = payload()
    assert "no" in as_table(data).splitlines()[-1]
    data["results"][0]["harness_health"]["context_truncated"] = True
    assert "YES" in as_table(data).splitlines()[-1]


def test_empty_run_says_so() -> None:
    data = payload(results=[], config={"context_sizes": [], "ctxbench": {"enabled": [], "repeats": 3}})
    assert "no context sizes measured" in as_table(data)


def test_markdown_names_the_run() -> None:
    text = as_markdown(payload())
    assert "ctxbench report v1" in text
    assert "m" in text
    assert "http://h/v1" in text
    assert "o200k_base" in text
    assert "16,000" in text


def test_notes_flag_missing_suites() -> None:
    data = payload()
    data["config"]["ctxbench"]["enabled"] = ["prefill"]
    notes = caveats(data)
    assert any("execution suite did not run" in note for note in notes)
    assert any("memory suite did not run" in note for note in notes)


def test_notes_flag_rows_the_endpoint_refused_as_oversized() -> None:
    """A refused request is not a measurement of the model, and the table's x-axis must
    not read as one."""
    data = payload()
    data["results"][0]["context_size_tokens"] = 128_000
    data["results"][0]["speed"]["prefill"] = {"error": "rejected"}
    data["results"][0]["harness_health"]["context_truncation"] = {
        "server_context_limit": 100_096,
        "exceeds_server_limit": True,
        "received_tokens": None,
    }
    note = " ".join(caveats(data))
    assert "100,096-token context window" in note
    assert "128,000" in note
    assert "carry no model measurement" in note


def test_no_limit_note_when_every_row_was_measured() -> None:
    assert not any("context window" in note for note in caveats(payload()))


def test_notes_state_that_ttft_is_uncached() -> None:
    """The TTFT column answers "what does a fresh context cost", not "what does a warm
    prefix cost". A reader comparing it against a session figure has to know which."""
    data = payload()
    data["results"][0]["speed"]["prefill"]["prefix_cache_busted"] = True
    assert any("prefix-cache-busted" in note for note in caveats(data))


def test_no_ttft_note_on_a_single_repeat() -> None:
    data = payload()
    data["results"][0]["speed"]["prefill"]["prefix_cache_busted"] = False
    assert not any("prefix-cache-busted" in note for note in caveats(data))


def test_notes_flag_unobservable_needle_trials() -> None:
    """A reasoning model that spent its whole budget never tested retrieval. Counting
    those as failures would report the token budget as a model property."""
    data = payload()
    data["results"][0]["correctness"]["needle"]["unobservable"] = 3
    note = " ".join(caveats(data))
    assert "3 needle trial(s)" in note
    assert "excluded from the pass rate" in note


def test_no_unobservable_note_when_every_trial_replied() -> None:
    assert not any("needle trial" in note for note in caveats(payload()))


def test_notes_flag_low_repeats() -> None:
    data = payload()
    data["config"]["ctxbench"]["repeats"] = 1
    assert any("repeats=1" in note for note in caveats(data))


def test_notes_flag_monotonic_memory_growth_as_a_candidate() -> None:
    """Growth is a candidate leak, not a verdict: a short trace can rise for boring
    reasons, so the wording has to leave room for a re-run to contradict it."""
    data = payload(memory={"iterations": 10, "growth_mb": 12.0, "monotonic_growth": True})
    note = " ".join(caveats(data))
    assert "monotonically" in note
    assert "re-run to confirm" in note


def test_notes_explain_a_skipped_memory_suite() -> None:
    """A leak trace needs repeated large requests; if the endpoint cannot hold the size,
    saying "0 samples" reads as a broken measurement rather than a skipped one."""
    data = payload(
        memory={
            "skipped": True,
            "iterations": 0,
            "note": "the requested size exceeds the endpoint's context window",
        }
    )
    note = " ".join(caveats(data))
    assert "memory suite did not run" in note
    assert "context window" in note


def test_no_memory_trace_note_when_the_suite_was_never_requested() -> None:
    """A "0 samples" note on top of "did not run" reads as a broken measurement rather
    than a suite that was never asked for."""
    data = payload()
    data["config"]["ctxbench"]["enabled"] = ["prefill"]
    note = " ".join(caveats(data))
    assert "memory suite did not run" in note
    assert "memory trace has" not in note


def test_no_unmeasurable_note_when_the_suite_was_skipped() -> None:
    """"RSS was not measurable" is a different fact from "the suite was skipped", and
    claiming the platform cannot measure RSS when it never tried is a wrong note."""
    data = payload(
        memory={"skipped": True, "iterations": 0, "note": "the size exceeds the window"}
    )
    assert not any("not measurable" in note for note in caveats(data))


def test_unmeasurable_note_when_the_suite_ran_but_produced_no_samples() -> None:
    data = payload(memory={"iterations": 0, "samples": 0, "samples_mb": []})
    assert any("not measurable" in note for note in caveats(data))


def test_notes_flag_a_short_memory_trace() -> None:
    data = payload(memory={"iterations": 2, "growth_mb": 1.0, "monotonic_growth": False})
    note = " ".join(caveats(data))
    assert "2 samples" in note
    assert "not evidence of a leak either way" in note


def test_notes_state_the_tokenizer_difference() -> None:
    """The `sent` column is a local count under a different vocabulary than the model's.
    A reader comparing it to the server's number has to know that."""
    note = " ".join(caveats(payload()))
    assert "labeled by local o200k_base counts" in note
    assert "1.27x" in note


def test_notes_are_quiet_when_counts_agree() -> None:
    data = payload()
    data["results"][0]["harness_health"]["context_truncation"] = {
        "received_tokens": 16_010,
        "received_over_sent": 1.0,
    }
    note = " ".join(caveats(data))
    assert "the endpoint agreed within" in note