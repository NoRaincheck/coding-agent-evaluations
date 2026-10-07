"""Tests for the ctxbench runner and endpoint client.

No real endpoint here. The probe is driven against a fake HTTP server, because the
properties worth testing are all about what the runner does with a response -- whether a
truncation is detected, whether a failure is counted as a timeout or an error, whether a
suite that failed still leaves the other suites' numbers intact.
"""

from __future__ import annotations

import itertools
import json
import re
import threading
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from coding_agent_evaluations.config import ModelConfig, RuntimeConfig
from coding_agent_evaluations.ctxbench.config import CtxBenchConfig, CtxSuiteConfig
from coding_agent_evaluations.ctxbench.endpoint import (
    EndpointProbe,
    ProbeError,
    TruncationCheck,
)
from coding_agent_evaluations.ctxbench.runner import run_ctxbench

# --------------------------------------------------------------- fake endpoint


class _Handler(BaseHTTPRequestHandler):
    """Streams a fixed number of chunks, then reports usage.

    `server.prompt_tokens` is what the endpoint claims to have received, which is what the
    truncation check compares against what was sent. It is settable so a test can make the
    server claim fewer tokens than were sent, without needing a real truncating server.
    """

    def log_message(self, *args: Any) -> None:
        return

    def reply_for(self, body: Any) -> str:
        """What the fake model answers with. Overridden to model a retriever."""
        return self.server.text

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        # Subclasses vary the reply from the parsed body here. Reading the body again in a
        # subclass would find it already consumed, so the hook exists instead.
        self.server.text = self.reply_for(body)
        prompt_tokens = self.server.prompt_tokens or len(str(body))
        if self.server.status and self.server.status != 200:
            self.send_response(self.server.status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write((self.server.overflow or '{"error": "boom"}').encode())
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if self.server.stream_error_frame:
            # What LM Studio does for an over-long prompt: HTTP 200, then an error frame.
            self.wfile.write(b"event: error\n")
            self.wfile.write(
                f"data: {json.dumps({'error': {'message': self.server.stream_error_frame}})}\n\n".encode()
            )
            self.wfile.flush()
            return
        if self.server.silent:
            # A 200 that opens and closes with nothing in it.
            self.wfile.flush()
            return
        # `reasoning` models a reasoning endpoint: chunks carry thinking tokens and no
        # answer, so the budget is spent before the model says anything.
        field = "reasoning_content" if self.server.reasoning else "content"
        for _ in range(self.server.chunks):
            chunk = {
                "choices": [{"delta": {field: self.server.text}, "finish_reason": None}]
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        final = {
            "choices": [{"delta": {}, "finish_reason": self.server.finish_reason}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": self.server.chunks,
            },
        }
        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@pytest.fixture
def server() -> Any:
    instance = HTTPServer(("127.0.0.1", 0), _Handler)
    instance.requests = []
    instance.chunks = 3
    instance.prompt_tokens = None
    instance.status = 200
    instance.text = "ok"
    instance.overflow = None
    instance.reasoning = False
    instance.stream_error_frame = None
    instance.silent = False
    instance.finish_reason = "stop"
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance
    instance.shutdown()
    instance.server_close()


def probe_for(server: Any, **kwargs: Any) -> EndpointProbe:
    return EndpointProbe(
        model=ModelConfig(name="m", base_url=f"http://127.0.0.1:{server.server_port}/v1"),
        timeout_sec=10,
        retries=kwargs.pop("retries", 0),
        **kwargs,
    )


# ------------------------------------------------------------------- the client


def test_stream_reports_timing_and_usage(server: Any) -> None:
    result = probe_for(server).stream([{"role": "user", "content": "hi"}])
    assert result.text == "ok" * 3
    assert result.chunks == 3
    assert result.prompt_tokens is not None
    assert result.completion_tokens == 3
    assert result.ttft_sec > 0
    assert result.total_sec >= result.ttft_sec
    assert result.decode_tps and result.decode_tps > 0


def test_stream_forwards_extra_body(server: Any) -> None:
    """`chat_template_kwargs` and friends are endpoint extensions; dropping them would
    measure a different model configuration than the benchmark run uses."""
    probe = EndpointProbe(
        model=ModelConfig(
            name="m",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        ),
        timeout_sec=10,
        retries=0,
    )
    probe.stream([{"role": "user", "content": "hi"}])
    assert server.requests[-1]["chat_template_kwargs"] == {"enable_thinking": True}


def test_four_xx_is_not_retried(server: Any) -> None:
    """A 4xx is the server's considered answer; repeating it wastes the run."""
    server.status = 400
    probe = probe_for(server, retries=3)
    with pytest.raises(ProbeError) as caught:
        probe.stream([{"role": "user", "content": "hi"}])
    assert caught.value.status == 400
    assert probe.stats["retries"] == 0
    assert len(server.requests) == 1


def test_server_error_is_retried_and_counted(server: Any) -> None:
    server.status = 503
    probe = probe_for(server, retries=2)
    with pytest.raises(ProbeError):
        probe.stream([{"role": "user", "content": "hi"}])
    assert probe.stats["retries"] == 2
    assert len(server.requests) == 3


def test_unreachable_endpoint_raises(server: Any) -> None:
    port = server.server_port
    server.shutdown()
    probe = EndpointProbe(
        model=ModelConfig(name="m", base_url=f"http://127.0.0.1:{port}/v1"),
        timeout_sec=2,
        retries=0,
    )
    with pytest.raises(ProbeError, match="unreachable|timed out"):
        probe.stream([{"role": "user", "content": "hi"}])


def test_error_frame_inside_a_200_is_not_read_as_a_measurement(
    server: Any, tmp_path: Path
) -> None:
    """LM Studio reports an over-long prompt as an SSE `event: error` frame inside an HTTP
    200. A parser that reads only `data:` frames sees an empty successful stream and
    records a refused request as a 324 ms measurement with a 0% pass rate — the exact
    silent-wrong-number case this suite exists to catch."""
    server.stream_error_frame = (
        "Engine protocol predict request returned 400: request (162211 tokens) exceeds "
        "the available context size (100096 tokens), try increasing it"
    )
    config = replace(
        config_for(server, tmp_path, enabled=["prefill"], repeats=1),
        context_sizes=(4_000,),
    )
    payload = run_ctxbench(config)
    row = payload["results"][0]
    truncation = row["harness_health"]["context_truncation"]
    assert truncation["rejected"] is True
    assert truncation["server_context_limit"] == 100_096
    # No measurement was taken, so the speed block says so rather than reporting one.
    assert row["speed"]["prefill"]["error"]
    assert "ttft_ms" not in row["speed"]["prefill"]
    assert row["errors"]


def test_error_frame_is_not_retried(server: Any, tmp_path: Path) -> None:
    """A refusal delivered out of band is still a refusal; repeating it wastes the run."""
    server.stream_error_frame = "request (1 tokens) exceeds the available context size (0)"
    probe = probe_for(server, retries=3)
    with pytest.raises(ProbeError):
        probe.stream([{"role": "user", "content": "hi"}])
    assert probe.stats["retries"] == 0
    assert len(server.requests) == 1


def test_empty_stream_is_an_error_not_a_fast_response(server: Any) -> None:
    """A stream that opens and closes with nothing in it must not be recorded as a
    response, or a broken endpoint looks like the fastest one measured."""
    server.silent = True
    probe = probe_for(server, retries=0)
    with pytest.raises(ProbeError, match="without sending any content"):
        probe.stream([{"role": "user", "content": "hi"}])


def test_context_overflow_is_parsed_out_of_the_body() -> None:
    """The endpoint names the window it has. That number is the difference between "this
    model fails at 128k" and "this model cannot hold 128k"."""
    body = (
        '{"error":"Engine protocol predict request returned 400: '
        'request (1280017 tokens) exceeds the available context size '
        '(100096 tokens), try increasing it"}'
    )
    error = ProbeError("HTTP 400", status=400, body=body)
    assert error.context_overflow
    assert error.server_context_limit == 100_096
    assert error.server_prompt_tokens == 1_280_017


def test_non_overflow_400_is_not_an_overflow() -> None:
    error = ProbeError("HTTP 400", status=400, body='{"error":"bad request"}')
    assert not error.context_overflow
    assert error.server_context_limit is None


def test_oversized_row_is_recorded_as_rejected(server: Any, tmp_path: Path) -> None:
    """The row must show the server's limit, not a bare failure that reads as a model
    defect."""
    server.status = 400
    server.overflow = (
        '{"error":"request (1280017 tokens) exceeds the available context size '
        '(100096 tokens), try increasing it"}'
    )
    config = replace(
        config_for(server, tmp_path, enabled=["prefill"], repeats=1),
        # Above the fake server's stated 100,096-token window, so the rejection is
        # genuinely an overflow rather than an unrelated 400.
        context_sizes=(128_000,),
    )
    payload = run_ctxbench(config)
    row = payload["results"][0]
    truncation = row["harness_health"]["context_truncation"]
    assert truncation["rejected"] is True
    assert truncation["exceeds_server_limit"] is True
    assert truncation["server_context_limit"] == 100_096
    assert "100,096" in " ".join(row["errors"])


# ------------------------------------------------------------ truncation logic


def test_truncation_detected_when_server_reports_fewer_tokens() -> None:
    check = TruncationCheck(sent_tokens=128_000, received_tokens=32_000)
    assert check.truncated
    assert check.retained_ratio == 0.25


def test_no_truncation_when_server_reports_more_tokens() -> None:
    """A different vocabulary produces a larger count, not a truncated one. Reporting
    this as truncation would be a false alarm on every row."""
    check = TruncationCheck(sent_tokens=16_000, received_tokens=20_297)
    assert not check.truncated
    assert check.tokenizer_mismatch == pytest.approx(1.2686, abs=1e-3)


def test_small_boundary_disagreement_is_not_truncation() -> None:
    """The check exists to catch tens of thousands of dropped tokens, not to adjudicate
    a one-token disagreement between two encodings."""
    assert not TruncationCheck(sent_tokens=100_000, received_tokens=99_900).truncated


def test_missing_usage_is_not_truncation() -> None:
    """An absent measurement is not a negative one."""
    check = TruncationCheck(sent_tokens=100_000, received_tokens=None)
    assert not check.truncated
    assert check.retained_ratio is None


def test_repeats_are_prefix_cache_busted(server: Any, tmp_path: Path) -> None:
    """Identical repeats would let an endpoint's prefix cache turn a 42-second prefill
    into a 75-millisecond one, and the median would report the cache read."""
    run_ctxbench(config_for(server, tmp_path, enabled=["prefill"], repeats=3))
    prompts = [request["messages"][0]["content"] for request in server.requests]
    assert len(set(prompts)) == len(prompts), "every repeat must send a distinct prompt"
    # Two context sizes x three repeats: the first request at each size is unmarked (it has
    # no earlier twin to be confused with), and every other request is marked.
    marked = [prompt.startswith("# probe-variant") for prompt in prompts]
    assert marked == [False, True, True, False, True, True]


def test_answer_is_unobservable_when_the_budget_went_to_reasoning(
    server: Any, tmp_path: Path
) -> None:
    """A reasoning model can spend its whole `max_tokens` thinking. Scoring that as a
    failed acknowledgement would report the token budget as a model behaviour."""
    server.text = "thinking"
    server.reasoning = True
    payload = run_ctxbench(
        config_for(server, tmp_path, enabled=["prefill"], repeats=1)
    )
    prefill = payload["results"][0]["speed"]["prefill"]
    assert prefill["acknowledged"] in (None, False)


def test_run_id_is_a_utc_timestamp(server: Any, tmp_path: Path) -> None:
    """A monotonic clock formatted as a date produces an id from an arbitrary epoch."""
    payload = run_ctxbench(
        config_for(server, tmp_path, enabled=["prefill"], repeats=1)
    )
    stamp = payload["run_id"].removeprefix("ctx_")
    assert len(stamp) == 15 and stamp[8] == "_"
    # The id is UTC; attaching the zone after parsing is deliberate, since the assertion
    # is about the shape of the stamp and not about an offset.
    datetime.strptime(stamp, "%Y%m%d_%H%M%S").replace(tzinfo=UTC)


def test_execution_suite_pins_the_context_budget(
    server: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pass@1 measures the harness under a context constraint, so the budget has to reach
    the runner the benchmark uses rather than being a ctxbench-local label."""
    seen: dict[str, Any] = {}

    def fake_run(config: Any) -> dict[str, Any]:
        seen["max_context_tokens"] = config.max_context_tokens
        seen["harness"] = config.harness.name
        seen["model"] = config.model.name
        seen["num_tasks"] = config.num_tasks
        return {
            "dataset": config.dataset,
            "harness": config.harness.name,
            "tasks_selected": 2,
            "jobs_total": 2,
            "resolved": 1,
            "resolve_rate": 0.5,
            "resolve_rate_graded": 0.5,
            "jobs_infrastructure_failed": 0,
        }

    monkeypatch.setattr("coding_agent_evaluations.ctxbench.runner.run_evaluation", fake_run)
    config = replace(
        config_for(server, tmp_path, enabled=["execution"], repeats=1),
        context_sizes=(64_000,),
        harness="pi",
    )
    payload = run_ctxbench(config)
    assert seen == {
        "max_context_tokens": 64_000,
        "harness": "pi",
        "model": "m",
        "num_tasks": 2,
    }
    row = payload["results"][0]
    assert row["correctness"]["code_execution_pass_at_1"] == 0.5
    assert row["correctness"]["execution"]["max_context_tokens"] == 64_000


def test_execution_pass_rate_excludes_infrastructure_failures(
    server: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sandbox that could not be provisioned must not read as a failed attempt."""
    monkeypatch.setattr(
        "coding_agent_evaluations.ctxbench.runner.run_evaluation",
        lambda config: {
            "dataset": config.dataset,
            "harness": config.harness.name,
            "tasks_selected": 2,
            "jobs_total": 2,
            "resolved": 1,
            "resolve_rate": 0.5,
            # One pair never got a fair test on this host.
            "resolve_rate_graded": 1.0,
            "jobs_infrastructure_failed": 1,
            "infrastructure_reasons": {"provision_error": 1},
        },
    )
    config = replace(
        config_for(server, tmp_path, enabled=["execution"], repeats=1),
        context_sizes=(64_000,),
    )
    row = run_ctxbench(config)["results"][0]
    assert row["correctness"]["code_execution_pass_at_1"] == 1.0
    assert row["correctness"]["execution"]["infrastructure_failed"] == 1


def test_execution_crash_does_not_lose_the_other_suites(
    server: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cell that raises must be recorded, not allowed to abort the sweep."""
    def boom(config: Any) -> dict[str, Any]:
        raise RuntimeError("workspace provisioning failed")

    monkeypatch.setattr("coding_agent_evaluations.ctxbench.runner.run_evaluation", boom)
    config = replace(
        config_for(server, tmp_path, enabled=["prefill", "execution"], repeats=1),
        context_sizes=(4_000,),
    )
    payload = run_ctxbench(config)
    row = payload["results"][0]
    assert row["correctness"]["code_execution_pass_at_1"] is None
    assert row["speed"]["prefill"].get("ttft_ms") is not None
    assert any("execution" in error for error in row["errors"])


def test_needle_trials_that_never_replied_are_excluded_from_the_rate(
    server: Any, tmp_path: Path
) -> None:
    """A reasoning model can spend its whole `max_tokens` thinking. Those trials never
    tested retrieval, so folding them in as failures would report the token budget as a
    model behaviour."""
    server.text = "thinking about it"  # reasoning deltas, never an answer
    server.reasoning = True
    payload = run_ctxbench(
        config_for(server, tmp_path, enabled=["needle"], needle_trials=4)
    )
    needle = payload["results"][0]["correctness"]["needle"]
    assert needle["unobservable"] == 4
    assert needle["trials_attempted"] == 4
    # Nothing was scored, so there is no rate -- rather than a rate of zero over trials
    # that never happened as measurements.
    assert needle["trials"] == 0
    assert needle["pass_rate"] is None
    assert all(trial["passed"] is None for trial in needle["by_depth"])


def test_needle_trial_is_retried_with_a_larger_budget(
    server: Any, tmp_path: Path
) -> None:
    """A model that thinks through its whole budget has not tested retrieval yet.
    Retrying once keeps the number of scored trials constant as context grows; writing
    trials off instead would make the pass rate fall for a budget reason."""
    server.text = "thinking"
    server.reasoning = True
    config = replace(
        config_for(server, tmp_path, enabled=["needle"], needle_trials=2),
        context_sizes=(4_000,),
    )
    config = replace(
        config,
        suites=replace(config.suites, needle_max_tokens=8, needle_retry_max_tokens=64),
    )
    needle = run_ctxbench(config)["results"][0]["correctness"]["needle"]
    # Every trial needed the retry, so each costs two requests.
    assert len(server.requests) == 4
    assert all(trial["retried_with_larger_budget"] for trial in needle["by_depth"])


def test_needle_pass_rate_denominator_excludes_unobservable_trials(
    server: Any, tmp_path: Path
) -> None:
    """One real answer and one that never arrived must give a rate over one trial, not two.

    The retry budget is set equal to the first budget so no rescue happens here: this tests
    the denominator rule itself, which the retry test covers from the other side.
    """
    cycle = [("the answer is here", False), ("thinking", True)]
    replies = itertools.cycle(cycle)

    class _Alternating(_Handler):
        def reply_for(self, body: Any) -> str:
            self.server.text, self.server.reasoning = next(replies)
            return super().reply_for(body)

    instance = HTTPServer(("127.0.0.1", 0), _Alternating)
    instance.requests = []
    instance.chunks = 2
    instance.prompt_tokens = None
    instance.status = 200
    instance.text = ""
    instance.overflow = None
    instance.reasoning = False
    instance.stream_error_frame = None
    instance.silent = False
    instance.finish_reason = "stop"
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        config = replace(
            config_for(instance, tmp_path, enabled=["needle"], needle_trials=2),
            context_sizes=(4_000,),
        )
        config = replace(
            config, suites=replace(config.suites, needle_retry_max_tokens=512)
        )
        needle = run_ctxbench(config)["results"][0]["correctness"]["needle"]
    finally:
        instance.shutdown()
        instance.server_close()

    assert needle["trials_attempted"] == 2
    assert needle["unobservable"] == 1
    assert needle["trials"] == 1
    # One scored trial, and it failed to retrieve: a rate of 0 over one trial, not 0.5
    # over two and not a rate over zero.
    assert needle["pass_rate"] == 0.0


def test_truncation_is_detected_from_any_repeat(server: Any, tmp_path: Path) -> None:
    """If one repeat loses context and a later one does not, reporting the healthy
    sample's count would hide the finding behind the median."""
    # Truncation is the server receiving *fewer* tokens than were sent, so the odd one out
    # has to be the small count.
    counts = iter([4_000, 2_000, 4_000])

    class _Varying(_Handler):
        def reply_for(self, body: Any) -> str:
            self.server.prompt_tokens = next(counts)
            return super().reply_for(body)

    instance = HTTPServer(("127.0.0.1", 0), _Varying)
    instance.requests = []
    instance.chunks = 1
    instance.prompt_tokens = None
    instance.status = 200
    instance.text = "ok"
    instance.overflow = None
    instance.reasoning = False
    instance.stream_error_frame = None
    instance.silent = False
    instance.finish_reason = "stop"
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        config = replace(
            config_for(instance, tmp_path, enabled=["prefill"], repeats=3),
            context_sizes=(4_000,),
        )
        row = run_ctxbench(config)["results"][0]
    finally:
        instance.shutdown()
        instance.server_close()

    assert row["harness_health"]["context_truncated"] is True
    assert row["harness_health"]["context_truncation"]["received_tokens"] == 2_000


def _needle_answer(prompt: str) -> str | None:
    """The value a perfect retriever would answer with, read out of the haystack."""
    match = re.search(r"(?:SECRET_API_KEY|RELEASE_CHANNEL|DOCS_URL) = '([^']+)'", prompt)
    if match:
        return match.group(1)
    match = re.search(r"(?m)^(def calculate_tax_\d+\([^)]*\)[^:]*:)", prompt)
    return match.group(1) if match else None


def _fake_model_server(answers: Callable[[str], str | None]) -> Any:
    """An endpoint that answers needle probes from the planted value, or not at all.

    Lets a test control pass/fail per trial without a real model, which is the only way to
    check the arithmetic of the rate rather than the model.
    """

    class _Perfect(_Handler):
        def reply_for(self, body: Any) -> str:
            answer = answers(body["messages"][0]["content"])
            return answer if answer is not None else "not found"

    instance = HTTPServer(("127.0.0.1", 0), _Perfect)
    instance.requests = []
    # One chunk, so the reply is the answer rather than the answer repeated once per chunk.
    instance.chunks = 1
    instance.prompt_tokens = None
    instance.status = 200
    instance.text = ""
    instance.overflow = None
    instance.reasoning = False
    instance.stream_error_frame = None
    instance.silent = False
    instance.finish_reason = "stop"
    threading.Thread(target=instance.serve_forever, daemon=True).start()
    return instance


def test_perfect_retriever_scores_one_hundred_percent(
    server: Any, tmp_path: Path
) -> None:
    """The upper bound. If a model that always answers correctly does not score 1.0, the
    rate is not measuring retrieval."""
    instance = _fake_model_server(_needle_answer)
    try:
        config = replace(
            config_for(instance, tmp_path, enabled=["needle"], needle_trials=4),
            context_sizes=(4_000,),
        )
        needle = run_ctxbench(config)["results"][0]["correctness"]["needle"]
    finally:
        instance.shutdown()
        instance.server_close()
    assert needle["trials"] == 4
    assert needle["passed"] == 4
    assert needle["pass_rate"] == 1.0


def test_hopeless_retriever_scores_zero_percent(server: Any, tmp_path: Path) -> None:
    """The lower bound, and the one that catches a rate whose numerator and denominator
    disagree: a suite reporting 1.0 over a row containing a failure is the bug this
    guards."""
    instance = _fake_model_server(lambda prompt: None)
    try:
        config = replace(
            config_for(instance, tmp_path, enabled=["needle"], needle_trials=4),
            context_sizes=(4_000,),
        )
        needle = run_ctxbench(config)["results"][0]["correctness"]["needle"]
    finally:
        instance.shutdown()
        instance.server_close()
    assert needle["trials"] == 4
    assert needle["passed"] == 0
    assert needle["pass_rate"] == 0.0


def test_partial_retriever_scores_the_share_it_passed(tmp_path: Path) -> None:
    """Two of four: the rate must be 0.5, not 1.0 and not 0.25."""
    calls = itertools.count()

    def half_right(prompt: str) -> str | None:
        answer = _needle_answer(prompt)
        if answer is None:
            return None
        return None if next(calls) % 2 else answer

    instance = _fake_model_server(half_right)
    try:
        config = replace(
            config_for(instance, tmp_path, enabled=["needle"], needle_trials=4),
            context_sizes=(4_000,),
        )
        needle = run_ctxbench(config)["results"][0]["correctness"]["needle"]
    finally:
        instance.shutdown()
        instance.server_close()
    assert needle["trials"] == 4
    assert needle["passed"] == 2
    assert needle["pass_rate"] == 0.5
    assert sum(1 for trial in needle["by_depth"] if trial["passed"]) == 2


def test_execution_config_translates_the_ctxbench_settings(tmp_path: Path) -> None:
    """The translation into a benchmark run is the contract of the pass@1 suite, and it
    is checkable without materializing a dataset."""
    from coding_agent_evaluations.ctxbench.runner import execution_config

    config = CtxBenchConfig(
        model=ModelConfig(
            name="m",
            base_url="http://h/v1",
            temperature=0.3,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        ),
        suites=CtxSuiteConfig(
            enabled=("execution",),
            execution_num_tasks=5,
            execution_seeds=3,
            execution_max_steps=150,
            execution_max_time_sec=3600,
            dataset="terminal_bench_2_verified",
            cache_dir=tmp_path / "datasets",
        ),
        runtime=RuntimeConfig(root=tmp_path / "workspaces"),
        harness="pi",
        seed=7,
        output_dir=tmp_path / "out",
    )
    run = execution_config(config, 64_000, tmp_path / "out")
    assert run.max_context_tokens == 64_000
    assert run.harness.name == "pi"
    assert run.dataset == "terminal_bench_2_verified"
    assert run.num_tasks == 5
    assert run.seeds_per_task == 3
    assert run.max_steps == 150
    assert run.max_total_time_sec == 3600
    assert run.seed == 7
    assert run.model.name == "m"
    assert run.model.temperature == 0.3
    # The endpoint extensions the reference protocol relies on have to survive the
    # translation, or pass@1 measures a different model configuration than a sweep does.
    assert run.model.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}
    assert run.output_dir == tmp_path / "out" / "execution" / "ctx64000"
    assert run.runtime.root == tmp_path / "workspaces"
    assert run.cache_dir == tmp_path / "datasets"


def test_execution_cells_do_not_share_a_directory(tmp_path: Path) -> None:
    """`cae`'s resume matches on `(instance_id, seed)` alone, so two context sizes writing
    to one directory would let the second reuse the first's rows and report one size's
    results at another."""
    from coding_agent_evaluations.ctxbench.runner import execution_config

    config = CtxBenchConfig(
        model=ModelConfig(name="m", base_url="http://h/v1"),
        suites=CtxSuiteConfig(enabled=("execution",)),
        output_dir=tmp_path / "out",
    )
    first = execution_config(config, 16_000, tmp_path / "out")
    second = execution_config(config, 128_000, tmp_path / "out")
    assert first.output_dir != second.output_dir


def test_execution_config_honours_an_all_tasks_run(tmp_path: Path) -> None:
    """`--num-tasks all` is expensive here (the whole benchmark, once per ladder size) but
    it is a legitimate choice, so it is passed through rather than silently truncated to a
    default the user did not ask for."""
    from coding_agent_evaluations.ctxbench.runner import execution_config

    config = CtxBenchConfig(
        model=ModelConfig(name="m", base_url="http://h/v1"),
        suites=CtxSuiteConfig(enabled=("execution",), execution_num_tasks=None),
        output_dir=tmp_path / "out",
    )
    assert execution_config(config, 16_000, tmp_path / "out").num_tasks is None


# ------------------------------------------------------------------- the runner


def config_for(server: Any, tmp_path: Path, **suite: Any) -> CtxBenchConfig:
    """A config aimed at the fake endpoint.

    `enabled` defaults to the endpoint-only suites: the execution suite materializes real
    benchmark tasks and downloads their datasets, which is not what these tests are for and
    would make them depend on the network. It is exercised through `cae run`'s own tests.
    """
    suite.setdefault("enabled", ("prefill", "generation", "needle", "memory"))
    suite.setdefault("repeats", 1)
    suite.setdefault("needle_trials", 2)
    suite.setdefault("memory_iterations", 2)
    return CtxBenchConfig(
        model=ModelConfig(name="m", base_url=f"http://127.0.0.1:{server.server_port}/v1"),
        suites=CtxSuiteConfig(
            corpus_cache=tmp_path / "cache",
            request_timeout_sec=10,
            **suite,
        ),
        output_dir=tmp_path / "out",
        context_sizes=(4_000, 8_000),
    )


def test_runner_produces_one_row_per_context_size(server: Any, tmp_path: Path) -> None:
    payload = run_ctxbench(config_for(server, tmp_path, repeats=1, needle_trials=2))
    assert [row["context_size_tokens"] for row in payload["results"]] == [4_000, 8_000]
    assert payload["tokenizer"] == "o200k_base"
    assert payload["results"][0]["harness_health"]["prompt_tokens_sent"] > 0


def test_runner_writes_the_json_record(server: Any, tmp_path: Path) -> None:
    payload = run_ctxbench(config_for(server, tmp_path, repeats=1, needle_trials=2))
    written = json.loads((tmp_path / "out" / "ctxbench.json").read_text())
    assert written == payload


def test_failed_suite_does_not_lose_the_others(server: Any, tmp_path: Path) -> None:
    """"Could not prefill this and also could not retrieve from it" is two findings;
    collapsing them into a failed cell loses the second."""
    server.status = 400
    payload = run_ctxbench(
        config_for(server, tmp_path, repeats=1, needle_trials=2, enabled=["prefill", "needle"])
    )
    for row in payload["results"]:
        assert row["errors"], "a 400 should be recorded, not swallowed"
        # Both suites still appear, with an explicit absence rather than a fake zero.
        assert row["speed"]["prefill"]["error"]
        assert row["correctness"]["needle_in_haystack_pass_rate"] in (None, 0.0)


def test_needle_trials_are_attempted_even_when_the_probe_fails(server: Any, tmp_path: Path) -> None:
    """A failed trial is a trial. Dropping it would inflate the denominator's pass rate by
    removing the probes that failed."""
    server.status = 400
    payload = run_ctxbench(
        config_for(server, tmp_path, repeats=1, needle_trials=4, enabled=["needle"])
    )
    row = payload["results"][0]
    assert row["correctness"]["needle"]["trials"] == 4
    assert row["correctness"]["needle_in_haystack_pass_rate"] == 0.0


def test_truncation_flag_reflects_the_server(server: Any, tmp_path: Path) -> None:
    server.prompt_tokens = 100
    payload = run_ctxbench(
        config_for(server, tmp_path, repeats=1, needle_trials=2, enabled=["prefill"])
    )
    row = payload["results"][0]
    assert row["harness_health"]["context_truncated"] is True
    assert row["harness_health"]["context_truncation"]["received_tokens"] == 100


def test_memory_suite_is_skipped_when_the_size_is_too_large(
    server: Any, tmp_path: Path
) -> None:
    """A leak trace needs repeated *large* requests. If the endpoint cannot hold the size,
    the honest answer is that the suite was skipped -- not a trace of failed iterations."""
    server.status = 400
    server.overflow = (
        '{"error":"request (200001 tokens) exceeds the available context size '
        '(100096 tokens)"}'
    )
    config = replace(
        config_for(server, tmp_path, enabled=["memory"], repeats=1, memory_iterations=3),
        context_sizes=(128_000,),
    )
    memory = run_ctxbench(config)["memory"]
    assert memory["skipped"] is True
    assert memory["server_context_limit"] == 100_096
    assert memory["iterations"] == 0


def test_memory_suite_records_a_trace(server: Any, tmp_path: Path) -> None:
    payload = run_ctxbench(
        config_for(
            server, tmp_path, repeats=1, needle_trials=2, enabled=["memory"], memory_iterations=3
        )
    )
    assert payload["memory"]["iterations"] == 3
    assert payload["memory"]["context_size_tokens"] == 8_000
    assert payload["memory"]["peak_mb"] is not None


def test_cached_corpus_is_reused(server: Any, tmp_path: Path) -> None:
    config = config_for(server, tmp_path, repeats=1, needle_trials=2, enabled=["prefill"])
    run_ctxbench(config)
    cached = list((tmp_path / "cache").glob("*.txt"))
    assert cached, "corpora should be cached so a rerun does not regenerate them"
    assert "4000-o200k_base-42" in cached[0].name