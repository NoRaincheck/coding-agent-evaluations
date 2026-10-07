"""Streaming probes against an OpenAI-compatible endpoint, timed.

Everything the speed suite reports comes from this module. Two decisions shape it.

**Timing is taken at the socket, per chunk.** `time.perf_counter()` is read immediately
before the request is written and again on each streamed chunk, so TTFT is the delay to
the first chunk the server emitted rather than to the first chunk the HTTP library
finished parsing. The distinction matters most exactly where the suite is looking:
prefill at 128k runs for seconds before anything is generated, and a timer started after
buffering would report that time as zero.

**The server's own accounting is recorded, never assumed.** `usage` in the final chunk
is the only evidence of how many tokens the endpoint actually received and produced.
That is what makes silent truncation detectable at all: if a harness quietly caps a 128k
prompt to 32k, the prompt tokens come back smaller than what was sent, and the report has
to be able to say so instead of assuming the send was intact.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from ..config import ModelConfig

logger = logging.getLogger(__name__)


class ProbeError(RuntimeError):
    """A probe could not be completed.

    Carries the HTTP status when there was one, because "the 128k request failed" and
    "the 16k request failed" need different fixes and the report has to tell them apart.
    A context-overflow rejection is parsed out of the body, since it names the model's
    actual window -- the single most useful fact a sweep can discover about a served
    checkpoint, and the difference between "this model fails at 128k" and "this model
    cannot hold 128k".
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        body: str = "",
        stream_error: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.body = body
        #: True when the failure arrived as an SSE frame inside an HTTP 200, or as an
        #: empty stream. Distinguished because the HTTP status is 200 and so cannot be used
        #: to decide whether a retry is worthwhile or whether the request was refused.
        self.stream_error = stream_error
        self.server_prompt_tokens = _first_int(body, r"request \((\d+) tokens\)")
        self.server_context_limit = _first_int(
            body, r"available context size \(?(\d+)"
        )

    @property
    def context_overflow(self) -> bool:
        """Whether the endpoint refused the request for exceeding its context window."""
        if self.server_context_limit is not None:
            return True
        return (
            self.status is not None
            and 400 <= self.status < 500
            and "exceeds" in self.body.lower()
        )


def _first_int(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


@dataclass
class StreamResult:
    """One streamed completion, timed and with the server's token accounting."""

    text: str
    ttft_sec: float
    total_sec: float
    first_token_sec: float | None = None
    chunks: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    error: str | None = None
    #: Tokens the model spent reasoning. A reasoning model can spend its entire
    #: `max_tokens` budget thinking and never emit an answer, which is a budget fact and
    #: not a refusal -- so the two are counted separately.
    reasoning_tokens: int = 0

    @property
    def answer_unobservable(self) -> bool:
        """True when the budget ran out before any answer text was produced."""
        return not self.text.strip() and self.reasoning_tokens > 0

    @property
    def generation_sec(self) -> float:
        """Decode window: first token to last.

        Throughput is measured over this rather than over the whole request, so prefill
        and decode are not summed into one number that hides which of the two regressed.
        """
        if self.first_token_sec is None or self.total_sec <= self.first_token_sec:
            return 0.0
        return self.total_sec - self.first_token_sec

    @property
    def decode_tps(self) -> float | None:
        if self.generation_sec <= 0 or not self.completion_tokens:
            return None
        return self.completion_tokens / self.generation_sec

    @property
    def prefill_tps(self) -> float | None:
        """Prompt tokens processed per second of time-to-first-token."""
        if self.ttft_sec <= 0 or not self.prompt_tokens:
            return None
        return self.prompt_tokens / self.ttft_sec

    def as_dict(self) -> dict[str, Any]:
        return {
            "ttft_ms": round(self.ttft_sec * 1000, 1),
            "first_token_ms": (
                round(self.first_token_sec * 1000, 1)
                if self.first_token_sec is not None
                else None
            ),
            "total_latency_ms": round(self.total_sec * 1000, 1),
            "decode_tps": round(self.decode_tps, 2) if self.decode_tps else None,
            "prefill_tps": round(self.prefill_tps, 1) if self.prefill_tps else None,
            "chunks": self.chunks,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
            "error": self.error,
            "text": self.text,
        }


@dataclass
class TruncationCheck:
    """Whether the endpoint received the context it was sent.

    `sent` is what the caller measured locally; `received` is what the server reported.
    A server that silently truncates shows up as `received < sent`. A server that rejects
    oversized input outright raises instead, and that is a different finding: a refusal is
    visible, a silent drop is not.
    """

    sent_tokens: int
    received_tokens: int | None
    status: int | None = None
    rejected: bool = False
    #: What the endpoint said it could hold, when it named it. A row above this number was
    #: never a measurement of the model; it was a request the server could not accept.
    server_context_limit: int | None = None
    server_prompt_tokens: int | None = None

    @property
    def exceeds_server_limit(self) -> bool:
        return self.server_context_limit is not None and self.sent_tokens > self.server_context_limit

    @property
    def truncated(self) -> bool:
        if self.received_tokens is None:
            return False
        # A small tolerance: tokenizers disagree on boundaries, and the point of this
        # check is to catch a harness that dropped tens of thousands of tokens, not to
        # adjudicate a one-token disagreement between two encodings.
        return self.received_tokens < self.sent_tokens * 0.98

    @property
    def tokenizer_mismatch(self) -> float | None:
        """How far the server's count sits from the local one, as a ratio.

        The local count is taken with `model.tokenizer` (`o200k_base` by default) while
        the endpoint counts with the model's own vocabulary, so the two never agree
        exactly. This is reported separately from `truncated` because it is not a defect:
        it means the `sent` figure is an estimate, and a reader comparing it against a
        server-reported number needs to know that. A harness that truncated to half would
        show up as `truncated`; a vocabulary difference shows up here and nowhere else.
        """
        if not self.sent_tokens or self.received_tokens is None:
            return None
        return round(self.received_tokens / self.sent_tokens, 4)

    @property
    def retained_ratio(self) -> float | None:
        if not self.sent_tokens or self.received_tokens is None:
            return None
        return round(min(self.received_tokens, self.sent_tokens) / self.sent_tokens, 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sent_tokens": self.sent_tokens,
            "received_tokens": self.received_tokens,
            "received_over_sent": self.tokenizer_mismatch,
            "retained_ratio": self.retained_ratio,
            "truncated": self.truncated,
            "rejected": self.rejected,
            "exceeds_server_limit": self.exceeds_server_limit,
            "server_context_limit": self.server_context_limit,
            "server_prompt_tokens": self.server_prompt_tokens,
            "status": self.status,
        }


@dataclass
class EndpointProbe:
    """The client: one model, one base URL, retries accounted for."""

    model: ModelConfig
    timeout_sec: int = 900
    retries: int = 1
    #: Counted across every request this probe made, for the harness health block.
    stats: dict[str, int] = field(
        default_factory=lambda: {"requests": 0, "retries": 0, "timeouts": 0, "errors": 0}
    )

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> StreamResult:
        """Stream one completion, timing to the first chunk and to completion.

        Retries are bounded and counted. An unbounded retry loop would hide exactly the
        failure this suite exists to find: an endpoint that times out at 128k and
        succeeds on retry reads as "slow" unless the retry is visible in the report.
        """
        payload: dict[str, Any] = {
            "model": self.model.name,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if self.model.extra_body:
            # `chat_template_kwargs` and friends are endpoint extensions, not OpenAI
            # fields, and a reasoning model needs them to behave like the reference run.
            for key, value in self.model.extra_body.items():
                payload.setdefault(key, value)

        url = f"{self.model.base_url.rstrip('/')}/chat/completions"
        last_error: ProbeError | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                self.stats["retries"] += 1
            try:
                return self._stream_once(url, payload)
            except ProbeError as exc:
                last_error = exc
                if exc.context_overflow:
                    # The server's considered answer, however it was delivered: a 4xx, or
                    # an error frame inside a 200. Repeating it wastes the run.
                    raise
                if (
                    exc.status is not None
                    and 400 <= exc.status < 500
                    and exc.status != 429
                ):
                    raise
                self.stats["errors"] += 1
                logger.warning(
                    "Probe attempt %s/%s failed: %s", attempt + 1, self.retries + 1, exc
                )
        assert last_error is not None
        raise last_error

    def _stream_once(self, url: str, payload: dict[str, Any]) -> StreamResult:
        body = json.dumps(payload).encode()
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.model.resolve_api_key()}",
            },
        )
        self.stats["requests"] += 1
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                result = StreamResult(text="", ttft_sec=0.0, total_sec=0.0)
                parts: list[str] = []
                # SSE frames carry an `event:` name and one or more `data:` lines. A server
                # that fails mid-request reports it as `event: error` inside an HTTP 200 --
                # LM Studio does exactly this for an over-long prompt -- so a parser that
                # reads only `data:` frames sees an empty, successful-looking stream and
                # records a refused request as a measurement.
                event = "message"
                for raw in response:
                    now = time.perf_counter()
                    line = raw.decode("utf-8", errors="replace").strip()
                    if line.startswith("event:"):
                        event = line[6:].strip()
                        continue
                    if not line.startswith("data:"):
                        continue
                    payload_json = line[5:].strip()
                    if not payload_json or payload_json == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(payload_json)
                    except json.JSONDecodeError:
                        continue
                    if event == "error" or (
                        isinstance(chunk, dict) and isinstance(chunk.get("error"), dict)
                    ):
                        raise ProbeError(
                            "endpoint sent an error frame",
                            body=json.dumps(chunk),
                            stream_error=True,
                        )
                    emitted = False
                    for choice in chunk.get("choices") or []:
                        if not isinstance(choice, dict):
                            continue
                        delta = choice.get("delta")
                        if isinstance(delta, dict):
                            if delta.get("content"):
                                parts.append(str(delta["content"]))
                                emitted = True
                            # A reasoning token is generation too. Excluding it would
                            # charge a reasoning model's thinking to its prefill and
                            # make TTFT at 128k look like a context-length regression
                            # when it is a thinking-budget one.
                            elif delta.get("reasoning_content"):
                                emitted = True
                                result.reasoning_tokens += 1
                        if choice.get("finish_reason"):
                            result.finish_reason = str(choice["finish_reason"])
                    # TTFT is the first chunk carrying a token. Servers open with an empty
                    # role-only preamble chunk that carries no work, and timing to it would
                    # report a slow prefill as instant.
                    if emitted and result.first_token_sec is None:
                        result.first_token_sec = now - started
                        result.ttft_sec = result.first_token_sec
                    if emitted:
                        result.chunks += 1
                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        if usage.get("prompt_tokens") is not None:
                            result.prompt_tokens = int(usage["prompt_tokens"])
                        if usage.get("completion_tokens") is not None:
                            result.completion_tokens = int(usage["completion_tokens"])
                result.total_sec = time.perf_counter() - started
                result.text = "".join(parts)
                if result.first_token_sec is None:
                    # The stream opened but never produced content. An endpoint that
                    # returns nothing must not look like a fast one, and must not be
                    # recorded as a measurement at all.
                    raise ProbeError(
                        "endpoint closed the stream without sending any content",
                        status=200,
                        body="no choices and no usage were received",
                        stream_error=True,
                    )
                return result
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:2000].decode(errors="replace")
            self.stats["errors"] += 1
            raise ProbeError(
                f"{url} returned HTTP {exc.code}", status=exc.code, body=detail
            ) from exc
        except TimeoutError as exc:
            self.stats["timeouts"] += 1
            raise ProbeError(
                f"{url} timed out after {self.timeout_sec}s",
                body=f"prompt was {len(json.dumps(payload))} bytes",
            ) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError):
                self.stats["timeouts"] += 1
            self.stats["errors"] += 1
            raise ProbeError(f"{url} unreachable: {reason}") from exc
        except OSError as exc:  # pragma: no cover - defensive
            self.stats["errors"] += 1
            raise ProbeError(f"{url} failed: {exc}") from exc


def truncated(
    sent_tokens: int,
    result: StreamResult,
    *,
    status: int | None = None,
) -> TruncationCheck:
    """Compare what was sent against what the endpoint reported receiving."""
    return TruncationCheck(
        sent_tokens=sent_tokens,
        received_tokens=result.prompt_tokens,
        status=status or None,
    )


__all__ = ["EndpointProbe", "ProbeError", "StreamResult", "TruncationCheck", "truncated"]