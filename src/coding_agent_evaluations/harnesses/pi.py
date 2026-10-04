"""The pi CLI as a harness.

pi owns its own system prompt, tools (read, bash, edit, write) and session
handling, so the adapter runs the real CLI headless in the task workspace and
reports its behaviour.

Hermeticity comes from ``PI_CODING_AGENT_DIR``: each rollout gets its own agent
directory containing a generated ``models.json`` for the evaluation endpoint,
plus its own session directory. The machine's global pi configuration, installed
extensions and credentials are therefore invisible to the run.

``--mode json`` gives a JSONL event stream, which is where steps, tool calls and
token usage come from.

One extension is loaded on purpose, by explicit path: the vendored
``pi-token-stats`` extension (``--no-extensions`` still honours ``--extension``).
It is an observer — no prompt, tool or sampling knob changes — and it appends
time-to-first-token and throughput entries to pi's session JSONL, which is the
only place pi reports *when* a token arrived rather than only how many there
were. Those entries are read back into ``trajectory["token_stats"]``; see
``pi_stats`` and ``harness.options.token_stats``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from .base import HarnessRun
from .cli import CliHarness, hermetic_env, parse_jsonl
from .pi_stats import extension_path, read_token_stats, token_stats_enabled

_PROVIDER_ID = "cae-eval"


class PiHarness(CliHarness):
    """Runs ``pi --print --mode json`` against the evaluation endpoint."""

    name = "pi"
    # pi owns its loop; it is bounded only by the wall-clock budget.
    enforces_max_steps = False
    # pi accepts no sampling knobs. `extra_body` is not forwarded, and the
    # reasoning level below is pi's own coarse setting, not the reference
    # `chat_template_kwargs.enable_thinking` toggle.
    ignored_model_params = (
        "temperature",
        "max_tokens_per_turn",
        "parallel_tool_calls",
        "extra_body",
    )

    def resolve_binary(self) -> str:
        binary = shutil.which("pi")
        if binary is None:
            raise FileNotFoundError(
                "pi is not on PATH; install it (e.g. `npm i -g "
                "@earendil-works/pi-coding-agent`) or set harness.options.binary"
            )
        return binary

    def session_id(self, run: HarnessRun) -> str:
        digest = hashlib.sha1(
            f"{run.instance_id}:{run.seed}".encode()
        ).hexdigest()[:12]
        return f"cae-{run.seed}-{digest}"

    def build_command(
        self, run: HarnessRun, config_dir: Path, session_id: str
    ) -> list[str]:
        options = dict(run.config.options)
        binary = str(options.get("binary") or self.binary or "pi")
        session_dir = self.session_dir(config_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        tools = options.get(
            "tools", ["read", "bash", "edit", "write"]
        )
        command = [
            binary,
            "--print",
            "--mode",
            "json",
            "--approve",
            "--no-skills",
            "--no-themes",
            "--no-prompt-templates",
            "--no-extensions",
            "--session-dir",
            str(session_dir),
            "--session-id",
            session_id,
            "--model",
            f"{_PROVIDER_ID}/{run.model.name}",
            "--api-key",
            run.model.resolve_api_key(),
            "--tools",
            ",".join(str(tool) for tool in tools),
            "--thinking",
            str(options.get("thinking", "medium")),
        ]
        # Loaded by path, not by discovery: pi's own extension directory stays out of the
        # measurement, and only the one pinned observer is in.
        stats_extension = extension_path(run.config)
        if stats_extension is not None:
            command.extend(["--extension", str(stats_extension)])
        if not options.get("context_files", False):
            command.append("--no-context-files")
        # The instruction is passed as the final positional argument.
        command.append(run.runtime.get_task_instruction())
        return command

    @staticmethod
    def session_dir(config_dir: Path) -> Path:
        """Where pi writes this rollout's session JSONL."""
        return config_dir / "sessions"

    def build_env(
        self, run: HarnessRun, config_dir: Path, session_id: str
    ) -> dict[str, str]:
        agent_dir = config_dir / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "models.json").write_text(
            json.dumps(_models_config(run), indent=2) + "\n", encoding="utf-8"
        )
        settings = {
            "defaultThinkingLevel": str(
                run.config.options.get("thinking", "medium")
            ),
            "quietStartup": True,
            "enableInstallTelemetry": False,
        }
        (agent_dir / "settings.json").write_text(
            json.dumps(settings, indent=2) + "\n", encoding="utf-8"
        )
        environment = hermetic_env(dict(os.environ), config_dir)
        environment["PI_CODING_AGENT_DIR"] = str(agent_dir)
        # The token-speed extension persists its chars-per-token calibration to
        # `~/.pi/agent` unless told otherwise; keep that inside the rollout.
        if token_stats_enabled(run.config):
            environment["PI_TOKEN_SPEED_STATS_DIR"] = str(agent_dir)
        environment.update(run.model.as_env())
        return environment

    def parse(self, run: HarnessRun, stdout: str, exit_code: int) -> dict[str, Any]:
        """Build a trajectory from pi's JSONL event stream.

        pi reports a tool call twice — `tool_execution_start` with the arguments,
        then `tool_execution_end` with the result — so the two are paired by
        `toolCallId` instead of being counted as separate steps.
        """
        events = parse_jsonl(stdout)
        steps: list[dict[str, Any]] = []
        messages: list[dict[str, Any]] = []
        pending: dict[str, dict[str, Any]] = {}
        final_message = ""
        input_tokens = 0
        output_tokens = 0
        reasoning_tokens = 0
        context_tokens = 0
        stop_reason = ""
        for event in events:
            kind = str(event.get("type") or "")
            if kind == "tool_execution_start":
                call_id = str(event.get("toolCallId") or "")
                pending[call_id] = {
                    "tool": event.get("toolName"),
                    "input": event.get("args"),
                    "observation": "",
                    "is_error": False,
                }
            elif kind == "tool_execution_end":
                call_id = str(event.get("toolCallId") or "")
                step = pending.pop(
                    call_id, {"tool": event.get("toolName"), "input": None}
                )
                step["observation"] = _text_of(event.get("result"))
                step["is_error"] = bool(event.get("isError"))
                steps.append(step)
            elif kind == "message_end":
                message = event.get("message") or {}
                role = str(message.get("role") or "")
                if role in {"assistant", "user"}:
                    messages.append(message)
                usage = message.get("usage") or {}
                if isinstance(usage, dict):
                    # pi reports per-message usage; the largest input seen is the
                    # context the model actually worked with.
                    input_tokens = max(input_tokens, int(usage.get("input") or 0))
                    output_tokens += int(usage.get("output") or 0)
                    reasoning_tokens += int(usage.get("reasoning") or 0)
                if role == "assistant":
                    stop_reason = str(message.get("stopReason") or stop_reason)
                    text = _text_of(_assistant_text(message.get("content")))
                    if text:
                        final_message = text
        context_tokens = input_tokens
        if exit_code != 0:
            reason = "harness_error"
        elif stop_reason in {"max_tokens", "length"}:
            reason = "max_tokens_per_turn"
        else:
            reason = "agent"
        trajectory = {
            "messages": messages,
            "steps": steps,
            "n_steps": len(steps),
            "exit_reason": reason,
            "error": None if exit_code == 0 else f"pi exited with {exit_code}",
            "final_message": final_message,
            "context_tokens": context_tokens,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
            },
        }
        stats = self.collect_token_stats(run)
        if stats is not None:
            trajectory["token_stats"] = stats
        return trajectory

    def collect_token_stats(self, run: HarnessRun) -> dict[str, Any] | None:
        """Read back the token-speed entries the extension left in the session JSONL.

        pi writes the session, not stdout, so this runs after the CLI exits. Returns ``None``
        when token stats are switched off, so a trajectory without the key means "not
        measured" rather than "measured zero".
        """
        if extension_path(run.config) is None:
            return None
        config_dir = run.runtime.paths.state / f"harness-{self.name}"
        stats = read_token_stats(self.session_dir(config_dir), self.session_id(run))
        if not stats["aggregate"]["requests"]:
            run.logger.warning(
                "no token_speed_stats entries for %s seed %s; the pi extension may have "
                "failed to load",
                run.instance_id,
                run.seed,
            )
        return stats


def _models_config(run: HarnessRun) -> dict[str, Any]:
    options = dict(run.config.options)
    model: dict[str, Any] = {
        "id": run.model.name,
        "name": options.get("model_name", run.model.name),
        "contextWindow": int(
            options.get("context_window", run.max_context_tokens or 32768)
        ),
        "maxTokens": int(
            options.get(
                "max_tokens", run.model.max_tokens_per_turn or 8192
            )
        ),
        "input": ["text"],
        "reasoning": bool(options.get("reasoning", True)),
    }
    provider: dict[str, Any] = {
        "baseUrl": run.model.base_url,
        "api": "openai-completions",
        "apiKey": run.model.resolve_api_key(),
        "models": [model],
    }
    return {"providers": {_PROVIDER_ID: provider}}


def _assistant_text(value: Any) -> str:
    """Join an assistant message's text blocks, skipping thinking blocks."""
    if not isinstance(value, list):
        return _text_of(value)
    parts = [
        str(item.get("text") or "")
        for item in value
        if isinstance(item, dict)
        and item.get("type") in (None, "text")
        and item.get("thinking") is None
    ]
    return "\n".join(part for part in parts if part)


def _text_of(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    if isinstance(value, dict):
        return _text_of(value.get("content"))
    return str(value)


__all__ = ["PiHarness"]