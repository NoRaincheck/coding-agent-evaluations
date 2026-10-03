"""The opencode CLI as a harness.

opencode owns its own agent loop, prompt, tool set and compaction. The adapter
therefore runs the real CLI headless inside the task workspace and reports what
it did, rather than reimplementing any of it.

Each rollout gets a hermetic ``XDG_*`` tree and its own ``opencode.json``, so the
machine's global opencode configuration and session history cannot influence a
measurement. The model is exposed as an ``@ai-sdk/openai-compatible`` provider
pointing at the evaluation endpoint, which is how a local OpenAI-compatible
server such as LM Studio is wired in.
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

_PROVIDER_ID = "cae-eval"


class OpenCodeHarness(CliHarness):
    """Runs ``opencode run`` headless against the evaluation endpoint."""

    name = "opencode"
    # opencode drives its own loop until the model stops calling tools; only the
    # wall-clock budget bounds it.
    enforces_max_steps = False
    # The CLI takes no sampling knobs: temperature, top_p, min_p, seed and the
    # reasoning toggle from the reference protocol cannot be set.
    ignored_model_params = (
        "temperature",
        "max_tokens_per_turn",
        "parallel_tool_calls",
        "extra_body",
    )

    def resolve_binary(self) -> str:
        binary = shutil.which("opencode")
        if binary is None:
            raise FileNotFoundError(
                "opencode is not on PATH; install it (e.g. `brew install opencode`) "
                "or set harness.options.binary"
            )
        return binary

    def session_id(self, run: HarnessRun) -> str:
        # opencode rejects session ids that do not start with "ses".
        return f"ses_cae_{run.seed}_{hashlib.sha1(run.instance_id.encode()).hexdigest()[:12]}"

    def build_command(
        self, run: HarnessRun, config_dir: Path, session_id: str
    ) -> list[str]:
        binary = str(run.config.options.get("binary") or self.binary or "opencode")
        # `--standalone` runs a private server instead of registering with the
        # background service: with a redirected XDG_* tree there is no service to
        # reach, and the shared one would leak state between rollouts.
        command = [
            binary,
            "run",
            "--standalone",
            "--print-logs",
            "--auto",
            "--format",
            "json",
            "--session",
            session_id,
            "--model",
            f"{_PROVIDER_ID}/{run.model.name}",
        ]
        for flag, key in (
            ("--agent", "agent"),
            ("--title", "title"),
        ):
            value = run.config.options.get(key)
            if value:
                command.extend([flag, str(value)])
        # The instruction is passed as the final positional argument.
        command.append(run.runtime.get_task_instruction())
        return command

    def build_env(
        self, run: HarnessRun, config_dir: Path, session_id: str
    ) -> dict[str, str]:
        environment = hermetic_env(dict(os.environ), config_dir)
        config = _opencode_config(run)
        config_path = config_dir / "opencode.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        environment["OPENCODE_CONFIG"] = str(config_path)
        environment.update(run.model.as_env())
        return environment

    def parse(self, run: HarnessRun, stdout: str, exit_code: int) -> dict[str, Any]:
        events = parse_jsonl(stdout)
        steps: list[dict[str, Any]] = []
        messages: list[dict[str, Any]] = []
        final_message = ""
        context_tokens = 0
        input_tokens = 0
        output_tokens = 0
        for event in events:
            kind = str(event.get("type") or "")
            if kind == "text":
                final_message = str(event.get("text") or final_message)
            elif kind == "tool_use":
                state = event.get("state") or {}
                steps.append(
                    {
                        "tool": event.get("tool"),
                        "input": state.get("input"),
                        "observation": _text_of(state.get("output")),
                        "status": state.get("status"),
                    }
                )
            elif kind == "step_finish":
                tokens = event.get("tokens") or {}
                context_tokens = max(context_tokens, int(tokens.get("input") or 0))
                input_tokens += int(tokens.get("input") or 0)
                output_tokens += int(tokens.get("output") or 0)
            elif kind in {"message", "step_start", "session"}:
                messages.append(event)
        if not final_message:
            for event in reversed(events):
                if str(event.get("type") or "") == "text" and event.get("text"):
                    final_message = str(event["text"])
                    break
        if exit_code != 0:
            reason = "harness_error"
        else:
            reason = "agent"
        return {
            "messages": messages,
            "steps": steps,
            "n_steps": len(steps),
            "exit_reason": reason,
            "error": None if exit_code == 0 else f"opencode exited with {exit_code}",
            "final_message": final_message,
            "context_tokens": context_tokens,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }


def _opencode_config(run: HarnessRun) -> dict[str, Any]:
    options: dict[str, Any] = dict(run.config.options)
    provider_id = _PROVIDER_ID
    extra: dict[str, Any] = {}
    if run.model.extra_body:
        # openai-compatible provider options are forwarded to the endpoint.
        extra = {
            key: value
            for key, value in run.model.extra_body.items()
            if isinstance(value, (str, int, float, bool))
        }
    config: dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            provider_id: {
                "npm": "@ai-sdk/openai-compatible",
                "name": options.get("provider_name", "CAE eval endpoint"),
                "options": {
                    "baseURL": run.model.base_url,
                    "apiKey": run.model.resolve_api_key(),
                    **extra,
                },
                "models": {
                    run.model.name: {
                        "name": options.get("model_name", run.model.name),
                        "tool_call": True,
                        # No `limit` block: opencode rejects the model outright
                        # ("provider.no-route") when a custom model declares one.
                    }
                },
            }
        },
        "model": f"{provider_id}/{run.model.name}",
        "autoupdate": False,
        "share": "disabled",
        "instructions": options.get("instructions", []),
        "permission": {
            "edit": "allow",
            "bash": "allow",
            "webfetch": "deny",
        },
    }
    if options.get("agent"):
        config["agent"] = options["agent"]
    return config


def _text_of(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, list):
            return "\n".join(
                str(item.get("text") or "")
                for item in content
                if isinstance(item, dict)
            )
    return str(value)


__all__ = ["OpenCodeHarness"]