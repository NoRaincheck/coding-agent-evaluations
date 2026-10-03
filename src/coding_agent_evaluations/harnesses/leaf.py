"""The reference Leaf harness, driven against the host runtime.

Leaf is used exactly as FrogNano ships it: same agent, same five tools, same
OpenAI tool schemas, same system prompt. The only change is the execution
substrate, so any score difference against a published FrogNano run comes from
the sandbox or the model, never from the agent loop.
"""

from __future__ import annotations

from typing import Any

from frognano.harness.leaf import LeafAgent, LeafConfig
from frognano.harness.leaf.environment import LeafEnvironment

from .base import HarnessRun


class LeafHarness:
    """Runs :class:`frognano.harness.leaf.LeafAgent` in a host workspace."""

    name = "leaf"
    enforces_max_steps = True
    # leaf sends the full sampling payload in extra_body.
    ignored_model_params: tuple[str, ...] = ()

    def run(self, run: HarnessRun) -> dict[str, Any]:
        api_key = run.model.resolve_api_key()
        config = LeafConfig(
            model=run.model.name,
            base_url=run.model.base_url,
            api_key=api_key,
            temperature=run.model.temperature,
            max_tokens_per_turn=run.model.max_tokens_per_turn,
            timeout_sec=run.model.timeout_sec,
            max_retries=run.model.max_retries,
            parallel_tool_calls=run.model.parallel_tool_calls,
            max_steps=run.max_steps,
            max_context_tokens=run.max_context_tokens,
            max_total_time_sec=run.max_total_time_sec,
            extra_body=run.model.extra_body or None,
        )
        environment = LeafEnvironment(run.runtime)
        agent = LeafAgent(config)
        trajectory = agent.run(
            environment,
            instance_id=run.instance_id,
            seed=run.seed,
            checkpoint_callback=run.checkpoint_callback,
        )
        trajectory["harness"] = self.name
        return trajectory


__all__ = ["LeafHarness"]