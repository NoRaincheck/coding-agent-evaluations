"""Smoke: the real pi harness against a served model, with no benchmark task involved.

This is a wiring check, not an evaluation: one trivial prompt in a temp directory, one
assistant turn, no provisioning and no verifier. It proves the two things the unit tests
cannot — that pi actually accepts `--extension` alongside `--no-extensions`, and that the
session JSONL it writes is in the shape the reader expects.

    .venv/bin/python scripts/smoke_pi_token_stats.py

Skips (exit 0) when the endpoint is not serving or pi is not installed.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from coding_agent_evaluations.config import ModelConfig, load_config
from coding_agent_evaluations.harnesses.base import HarnessRun

REPO = Path(__file__).resolve().parents[1]
PROMPT = "Write the file hello.txt containing the word hello, then stop."


class SmokeRuntime:
    """The slice of the runtime contract the CLI harness touches, in a temp directory."""

    def __init__(self, root: Path) -> None:
        self.paths = type(
            "Paths", (), {"state": root / "state", "workspace": root / "work", "root": root}
        )()
        for path in (self.paths.state, self.paths.workspace):
            path.mkdir(parents=True, exist_ok=True)

    def get_task_instruction(self) -> str:
        return PROMPT


def endpoint_is_up(model: ModelConfig) -> bool:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        f"{model.base_url}/chat/completions",
        data=json.dumps(
            {"model": model.name, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 4}
        ).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {model.resolve_api_key()}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"endpoint not serving ({exc}); skipping")
        return False


def main() -> int:
    config = load_config(REPO / "configs/harness/pi.yaml")
    model = config.model
    print(f"model: {model.name} @ {model.base_url}")
    if not endpoint_is_up(model):
        return 0

    from coding_agent_evaluations.harnesses.pi import PiHarness

    harness = PiHarness()
    with tempfile.TemporaryDirectory(prefix="cae-pi-smoke-") as directory:
        run = HarnessRun(
            runtime=SmokeRuntime(Path(directory)),
            task={"instance_id": "smoke", "dataset": "swebench_verified"},
            model=model,
            config=config.harness,
            instance_id="smoke",
            seed=0,
            max_steps=10,
            max_context_tokens=config.max_context_tokens,
            max_total_time_sec=300,
        )
        trajectory = harness.run(run)

    stats = trajectory.get("token_stats")
    assert stats is not None, "token_stats missing: was the extension loaded?"
    aggregate = stats["aggregate"]
    print(f"steps: {trajectory['n_steps']}  exit: {trajectory['exit_reason']}")
    print(
        "requests={requests} input={input_tokens} output={output_tokens} "
        "ttft_p50={ttft_p50} decode_tps={decode_tps} prefill_tps={prefill_tps}".format(
            requests=aggregate["requests"],
            input_tokens=aggregate["input_tokens"],
            output_tokens=aggregate["output_tokens"],
            ttft_p50=aggregate["ttft_ms_p50"],
            decode_tps=aggregate["decode_tps_mean"],
            prefill_tps=aggregate["prefill_tps_mean"],
        )
    )
    assert aggregate["requests"] >= 1, f"no stats entries: {stats['session_files']}"
    assert aggregate["ttft_ms_p50"], "TTFT not recorded"
    assert aggregate["output_tokens"] > 0, "no output tokens recorded"

    # The same rollout with the extension switched off must record nothing at all.
    from dataclasses import replace as _replace

    off = _replace(
        run,
        config=replace(run.config, options={**run.config.options, "token_stats": False}),
    )
    with tempfile.TemporaryDirectory(prefix="cae-pi-smoke-off-") as directory:
        off_run = _replace(off, runtime=SmokeRuntime(Path(directory)))
        off_trajectory = PiHarness().run(off_run)
    assert "token_stats" not in off_trajectory, "token_stats recorded while switched off"

    print("ok: pi loads the pinned extension, records token speed, and can be run without it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
