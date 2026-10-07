---
name: ctxbench
description: Sweep context sizes (16k-128k) against a served model with cae and print the standard report. Use when asked how a model or harness scales with context, to check whether a harness truncates or leaks at long context, to measure TTFT/tokens-per-second across context sizes, to run a needle-in-a-haystack retrieval test, or to compare context-scaling behaviour between models or harnesses. Complements run-benchmarks, which answers whether a task was solved rather than what the context cost.
license: MIT
metadata:
  runner: cae
  report_version: "1"
---

# Context-scaling sweep

One script runs a context sweep. Same standardisation argument as `run-benchmarks`: a
model × context-size number is only readable next to another if both were produced the
same way.

```bash
uv run --script .agents/skills/ctxbench/scripts/ctxbench.py run \
  --model <served-model-id> \
  --base-url <openai-compatible-endpoint> \
  [scale flags]
```

The script is a PEP 723 script: `uv run --script` resolves and pins `cae` from
`git+https://github.com/NoRaincheck/coding-agent-evaluations.git` on first use.

## When this is the right tool

`run-benchmarks` answers *can this model solve this task*. This answers *what does the
context cost, and does anything break as it grows*. Use this one for:

- time to first token and decode throughput as context grows
- whether a definition buried at depth N is still retrievable
- whether the harness silently dropped context the endpoint says it never received
- whether repeated large requests leak memory

Use `run-benchmarks` instead when the question is pass@1 on a real task at a fixed
context budget. The two overlap on pass@1 — this repo's sweep delegates that suite to the
same benchmark runner — but the benchmark run is the cheaper way to get that one number.

## Inputs

**Model and endpoint are the only two you must never guess.** If they were not given,
ask. Everything else has a default; use the default and say so in the report preamble.

| Input | Flag | Default |
|---|---|---|
| Served model id | `--model` | *required* |
| Endpoint | `--base-url` | *required* |
| Context ladder | `--contexts` | `16000,32000,64000,128000` |
| Suites | `--suites` | all five |
| Repeats per measurement | `--repeats` | `3` |
| Needle trials per size | `--needle-trials` | `6` |
| Harness for pass@1 | `--harness` | `leaf` |
| Benchmark tasks for pass@1 | `--num-tasks` | `2` |
| Memory iterations | `--memory-iterations` | `10` |
| Parent directory for results | `--output-root` | `eval-results/ctxbench` |

Suites: `prefill`, `generation`, `needle`, `execution`, `memory`. They are independent —
drop `--suites execution` to skip the expensive one and keep the rest.

### What a sweep costs

Worth knowing before starting one, because the numbers are dominated by decode, not by
the ladder:

- `prefill` is uncached by construction, so it is the honest cost of a fresh context. On
  a local 4B model that is tens of seconds at 16k and several minutes at 64k.
- `needle` retries any trial the model failed to answer inside its budget, and a
  reasoning model needs a large budget at long context — minutes per trial at 64k. Lower
  `--needle-trials`, or accept the cost.
- `execution` provisions real task workspaces and runs their verifiers. It dominates
  everything else; drop it unless pass@1 is the point.
- A ladder row the endpoint refuses is immediate, not slow.

For a quick look, use one size and `--suites prefill,generation`.

## Run

```bash
# Prove the wiring before committing to a long run (~2 min at one size, prefill only)
uv run --script .agents/skills/ctxbench/scripts/ctxbench.py run \
  --model frognano-4b-2609 --base-url http://127.0.0.1:1234/v1 \
  --contexts 8000 --suites prefill --repeats 1

# A real sweep, no pass@1 (the long pole)
uv run --script .agents/skills/ctxbench/scripts/ctxbench.py run \
  --model qwen3.6-35b-a3b-mtp --base-url http://127.0.0.1:1234/v1 \
  --suites prefill,generation,needle,memory

# Full sweep including pass@1
uv run --script .agents/skills/ctxbench/scripts/ctxbench.py run \
  --model frognano-4b-2609 --base-url http://127.0.0.1:1234/v1 --num-tasks 5
```

Other subcommands:

```bash
ctxbench.py doctor --model <id> --base-url <url>   # preflight only
ctxbench.py report --model <id> --base-url <url>   # reprint a finished run
  --plan        # print the resolved config, run nothing
  --dry-run     # print the ladder and estimated cost, run nothing
  --json        # machine-readable form
```

`report` locates results from the scale directory, which the ladder and suite flags
decide — so pass the same flags the run used. With the defaults it just reads
`eval-results/ctxbench/<scale>/ctxbench.json`.

## Before you report a number, check what it means

Four things routinely turn a sweep into a wrong conclusion. The script flags the first
two; you have to check the rest.

1. **The ladder must fit the served model.** If the endpoint refuses a size, the report
   prints the window it reported and marks the row as carrying no measurement. A 128k row
   against a 100k-token model is not a finding about the model. Note that some servers
   refuse out of band — an SSE `event: error` frame inside an HTTP 200 — which this runner
   detects, but a client that does not will report the refusal as a very fast empty
   success.
2. **Context sizes are labelled by a local tokenizer**, `o200k_base` by default, which is
   not the model's own vocabulary. On this project it under-counts by ~1.23x. The server's
   own count is in the `received` column; read that one for what the endpoint saw.
3. **TTFT is uncached by construction.** Repeats 2..n are prefix-cache-busted, because an
   identical repeat lets a cached prefix turn a 42-second prefill into a 75-millisecond
   one. If you want the warm-prefix number, it is not in this table.
4. **A needle pass rate needs its per-depth breakdown.** A model that reads the head of a
   long context and gives up scores well at 16k and badly at 128k; the average hides that
   shape. The breakdown is in `ctxbench.json`. Trials the model never answered are counted
   separately, so the rate's denominator is the number of trials that actually ran.

## Output

Everything lands under `<output-root>/<scale>/`:

```
ctxbench.yaml      the input, exactly as resolved
ctxbench.json      the full record: every probe, every needle trial, every sample
REPORT.md          the standard report
REPORT.json        the same payload as JSON
execution/         per-size benchmark cells, when that suite runs
```

`<scale>` encodes the ladder and the suite limits, so runs at different scales cannot
contaminate each other's directory.

## Interpreting the table

`ttft_ms` and `prefill_tps` move together; if TTFT falls while the ladder grows, the
endpoint is caching and the run is suspect. `decode_tps` falling while `ttft_ms` holds
steady points at attention over long contexts rather than at prefill. A `truncated` value
of `YES` means the server reported fewer prompt tokens than were sent — a harness or
proxy dropped context, which is a different and more serious finding than anything in the
speed columns.