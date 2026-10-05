---
name: run-benchmarks
description: Run a model x harness benchmark with cae and print the standard report. Use when asked to benchmark, evaluate, or test a model or an agent harness (leaf, opencode, pi) on SWE-bench Verified or Terminal-Bench 2, to compare harnesses, or to check that benchmark wiring works before a long run. Every run goes through one script so the config contract, the output paths and the report format are identical across models, harnesses and people.
license: MIT
metadata:
  runner: cae
  report_version: "1"
---

# Run benchmarks

One script runs a benchmark. There is no other way to run one.

The reason is standardisation. A model x harness number is only worth reading next to
another model x harness number if the two were produced the same way — same tasks,
same seeds, same limits, same config, same report. This skill pins all of that, so
reports from different models, harnesses, machines and people diff line by line.

```bash
uv run --script .agents/skills/run-benchmarks/scripts/benchmark.py run \
  --model <served-model-id> \
  --base-url <openai-compatible-endpoint> \
  [scale and matrix flags]
```

The script is a PEP 723 script: `uv run --script` resolves and pins `cae` from
`git+https://github.com/NoRaincheck/coding-agent-evaluations.git` on first use. No
install step, no virtualenv to manage, and every run uses the same runner version.

Run it from the project you want the results in — results land under
`--output-root`, relative to the working directory. Nothing else about the project
matters; in particular the script needs neither this repository's `configs/` nor its
`src/`.

## Inputs

Collect these before starting. **Model and endpoint are the only two you must never
guess** — if they were not given, ask. Everything else has a default; use the default
and say so in the report preamble rather than asking a round of questions.

| Input | Flag | Default |
|---|---|---|
| Served model id | `--model` | *required* |
| Endpoint | `--base-url` | *required* |
| Dataset | `--dataset` | `swebench_verified` |
| Harnesses | `--harnesses` | `leaf,opencode,pi` |
| Tasks | `--num-tasks` (or `all`) | `2` |
| Seeds per task | `--seeds` | `1` |
| Agent step limit | `--max-steps` | `10` |
| Per-job wall clock | `--max-time` | `900` |
| Concurrency | `--max-workers` | `1` |
| Extra models for the same matrix | `--models` | — |
| Parent directory for results | `--output-root` | `eval-results/bench` |

Datasets: `swebench_verified`, `terminal_bench_2_verified`. Harnesses: `leaf` (the
reference), `opencode`, `pi`. Each dataset has fixed reference-protocol values
(context window, step limit, per-turn tokens, interpreter) that the script applies —
they are part of the contract, not per-run choices.

## Run

```bash
# Prove the wiring before committing to a long run (default scale, ~30 min)
uv run --script .agents/skills/run-benchmarks/scripts/benchmark.py run \
  --model frognano-4b-2609 --base-url http://127.0.0.1:1234/v1

# A real comparison: 5 tasks, 3 seeds, reference limits
uv run --script .agents/skills/run-benchmarks/scripts/benchmark.py run \
  --model qwen3.6-35b-a3b-mtp --base-url http://127.0.0.1:1234/v1 \
  --num-tasks 5 --seeds 3 --max-steps 150 --max-time 10800
```

`run` does three things in order, and you do not need to invoke them separately:

1. **`cae doctor`** — Python, `uv`, harness CLIs, disk, and the endpoint actually
   serving the model by name. A model the server lists but cannot load is caught in
   seconds here instead of hours into a run. Skip with `--skip-doctor`.
2. **`cae run`** — every cell, one after another, then the comparison.
3. **report** — the standard block, printed and written to disk.

Other subcommands:

Subcommands and flags:

```bash
benchmark.py doctor --model <id> --base-url <url>   # preflight only, runs nothing
benchmark.py report --model <id> --base-url <url>   # reprint a finished run, runs nothing
  --dry-run     # print the cells and their output dirs, run nothing
  --skip-doctor # skip the preflight
  --no-progress # no progress bar (for a log file)
  --json        # machine-readable form of the same data
```

The report is printed on stdout even when a cell fails, so a failed run still
reports what happened. The script exits non-zero in that case, and `cae`'s own
per-cell output is restored to stderr as the diagnosis.

Runs are long. Start them in the background rather than under a foreground timeout,
and never start a second run of the same scale while one is in flight — `resume`
already picks up where a stopped run left off.

## Output

Everything lands under `<output-root>/<scale>/<dataset>/<harness>/<model>/`:

```
benchmark.yaml    the input, exactly as the script resolved it
config.json       cae's fully resolved per-cell config
results.jsonl     one row per task-seed pair
summary.json      the run's numbers
trajectories/     messages, steps, tokens, exit reason, generated patch
REPORT.md         the standard report
```

`<scale>` is `n<tasks>s<seeds>st<steps>to<time>w<workers>`, e.g.
`n5s3st150to10800w1`. It is part of the path on purpose: `cae`'s resume matches on
`(instance_id, seed)` alone and does not notice that limits changed, so runs at
different scales must not share a directory. Changing any limit changes the path, and
the old run stays exactly where it was for comparison.

The report is the deliverable:

```
benchmark report v1
generated    2026-10-06T09:14:02Z
runner       cae 0.1.0
dataset      swebench_verified
scale        tasks=2 seeds=1 max_steps=10 max_total_time_sec=900 max_workers=1
harnesses    leaf, opencode, pi
output       eval-results/bench/n2s1st10to900w1

dataset            harness   model                tasks  jobs  resolved  resolve_rate  graded  infra  failed  mean_steps  mean_sec
-----------------  --------  -------------------  -----  ----  --------  ------------  ------  -----  ------  ----------  --------
swebench_verified  leaf      frognano-4b-2609     2      2     1         50.00%        50.00%  0      0       10.00       69.3
swebench_verified  pi        frognano-4b-2609     2      2     1         50.00%        50.00%  0      0       30.00      235.2

notes:
  - opencode: max_steps=10 is NOT enforced (the harness owns its loop); bounded by max_total_time_sec=900 only
```

One row per cell: `resolved`, `resolve_rate` (over every scheduled pair),
`resolve_rate_graded` (over pairs that were actually graded), `infra` (pairs that
never got a fair test — provisioning, verifier or endpoint failure, not a model
failure), and `failed`. `--json` carries the per-reason breakdown of `infra` and the
full resolved config of each cell.

## Rules

- **Print the report, then stop.** It is the whole answer. Do not rank the rows, call
  a winner, explain what the numbers mean, or add advice after the notes block. If
  the user asks for an interpretation, that is a separate request.
- **Report `infra` before anything else if it is non-zero** — those rows are not
  model results. Say so in one line, still without editorialising the rest.
- **Pass the scale explicitly** for anything you intend to compare or keep, rather
  than leaning on the defaults, so the run is reproducible from the command alone.
- **Never call `cae` directly or hand-write a config.** Both are how runs drift apart:
  the shipped `configs/*.yaml` are not in the published wheel, and a hand-written file
  is unversioned and unbounded. The script exists to be the only door.
- **Do not edit `benchmark.py`** to accommodate one run. If a benchmark needs
  something the contract lacks, change the contract — in the skill — so every
  subsequent run gets it too.