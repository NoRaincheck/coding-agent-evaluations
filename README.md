# coding-agent-evaluations

Evaluate an OpenAI-compatible model on the benchmarks from
[microsoft/FrogNano](https://github.com/microsoft/FrogNano) — **SWE-bench
Verified** and **Terminal-Bench 2.0 Verified** — once per agent harness:

| Harness | What it is |
|---|---|
| `leaf` | The reference harness from FrogNano, vendored at a pinned commit. The baseline. |
| `opencode` | The opencode CLI, driven headless in the task workspace. |
| `pi` | The pi CLI, driven headless in the task workspace. |

The same tasks, model and limits go to every harness, so the resulting
`summary.json` files are directly comparable.

There is a second runner, `cae ctxbench`, for a question the benchmarks do not
answer: not *can the model solve this*, but *what does a 64k context cost, does
retrieval still work at depth, did the harness drop context it could not fit*.
See [Context scaling](#context-scaling).

Two properties shaped the design:

- **No container runtime, no cluster.** Tasks run natively on the machine, in a
  provisioned workspace directory. No Docker, no Rancher Desktop, no Kubernetes.
  See [Fidelity](#fidelity-what-this-does-and-does-not-reproduce) for what that
  costs.
- **The model is configuration, not code.** Only `model.name` and
  `model.base_url` name a checkpoint, so re-running against a different served
  model is a one-line change.

## Install

```bash
uv venv --python 3.13
uv pip install -e '.[dev]'
```

FrogNano is **vendored, not installed** — see
[Licence and provenance](#licence-and-provenance). Harnesses are invoked from
your own `PATH`:

```bash
brew install opencode                                    # or your preferred install
npm i -g @earendil-works/pi-coding-agent                  # pi
```

No model is ever downloaded. The endpoint is reached over the OpenAI chat
completions API only, and token accounting runs offline (no tokenizer fetch).

## Running it through an agent

Two skills live under [`.agents/skills/`](.agents/skills/), so any agent that
reads the shared `.agents/skills/` convention — OpenCode, Copilot, Codex,
Cursor, pi — can run an evaluation without knowing this repository. Install
them into another project with:

```bash
npx skills add NoRaincheck/coding-agent-evaluations
```

| Skill | Question it answers |
|---|---|
| [`run-benchmarks`](.agents/skills/run-benchmarks/) | Can this model solve these tasks, through this harness? |
| [`ctxbench`](.agents/skills/ctxbench/) | What does the context cost, and does anything break as it grows? |

Each skill drives one script, which wraps `cae` so that every run anywhere gets
the same config contract, the same output paths and the same report format:

```bash
uv run --script .agents/skills/run-benchmarks/scripts/benchmark.py run \
  --model <served-model-id> --base-url <endpoint>

uv run --script .agents/skills/ctxbench/scripts/ctxbench.py run \
  --model <served-model-id> --base-url <endpoint>
```

Both resolve `cae` from this repository through `uv run --script`, so they need
no install step and always use the same runner version. Both key their output
paths by the run's scale, because a resume or a rerun that ignored the limits
would otherwise reuse results across runs whose limits differ.

## Quick start

```bash
export EVAL_MODEL_NAME=frognano-4b-2609
export EVAL_MODEL_BASE_URL=http://127.0.0.1:1234/v1

cae doctor --config configs/dataset/swebench-verified.yaml
scripts/smoke.sh
```

`cae doctor` checks Python, `uv`, both harness CLIs, free disk and that the
endpoint answers with the model you asked for. Fix anything it reports before a
long run — it finds in seconds what a run would find hours later.

## Usage

```bash
# One harness, one dataset
cae run --config configs/dataset/swebench-verified.yaml --harness leaf

# Terminal-Bench 2
cae run --config configs/dataset/terminal-bench-2-verified.yaml --harness pi

# Every model on every harness, from one config file
cae run --config configs/matrix/local-models.yaml

# A smoke slice
cae run --config configs/harness/opencode.yaml \
  --set num_tasks=2 --set seeds_per_task=1 --set max_steps=10

# A specific task
cae run --config configs/dataset/swebench-verified.yaml \
  --task-id django__django-14672

# Inspect before committing to a run
cae tasks  --config configs/dataset/swebench-verified.yaml
cae config --config configs/dataset/swebench-verified.yaml
cae run   --config configs/dataset/swebench-verified.yaml --dry-run
cae run   --config configs/matrix/local-models.yaml --cells

# Compare finished runs
cae compare eval-results/ --markdown
```

`scripts/run-matrix.sh` runs one model through all three harnesses and prints the
comparison table; `scripts/smoke.sh` is the fast version. For more than one
model, use a [matrix config](#running-a-matrix) instead of a shell loop.

## Running a matrix

A matrix is "every model on every harness, same tasks, same limits". It is one config file
rather than a shell loop, so the cells cannot drift apart:

```yaml
# configs/matrix/local-models.yaml
extends: [../base.yaml]

num_tasks: 2          # every other setting is inherited, and applies to every cell
max_steps: 10

matrix:
  datasets: [swebench_verified]
  harnesses: [leaf, opencode, pi]
  models:
    - frognano-4b-2609                        # inherits base_url from base.yaml
    - ling-3.0-tiny
    - name: qwen3.6-35b-a3b-mtp               # or name any field of `model:`
      base_url: http://127.0.0.1:1234/v1
```

```bash
# What would run, without running it
cae run --config configs/matrix/local-models.yaml --cells

# Check the endpoint can serve every model before spending hours finding out it cannot
cae doctor --config configs/matrix/local-models.yaml

# Run it
cae run --config configs/matrix/local-models.yaml

# Then the comparison, from anywhere in the tree
cae compare eval-results/matrix/ --markdown
```

Which models are on the endpoint is the server's business, not the config's:

```bash
curl -s "${EVAL_MODEL_BASE_URL:-http://127.0.0.1:1234/v1}/models" | jq -r '.data[].id'
```

The run prints one progress bar over the **task-seed pairs of the whole matrix**, not over
the cells — a cell is a full evaluation, so "3/9 cells" would read as 33% while the fourth
cell is still grinding through its last forty pairs. The bar's description carries the cell
that is running, and when it finishes the whole comparison is printed:

```
swebench_verified/pi/frognano-4b-2609:  38%|████▊      | 190/500 [2:14<3:36, 1.11pair/s]
```

Notes worth knowing before you scale one up:

- **Cells run one after another.** A cell is already a full evaluation; overlapping cells
  would multiply disk and endpoint load by the cell count and make the comparison measure
  contention. Use `max_workers` inside a cell for concurrency.
- **Each cell gets its own directory**, `<output_dir>/<dataset>/<harness>/<model>/`, which is
  what makes `cae compare` able to line the whole matrix up afterwards.
- **One failing cell does not void the matrix** (`matrix.continue_on_error: false` to change
  that): a model the server lists but cannot load would otherwise discard the measurements
  taken for the others.
- **`resume` works per cell**, so an interrupted matrix picks up where it stopped, and the
  bar counts only the work that is left.
- **`--set` reaches every cell** (`--set num_tasks=50 --set seeds_per_task=3`), because the
  cells share one base config. `--harness pi` narrows the matrix instead of overriding it.
- `--no-progress` drops the bar (for a log file or a CI job); `--markdown` prints the
  comparison as a table.
- Matrix runs are the slow case. Start with the shipped `num_tasks: 2 max_steps: 10` to
  prove the wiring, then scale.

### Configuration

Configs compose with `extends`, and every value accepts `${VAR}` /
`${VAR:-default}`:

```
configs/
  base.yaml                            model + limits + runtime
  dataset/swebench-verified.yaml       the benchmark preset
  dataset/terminal-bench-2-verified.yaml
  harness/leaf.yaml  opencode.yaml  pi.yaml
  matrix/local-models.yaml             several models × several harnesses
  profile/smoke.yaml                   small, fast slice
```

The shipped dataset presets reproduce the FrogNano protocol — three seeds,
150 agent steps, a 131,072-token context limit and a 10,800-second budget — and
differ only where the benchmark differs (completion tokens per turn).

Useful environment variables:

| Variable | Purpose |
|---|---|
| `EVAL_MODEL_NAME` | Served model id. **The knob for re-running another model.** |
| `EVAL_MODEL_BASE_URL` | OpenAI-compatible endpoint, e.g. `http://127.0.0.1:1234/v1`. |
| `EVAL_API_KEY` | Endpoint key; any non-empty placeholder works for LM Studio. |
| `EVAL_OUTPUT_ROOT` | Parent directory for run outputs. |
| `EVAL_WORK_ROOT` | Workspace root (default `~/.cache/cae/workspaces`). |
| `EVAL_CACHE_DIR` | Dataset cache (default `~/.cache/cae/datasets`). |
| `EVAL_MAX_WORKERS` | Task-seed pairs in flight. |

### Changing model

Only the model block is model specific. Anything already loaded works:

```bash
EVAL_MODEL_NAME=qwen3.8-27b-splash scripts/run-matrix.sh
```

If a model needs different sampling, override it on the command line rather than
editing configs — the resolved settings land in `config.json` either way:

```bash
cae run --config configs/dataset/swebench-verified.yaml \
  --set model.temperature=0.7 \
  --set model.max_tokens_per_turn=16000 \
  --set model.extra_body='{"chat_template_kwargs": {"enable_thinking": true}}'
```

Sampling settings the harness cannot express are reported rather than silently
dropped; see [Comparing harnesses](#comparing-harnesses).

## Output

```
<output_dir>/
  config.json                        the fully resolved config
  results.jsonl                      one row per task-seed pair, appended
  summary.json                       the run's numbers
  trajectories/<instance_id>/
    trajectory_seed-<n>.json         messages, steps, tokens, exit reason
    generated_seed-<n>.patch         the diff the agent produced
    harness_seed-<n>/                the harness's raw stdout/stderr
```

`summary.json` keeps the reference runner's accounting: the resolve-rate
denominator is every scheduled task-seed pair (not pass@k), and completed
unresolved pairs are preserved on resume.

It adds two fields that matter on a single machine:

- `jobs_infrastructure_failed` — pairs that never got a fair test, broken out by
  `infrastructure_reasons` (`provision_error`, `verifier_error`, …).
- `resolve_rate_graded` — the same numerator over only the pairs that were
  actually graded. Compare `resolve_rate` with upstream numbers and
  `resolve_rate_graded` when you want to know the model's own result.

`limits` records what was actually in force: `max_steps_enforced` and
`model_params_ignored`. `token_stats` records token speed where the harness
measures it — see [Token speed](#token-speed).

## Token speed

Token *counts* do not say whether a run was slow or the endpoint was slow. The
pi harness therefore loads one pinned extension, [pi-token-stats][pi-token-stats],
which hooks pi's message lifecycle and appends per-message and per-turn
statistics to the session JSONL: time to first token, decode and prefill
throughput, tokens, and generation wall time.

It is an **observer**. No prompt, tool, sampling knob or sampling knob the
harness would otherwise set changes, and the extension is loaded by explicit path
with `--no-extensions` still in force, so a globally installed pi extension cannot
enter a measurement. `harness.options.token_stats: false` turns it off.

Where the numbers end up:

| Where | What |
|---|---|
| `trajectories/<id>/trajectory_seed-<n>.json` | `token_stats.messages` and `token_stats.turns` — the extension's own entries, unchanged, so they can be re-aggregated later or read with upstream's `jq` recipes |
| same file, `token_stats.aggregate` | the rollout's numbers: `ttft_ms_mean` / `_p50` / `_p90`, `decode_tps_mean` and `decode_tps_total`, `prefill_tps_mean`, token totals, `generation_sec`, `span_sec` |
| `results.jsonl`, `token_stats` | the same aggregate per task-seed pair |
| `summary.json`, `token_stats` | the run-level rollup, rates weighted by request count so one long rollout cannot set the headline |
| `cae compare` | a second table, since speed is only measured by some harnesses |

Latency is reported as a distribution rather than a mean, because a mean hides
exactly the tail that makes an evaluation slow. Throughput is reported both as a
per-request mean and as a ratio of totals — the mean rewards a run made of many
tiny requests, the ratio does not. A metric the endpoint did not report reads as
`null`, not `0`, so "never measured" is never mistaken for "measured zero".

The extension is vendored at a pinned commit; see
[`_vendor/pi_token_stats/PROVENANCE.md`](src/coding_agent_evaluations/_vendor/pi_token_stats/PROVENANCE.md)
for the commit, why it is vendored rather than installed, and how to update it.
Point `harness.options.token_stats_extension` at another copy to evaluate a newer
revision without editing the tree.

To check the wiring against a served model without running a benchmark:

```bash
.venv/bin/python scripts/smoke_pi_token_stats.py
```

[pi-token-stats]: https://github.com/NoRaincheck/pi-token-stats

## Context scaling

`cae run` answers *can this model solve this task*. `cae ctxbench` answers a
different question: as context grows from 16k to 128k, what does it cost, what
stops working, and does the harness quietly drop what it could not fit?

```bash
cae ctxbench --config configs/ctxbench.yaml
cae ctxbench --config configs/ctxbench.yaml --contexts 16000,64000 --json
```

Five suites, each independent so a failure names itself:

| Suite | What it measures |
|---|---|
| `prefill` | Time to first token against a padded prompt asking for one word |
| `generation` | Decode throughput and end-to-end latency on a fixed coding prompt |
| `needle` | Retrieval of a definition planted at a known depth |
| `execution` | pass@1 from real benchmark tasks, at that context budget |
| `memory` | RSS across repeated large requests |

`prefill` and `generation` talk to the endpoint directly, so they measure the
endpoint. `execution` pins `max_context_tokens` to the size under test and
delegates to the same runner `cae run` uses, so pass@1 is graded by the
benchmark's own verifiers rather than a reimplementation of them.

### Filler that hits the target exactly

A prompt padded to "about 16k" produces a row labelled 16k that is really a
14,300-token measurement, and the TTFT curve it draws is the curve for a smaller
context than the one reported. The corpus generator therefore bisects to a
count and closes the last few tokens with a comment, and the whole corpus is
`compile()`d. Every generated module parses on its own, so a needle planted at a
module boundary lands in a valid file. Context sizes are therefore integers, not
approximations, and the filler is seeded so a rerun reproduces it.

### Five ways a sweep lies to you

Each of these was found by running this against a real endpoint, and each is now
either prevented or stated in the report:

- **Prefix caching turns repeats into cache reads.** Repeats 2..n send
  byte-identical prompts, so a cached prefix made a 42-second prefill read as
  75 milliseconds — and the median of three reported the cache read as the cost
  of a context. Repeats after the first now carry a unique marker, so TTFT is
  the cost of a *fresh* context, and the report says so.
- **A refusal delivered out of band looks like a fast success.** This endpoint
  rejects an over-long prompt with an SSE `event: error` frame inside an **HTTP
  200**. A client that reads only `data:` frames sees a 200, an empty stream and
  no usage — and reports a refused request as a 324 ms prefill with a 0% needle
  pass rate and no errors at all. The client now reads frame event names and
  treats an error frame, or any stream that closes without content, as a
  failure.
- **A row the endpoint refused is not a measurement of the model.** A 128k row
  against a model with a 100,096-token window is rejected, not scored. The
  runner parses the limit out of the error and marks the row as carrying no
  measurement.
- **A reasoning model can spend its whole answer budget thinking.** Needle
  trials that produce no reply are recorded as *unobservable* and kept out of
  the pass-rate denominator, and are retried once with a larger budget first.
  Counting them as failures would report the token budget as a model property.
- **Local token counts are not the model's vocabulary.** Sizes are labelled by
  `o200k_base` by default, which on this project under-counts by ~1.23x against
  the endpoint. The server's own count is reported separately, so a reader can
  tell which number describes what the endpoint saw.

Anything a suite did not measure reads `-`, never `0`, and every rate carries its
denominator. `cae ctxbench` exits non-zero when any suite recorded an error, so
a pipeline can tell a clean sweep from a half-measured one.

## How it works

```
config  ──▶ datasets ──▶ runtime (host) ──▶ harness ──▶ verifier ──▶ summary
                            │
                            └─ provisions the workspace the image would provide
```

`ctxbench` is a separate path over the same pieces — `ModelConfig`, the dataset
loader, the host runtime and the runner — because the speed and retrieval suites
need no workspace at all:

```
ladder ──▶ filler (exact tokens) ──▶ endpoint probe ──▶ per-size row
   │                                                        │
   └────────────────────▶ execution ──▶ cae run ────────────┘
```

**Datasets** come from FrogNano's pinned sources, so dataset revisions,
instructions, timeouts and task selection match the reference evaluation. The
`seed: 42` shuffle is preserved, so a `--set num_tasks=5` smoke slice selects the
same five tasks here as it would upstream.

**The runtime** provisions one workspace per task-seed pair:

- *SWE-bench Verified* — clone `repo` at `base_commit` (metadata from the task's
  `tests/config.json`) and install the repo plus its test extra into a
  workspace virtual environment.
- *Terminal-Bench 2* — parse the task `Dockerfile`, pick the interpreter its
  base image names, replay `COPY`, `pip`/`uv` and filesystem steps, and log
  whatever could not be translated.

The runtime implements the same interface as FrogNano's `KubernetesTaskRuntime`,
so the upstream `LeafAgent`, its tool runner and its patch capture all run
unmodified. Command execution keeps the reference guarantees: the command is
written to a file, output is captured with its exit code, and the output is
SHA-256 verified before it is handed back. A command that times out raises
rather than silently resubmitting.

**Harnesses** get the workspace and the instruction, and return a trajectory in
a shared shape. `opencode` and `pi` are real CLIs with their own prompts, tools
and compaction, so the adapters run the CLIs headless rather than reimplementing
any of it. Each rollout is hermetic — a private `XDG_*` tree and, for pi, a
private `PI_CODING_AGENT_DIR` — so your global agent configuration, extensions
and credentials cannot leak into a measurement. The one extension a pi rollout
loads is named explicitly and vendored, so the measurement has exactly one
observer in it and that observer is pinned ([Token speed](#token-speed)).

## Fidelity: what this does and does not reproduce

Running on the host instead of in the benchmark image is a real deviation. It is
worth being explicit about what it changes.

**Reproduced faithfully**

- The leaf agent, its five tools, its system prompt and its tool schemas —
  vendored from the pinned FrogNano commit, not reimplemented.
- Task selection, shuffling, seeds, resume semantics and summary accounting.
- Patch capture: `git add -A` + `git diff --cached --binary`, index reset so
  agent-created files survive grading.
- Grading: the benchmark's own `tests/test.sh`, unmodified except for path and
  package-manager translation, and its `logs/verifier/reward.{json,txt}`.

**Translated, and logged as it happens**

Benchmark verifiers and task payloads assume a container: they install packages
with `apt-get`, activate conda under `/opt/miniconda3`, and use absolute paths
like `/tests` and `/logs/verifier`. The host runtime:

- rewrites container paths to workspace paths in the verifier, in copied task
  payloads and in verifier test files;
- turns image-only package managers into logged no-ops — this means system
  packages an image would have installed are **absent**;
- drops `conda activate` in favour of the workspace virtual environment.

Every substitution is recorded per task in `provision.json` and in the
trajectory's `verifier_translation`, next to the original script, so a run can be
audited rather than trusted.

**Not reproduced**

- **Dependency versions.** The images pin them; the host resolves them. A task
  can fail on the host and pass in the image. This is the largest source of
  divergence and it is one-directional noise against the model.
- **System packages.** Anything installed with `apt-get`/`apk` is missing.
- **Encrypted task payloads.** Terminal-Bench tasks shipping
  `environment/protected*.enc` can only be materialized by their image; they fail
  as `provision_error` rather than scoring zero.
- **Network isolation.** Benchmarks marked `no-network` are not isolated on the
  host. Both supported benchmarks use public networking for the agent, so this
  does not apply to them, but a task relying on it would not be enforced.
- **Architecture.** SWE-bench's images are x86_64-only. Running them natively on
  Apple Silicon is not possible here; the host runtime instead reproduces the
  repository state from metadata, so results are not comparable to
  image-based numbers on such a host.
- **Absolute task paths.** Tasks describe their own layout with container paths
  (`/app/logs`, `/testbed`). On the host the agent finds those files relative to
  its workspace, so it spends a turn or two discovering the mapping. This applies
  equally to all three harnesses — the workspace is the working directory for
  every one of them — so it does not bias a comparison between them.
- **Concurrency.** The reference protocol uses 150 shared workers. On one
  machine, `max_workers` is bounded by disk and by how many requests your
  endpoint serves concurrently.

Because of the first two points, treat host numbers as a **harness comparison**,
not as a reproduction of a published FrogNano score. `jobs_infrastructure_failed`
tells you how much of the run never got a fair test.

## Comparing harnesses

```bash
cae compare eval-results/
```

```
dataset                    harness   model             tasks  jobs  resolved  resolve_rate  mean_steps
-------------------------  --------  ----------------  -----  ----  --------  ------------  ----------
swebench_verified          leaf      frognano-4b-2609  5      15    1         6.7%         74.00
swebench_verified          opencode  frognano-4b-2609  5      15    2         13.3%        112.00
swebench_verified          pi        frognano-4b-2609  5      15    1         6.7%         68.00

notes:
  - opencode: max_steps=150 is NOT enforced (the harness owns its loop); bounded by max_total_time_sec=10800 only
  - opencode: model settings not applied: temperature, max_tokens_per_turn, parallel_tool_calls, extra_body
```

Read the notes. Two rows are only comparable if the limits in force were the
same:

- **leaf** owns its loop, so `max_steps` is a real bound and it receives the full
  sampling payload (`temperature`, `max_tokens_per_turn`, `extra_body`).
- **opencode** and **pi** own their own loops. They are bounded by the wall
  clock, not by a step count, and they expose no sampling knobs — `temperature`
  and friends are not applied. A higher step count for those harnesses is
  therefore expected and is not by itself a better result.

For a fair comparison, set the time budget so tight that every harness is
time-bound, or accept that leaf is step-bound and the CLIs are clock-bound.

A second table appears when any run measured token speed. It is not comparable
across harnesses that did not measure it, and `cae compare` says so in its notes
— see [Token speed](#token-speed).

## Development

```bash
.venv/bin/python -m pytest tests -q
```

The test suite runs offline: it covers config composition and env expansion,
verifier translation, the command executor's exit-code/timeout/checksum
behaviour, Dockerfile parsing, patch capture, reward extraction, matrix expansion
and progress accounting, pi token-speed reading and roll-up, harness
configuration generation and the comparison report. It does not call a model.

`ctxbench`'s tests are offline too. The filler generator is checked against the
exact-token guarantee at every ladder size, the needle scorer against every way a
reply can nearly match, and the endpoint client against a fake HTTP server rather
than a served model — which is what makes the properties worth testing checkable
here: that a truncation is detected, that a refused request is not scored as a
model failure, that a trial with no reply leaves the denominator, and that a
prefix-cache-busted repeat really is a distinct prompt.

### Running the tests against a config matrix

Harness tests are parametrised over the same matrix a run uses — local models ×
harnesses — and each cell is resolved from that harness's own shipped config, so a
test asserting a harness is wired for a model asserts what the runner asserts. By
default the matrix is the configured model crossed with all three harnesses; point
it elsewhere with the environment:

```bash
# The default: the model in EVAL_MODEL_NAME (or configs/base.yaml) × leaf, opencode, pi
.venv/bin/python -m pytest tests -q

# A different set of local models, and only one harness
EVAL_MATRIX_MODELS='qwen3.6-35b-a3b-mtp,ling-3.0-tiny' \
EVAL_MATRIX_HARNESSES=pi \
  .venv/bin/python -m pytest tests -q

# A different endpoint for the whole matrix
EVAL_MATRIX_BASE_URL=http://127.0.0.1:1235/v1 .venv/bin/python -m pytest tests -q

# Just the matrix tests, or one harness's cells
.venv/bin/python -m pytest tests -q -k matrix
.venv/bin/python -m pytest tests -q -k "matrix and pi"
.venv/bin/python -m pytest tests --collect-only -q -k matrix   # show the cells
```

A model entry may be a bare served id (`- a, b`) or pin its own endpoint
(`- name: b` with `base_url:`); an omitted `base_url` comes from
`EVAL_MATRIX_BASE_URL`, then `EVAL_MODEL_BASE_URL`, then the config default.

```bash
# Against a real endpoint and the real pi CLI (skips when either is unavailable)
.venv/bin/python scripts/smoke_pi_token_stats.py
```

Layout:

```
src/coding_agent_evaluations/
  config.py        config composition, env expansion, model settings
  matrix.py        matrix configs: models × harnesses → run configs
  matrix_runner.py runs the cells in order, with the progress bar
  datasets.py      pinned task sources and selection
  runner.py        seeds, workers, resume, results and summary
  report.py        cross-harness comparison
  doctor.py        preflight checks
  cli.py           the cae command
  runtimes/
    base.py        sandbox contract and the command executor
    host.py        host task runtime
    provision.py   workspace provisioning per benchmark
    translate.py   container-to-host path and command translation
  harnesses/
    leaf.py        reference harness (drives the vendored FrogNano Leaf agent)
    cli.py         shared CLI harness plumbing
    opencode.py    opencode adapter
    pi.py          pi adapter
    pi_stats.py    reads the token-speed entries out of a pi session
  _vendor/
    frognano/      pinned FrogNano subset (MIT, Microsoft Corporation)
    pi_token_stats/ pinned pi-token-stats extension (see its PROVENANCE.md)
```

## Licence and provenance

The benchmarks, the reference harness and the task definitions come from
microsoft/FrogNano and laude-institute/harbor-datasets, pinned by commit. This
project vendors no model weights and contacts no model host other than the
OpenAI-compatible endpoint you configure.

### Why FrogNano is vendored

FrogNano was originally a pinned git **dependency**
(`frognano @ git+...@7077a19d93f38fa0a7afd58b90a0774b25f6b629`). It is now
**vendored** into `src/coding_agent_evaluations/_vendor/frognano/`, at that same
commit, and imported through that path rather than as a top-level package.

The reason is the dependency graph, not preference. As a dependency, FroNano
forced this project to install `kubernetes==36.0.3` — 3,650 lines of a runtime
this project never executes, because the host runtime replaces it — purely to
satisfy a module-level import and a type annotation in
`frognano/harness/leaf/environment.py`. Vendoring lets that seam be cut. FroNano
also pinned `openai==3.13.0` and pulled `transformers` for a Hugging Face
tokenizer fallback that this project never reaches. Dropping the dependency
removes `kubernetes` and `transformers` from the install entirely; the remaining
three dependencies are the ones actually imported.

Vendored code is excluded from `ruff check` and `ruff format` so it stays
diffable against upstream. Upstream is MIT licensed (Copyright (c) Microsoft
Corporation); the notice ships in `_vendor/frognano/LICENSE`.

### Why pi-token-stats is vendored

The pi extension from
[NoRaincheck/pi-token-stats](https://github.com/NoRaincheck/pi-token-stats) is
vendored into `src/coding_agent_evaluations/_vendor/pi_token_stats/` at commit
`fb1abd2`, unmodified, and loaded from there by explicit path.

Which metrics exist and what they mean is part of what a run reports, so the
extension a run measured with has to be pinned rather than fetched from `main` at
run time — and rollouts run with `PI_OFFLINE=1` anyway. Vendoring also keeps it out
of the machine's global pi installation, which the harness otherwise makes invisible
to a measurement. `_vendor/pi_token_stats/PROVENANCE.md` carries the commit, the
upstream README, the two upstream behaviours the harness does not paper over
(`stopReason` is assigned after the entry is serialised, so it never lands in the
session; `cacheAwarePrefillTokens` only exists when the provider reports cache
reads, which a local endpoint usually does not), and the update recipe.

**What was vendored** — the closure of what this project imports, and nothing
more: `datasets/` (task sources and revisions), `harness/leaf/` (the agent, its
five tools, its tool runner, its system prompt), `runtimes/errors.py` and
`runtimes/python.py`, and `config.py` for `DEFAULT_MAX_TOKENS_PER_TURN`.
**Not vendored** — `runtimes/kubernetes.py`, `runner.py`, `cli.py`, `wandb.py`
and `configs/`: all reachable only through the reference runner or the cluster
path, none of it on this project's import path.

**Divergences from upstream**, both mechanical and marked `VENDORED DIVERGENCE`
in-file:

| Divergence | Why it is safe |
|---|---|
| `from frognano.` → `from coding_agent_evaluations._vendor.frognano.` | Import path only. |
| `LeafEnvironment.__init__`'s `KubernetesTaskRuntime` annotation left unresolved, and `runtimes/__init__.py` no longer re-exports it | `from __future__ import annotations` makes the annotation a string, so it is never evaluated. The runtime is duck-typed — Leaf calls only `get_task_instruction`, `run`, `copy_to_container`, `get_patch`, `compute_reward`, `recreate`, and reads `.logger` and `.task`. |

Nothing else differs. The agent loop, the five tools, the tool schemas, the
system prompt and the tool runner are unmodified, so the leaf baseline still
measures the reference agent. One consequence is worth stating plainly:
`config.py` reaches `files("frognano.configs.eval")` to load packaged eval YAML,
and `configs/` is not vendored, so those two loader functions are unreachable.
Only `DEFAULT_MAX_TOKENS_PER_TURN` is consumed from that module.
