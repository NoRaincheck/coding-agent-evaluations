# coding-agent-evaluations

Evaluate an OpenAI-compatible model on the benchmarks from
[microsoft/FrogNano](https://github.com/microsoft/FrogNano) — **SWE-bench
Verified** and **Terminal-Bench 2.0 Verified** — once per agent harness:

| Harness | What it is |
|---|---|
| `leaf` | The reference harness from FrogNano, used unmodified. The baseline. |
| `opencode` | The opencode CLI, driven headless in the task workspace. |
| `pi` | The pi CLI, driven headless in the task workspace. |

The same tasks, model and limits go to every harness, so the resulting
`summary.json` files are directly comparable.

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

The reference implementation is pinned as a dependency
(`frognano @ git+...@7077a19`), which is what keeps the leaf harness byte-identical
to upstream. Harnesses are invoked from your own `PATH`:

```bash
brew install opencode                                    # or your preferred install
npm i -g @earendil-works/pi-coding-agent                  # pi
```

No model is ever downloaded. The endpoint is reached over the OpenAI chat
completions API only, and token accounting runs offline (no tokenizer fetch).

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

# Compare finished runs
cae compare eval-results/ --markdown
```

`scripts/run-matrix.sh` runs the whole matrix and prints the comparison table;
`scripts/smoke.sh` is the fast version.

### Configuration

Configs compose with `extends`, and every value accepts `${VAR}` /
`${VAR:-default}`:

```
configs/
  base.yaml                            model + limits + runtime
  dataset/swebench-verified.yaml       the benchmark preset
  dataset/terminal-bench-2-verified.yaml
  harness/leaf.yaml  opencode.yaml  pi.yaml
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
`model_params_ignored`.

## How it works

```
config  ──▶ datasets ──▶ runtime (host) ──▶ harness ──▶ verifier ──▶ summary
                            │
                            └─ provisions the workspace the image would provide
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
and credentials cannot leak into a measurement.

## Fidelity: what this does and does not reproduce

Running on the host instead of in the benchmark image is a real deviation. It is
worth being explicit about what it changes.

**Reproduced faithfully**

- The leaf agent, its five tools, its system prompt and its tool schemas —
  imported from the pinned FrogNano commit, not reimplemented.
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

## Development

```bash
.venv/bin/python -m pytest tests -q
```

The test suite runs offline: it covers config composition and env expansion,
verifier translation, the command executor's exit-code/timeout/checksum
behaviour, Dockerfile parsing, patch capture, reward extraction, harness
configuration generation and the comparison report. It does not call a model.

Layout:

```
src/coding_agent_evaluations/
  config.py        config composition, env expansion, model settings
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
    leaf.py        reference harness (imported from frognano)
    cli.py         shared CLI harness plumbing
    opencode.py    opencode adapter
    pi.py          pi adapter
```

## Licence and provenance

The benchmarks, the reference harness and the task definitions come from
microsoft/FrogNano and laude-institute/harbor-datasets, pinned by commit. This
project vendors no model weights and contacts no model host other than the
OpenAI-compatible endpoint you configure.