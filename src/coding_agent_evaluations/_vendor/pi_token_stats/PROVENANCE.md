# Vendored: pi-token-stats

- Upstream: https://github.com/NoRaincheck/pi-token-stats
- Commit: `fb1abd2fe7649f3fb9315511cc11a84460238bca` ("Rename extension to index.ts; add
  prefillTps & cacheAwarePrefillTokens metrics")
- `index.ts`: verbatim copy of upstream `index.ts` at that commit.
- `UPSTREAM_README.md`: upstream README, unmodified, kept so the metric definitions travel
  with the code.

## Why it is vendored

The pi harness must load a **pinned** extension to measure token speed: which metrics exist,
and what they mean, is part of what a run reports. Fetching `index.ts` from GitHub at run
time would make a measurement depend on whatever `main` happens to hold, and the rollout runs
with `PI_OFFLINE=1`. Vendoring also keeps the run hermetic — the extension is written into the
rollout's own agent directory and never read from a global install.

## Divergence from upstream

None in the code. `index.ts` is byte-identical to upstream at the pinned commit. The harness
only sets the environment the extension already honours:

| Knob | Value | Why |
|---|---|---|
| `PI_TOKEN_SPEED_STATS_DIR` | the rollout's hermetic agent directory | Upstream defaults to `~/.pi/agent`, which would write `token-speed-stats-state.json` into the operator's real pi directory. Pointing it at the rollout's own directory keeps the calibration state (chars-per-token ratios) inside the measurement. |

Behaviour the harness does not change and does not hide:

- `TokenSpeedStatsEntry.stopReason` is assigned *after* `pi.appendEntry` serialises the entry,
  so `stopReason` never reaches the session file upstream. The harvested entries are reported
  exactly as written.
- `intermediateToolCalls` is only present when the extension's `tool_call` hook has fired.
- `cacheAwarePrefillTokens` is only present when the provider reported cache reads. A local
  OpenAI-compatible endpoint normally reports none, so expect it to be absent.

## Updating

```bash
git clone https://github.com/NoRaincheck/pi-token-stats /tmp/pi-token-stats
git -C /tmp/pi-token-stats checkout <commit>
cp /tmp/pi-token-stats/index.ts src/coding_agent_evaluations/_vendor/pi_token_stats/index.ts
cp /tmp/pi-token-stats/README.md src/coding_agent_evaluations/_vendor/pi_token_stats/UPSTREAM_README.md
```

Then update the commit above and the entry types read in
`src/coding_agent_evaluations/harnesses/pi_stats.py` if upstream renames them.
