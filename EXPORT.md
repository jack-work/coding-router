# Using the exported router

## Files (1.05 MB total, self-contained)
- `results/router_v0.npz` — task embeddings, per-arm outcomes, per-arm median cost
- `results/router_v0.json` — arm list, model+effort mapping, hyperparameters, provenance, scope
- `router/router_core.py` — the only code needed at inference (`Router`/`Decision`). Deps: `numpy`, `openai` (embeddings only).

## Use
```python
from router.router_core import Router
r = Router("results")
d = r.route(issue_text)           # or r.route_embedding(vec) if you embed yourself
if d.off_distribution:
    ...                            # it abstained; it returned the strongest arm
client.messages.create(**d.request_kwargs, messages=[...])
```
`request_kwargs` is ready to splat into the Anthropic or OpenAI SDK and already encodes the
measured API constraints (adaptive thinking + `output_config.effort` for Anthropic,
`reasoning.effort` for OpenAI).

## What it is
kNN over 110 labelled long-horizon SWE tasks (41 OpenAI/Anthropic arms, 88 repos) from
DeepSWE v1.1. No fitted weights — the lookup table *is* the model. Embeds the task, takes the
12 nearest labelled neighbours, and picks the cheapest arm those neighbours say will likely
solve it; escalates when nothing clears the bar or nothing is close enough.

## Measured
| policy | graded | cost | vs always-best |
|---|---|---|---|
| always `claude-opus-5@high` | 0.954 | $679.9 | baseline |
| exported router (guard on) | 0.939 | $375.3 | **1.81×** |
| ungated policy | 0.933 | $316.1 | 2.15× — CI [1.87, 2.46] |

Graded-delta 95% CI **[−0.044, +0.000]** (repo-clustered bootstrap), i.e. not distinguishable
from parity at n=110. Nested repo-grouped CV; hyperparameters chosen inside training folds only.

## Limits — read before trusting it
- **Input must be repo-issue-sized** (p50 ≈ 2,000 chars). One-line prompts score 0.14–0.34
  similarity, below the in-distribution minimum, and always abstain. Safe, but no saving.
- **One dataset, 110 tasks.** No held-out confirmation yet.
- **Long-horizon only** (median 61 agent steps). Says nothing about short interactive asks.
- Quality loss concentrates on the hardest third (−0.025) and is flat-to-positive elsewhere.
- Known defect: `gpt-5.6-sol@xhigh` takes 10% of traffic and 19.5% of spend at the worst
  graded score of any arm used. Rerouting it should improve both axes.
