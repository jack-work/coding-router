# Using the exported router

Two artifact kinds share one file pair (`router.json` + `router.npz`); `meta["kind"]`
selects the loader. `load_router()` returns the right class either way — the kNN kind
keeps working unchanged and is the rollback path.

## kNN kind (`kind` absent or `"knn"`)
- `router.npz` — task embeddings (`emb`), per-arm outcomes (`resolved`, bool), per-arm median cost (`med_cost`)
- `router.json` — arm list, `arm_spec`, `k`/`tau`/`sim_floor`, provenance, scope
- Decision: weighted k-nearest vote → cheapest arm with P(solve) ≥ `tau`, escalate on
  fallback/off-distribution.

## Trained kind (`kind == "trained"`, EXP-012 winner `reward_lcb_b0.2`)
- `router.npz` — `emb` (110-task DeepSWE bank in the TUNED encoder's space, fp32),
  `graded` (41×110 float outcomes — the calibration side of the vote), `med_cost` (41,)
- `router.json` — everything above plus `T` (trained soft-vote temperature), `lam`
  (selected cost weight), `embed_model_mlx`/`embed_model_torch` (encoder dir names
  beside the artifact in `hf_repo`), and provenance incl. the selection policy
- Encoder, one artifact, two formats (bank and queries must share one vector space):
  - `encoder-fp16/` — merged LoRA→base Qwen3-Embedding-0.6B, safetensors; loads via
    sentence-transformers/transformers (CUDA/CPU)
  - `encoder-mlx-4bit/` — the same merged encoder quantized 4-bit (group 64) for
    Apple Silicon; loads via `mlx_embeddings`
- Decision: softmax(sims / `T`) vote over the bank → per-arm P(solve) → argmax of
  u = P(solve) − `lam`·`med_cost` (the exact rule EXP-012 certified — deliberately NOT
  the kNN cheapest-above-tau walk). Abstention: `nearest_sim < sim_floor` escalates to
  the strongest arm; that is the trained kind's only abstention.

## Use
```python
from router.router_core import load_router
r = load_router("results")        # kNN or trained, per the artifact's meta
d = r.route_embedding(vec)         # embed with the artifact's own encoder (see below)
if d.off_distribution:
    ...                            # it abstained; it returned the strongest arm
client.messages.create(**d.request_kwargs, messages=[...])
```
Embedding must happen in the artifact's own vector space:
`load_local_embedder(*(r.embed_models() or (EMBED_MODEL_MLX, EMBED_MODEL_TORCH)))`
handles both kinds (a trained artifact's tuned encoder downloads on first use).
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
