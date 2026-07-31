# Router experiment journal

Mirror of the Notion research journal (router lane). Notion MCP is configured
(notion.com/mcp) but not connected in the authoring session — copy each entry to the
Research page's router section when a connected session is available. One entry per
experiment; format: motivation / setup / result / verdict / next.

All experiments share the evaluation protocol fixed by EXP-001 unless stated:
DeepSWE v1.1, 41-arm frontier pool x 110 tasks, 6 seeds of 80/20 repo-split holdouts,
hyperparameters + checkpoints selected ONLY on an inner 75/25 repo split of the train
side (feasibility: inner graded >= inner-best-arm - 0.02, then max cost ratio), pooled
holdout decisions, repo-clustered bootstrap CIs. Baseline = always-best train arm
(claude_opus_5_high on every seed): $5.53/task, graded 0.934 on the pooled cells.

---

## EXP-001 — LR baseline + embedding-model comparison (2026-07-30)

**Motivation.** Before any RL: does a per-arm logistic regression on task embeddings beat
the kNN incumbent, and which embedding space is best? (CARROT-style plug-in router;
soft labels per Hybrid-LLM.)

**Setup.** Branch `exp/lr-baseline-embeddings` commit dba6343, `router/experiments.py
lr-baseline`. Per-arm L2 LR (numpy dual IRLS, graded soft targets, prior-log-odds
offset), decision rule identical to holdout-deepswe kNN. 120 E2B jobs: 3 experiments
(dswe80, lcb2dswe, srb2dswe transfer via source difficulty direction) x 4 embeddings
(te3-large/small, qwen3-emb-0.6b/8b) x 6 seeds. Results: `results/lr_baseline.json`.

**Result.** kNN beats LR at every embedding: kNN qwen3-0.6b 4.02x (-0.027), te3-large
3.94x (-0.021), 8b 3.32x (-0.007) vs LR best (te3-small) 3.05x (-0.053). Transfer dead:
z-only collapses to per-seed constant picks; emb+z == emb. Coarse tau grids (0.2-spaced)
collapse LR to cheap arms (-0.205) — grids need 0.05 resolution.

**Verdict.** Frozen-encoder kNN at 4.02x/-0.027 (qwen3-emb-0.6b) is the bar. LR head
rejected. Qwen3-Embedding-0.6B validated as open substrate (parity with te3-large).

**Next.** Offline RL through the tabular (soft-kNN) policy.

---

## EXP-002 — softknn-rl v1: exact expected-reward gradients (2026-07-31)

**Motivation.** Keep the winning policy class, make its geometry trainable. Full
41x110 outcome matrix => expected reward under the policy is EXACTLY computable and
differentiable (offline RL with zero sampling variance).

**Setup.** Box 6 (2xH100), `/nvme/work/router-rl/train_softknn.py`, tmux rlrouter0/1.
Policy: p_hat[a](x) = softmax-weighted (T trainable) neighbor vote over train fold;
u = p_hat - lam*medcost; train dist = softmax(u/tau). Loss = -E_pi[graded - lam*cost],
query's repo masked from memory (NCA-style, mirrors grouped CV). lam trained at 0.02,
swept {0.005..0.1} in the decision rule at eval. Modes: frozen (T,tau only) and LoRA
r=16 on Qwen3-Embedding-0.6B (grad checkpointing; note: checkpointing silently no-ops
in eval mode — needs model.train()). Deployment rule: hard argmax(u), memory = full 80%.

**Result (pooled, 6 seeds).**
- frozen: 3.75x, graded 0.943 vs base 0.934 (delta +0.009, CI [-0.023, +0.044]) —
  statistical parity at 3.75x cheaper. Beats EXP-001 kNN on quality at similar ratio.
- lora: 5.72x, graded 0.885 (delta -0.049, CI [-0.080, -0.021]) — provably cheaper AND
  provably worse. All seeds selected lam=0.1 + early checkpoints.

**Failure mode diagnosed.** LoRA overfits monotonically: holdout graded decays with
training steps (lam=0.02: 0.933@10 -> 0.897@300; lam=0.1: 0.906 -> 0.874) while the
frozen curve is flat (~0.94). Classic offline over-optimization of the encoder against
an 88-task matrix.

**Verdict.** Two new Pareto points. Trained-temperature utility rule (frozen) is the
new quality-parity champion: 3.75x at parity. Encoder fine-tuning needs anchoring.

**Next.** EXP-003 KL-anchored LoRA; EXP-004 8B encoder under the trained rule.

---

## EXP-003 — anchored LoRA: KL(pi || pi_ref) regularization (RUNNING, 2026-07-31)

**Motivation.** EXP-002's LoRA over-optimization is the textbook case for reference-
policy anchoring (the offline-RL/PPO-family fix): penalize divergence from the frozen-
geometry policy so the encoder can only move where reward justifies it.

**Setup.** `train_softknn_v2.py`, GPU0 tmux rlv2gpu0. loss += beta * KL(pi || pi_ref),
pi_ref = init-geometry policy (LoRA B=0 at step 0), beta in {0.05, 0.2}, 6 seeds each.
Success = holdout quality decay eliminated AND a point dominating either EXP-002 mode.

## EXP-004 — 8B encoder under the trained decision rule (2026-07-31)

**Motivation.** EXP-001 showed 8B embeddings buy quality (kNN -0.007). Does the
trained-temperature utility rule on frozen 8B embeddings push the parity point past
3.75x, or graded strictly above baseline?

**Setup.** `train_softknn_v2.py`, GPU1 tmux rlv2gpu1, frozen mode, Qwen3-Embedding-8B,
6 seeds.

**Result.** 3.74x, graded 0.921 (delta -0.013, CI [-0.039, +0.011]). Parity, but NOT
better than 0.6B frozen (3.75x, +0.009) — the kNN-era "8B protects quality" pattern
does not survive the trained decision rule. Per-seed ratios 2.28-5.93.

**Verdict.** 0.6B stays champion; 8B adds cost (encode latency, VRAM) for nothing here.

---

## EXP-005 — low-rank factor head vs tabular memory (RUNNING, 2026-07-31)

**Motivation.** Research synthesis recommends a rank-r arm-embedding head (EmbedLLM/
IRT-style) over independent heads: 4,510 graded cells support a factor model better
than 41 separate 110-sample fits. Same exact-expected-reward objective, frozen 0.6B
embeddings, KL anchor to the base-rate policy. Tests parametric-vs-tabular at fixed
objective/protocol.

**Setup.** `train_factor.py`, GPU1 tmux rlv3gpu1. logit[a](x) = v_a.(P e_x) + b_a,
b_a init at train base-rate log-odds. (rank, beta) in {(4, 0.1), (16, 0.1), (4, 0)},
6 seeds each, lam swept in decision rule.

## EXP-006 — 8B anchored LoRA (QUEUED after EXP-005, 2026-07-31)

**Setup.** `train_softknn_v2_150.py` (150 steps), lora mode, 8b, beta=0.2, 6 seeds,
GPU1 same tmux chain. The quality-frontier bet: anchored geometry training on the
bigger encoder.
