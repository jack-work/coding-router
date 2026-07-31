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

**Result (beta=0.05, 6 seeds).** 5.84x, graded 0.882 (delta -0.052, CI
[-0.087, -0.021]) — statistically indistinguishable from UNANCHORED LoRA
(5.72x/-0.049): the weak anchor changed nothing. All seeds again selected lam=0.1.

**Result (beta=0.2, 6 seeds).** 5.82x, delta -0.030 (CI [-0.061, -0.002]), selected
checkpoints late (150-270): the STRONG anchor tamed the overfit decay and recovered
~2pp of quality at the same ratio tier. Directionally the offline-RL anchoring story
holds; it took beta=0.2, not 0.05.

**Verdict.** Anchored LoRA at beta=0.2 (5.82x/-0.030) sits on the frontier next to
the factor champion (5.33x/-0.027) — at ~100x the training compute. Anchoring works;
it is not worth the GPUs at n=110.

---

## EXP-009 — encoder training-algorithm sweep results (2026-07-31)

Seeds 0-5, LoRA 0.6B, beta=0, 150 steps, identical protocol; reward control replicates
unanchored LoRA (sanity holds):

| algo | ratio | graded delta (CI) | note |
|---|---|---|---|
| rank (pairwise utility) | 6.63x | -0.051 [-0.089, -0.017] | highest ratio in program |
| reward (control) | 5.77x | -0.047 [-0.080, -0.017] | replicates EXP-002 lora |
| grpo (sampled) | 5.60x | -0.037 [-0.065, -0.011] | noise-as-regularizer: mild, directional |
| nca (metric only) | 4.55x | -0.019 [-0.043, +0.004] | best encoder-trained quality; no reward signal |

**Reading.** The training signal matters more than the estimator: ranking pushes the
cost axis; pure representation learning (NCA) protects quality. GRPO ~ exact gradients
(overlapping CIs). Fresh-seed confirmations (seeds 6-11) for rank and nca launched,
pre-registered: same acceptance rule as EXP-007 (delta CI contains 0 for a parity
claim; report cost otherwise).

**Current per-task Pareto set (DeepSWE, pooled 6-seed unless noted):**
frozen-temps 3.75x/+0.009 -> nca 4.55x/-0.019 -> factor champion 5.33x/-0.027
(fresh-confirmed) -> anchored-lora-b0.2 5.82x/-0.030 -> rank-lora 6.63x/-0.051.

**Fresh-seed confirmations (seeds 6-11).**
- rank: 8.36x, delta -0.052 (CI [-0.086, -0.019]) — CONFIRMED, even stronger ratio;
  quality cost stable at ~5pp. The program's max-savings point.
- nca: 5.01x, delta -0.036 (CI [-0.073, -0.003]) — parity claim DEMOTED (CI excludes
  0 on fresh seeds); lands between champion and anchored-LoRA.

**Selection-bias tax, quantified.** Every quality-selected variant degrades ~0.02
graded from selection seeds to fresh seeds (champion -0.003 -> -0.027; nca -0.019 ->
-0.036), while rank — selected for ratio, not quality — held (-0.051 -> -0.052).
Read all seeds-0-5 deltas with a ~-0.02 correction. Frozen-temps' +0.009 parity floor
rests on seeds 0-5 only; fresh confirmation launched (with grpo, completing the
sweep's fresh picture), plus rank+beta-0.2 (does anchoring buy back rank's quality?).

**Honest frontier on FRESH evidence:** factor champion 5.33x/-0.027 (balanced pick),
nca 5.01x/-0.036, rank-LoRA 8.36x/-0.052 (aggressive pick).

**Round 2 fresh results.** frozen-temps 3.87x/-0.018 (CI [-0.049, +0.010]) — PARITY
FLOOR CONFIRMED. grpo 7.12x/-0.030 (CI [-0.062, -0.000]) — better than its selection
seeds (reverse luck possible; seeds 12-17 tiebreak launched for grpo vs rank).
rank+anchor b0.2: 6.76x/-0.050 — anchoring does NOT buy back rank's quality.
Offline per-task program CONVERGED: ~3.9x parity / 5.3x -0.027 / 7-8x -0.03..-0.05.

---

## EXP-012 — canonical re-run: quality-first, one eval, generalization axis (RUNNING)

**Directive (Kion, 2026-07-31).** Quality bound under 1 graded point even at only 2x
savings; the router must run locally on a user's machine; EXACTLY ONE eval setting =
DeepSWE holdout; vary the TRAINING dataset to measure generalization to DeepSWE;
measure tokens/task and speed/task. Roster: always-best, oracle, static luna_max,
kNN, soft-kNN temps, GRPO-LoRA (most promising), REINFORCE-LoRA, each +/- KL anchor.

**Changes from prior protocol.** Feasibility margin -0.02 -> -0.01; lam grid extended
to 0.001; fallback selection = max inner graded (quality-first). Token (input+output,
cache separate) and agent-duration matrices built from trials.json (100% filled) —
every routed decision now reports tokens/task and seconds/task. Statistical honesty:
at n=110 the <1% bound is a POINT-ESTIMATE criterion with CI containing 0; certifying
<1% at 95% needs ~5-10x more eval data.

**Tiebreak input (seeds 12-17).** grpo 5.84x/-0.036 vs rank 6.52x/-0.068 — GRPO's
quality is stable (~-0.034 across three batches), rank's is worse and drifting; rank
DROPPED from roster per directive.

**Sweep.** v5: {frozen, reward, grpo} x {dswe, lcb, srb} x beta {0, 0.2} (frozen:
beta 0 only) = 15 configs x seeds 0-5 on both GPUs (~7h). SRB trains quality-only
(no cost field); LCB cost term rescaled (lam_train=3.0). Eval side always identical:
DeepSWE-train memory, DeepSWE costs, inner-split selection. Winners get fresh-seed
confirmation on 6-11 before the final table.

## EXP-011 — per-turn routing under injected difficulty (RUNNING, 2026-07-31)

**Motivation.** EXP-010 found no per-turn advantage on LCB because easy episodes offer
nothing to escalate into. Inject the escalation need: FORCE the weakest arm
(claude-haiku-4-5@budget, 0.632 LCB resolve) for the first 4 turns of every episode;
the router takes over from turn 5. If per-turn routing has value anywhere on LCB, it
is here: the policy must read live struggle signals (public-test failures, exit codes)
and decide whether/where to escalate. Forced turns are excluded from REINFORCE.

**Setup.** `rl/perturn.py train --sticky --handicap 4` (5 iters x 20 tasks x 2), then
`eval --handicap 4`: per-turn vs turn0-frozen(post-prefix) vs continue-weak (never
escalate) vs escalate-opus (always escalate) — the last two bracket the value of
DECIDING. Pre-registered success: per-turn beats turn0-frozen on paired graded or
cost at matched other-axis; both must beat continue-weak to show escalation matters.

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

## EXP-005 — low-rank factor head vs tabular memory (2026-07-31)

**Motivation.** Research synthesis recommends a rank-r arm-embedding head (EmbedLLM/
IRT-style) over independent heads: 4,510 graded cells support a factor model better
than 41 separate 110-sample fits. Same exact-expected-reward objective, frozen 0.6B
embeddings, KL anchor to the base-rate policy. Tests parametric-vs-tabular at fixed
objective/protocol.

**Setup.** `train_factor.py`, GPU1 tmux rlv3gpu1. logit[a](x) = v_a.(P e_x) + b_a,
b_a init at train base-rate log-odds. (rank, beta) in {(4, 0.1), (16, 0.1), (4, 0)},
6 seeds each, lam swept in decision rule.

**Result.** NEW CHAMPION: r4 beta=0 -> 4.25x, graded 0.931 (delta -0.003,
CI [-0.031, +0.027]) — parity at a better ratio than every prior policy, dominating
the EXP-001 kNN incumbent (4.02x, -0.027) on quality. Routes to only 3 arms
(terra_high 50.4%, opus_5_low 30.7%, luna_xhigh 19.0%); never uses the $5.53 baseline
arm. The KL-to-base-rate anchor HURT here: r4/r16 beta=0.1 give ~5.3x but delta CIs
[-0.051,+0.001]/[-0.045,+0.001] — real quality cost. Weight decay + base-rate bias
init are regularization enough for a head this small.

**Verdict.** Partially overturns EXP-001's "kNN beats learned heads": what changed is
the reward objective + rank-4 cross-arm sharing + base-rate init, not the head class.
Champion promoted pending EXP-007 fresh-seed confirmation (multiple-comparisons guard:
many variants have now been selected against the same 6 seeds).

---

## EXP-007 — champion confirmation on fresh seeds 6-11 (RUNNING, 2026-07-31)

**Motivation.** Integrity: the 4.25x champion was one of many variants compared on the
same 6 holdout seeds; its edge could be selection luck. Fresh, never-touched repo
splits (seeds 6-11) give an unbiased estimate of the promoted config exactly as-is
(factor r4, beta=0, same lam/checkpoint selection protocol).

**Setup.** `train_factor.py 6,7,8,9,10,11 4 0`, CPU on box 6 (GPUs busy), tmux
rlconfirm. Pre-registered acceptance: pooled fresh-seed delta CI must contain 0 and
ratio must stay >= 3.5x; anything less demotes the champion to "seed-overfit".

**Result.** PASS: 5.33x (CI [4.86, 5.79]), delta -0.027 (CI [-0.059, +0.005]),
per-seed ratios 3.52-8.80. Honest read: ratio is robust (4.2-5.3x across seed
batches); quality delta's central value moved from -0.003 to -0.027 — parity holds
by the pre-registered test, but a small real quality cost (~0.02-0.03) is likely.
The original 6-seed -0.003 was probably the optimistic tail of selection.

**Verdict.** Champion CONFIRMED as the per-task offline configuration: factor r4,
beta=0, reward objective, frozen 0.6B embeddings — ~4.3-5.3x at <=0.03 quality cost.

---

## PIVOT (Kion, 2026-07-31): PER-TURN routing is the primary objective

Route every agent turn (each LLM call), switching arms mid-trajectory. All experiments
above are per-task (one arm per whole episode) and become baselines. Per-turn
counterfactuals do not exist in any offline matrix we hold, so this requires ON-POLICY
LIVE rollouts with the models in the loop (Kion endorsed). Live environment:
LiveCodeBench agentic episodes in E2B (repo harness), where episodes are minutes and
cents, vs DeepSWE's 15-min/$1+ episodes.

## EXP-008 — on-policy per-turn routing on LiveCodeBench (BUILDING, 2026-07-31)

**Motivation.** First per-turn result + first models-in-the-loop RL. Does a per-turn
policy (state = task difficulty priors + live episode signals) beat (a) the best
static arm and (b) the same policy frozen to its turn-0 decision (per-task routing),
on held-out contests at matched cost?

**Design.**
- Episode runner: stateless per-turn calls — canonical TEXT history (tool calls and
  outputs flattened to text, mini-swe-agent-style), native tool EMISSION per call, so
  any turn can switch arm/provider without provider-native history round-tripping.
- Arms: the 7 LCB-matrix arms (all cheap: med $0.0016-$0.03/episode).
- State per turn: kNN per-arm p_solve priors from the TRAIN-contest matrix only,
  turn fraction, log cost-so-far, public-test pass fraction, wrote-solution flag,
  last-exit-ok.
- Policy: linear softmax head; sampled during training (REINFORCE, task-mean baseline
  from M=2 rollouts, GRPO-style), argmax at eval. Reward = graded - lam*cost, lam=8.
- Split by CONTEST (train contests for rollouts, held-out contests for eval).
- Budget cap: abort if cumulative episode spend exceeds $40.

**Cross-lane note (2026-07-31).** A parallel lane (`exp/per-turn-routing`, worktree
../coding-router-perturn) probed per-turn routing with the DeepSWE-trained router on
LCB and found every decision off_distribution (DeepSWE router is OOD on AtCoder
puzzles) — EXP-008 avoids that confound by using LCB-native kNN priors. Its second
insight (Kion): a mid-episode switch pays the candidate arm's COLD prefill, so
divergence is only worth acting on if the confidence gain clears that cost. EXP-008's
stateless-turn design makes all arms pay full prefill uniformly (cost internalized in
reward), and the `--sticky` variant (prev-arm one-hot feature) lets the policy LEARN
switch economics. Caveat carried forward: LCB quality is near-saturated with these
arms (graded ~0.99 in training iterations), so per-turn results here demonstrate
mechanism + cost optimization, not the DeepSWE-regime quality-cost tension.

**Training progress.** it0-it3: graded 0.986-0.993 stable, mean reward +0.65 -> +0.78
(cost per iteration $1.70 -> $1.05 — the policy is learning to shed cost at flat
quality). Spend so far ~$6 of the $40 cap.

---

## EXP-009 — encoder training-algorithm sweep (QUEUED for GPU0, 2026-07-31)

**Motivation.** Encoder training showed signal (EXP-002 LoRA moved the frontier;
anchoring diagnosis pending in EXP-003). Compare TRAINING SIGNALS through the same
soft-kNN policy, same protocol, 150 steps, beta=0: exact expected reward (control),
pairwise utility ranking (EquiRouter-flavored, robust to near-ties), GRPO sampled
estimator (does gradient noise regularize?), NCA metric learning (representation-only
control: no reward signal, positives = tasks with correlated arm-outcome profiles).
`rl/train_softknn_v4.py`, 4 algos x 6 seeds, launches when EXP-003 vacates GPU0.

## EXP-010 — per-turn router variants vs current bests (2026-07-31)

**Comparison set, all LIVE on held-out contests (greedy):** trained per-turn linear
policy, turn0-frozen control (per-task routing with the same policy), hand ladder
cascade (cheap->escalate on failing public tests), static nano@high, static
opus@medium. Plus a `--sticky` retrain (prev-arm feature). Success metric: per-turn
beats turn0-frozen at matched-or-better graded — that is the direct per-task-vs-
per-turn granularity answer on live episodes.

**Result (21 held-out live tasks, paired bootstrap vs always-opus@medium):**

| variant | graded | $/task | paired dg (CI) | x cheaper |
|---|---|---|---|---|
| sticky-it4 turn0-frozen | 0.949 | 0.0051 | -0.046 [-0.137, +0.000] | 18.2x |
| sticky-it4 per-turn | 0.949 | 0.0080 | -0.046 [-0.137, +0.000] | 11.6x |
| sticky-it3 turn0-frozen | 0.939 | 0.0045 | -0.056 [-0.151, +0.000] | 20.7x |
| ladder (hand cascade) | 0.901 | 0.0457 | -0.094 [-0.232, +0.000] | 2.0x |
| non-sticky it3/it4 per-turn | 0.856-0.889 | 0.028-0.047 | -0.10..-0.14 | 2.0-3.3x |
| static nano@high | 0.808 | 0.0481 | -0.187 [-0.370, -0.044] | 1.9x |
| static opus@medium (ref) | 0.995 | 0.0928 | 0 | 1.0x |

**Findings.**
1. Trained policies reach the cheap corner: ~0.94-0.95 graded at $0.005-0.008/task —
   11-20x cheaper than always-opus at a small (CI-touching-zero) quality cost. They
   dominate the hand ladder and both static arms.
2. PER-TURN SHOWS NO ADVANTAGE OVER TURN0-FROZEN here: same policy frozen at turn 0
   is equal quality and CHEAPER (per-turn pays switch re-prefill, visible as
   +60% cost for sticky-it4 per-turn vs its turn0 twin). Consistent with the
   saturation caveat: short, easy LCB episodes offer no mid-episode escalation
   opportunities; the granularity question needs the DeepSWE regime to be decisive.
3. The sticky (prev-arm) feature was the difference between the cheap corner and
   mediocrity — non-sticky policies landed at 2-3x. Arm-commitment coherence matters.
4. Live-vs-matrix scaffold gap: static nano@high costs $0.048/task live vs $0.0034
   matrix median — the stateless per-turn scaffold re-prefills every turn and hurts
   reasoning-heavy arms most. Matrix priors and live costs are not interchangeable.
5. Ops lessons: concurrent evals raced the episode cache (ladder resamples differ
   0.901/$0.046 vs 0.949/$0.032 — that spread IS the live-eval noise floor at n=21);
   mid-wave reads are biased (it3 looked 0.998 at 16/21, fell to 0.856 at 21/21 —
   stragglers are hard tasks); checkpoint selection by train reward is mandatory
   (it4 was one noisy update past it3 and markedly worse).

**Verdict.** On-policy per-turn training WORKS mechanically end-to-end (cross-provider
mid-episode switching, $25 total spend), and trained routing crushes statics/ladder on
LCB — but per-task (turn0) granularity is not beaten on this benchmark. Next per-turn
test needs either injected mid-episode difficulty on LCB or a live long-horizon domain.

## EXP-006 — 8B anchored LoRA (2026-07-31)

**Setup.** `train_softknn_v2_150.py` (150 steps), lora mode, 8b, beta=0.2, 6 seeds,
GPU1 same tmux chain. The quality-frontier bet: anchored geometry training on the
bigger encoder.

**Result.** 5.69x, graded 0.897 (delta -0.036, CI [-0.065, -0.011]). Best LoRA
quality profile in the 5-6x tier, and selected checkpoints skew LATE (140-150 vs
0.6B's 10-100) — the bigger encoder + anchor overfits much slower per step. Still a
real quality cost (CI excludes 0) and does not dominate the factor-head champion
(5.33x/-0.027 fresh-seed), which trains in seconds instead of 13 GPU-hours.

**Verdict.** Geometry fine-tuning at 8B is viable but not worth it on this data size;
champion unchanged. If more labelled tasks ever arrive, revisit (the slow-overfit
trend suggests 8B LoRA scales with n better than 0.6B).
