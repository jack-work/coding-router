# Router experiment journal

In-repo source of truth for the RL routing lane; mirrored to Notion (parent "DeepSWE
Router" page + child "RL routing lane — experiment log", page id
3af0f8b3f591812598a6efa3fb9e66fe). Keep both in sync per-experiment. One entry per
experiment; format: motivation / setup / result / verdict / next.

All experiments share the evaluation protocol fixed by EXP-001 unless stated:
DeepSWE v1.1, 41-arm frontier pool x 110 tasks, 6 seeds of 80/20 repo-split holdouts,
hyperparameters + checkpoints selected ONLY on an inner 75/25 repo split of the train
side (feasibility: inner graded >= inner-best-arm - 0.02, then max cost ratio), pooled
holdout decisions, repo-clustered bootstrap CIs. Baseline = always-best train arm:
$5.53/task, graded 0.934 on the pooled cells.

**CORRECTION (2026-07-31, luna_max audit).** Two claims previously in this file were
wrong. (1) The always-best-train arm is NOT "opus_5_high on every seed" — opus is the
train argmax on only 3 of 6 seeds; the policy suffers winner's curse (e.g. seed 0
picks gpt_5_5_high at train 0.962, which collapses to 0.880 on test). (2) The
statement "static luna_max beat always-opus on pooled holdout cells (0.944 vs 0.934)"
misattributed the 0.934: that number is the always-best-TRAIN-ARM POLICY, not static
opus. Static opus on the identical pooled cells scores 0.963 and beats luna_max
(0.9445) on 5 of 6 seeds. Corrected claims: luna_max beats the IMPLEMENTABLE
always-best-train policy (fresh seeds 6-17: +0.0125 graded at 2.03x cheaper), and is
at statistical parity with each single frontier arm (all repo-clustered CIs span 0);
static opus remains the strongest static quality reference. All router deltas in this
file are vs the always-best-train POLICY — a winner's-curse-weakened baseline; the
EXP-012 table reports static-arm baselines alongside.

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

## EXP-014 — live DeepSWE via Pier/Modal: harness validation + drift (2026-07-31)

**Budget:** Kion approved Modal <$10k, APIs $20k. Spend ledger (cumulative, live lanes):
LCB live episodes ~$62; DeepSWE smokes ~$0.4; Modal compute ~$1.

**Harness.** DeepSWE tasks are Harbor-format; Datacurve's official runner Pier
(datacurve-pier 0.3.0, PyPI) reproduces the leaderboard setup exactly: mini-swe-agent,
Modal environments (CPU-only: tasks declare cpus=2, mem=8GB, gpus=0), separate
verifier env, reward.json with the same f2p fields as trials.json, per-trial
cost_usd/tokens/steps. Arm mapping: --model <litellm id with DOTS, e.g.
openai/gpt-5.6-luna> --ak reasoning_effort=<effort>. (Trials' internal names use
dashes; gpt-5-6-luna 404s on the live API.) Matrix opus arms ran via vertex_ai; we
call Anthropic directly — minor provenance delta.

**Zero-step trap (batch-runner rule).** A bad model id -> agent runs 0 steps -> empty
patch -> verifier scores the unmodified repo (partial ~0.13, p2p green) — looks like a
plausible bad score. Any trial with n_agent_steps==0 is MISSING DATA: halt and alert,
never record.

**Replication probe (luna_max x abs-module-cache-flags).** Live: f2p 0.95 (19/20),
$0.017, 262k tokens, 16 steps. Matrix (2026-06-30): 1.000, $1.57, 7.6M tokens, ~100
steps. Quality replicates; cost/tokens show ~30-90x efficiency drift — either the
model got drastically more efficient since matrix collection, or reasoning_effort=max
is being silently dropped (6k output tokens over 16 steps looks reasoning-free).
low-vs-max A/B on the same task running to separate the two.

**Consequence either way: matrix cost/token structure is stale for LIVE routing.**
Relative arm ordering may survive; absolutes do not. The live confirmation batch must
therefore re-benchmark the router's candidate arms live (budgeted), and live rewards
always use live costs.

## EXP-018 — held-out oracle gate: QUALITY routing on DeepSWE 1.1 is dead (2026-07-31)

**Trigger.** The sibling lane's Notion verdict (DeepSWE Router parent page, 2026-07-30):
naive per-task oracles are winner's-curse artifacts; with a HELD-OUT oracle (choose each
task's arm on half the attempts, score on the other half) their 9-arm headroom collapsed
to +1.18pts, CI includes 0. Kion had independently flagged the same ill-posedness.

**Replication on OUR 41-arm pool (113 tasks, 400 resamples, trials.json per-attempt):**
naive oracle 0.9987 -> held-out oracle 0.9619 [0.9458, 0.9757]; best static arm
0.9554 (opus_5_high). Honest quality headroom +0.0065 — includes zero. CONFIRMED.

**RETRACTION (canonical, single):** every naive-oracle quantity in this journal is
withdrawn — the luna-audit "best-of-frontier 0.987 / -0.042 headroom / 46 more solved
tasks", EXP-015's planned oracle rows, and any "routing quality prize" framing. The
held-out numbers above replace them.

**What SURVIVES — the three-layer correction (Kion: "don't overfit to their lessons").**
Layer 1 (our original claims): 4-8x cheaper vs always-best-train — INFLATED; that
baseline is a winner's-curse-weakened policy and an expensive point on the frontier.
Layer 2 (my first correction): <=1.22x vs the static Pareto frontier — DEFLATED; that
check selected the comparison statics ON THE EVAL CELLS (argmax-on-test in the statics'
favor — the mirror-image bias).
Layer 3 (the honest control, all selection train-side): a STATIC-SELECTION POLICY
(cheapest arm within a train-parity margin) scores 0.933 at $3.84/task (margin 0.01;
0.936/$2.55 at margin 0.02) on fresh cells. Against it:
| router | graded | $/task | dgraded (CI) | cost saving |
|---|---|---|---|---|
| frozen temps | 0.920 | $1.74 | -0.012 [-0.032, +0.009] | 2.2x |
| factor champion | 0.910 | $1.27 | -0.022 [-0.053, +0.008] | 3.0x |
| GRPO-LoRA | 0.907 | $0.95 | -0.025 [-0.050, +0.003] | 4.1x |
| rank-LoRA | 0.886 | $0.81 | -0.046 [-0.073, -0.017] | 4.8x (real quality loss) |
**Task-conditioned routing buys ~1.5-4x over honest static selection (margin-dependent)
at quality deltas whose CIs include zero.** More modest than layer 1, far better than
layer 2 or the sibling's 1.07x estimate. Quality-BEATING routing stays dead (held-out
oracle, verified on our data — that part of the sibling lesson is sound and adopted);
their other conclusions (cost prize 1.07x, "task statement carries no signal") are
NOT imported — our data contradicts them under honest controls.

**Consequences.**
1. EXP-012's quality-first framing is correct and unchanged (parity is the ceiling, so
   guard it and maximize savings).
2. Per-turn routing for QUALITY on DeepSWE 1.1 published data: dead — if per-task
   exploitable interaction is ~0, trajectory-conditioned quality gains cannot be shown
   there. Per-turn survives as (a) a cost play, (b) on live current models IF drift
   reopened headroom (requires >=2 live trials/cell for a held-out oracle — EXP-015 is
   1-trial; a second-trial pass on a subset is budgeted), or (c) on a benchmark with
   real headroom (sibling lane points at the ACRouter regime, ~13pt static-to-oracle gap).
3. Standing gate (adopted): compute a held-out oracle BEFORE designing any router
   against any matrix.

## EXP-015 results — live DeepSWE matrix: the published matrix is QUALITY-stale (2026-07-31)

1,125/1,130 clean trials (1 zero-step dropped, 4 missing rewards), 10 arms x 113 tasks,
1 trial/cell, **$3,576 total** (my $100-600 estimate was ~6x off: luna's 90x efficiency
drift does NOT generalize — sol/opus/fable barely got cheaper; ledger updated: ~$3.7k
of $20k API spent).

**Live per-arm (f2p, $/task):** luna_low 0.369/$0.011; luna_medium 0.662/$0.031;
**luna_max 0.687/$0.031 (COLLAPSED — matrix said 0.946)**; luna_high 0.912/$0.141;
terra_max 0.839/$0.421; terra_high 0.930/$0.770; sol_xhigh 0.939/$4.68;
sonnet5_high 0.899/$5.05; opus5_high 0.951/$6.33; fable5_xhigh 0.943/$14.36.

**Findings.**
1. The June matrix is not just economically stale — it is QUALITY-stale. luna_max fell
   0.946 -> 0.687; effort ladders INVERTED live (luna high >> max; terra high >> max).
   Any router (or static pick) built on June data that selects luna_max is broken today.
2. The live static frontier is stark: luna_high 0.912 at $0.14 vs opus 0.951 at $6.33 —
   a 45x cost gap for 3.9pp of quality. Where a deployment sits on that line is a real
   product decision, and it moved in four weeks.
3. Whether LIVE routing headroom exists (quality or cost, vs the live static frontier)
   is UNANSWERABLE at 1 trial/cell — the 1-trial naive oracle (0.999) is pure winner's
   curse and is reported only as a ceiling artifact. A >=2-trial pass on the relevant
   arms is required for a held-out analysis (~$3-4k; decision pending).

**Reframed value proposition.** Static-arm choice equals routing on a FROZEN matrix —
but nothing is frozen: models drift in quality, cost, and effort-response monthly. The
defensible product story after EXP-018 + EXP-015 is CONTINUOUS live benchmarking +
adaptive arm selection (the router as drift-tracker), not "router beats statics on a
snapshot." The Pier/Modal harness makes the refresh loop cheap and provenance-clean.

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

**Result (21 eval tasks, all with the same 4 forced haiku@budget turns):**
| variant | graded | $/task |
|---|---|---|
| trained, turn0-frozen (one decision post-prefix) | 0.949 | $0.038 |
| trained, per-turn | 0.956 | $0.046 |
| static nano@high post-prefix | 0.988 | $0.052 |
| always-escalate to opus | 0.968 | $0.236 |
| never escalate (continue weak) | 0.920 | $0.149 |
| ladder | 0.994 | $0.088 |

Accidental but valuable: `static-opus` and `escalate-opus` were the IDENTICAL policy
run twice live — they differ by 0.027 graded and 1.7x cost. That is the measured
noise floor for single live samples at n=21; differences under ~0.03 are not real.

**Verdicts against pre-registered criteria.**
1. Escalation PAYS: never-escalate is dominated by everything (worst quality AND
   among the most expensive — a weak model burning turns is not cheap).
2. DECIDING pays: trained policies hit ~0.95 at $0.04-0.05/task — 5-6x cheaper than
   always-escalate at within-noise quality.
3. PER-TURN GRANULARITY STILL DOES NOT: per-turn vs turn0-frozen is +0.007 graded for
   +20% cost — inside the measured noise floor. Two experiments (EXP-010, EXP-011)
   now agree: on LCB the value is one good (re)decision, not continuous re-decision.

**Lane disposition.** Per-turn granularity on LCB: answered (no measurable value,
even with injected difficulty). The remaining per-turn question requires a live
long-horizon domain (DeepSWE Docker harness — a separate build decision). The trained
turn0 policy remains the deployable form of the live-lane router.

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

## EXP-019 — source-transfer per-turn routing on all DeepSWE (2026-07-31)

**Motivation.** DeepSWE must be evaluation-only. Fit the router on a separate,
correlated coding-trace dataset, then test cache-aware per-turn routing on every
DeepSWE v1.1 task. The target is to beat a matched always-Luna-max baseline while
clearing the 2% savings gate.

**Setup.** Fit on 1,424 graded coding trajectories from
`nebius/SWE-rebench free graded coding trajectories`, with 5-fold grouping by source
repository. Frozen `te3-large` embeddings, kNN `k=64`, and two arms only:
`gpt-5.6-luna@low` and `gpt-5.6-luna@max`. DeepSWE outcomes and costs were not used
for fitting. Evaluate all 113 DeepSWE tasks online with cache/prefill-aware per-turn
routing on Azure, using the same direct evaluator for a fresh always-Luna-max control.

**Result.**

| system | graded quality | cost | $/task | comparison |
|---|---:|---:|---:|---|
| source-trained per-turn router | 0.8672 (112/113 scored; conservative all-task lower bound 0.8595) | $120.80 | $1.069 | 4.93% cheaper than Luna max |
| always Luna max, matched live control | 0.8126 (112/113 scored; range 0.8054-0.8142) | $127.06 | $1.124 | baseline |
| always Opus max, published DeepSWE matrix | 0.9429 | $1,355.46 | $11.995 | 91.1% more expensive than router |

The router won 30 tasks, tied 60, and lost 22 among the 112 paired scored tasks.
It used 117 low-effort turns and 6,194 max-effort turns across 6,311 turns, with 50
switches. The `pwntools-tube-multiplexing` verifier timed out in both live runs and
is reported as unscored, not dropped. A fresh full Opus run was not launched because
the published all-task cost already exceeds the $1,000 cap.

**Verdict.** PASS for the stated gates in this matched live run: the source-trained
per-turn router beats the fresh Luna-max control by 5.46 graded points on scored
tasks, remains ahead by at least 4.53 points under the conservative missing-task
bound, and saves 4.93% cost. It is not Opus-quality: it is 7.57 graded points below
the published Opus-max matrix result, while costing 91.1% less. DeepSWE remains
evaluation-only.

Artifacts: `results/source_transfer_perturn_luna_only_20260731.json`,
`/private/tmp/deepswe-source-transfer-luna-only-full-20260801/summary.json`, and
`/private/tmp/deepswe-luna-max-direct-full-20260801/summary.json`.

---

## Notion mirror (2026-07-31)

Parent "DeepSWE Router" page recalibrated (status header + standing rules; sibling's
held-out-oracle analysis preserved as canonical). Full experiment log mirrored to child
page "RL routing lane - experiment log (EXP-001-018)":
https://app.notion.com/p/3af0f8b3f591812598a6efa3fb9e66fe
Per-experiment updates continue there per the standing goal.

## EXP-012 final table (2026-08-01)

All 15 configs, seeds 0-5, vs static-select control (0.929/$3.28/6.8Mtok/847s):
top = reward_lcb_b0.2 0.948/$2.64 (+0.019 [-0.011,+0.060]); best-balanced =
grpo_dswe_b0.2 0.938/$1.43/3.0Mtok/490s (+0.009, 2.3x); cost-max = reward_dswe
0.914-0.923/$1.26-1.29 (2.5-2.6x). 13/15 configs at/above control quality point.
Static fable_5_xhigh (named baseline: 0.936/$13.07/7.2Mtok/1409s) dominated by every
config on all four axes. Routers cut tokens 40-55% and latency 30-45% vs control.
Full table mirrored to the Notion lane page. Fresh confirmations (5 winner configs,
seeds 6-11) + EXP-013 slate-GRPO (8 held-out arms) launched; reward_lcb_b0.2 to be
appended to the confirmation queue; SRB masking fix pending.

## EXP-017 — trajectory-prefix information test (2026-08-01) — NEGATIVE, decisive

**Question (Kion's ill-posedness hypothesis).** Task difficulty may depend on early
trajectory state, not the task text ("run airbnb is easier in a folder containing
airbnb"). If so, routing AFTER K reconnaissance steps should predict outcomes better.

**Setup.** 1,126 live DeepSWE trajectories from EXP-015 (median 42 steps). Build text
prefixes K=0..5 (task text; task + first K assistant actions/outputs, system prompt and
boilerplate stripped — also the serve-shaped input per the production degeneracy note),
embed with Qwen3-Embedding-0.6B (max_seq 4096), pooled ridge with arm one-hots,
leave-one-REPO-out. Metric: out-of-fold R^2 predicting final f2p. $0 (data already paid).

**Result.** K=0 +0.2038 | K=1 +0.2010 | K=2 +0.2004 | K=3 +0.1986 | K=5 +0.1859.
Flat then declining: early steps carry no incremental outcome signal and eventually
dilute the task-text signal.

**SCOPE CORRECTION (2026-08-01).** This result covers the first ~5 steps ONLY. A
follow-up depth-bin run (EXP-017b) was invalid: prefixes were head-truncated at 12,000
chars while trajectories are a median 911,000 chars (100% exceed the cap), so all bins
past ~20% embedded the identical opening fragment — the flat curve there was an
artifact, not evidence. EXP-017c re-runs it conditioning on the most-recent 10k-char
window at each depth (what a router deciding there would see); deep-trajectory signal
is UNRESOLVED until it lands.

**Verdict (early steps only).** Route-after-reconnaissance is NOT justified from the
first ~5 steps on DeepSWE; per-task routing is
the correct decision object. Explains the EXP-010/011 per-turn nulls information-
theoretically (nothing to learn by waiting) and matches the literature note that failure
evidence appears at 59-84% of trajectory depth. Flagship live run = per-task closed-loop
trainer, no watch-then-route detour. Caveat: pooled-linear probe of prefix embeddings;
engineered prefix features (explicit test-pass/error state) untested, but the monotone
decline is a strong prior against.

## EXP-017d/019 — MAJOR NEGATIVE: no task-conditional routing signal (2026-08-01)

Triggered by Kion: "R^2 is probably not capturing relationship." Correct — the earlier
probe was an ADDITIVE ridge (embedding + arm one-hots), which by construction cannot
represent the arm x task interaction routing depends on, and R^2 on bounded,
zero-inflated f2p measures mass-fitting not discrimination. Redone with per-arm models
(full interaction), AUC, within-task rank correlation, and a shuffled-label null.

**1. Arm discrimination — nothing.** Within-task Spearman across arms: task-text model
0.638 vs TASK-BLIND arm-base-rates-only 0.645. Reading the task adds nothing to
deciding which arm suits it. (Per-arm R^2: 0.332 vs task-blind 0.319.)

**2. Difficulty prediction — nothing.** Predicting per-task mean f2p from task text,
leave-one-repo-out: R^2 = -0.013 (worse than the mean), Spearman +0.135, n=113.

**3. Per-arm success prediction on LIVE data — nothing, vs a proper null.** 5-fold
grouped CV (leave-one-repo-out was invalid here: singleton repos make it leave-one-out,
whose base-rate shift is anti-correlated with the held-out label — that produced the
absurd AUC~0.00 readings). Real AUCs 0.292-0.488 vs shuffled-label controls
0.370-0.421: every arm at or below its own null.

**4. Trajectory depth — nothing, at any depth.** Per-arm AUC by prefix depth:
0% 0.744 | 20% 0.746 | 40% 0.746 | 60% 0.747 | 80% 0.747 (within-task rho 0.638-0.645
throughout). Flat. NB the 0.744 level is itself mostly arm-base-rate ranking, not task
prediction.

**5. Routers vs a PRICE-MATCHED static policy.** Sweeping the static-selection margin
traces a control curve; interpolating it at each router's price: grpo-dswe-anchored
+0.017, grpo-srb +0.014, reward-lcb +0.008, frozen +0.008, grpo-dswe -0.000,
reward-dswe -0.003. But the static curve is non-monotonic with ~±0.02 of its own noise
(margin 0.005 -> 0.922 while margin 0.02 -> 0.944), so these are NOT resolvable.

**RETRACTION.** The lane's headline "1.5-2.5x cost reduction at quality parity vs
honest static selection" is withdrawn. That control was pinned at an expensive
operating point (margin 0.01, $3.28/task); against a price-matched static policy the
advantage is inside the noise, and the information analyses above say there is no
task-conditional signal to have exploited in the first place. The apparent saving was
OPERATING-POINT selection (choose a cheaper arm), not routing.

**Independently replicates the sibling lane's Method C** ("three task representations
have failed... per-task arm preference appears environment/repo-specific, not inferable
from the problem statement") with two more representations (Qwen3-0.6B tuned/untuned),
on LIVE current-model data, and with a shuffled-label null they did not run.

**What survives.** (a) Live-harness infrastructure. (b) The drift finding: published
matrices go quality-stale in weeks. (c) Methodology: held-out oracles, price-matched
controls, shuffled-label nulls, the ~0.02 selection tax. (d) A robust negative result.
(e) Product implication: the defensible artifact is continuous re-benchmarking +
static arm selection at a chosen price point, NOT per-task routing.

**Do not spend further on per-task routing against DeepSWE.** A benchmark with
demonstrated task-conditional signal is a precondition for any resumption.

## EXP-020 — THE RETRACTION ABOVE IS ITSELF WITHDRAWN (2026-08-01)

Kion stopped the EXP-017d/019 retraction before it was published. Correctly: that
retraction was a FALSE NEGATIVE produced by the measurement instrument, not a finding.

**The decisive test.** Run the ACTUAL deployed policy (kNN vote over the train memory,
cheapest arm by utility) twice: once with real task embeddings, once with embeddings
PERMUTED across tasks — destroying the task<->outcome correspondence while preserving
every marginal (same arms, same base rates, same cost structure, same policy). If task
text carries no usable signal, the two must score the same.

| lam | real | shuffled twin | signal |
|---|---|---|---|
| 0.005 | 0.942 @ $2.90 | 0.932 @ $3.34 | +0.010 +/- 0.008 |
| 0.020 | 0.948 @ $1.84 | 0.928 @ $1.87 | +0.020 +/- 0.008 |
| 0.050 | 0.934 @ $1.36 | 0.916 @ $1.33 | +0.018 +/- 0.007 |

Paired repo-clustered bootstrap at lam=0.02 (30 shuffles): **graded +0.0233,
95% CI [+0.0009, +0.0500], cost delta -$0.06 (matched), 98.1% of draws positive.**

**Task-conditional routing signal EXISTS on DeepSWE.** It is small (~2 graded points at
matched cost) and it is INVISIBLE TO LINEAR PROBES: ridge/logistic on embeddings, with
or without arm interaction, scored at the shuffled-label null (EXP-017d/019), while the
nonparametric kNN over the same embeddings extracts it. The relationship is local, not
linear — neighbourhood structure, not a direction in embedding space.

**Consequences.**
1. The "1.5-2.5x at quality parity" headline is REINSTATED at the lower end: +0.023
   graded at matched cost converts to roughly 1.5-2x cheaper at matched quality against
   the static-policy curve (that curve's own +-0.02 noise is why the price-matched
   comparison could not resolve it, and why I misread it as absence).
2. EXP-017's per-turn/prefix negative STANDS — it was a within-instrument comparison
   (same probe at every depth), so the flatness is informative even though the absolute
   level was probe-limited. Deep-trajectory signal remains untested by a kNN-class
   method.
3. CROSS-LANE: the sibling lane's Method C negative ("three task representations have
   failed; the feature is worse than ignoring the task") used PCA + linear/MLP per-arm
   predictors — the same instrument class that just produced a false negative here.
   That conclusion should be re-tested with a nonparametric predictor before it is
   treated as settled. Their held-out-oracle QUALITY-ceiling result is unaffected (it
   is model-free) and still stands.

**Methodology rule adopted:** a negative result about signal must be demonstrated with
the SAME model class that would exploit it, and against a shuffled-input control of the
real system — never with a linear proxy alone.

## EXP-012 fresh-seed confirmations + anchor verdict (2026-08-01)

Seeds 6-11, never used for any selection. Control = static-select margin 0.01:
0.933 graded, $3.84/task.

| config | graded | $/task | dq vs control (CI) | cheaper |
|---|---|---|---|---|
| **grpo_dswe anchored b0.2** | **0.931** | **1.38** | **-0.002 [-0.027, +0.029]** | **2.79x** |
| grpo_dswe unanchored | 0.901 | 1.13 | -0.032 [-0.060, -0.002] | 3.39x |
| frozen temps | 0.942 | 3.18 | +0.010 [-0.015, +0.040] | 1.21x |
| grpo srb | 0.939 | 3.20 | +0.007 [-0.017, +0.037] | 1.20x |
| reward srb | 0.938 | 3.18 | +0.005 [-0.016, +0.031] | 1.21x |
| reward lcb anchored | 0.939 | 2.61 | +0.007 [-0.018, +0.037] | 1.47x |

**HEADLINE (fresh-confirmed): anchored GRPO gives 2.79x cost reduction at statistical
parity** (-0.002, CI spans zero) after paying the selection tax (+0.009 -> -0.002, i.e.
0.011, consistent with the ~0.02 estimate).

**Anchor verdict.** Pooled 12 seeds, paired on identical cells: anchored minus
unanchored = +0.0258 graded, 95% CI [-0.0037, +0.0621], 95.5% of repo-clustered
bootstrap draws positive, at +$0.197/task. Two independent batches agree (+0.021 on
0-5, +0.030 on 6-11) and the mechanism is measured (anchor flattens the holdout-decay
curve). Verdict: PROBABLY REAL, not conclusive at 95%. Unanchored GRPO is the only
config provably worse than control on fresh seeds.

Note the srb/lcb/frozen configs land at ~1.2x here vs 1.8x on selection seeds — their
lam/checkpoint choices transferred to a more conservative operating point. Anchored
GRPO is the config whose cost advantage held up.

## EXP-022 — prefix depth, re-tested with the DEPLOYED policy class (2026-08-01)

EXP-017's depth negative used linear probes — the instrument class that produced a
false negative in EXP-020. Re-ran with kNN over prefix embeddings (one embedding per
task, averaged over that task's runs), each depth against its own shuffled-INPUT
control, lam=0.05, 15 shuffles, 5-fold repo-grouped.

| depth | real | shuffled twin | signal |
|---|---|---|---|
| 0% | 0.924 @ $0.33 | 0.919 | +0.005 +/- 0.006 |
| 20% | 0.928 @ $0.35 | 0.921 | +0.007 +/- 0.009 |
| 40% | 0.931 @ $0.35 | 0.919 | +0.012 +/- 0.012 |
| 60% | 0.930 @ $0.35 | 0.920 | +0.011 +/- 0.009 |
| 80% | 0.931 @ $0.35 | 0.920 | +0.011 +/- 0.011 |

**Correction to EXP-017.** "Flat at every depth" was instrument-limited. With kNN the
signal roughly doubles from task-only (+0.005) to mid-episode (+0.011) and absolute
quality rises +0.007 at matched cost. Trajectory information EXISTS.

**It still does not pay, on economics not statistics.** (1) All bins sit within ~1 SD
of each other. (2) The measurement GIVES the router the prefix for free; deployment
must buy it by running an arm 40-80% of the way through the episode, then either switch
(re-prefill + discarded work) or not. Paying ~40-80% of an episode for ~+0.007 graded
is not a trade that closes. This independently reproduces the cost-side verdict of the
two live LCB A/Bs (EXP-010, EXP-011): per-turn == turn0 quality at higher cost.

**Per-turn verdict (final, three methods agreeing).** Live A/B, injected-difficulty
A/B, and now an information analysis with the correct instrument: route ONCE from the
task text. Reopen only if switch cost collapses (same-provider, cache-preserving
handoff) or on a domain with far larger arm spread than DeepSWE.

## EXP-023 — anchor-strength sweep + slate variants (RUNNING, 2026-08-01)

GPUs were idle after the confirmation batch; refilled with the free (GPU-only) queue,
now using the parallel-per-GPU pattern (each job ~10GB of 94GB, 3 co-resident).

**GPU0 — anchor-strength sweep.** beta in {0.1, 0.35, 0.6} x grpo/dswe x seeds 0-11
(selection AND fresh in one pass, since fresh confirmation is required anyway). The
confirmed champion is beta=0.2 at 2.79x/parity; this locates the optimum and tests
whether the anchor effect is monotone (evidence it is mechanistic rather than lucky:
beta=0.05 did nothing, 0.2 worked).

**GPU1 — slate randomization.** (a) anchored slate (grpo/dswe/beta0.2, 8 held-out
arms), (b) slate on the srb source, (c) fresh-seed slate confirmation (seeds 6-11).
Tests whether arm-set-invariant training composes with the anchor, and whether the
zero-shot-new-arm property (EXP-013: unseen 8-arm pool routed at 0.920) holds on fresh
seeds.

**Bug fixed:** v6's slate-size floor was hardcoded at 8, impossible for the 4-arm srb
pool (ValueError: low >= high). Now adapts: lo = min(8, max(2, n_trainable//2)).

Gated on Kion (external spend, not GPU): live closed-loop GRPO (~$1.5-3k) and the
>=2-trial live pass (~$3-4k). After EXP-021 the 2-trial pass ranks first — single-trial
live cells are why live routing works at the cheap end but not the quality end.

## EXP-023 results — anchor dose-response settles the anchor question (2026-08-01)

grpo/dswe, FRESH seeds 6-11, vs static-select control (0.933, $3.84/task):

| beta | graded | $/task | dq vs control | cheaper |
|---|---|---|---|---|
| 0.0 | 0.901 | 1.13 | -0.032 [-0.060, -0.002] | 3.39x |
| 0.1 | 0.930 | 1.33 | -0.003 [-0.036, +0.031] | 2.88x |
| 0.2 | 0.931 | 1.38 | -0.002 [-0.027, +0.028] | 2.79x |
| 0.35 | 0.944 | 1.36 | +0.011 [-0.013, +0.040] | 2.82x |
| 0.6 | 0.936 | 1.88 | +0.003 [-0.020, +0.031] | 2.04x |

**Monotone rise to a peak at 0.35, then decline — the shape of a real regularization
parameter.** This settles the anchor question affirmatively; the earlier evidence was a
95.5%-positive bootstrap (CI marginally spanning zero), which a dose-response curve of
this shape is much harder to produce by chance. Unanchored is the only setting provably
worse than control.

**Selection discipline.** beta=0.35 looks like a new champion (0.944 at 2.82x, above
control on BOTH axes) but it was chosen by looking at seeds 6-11, which are therefore
no longer fresh for that choice. Confirmation launched on untouched seeds 12-17 for
beta in {0.25, 0.35, 0.45}; also testing whether the 0.35 optimum transfers to the srb
and lcb training sources (12 seeds each).

**Slate randomization (v6, 8 held-out arms).** full-pool 0.908 @ $1.12 (seeds 0-5),
0.905 @ $1.07 fresh (6-11) — the ~1pp in-pool robustness tax reproduces on fresh seeds.
Anchored slate 0.904 @ $1.11: the anchor does NOT compose with slate randomization
(both are regularizers; together they over-constrain). srb-source slate 0.937 @ $1.59
is the best slate variant. Zero-shot unseen-arm pool (8 arms never in any training
slate): 0.920 @ $1.61 — the new-model-as-data-update property holds.

## EXP-023 confirmation — beta=0.35's "+0.011" was seed luck; pooled result stands

Untouched seeds 12-17 vs control (0.936, $4.41): beta 0.25 -0.027, 0.35 -0.029,
0.45 -0.029 — all identical, none reproducing the +0.011 seen on seeds 6-11. The peak
was a seed artifact; within 0.1-0.6 the exact anchor value does not matter.
(Same on the lcb source: b0.2 0.944/$2.62 vs b0.35 0.943/$2.30, 12 seeds.)

**FINAL POOLED NUMBERS — the ones to quote:**

| config | seeds | graded | $/task | dq vs control (CI) | cheaper |
|---|---|---|---|---|---|
| **anchored GRPO b0.35** | **18** | **0.932** | **1.46** | **-0.001 [-0.024, +0.026]** | **2.63x** |
| anchored GRPO b0.2 | 12 | 0.935 | 1.41 | +0.004 [-0.024, +0.037] | 2.53x |
| UNanchored GRPO | 12 | 0.909 | 1.21 | -0.022 [-0.044, +0.001] | 2.94x |

Headline, over 18 independent repo-splits: **2.6x cheaper than the honest static-select
control at statistical parity (-0.001)**. The anchor contributes ~+0.025 graded
(anchored 0.932-0.935 vs unanchored 0.909) and is what converts a provably-worse
cheap router into a parity one — this is the settled version of the dose-response
finding; the specific beta within 0.1-0.6 is not identifiable at this n.

**Methodology note.** This is the third time a peak selected on one seed batch failed
to reproduce on the next (factor champion -0.003 -> -0.027; nca parity -> demoted;
beta=0.35 +0.011 -> -0.029). The ~0.02 selection tax is not a correction to apply
mentally, it is a hard rule: NO number gets promoted without a fresh-seed batch, and
pooled-over-all-seeds is the only quotable form.

## EXP-024 — continuous queue runner (2026-08-01)

Replaces fire-and-refill (which left idle gaps between batches) with a self-sustaining
runner on box 6: `/nvme/work/router-rl/queue_runner.sh` in tmux session `runner`. It
keeps 3 jobs per GPU alive at all times, popping the next config from `queue.txt` the
moment a slot frees, and appends LAUNCH/DONE lines to `queue_done.txt`. A Monitor tails
that file so each completed config emits an event -> aggregate -> journal + Notion entry
per experiment rather than in bulk.

**Queued grid (32 configs, all free/GPU-only, each = one 6-seed experiment):**
1. anchor(0.35) x {grpo, reward} x {dswe, srb, lcb} x both fresh seed batches (6-11 and
   12-17) — every cell ships with its own confirmation batch by construction, so the
   selection tax cannot bite again.
2. v4 algorithm family (rank, nca) UNDER the anchor — both were only ever run
   unanchored, and the anchor is now known to be worth ~+0.025.
3. slate randomization x {dswe(8 held-out), srb(1 held-out)} x {0, 0.35} on fresh
   batches — does arm-set-invariant training compose with the anchor at the settled
   value (it did not at 0.2).
4. factor head x rank {4, 8, 16} x anchor {0, 0.1} on seeds 12-17 — the CPU-cheap
   parametric baseline never got an anchored variant.

## EXP-025 — closed-loop per-turn router on live DeepSWE (2026-08-02)

Primary objective per Kion. The router picks a model for EVERY agent turn inside the
official Pier/mini-swe-agent scaffold on Modal; episode reward from the verifier + real
billed tokens; policy trains on its own rollouts. No offline matrix in the loop.

**Infrastructure.** `rl/router_proxy.py` = Modal app `coding-router-proxy`, an
OpenAI-compatible endpoint. Per call: rebuild state (turn index, cost so far, struggle
signals from recent tool output, previous arm) -> linear policy w/ exploration floor ->
forward to real provider -> log (state, action, probs, usage). Pier auto-allowlists the
host from OPENAI_API_BASE, so the integration is supported, not a hack; episode id
travels in the URL path since the agent cannot send custom headers.
`rl/perturn_live.py` (box 6) = the training driver.

**Two bugs found by running it live, both now fixed.**
1. mini-swe-agent DOES send function tools, and the gpt-5.6 family rejects tools +
   reasoning_effort on /v1/chat/completions ("use /v1/responses"). Before this was
   found, all 45 OpenAI calls failed and only the Anthropic arm worked - the episode
   looked like it ran (12 steps) but was really always-opus with 45 retries. Fix:
   Responses API + OpenAI-only phase-1 pool (Anthropic needs a Responses<->Messages
   translation, phase 2).
2. Stacked FastAPI route decorators silently did not register; replaced with a
   catch-all route tolerant of any base-URL shape litellm builds.

**First working episode:** 34 routed turns across 5 arms, task SOLVED (f2p 1.0).
**First training iteration (4 episodes):** f2p 1.000, $3.93/ep, reward +0.803, 79
routed turns spanning all 5 arms.

**Immediate finding - the switch cost is the whole game.** $3.93/episode against static
luna_high at $0.141 = 28x more expensive. Cache is per-model, so a near-random policy
that changes arm almost every turn forces a cold re-prefill of the full transcript every
time. Quality is fine; the economics are not, until the policy learns stickiness. The
prev-arm feature exists precisely for this (it was the largest lever on LCB). Whether it
can learn to switch only when switching pays IS the per-turn question.

**Run launched:** 8 iterations x 10 tasks x 2 rollouts, lam=0.05, explore 0.12,
repo-split train/eval, $2.5k cap. Pre-registered eval: greedy policy vs turn0-frozen
and static controls on held-out repos.
