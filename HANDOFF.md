# Coding-Router RL Program — Collaborator Handoff (2026-07-31)

## Mission & standing directives (from Kion)
Build an RL-trained router over (model x reasoning-effort) arms for coding agents.
- PER-TURN routing is the primary objective (route every agent turn, mid-episode switching);
  per-task routers are baselines. So far per-turn has NOT beaten per-task on LiveCodeBench
  (2 experiments agree) — the open question is the long-horizon DeepSWE regime.
- ONE canonical eval: DeepSWE v1.1 holdout (repo-grouped 80/20, seeds 0-5 select, 6-11
  confirm, 12-17 spare). Training datasets vary (that's the generalization axis).
- Quality-first: <1 graded point worse than baseline, even at only 2x cost savings.
  Baselines to report: always-best-train policy (beware: winner's curse — it's WEAK),
  static claude_fable_5_xhigh (Kion's named primary), static sol_xhigh / opus_5_high /
  luna_max, and the per-task oracle. Measure tokens/task and seconds/task.
- Target: router runs locally on a user's machine (favors Qwen3-Embedding-0.6B or smaller).
- Budgets approved: Modal <$10k, model APIs $20k. Both providers at highest rate limits —
  default to hundreds of concurrent episodes. Spend so far: ~$65 APIs, ~$2 Modal.
- Report every result batch as: Method | What it does | $/task | Cheaper | Quality D (CI) |
  Tradeoffs — baselines in-table, bold = fresh-seed confirmed, * = selection seeds only.

## Where things live
- Worktree ../coding-router-lrbase, branch exp/lr-baseline-embeddings (local only — the
  auto-mode classifier blocks git push; ask Kion to push or add an allow rule).
  EXPERIMENTS.md = the journal (EXP-001..016, every verdict + corrections). Mirror to the
  Notion research page (router section) when a Notion-connected session exists — none here.
- Parallel worktree ../coding-router-perturn (branch exp/per-turn-routing, PAUSED): on_turn
  hook + divergence-cost probe. Its findings: DeepSWE-trained router is OOD on LCB; mid-episode
  switches pay cold-prefill.
- Box 6 = azureuser@40.80.93.150 (ours; 2xH100). /nvme is EPHEMERAL (everything mirrored to
  the branch). Training scripts: /nvme/work/router-rl/ (train_softknn v1/v2/v4/v5/v6,
  train_factor.py, payloads); live harness: /nvme/work/deepswe-live/ (Pier, tasks, keys.env).
  Embeddings cache: results/emb/*.json (gitignored, reproducible).
- Memory files (Claude): coding-router-rl-lane, coding-router-perturn-lane,
  results-table-format, feedback_experiment_branching.

## Key results (all DeepSWE per-task unless noted; deltas vs always-best-train policy)
- Frontier (fresh-confirmed): frozen soft-kNN temps 3.87x/-0.018 (only confirmed-parity);
  factor head r4 5.33x/-0.027 (balanced champion; trains in seconds on CPU; 3-arm policy);
  NCA 5.01x/-0.036; GRPO-LoRA 7.12x/-0.030 (borderline parity); rank-LoRA 8.36x/-0.052.
- Selection tax: quality-selected variants lose ~0.02 graded on fresh seeds. Apply mentally.
- LoRA overfits monotonically at n=88 train tasks; KL-anchor beta=0.2 tames it (beta=0.05
  doesn't); anchoring != worth 100x the compute of the factor head.
- EXP-012 (RUNNING, ~75/90): {frozen,reward,grpo} x {dswe,lcb,srb} x beta{0,0.2}, quality-
  first selection (feasibility -0.01, lambda down to 0.001). PARTIAL headline: GRPO trained
  on SWE-rebench transfers BETTER than in-domain (0.946/+0.025/2.62x vs dswe 0.910/-0.010,
  4 seeds) — quality-only signal + 16x data beats overfit. Known flaw: v5 treats SRB's 45%
  missing cells as failures (fix = mask; can only improve it).
- Per-turn on LCB (live, our E2B harness, ~$60): trained sticky policy ~0.95 at 11-18x under
  always-opus, but turn0-frozen (per-task) equals/beats per-turn (EXP-010 + EXP-011, incl.
  4-turn weak-arm handicap). Live noise floor ~0.03 graded at n=21. Escalation pays; deciding
  pays; granularity doesn't (there).
- luna_max audit (subagent workflow): parity with every single frontier arm at $3.17/task,
  but NOT token-frugal (2-3.7x more tokens), flakiest pass@1, loses -0.042 to the per-task
  oracle (0.987). Losses concentrated in JS/TS (-0.061 vs opus); wins Go. => language-aware
  2-line router is a mandatory baseline. CORRECTION in journal: "luna beat always-opus" was
  a baseline misattribution; static opus 0.963 > luna 0.944 on pooled cells.
- Zero-shot embedding comparison: Qwen3-Emb-0.6B ~ te3-large (kNN 4.02x vs 3.94x);
  LFM2.5-Encoder-350M weak zero-shot (2.40x; MLM, no contrastive training) — queued for
  fine-tune track (mean pooling, all-linear LoRA targets).

## Live DeepSWE infra (EXP-014/015) — the big unlock
- DeepSWE tasks are Harbor-format; official harness = Pier (uv tool install datacurve-pier).
  pier run -p deep-swe-main/tasks --agent mini-swe-agent --model <litellm id WITH DOTS,
  e.g. openai/gpt-5.6-luna> --ak reasoning_effort=<effort> --env modal. CPU-only.
  Trial output schema == published trials.json (cost_usd, tokens, steps, reward.json f2p).
- ZERO-STEP TRAP: bad model id -> 0 agent steps -> verifier scores unmodified repo (partial
  ~0.13, looks legit). n_agent_steps==0 is MISSING DATA; halt and alert. Canary (all arms x
  2 tasks) before every batch.
- DRIFT: live gpt-5.6-luna@max solved a task at $0.017/262k tok/16 steps that the June matrix
  records at $1.57/7.6M tok/~100 steps (quality replicated: 0.95 vs 1.0). Effort dose-response
  verified (low: f2p 0.50 vs max: 0.95). => matrix economics are STALE; live rewards must use
  live costs; relative orderings need live re-verification.
- EXP-015 (RUNNING): fresh live matrix, 10 arms x 113 tasks x 1 trial on Modal (~500
  concurrent). Arms: luna low/med/high/max, terra high/max, sol xhigh, opus5/sonnet5 high,
  fable5 xhigh (all canary-verified incl. anthropic/claude-opus-5). Watcher = job-level
  result.json finished_at (it's written at START and updated live — don't count files).
- Pier job dirs: jobs-full/<arm>/<task>__<id>/{result.json, verifier/reward.json, agent/...}.

## Queued / open decisions
- EXP-013 (built, queued): slate-randomized GRPO (train_softknn_v6.py) + zero-shot arm-holdout
  eval — tests "new arms without retraining".
- EXP-016 (proposed): self-label SWE-rebench V2 (7,243 unlabeled Python tasks, per-instance
  docker images) with OUR arms via cheap live episodes => large on-arm training corpus.
- Per-turn on DeepSWE: router-proxy (mini-swe-agent -> OpenAI-compatible endpoint that picks
  the arm per request). The decisive granularity experiment. Not started.
- Boxes 7/8 (Azure, ~$15/hr total): offered to parallelize GPU queue; awaiting Kion.
- E2B fallback for Harbor tasks: validated by research (cloud-side template builds, public
  ECR pulls, docker-in-sandbox) — use if Modal becomes a constraint.

## Integrity conventions (non-negotiable)
Repo-grouped everything; selection ONLY on inner splits of train; pre-register acceptance
before confirmations; fresh seeds for any promotion; infra failures are missing data, never
zeros; don't read mid-wave results (stragglers are hard tasks); episode caches are keyed by
tag — stamp policy identity into tags; state the noise floor next to any live comparison.
