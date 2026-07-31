"""Routing-policy experiment CLIs: races, holdout evaluations, and arm probing.

Subcommands:
  holdout-deepswe  -- clean 80/20 repo-split holdout on DeepSWE (41-arm pool).
  exp1-holdout9    -- EXP 1 headline result: 9-arm pruned-frontier pool, 3 seeds.
  race-router      -- race every routing policy against the cascade on LiveCodeBench.
  race-deepswe     -- race routing policies on DeepSWE v1.1 (graded objective).
  probe-arms       -- pre-flight probe of every candidate (model, effort) arm.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import pathlib
import random
import sys
import time

import anthropic
import numpy as np
import openai
from tabulate import tabulate

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from router import datasets  # noqa: E402
from router import harness as sandbox  # noqa: E402
from router import router_core as route  # noqa: E402
from router.harness import load_env  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

# ============================================================ race-deepswe
DEEPSWE_EMB = ROOT / "results" / "deepswe_embeddings.json"


def build() -> route.Matrix:
    """Build the full 50-arm DeepSWE matrix, dropping tasks with any missing cell.

    Returns:
        The (arm x task) matrix. `resolved` is graded score >= 1.0 (exact solve).
    """
    d = datasets.load_deepswe()
    g = np.array(d["score"], dtype=float)
    c = np.array(d["cost"], dtype=float)
    # 2 score cells and 7 cost cells are missing. Silently propagating them made argmin
    # and argmax both return the same NaN arm and every ratio NaN. Drop the few affected
    # TASKS rather than the 6 affected ARMS: that keeps all 50 arms at ~96% of the matrix,
    # where dropping arms would cost 12% of it for the sake of 9 cells.
    bad = np.isnan(g).any(axis=0) | np.isnan(c).any(axis=0)
    if bad.any():
        print(f"dropping {int(bad.sum())} tasks with missing cells: "
              f"{[d['tasks'][j] for j in np.where(bad)[0]]}")
    ok = ~bad
    tasks = [q for q, k in zip(d["tasks"], ok) if k]
    g, c = g[:, ok], c[:, ok]
    assert not (np.isnan(g).any() or np.isnan(c).any()), "matrix still has NaN"
    return route.Matrix(
        arms=list(d["arms"]), qids=tasks,
        resolved=(g >= 1.0), graded=g, cost=c,
        difficulty=[str(d["difficulty"].get(q)) for q in tasks],
        group=[d["group"].get(q, q) for q in tasks])


def embed(m: route.Matrix, text: dict[str, str], *, cache_path: pathlib.Path | None = None,
          embed_model: str = "text-embedding-3-large", base_url: str | None = None) -> None:
    """Embed problem statements with the given model, cached on disk.

    Args:
        m: The matrix to attach embeddings to; mutates `m.emb` in place.
        text: Task id -> statement text, for ids in `m.qids` missing from cache.
        cache_path: Where to cache embeddings; defaults to `DEEPSWE_EMB`.
        embed_model: Embedding model name.
        base_url: If set, use this OpenAI-compatible base URL instead of OpenAI's
            (e.g. a local embedding server); auth is skipped in that case.
    """
    cache_path = cache_path or DEEPSWE_EMB
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [q for q in m.qids if q not in cache]
    if todo:
        cl = openai.OpenAI(api_key="not-needed" if base_url else None, base_url=base_url)
        for i in range(0, len(todo), 64):
            ch = todo[i:i + 64]
            out = cl.embeddings.create(model=embed_model,
                                       input=[(text.get(q) or q)[:8000] for q in ch])
            for q, e in zip(ch, out.data):
                cache[q] = e.embedding
        cache_path.write_text(json.dumps(cache))
    e = np.array([cache[q] for q in m.qids], dtype=float)
    m.emb = e / np.linalg.norm(e, axis=1, keepdims=True)


def deepswe_run(m: route.Matrix, policy, folds, pricey: int):
    """Run a policy under repo-grouped CV, tracking graded payoff, cost, and pricey-arm share.

    GRADED score (not binary resolve) is the objective: a policy's payoff on a task
    is the best graded score among the arms it actually invoked, and cost is the sum
    of those arms' costs. `hit` (share routed to the priciest arm) is a collapse
    diagnostic -- a policy that has silently degenerated into "always escalate" looks
    fine on accuracy alone and is useless.

    Args:
        m: The matrix to evaluate on.
        policy: A callable of `(matrix, train_indices, test_index) -> Plan`.
        folds: Problem-index groups from `route.grouped_folds`.
        pricey: Index of the priciest arm, for the collapse diagnostic.

    Returns:
        Per-task `(graded_payoff, cost, hit_pricey_arm)` arrays, all out-of-fold.
    """
    gr = np.zeros(m.n)
    co = np.zeros(m.n)
    hit = np.zeros(m.n, dtype=bool)
    for te in folds:
        tr = np.array([j for j in range(m.n) if j not in set(te.tolist())])
        for j in te:
            best = 0.0
            for i in policy(m, tr, int(j)):
                co[j] += m.cost[i, j]
                best = max(best, m.graded[i, j])
                if i == pricey:
                    hit[j] = True
                if m.resolved[i, j]:
                    break
            gr[j] = best
    return gr, co, hit


def cmd_race_deepswe() -> None:
    """CLI (`race-deepswe`): race routing policies on DeepSWE v1.1 under repo-grouped CV.

    Repo-grouped CV (91 repos over 113 tasks) is used because tasks from one repo
    share code, so a random split would leak neighbours into train.
    """
    load_env()
    m = build()
    d = datasets.load_deepswe()
    embed(m, d["text"])
    print(f"DeepSWE: {len(m.arms)} arms x {m.n} tasks, {len(set(m.group))} repos (CV groups)")

    gmean = m.graded.mean(axis=1)
    total = m.cost.sum(axis=1)
    best = int(np.argmax(gmean))
    cheap = int(np.argmin(total))
    pricey = int(np.argmax(total))
    print(f"  best graded : {m.arms[best]}  {gmean[best]:.3f}  ${total[best]:.0f}")
    print(f"  cheapest    : {m.arms[cheap]}  {gmean[cheap]:.3f}  ${total[cheap]:.0f}")
    print(f"  priciest    : {m.arms[pricey]}  ${total[pricey]:.0f}\n")

    folds = route.grouped_folds(m, n_folds=5)
    bg, bc, _ = deepswe_run(m, route.p_always(best), folds, pricey)
    B = bc.sum()

    pols = [("always-best", route.p_always(best)),
            ("always-cheapest", route.p_always(cheap)),
            ("random", route.p_random(0)),
            ("CASCADE cheap->best", route.p_cascade(cheap, best))]
    for tau in (0.5, 0.7, 0.9):
        pols.append((f"knn-threshold t={tau}", route.p_knn_threshold(12, tau)))
    for tau in (0.7, 0.9):
        pols.append((f"knn-two-sided t={tau}", route.p_knn_two_sided(12, tau)))
        pols.append((f"knn+cascade t={tau}", route.p_knn_cascade(12, tau)))

    rows, table = [], []
    for name, p in pols:
        g, c, h = deepswe_run(m, p, folds, pricey)
        r = B / c.sum() if c.sum() else float("inf")
        lo, hi = route.boot_ratio(bc, c, m.group)
        rows.append((name, g.mean(), c.sum(), r, lo, hi, h.mean()))
        table.append((name, f"{g.mean():.3f}", f"{g.mean()-bg.mean():+.3f}", f"{c.sum():.1f}",
                      f"{r:.2f}", f"[{lo:.2f},{hi:.2f}]", f"{h.mean()*100:.1f}%"))

    og, oc = route.oracle(m)
    # Oracle's own use of the priciest arm is the yardstick for collapse.
    o_pricey = np.mean([bool(m.resolved[pricey, j]
                             and m.cost[pricey, j] <= m.cost[np.where(m.resolved[:, j])[0], j].min())
                        for j in range(m.n)])
    table.append(("ORACLE (cheapest win)", f"{m.graded.max(axis=0).mean():.3f}", "",
                  f"{oc.sum():.1f}", f"{B/oc.sum():.2f}", "", f"{o_pricey*100:.1f}%"))
    print(tabulate(table, headers=["policy", "graded", "vs base", "cost$", "x cheap",
                                    "95% CI", "->pricey"], disable_numparse=True))

    # Give the cascade its strongest form before claiming kNN beats it. Picking the first
    # rung as argmin(cost) chose the worst arm in the pool (graded 0.370), which is a straw
    # man -- the earlier lesson was that this pool is NOT ordered by price, so the rung has
    # to be chosen on measured value. Sweep every arm as the first rung and report the best.
    print("\n  tuned cascade -- best first rung over all 50 arms:")
    tuned = []
    for i in range(len(m.arms)):
        if i == best:
            continue
        g, c, _ = deepswe_run(m, route.p_cascade(i, best), folds, pricey)
        tuned.append((B / c.sum(), g.mean(), m.arms[i]))
    tuned.sort(reverse=True)
    for r, gm, nm in tuned[:4]:
        print(f"    {nm:46s} {r:5.2f}x at graded {gm:.3f}")
    bestc = tuned[0]
    print(f"  -> BEST TUNED CASCADE: {bestc[2]} = {bestc[0]:.2f}x at graded {bestc[1]:.3f}")

    casc = next(r for r in rows if r[0].startswith("CASCADE"))
    knn = [r for r in rows if r[0].startswith("knn")]
    beat = [r for r in knn if r[1] >= casc[1] - 0.01 and r[3] > casc[3]]
    print(f"\n  cascade: {casc[3]:.2f}x cheaper at graded {casc[1]:.3f}")
    if beat:
        b = max(beat, key=lambda r: r[3])
        print(f"  kNN BEATS IT: {b[0]} -> {b[3]:.2f}x at graded {b[1]:.3f} "
              f"(CI [{b[4]:.2f},{b[5]:.2f}]) = {b[3]/casc[3]:.2f}x relative")
    else:
        print("  no kNN variant beats the cascade at matched graded score.")


# ============================================================ holdout-deepswe
# Clean 80/20 repo split on DeepSWE: split the 88 repos 80/20 (tasks follow their repo,
# so none spans both sides); inside the 80 ONLY, one train/val split picks (k, tau) --
# the holdout never informs the hyperparameters, which is the leak nested CV could not
# fully rule out; then evaluate once on the untouched 20 with a repo-clustered bootstrap
# CI. Repeated across several split seeds, since a single 80/20 at n~22 can be lucky and
# the spread across seeds is the honest picture of how much one split can be trusted.

GRID_HOLDOUT = [(k, t) for k in (6, 12, 20) for t in (0.3, 0.5, 0.7, 0.9)]


def sub_holdout(m: route.Matrix, idx: np.ndarray) -> route.Matrix:
    """Slice a matrix down to task indices `idx`, keeping all arms."""
    return route.Matrix(arms=m.arms, qids=[m.qids[j] for j in idx],
                        resolved=m.resolved[:, idx], graded=m.graded[:, idx],
                        cost=m.cost[:, idx], difficulty=[m.difficulty[j] for j in idx],
                        group=[m.group[j] for j in idx], emb=m.emb[idx])


def split_by_repo_holdout(m: route.Matrix, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split tasks into train/test by repo, so no repo spans both sides.

    Args:
        m: The matrix whose `group` (repo) labels define the split.
        frac: Fraction of repos assigned to train.
        seed: Shuffle seed, for reproducibility.

    Returns:
        A (train_indices, test_indices) pair, over task indices.
    """
    repos = sorted(set(m.group))
    random.Random(seed).shuffle(repos)
    n_tr = int(round(len(repos) * frac))
    tr_r = set(repos[:n_tr])
    tr = np.array([j for j in range(m.n) if m.group[j] in tr_r])
    te = np.array([j for j in range(m.n) if m.group[j] not in tr_r])
    return tr, te


def cmd_holdout_deepswe() -> None:
    """CLI (`holdout-deepswe`): clean 80/20 repo-split holdout on the 41-arm DeepSWE pool.

    See the module-section comment above for the split/hyperparameter-selection design.
    Reported across 6 split seeds so the spread (not just one lucky split) is visible.
    """
    load_env()
    full = build()
    d = datasets.load_deepswe()
    embed(full, d["text"])
    keep = [i for i, a in enumerate(full.arms)
            if any(t in a for t in ("gpt_5", "claude_", "codex"))]
    m = route.Matrix(arms=[full.arms[i] for i in keep], qids=full.qids,
                     resolved=full.resolved[keep], graded=full.graded[keep],
                     cost=full.cost[keep], difficulty=full.difficulty,
                     group=full.group, emb=full.emb)
    print(f"DeepSWE: {len(m.arms)} arms x {m.n} tasks x {len(set(m.group))} repos\n")

    ratios, deltas, table = [], [], []
    for seed in range(6):
        tr, te = split_by_repo_holdout(m, 0.8, seed)
        TR, TE = sub_holdout(m, tr), sub_holdout(m, te)

        # --- pick (k, tau) using ONLY the 80: one internal train/val split, no CV ---
        itr, ite = split_by_repo_holdout(TR, 0.75, 100 + seed)
        A, V = sub_holdout(TR, itr), sub_holdout(TR, ite)
        vbest = int(np.argmax(A.graded.mean(axis=1)))
        vb_c = sum(V.cost[vbest, j] for j in range(V.n))
        vb_g = V.graded[vbest].mean()
        cand = []
        for k, t in GRID_HOLDOUT:
            g = c = 0.0
            for j in range(V.n):
                plan = route.p_knn_threshold(k, t)(
                    route.Matrix(arms=A.arms, qids=A.qids + [V.qids[j]],
                                 resolved=np.c_[A.resolved, V.resolved[:, j]],
                                 graded=np.c_[A.graded, V.graded[:, j]],
                                 cost=np.c_[A.cost, V.cost[:, j]],
                                 difficulty=A.difficulty + [V.difficulty[j]],
                                 group=A.group + [V.group[j]],
                                 emb=np.vstack([A.emb, V.emb[j]])),
                    np.arange(A.n), A.n)
                i = plan[0]
                g += V.graded[i, j]
                c += V.cost[i, j]
            if c > 0 and g / V.n >= vb_g - 0.02:
                cand.append((vb_c / c, k, t))
        k, t = (max(cand)[1], max(cand)[2]) if cand else (12, 0.5)

        # --- evaluate ONCE on the untouched 20, table = the 80 ---
        gbest = int(np.argmax(TR.graded.mean(axis=1)))
        gr = np.zeros(TE.n)
        co = np.zeros(TE.n)
        bg = np.zeros(TE.n)
        bc = np.zeros(TE.n)
        for j in range(TE.n):
            merged = route.Matrix(arms=TR.arms, qids=TR.qids + [TE.qids[j]],
                                  resolved=np.c_[TR.resolved, TE.resolved[:, j]],
                                  graded=np.c_[TR.graded, TE.graded[:, j]],
                                  cost=np.c_[TR.cost, TE.cost[:, j]],
                                  difficulty=TR.difficulty + [TE.difficulty[j]],
                                  group=TR.group + [TE.group[j]],
                                  emb=np.vstack([TR.emb, TE.emb[j]]))
            i = route.p_knn_threshold(k, t)(merged, np.arange(TR.n), TR.n)[0]
            gr[j], co[j] = TE.graded[i, j], TE.cost[i, j]
            bg[j], bc[j] = TE.graded[gbest, j], TE.cost[gbest, j]
        lo, hi = route.boot_ratio(bc, co, TE.group)
        ratios.append(bc.sum() / co.sum())
        deltas.append(gr.mean() - bg.mean())
        table.append((seed, f"{TR.n}/{TE.n}", f"{len(set(TR.group))}/{len(set(TE.group))}",
                      f"({k},{t})", f"{bg.mean():.3f}", f"{gr.mean():.3f}",
                      f"{gr.mean()-bg.mean():+.3f}", f"{bc.sum()/co.sum():.2f}",
                      f"[{lo:.2f},{hi:.2f}]"))

    print(tabulate(table, headers=["seed", "tr/te tasks", "tr/te repos", "(k,tau)", "base",
                                    "knn", "delta", "x cheap", "95% CI"], disable_numparse=True))
    print(f"\n  across 6 splits: cost ratio median {np.median(ratios):.2f} "
          f"(min {min(ratios):.2f}, max {max(ratios):.2f})")
    print(f"                   graded delta median {np.median(deltas):+.3f} "
          f"(min {min(deltas):+.3f}, max {max(deltas):+.3f})")
    print(f"\n  For comparison, nested CV over all {m.n} tasks gave 2.15x, "
          f"delta -0.021, CI [1.87,2.46].")
    print("  The spread above is the cost of a single 20% holdout at this n.")


# ============================================================ exp1-holdout9
# EXP 1 -- headline result: 9-arm pool, DeepSWE 80/20 clean repo split, 3 seeds. The 9
# arms are the pruned frontier from the 41-arm race (the router selected only 13 of 41,
# and dropping the dominated ones improved the ratio), plus claude-fable-5@xhigh, which
# DeepSWE marks dominated but which leads coding elsewhere -- kept so a single benchmark
# does not get to decide the pool permanently.

NINE = ["gpt_5_6_terra_high", "gpt_5_6_luna_xhigh", "gpt_5_6_luna_max",
        "gpt_5_6_sol_medium", "gpt_5_6_sol_high", "claude_opus_5_low",
        "claude_opus_5_medium", "claude_opus_5_high", "claude_fable_5_xhigh"]
GRID_EXP1 = [(k, t) for k in (6, 12, 20) for t in (0.3, 0.5, 0.7, 0.9)]
EXP1_SEEDS = (0, 1, 2)


def sub_exp1(m, idx):
    """Slice a matrix down to task indices `idx`, keeping all arms."""
    return route.Matrix(arms=m.arms, qids=[m.qids[j] for j in idx], resolved=m.resolved[:, idx],
                        graded=m.graded[:, idx], cost=m.cost[:, idx],
                        difficulty=[m.difficulty[j] for j in idx],
                        group=[m.group[j] for j in idx], emb=m.emb[idx])


def split_by_repo_exp1(m, frac, seed):
    """Split tasks into train/test by repo, so no repo spans both sides."""
    repos = sorted(set(m.group))
    random.Random(seed).shuffle(repos)
    tr_r = set(repos[: int(round(len(repos) * frac))])
    return (np.array([j for j in range(m.n) if m.group[j] in tr_r]),
            np.array([j for j in range(m.n) if m.group[j] not in tr_r]))


def one_task(TR, q_res, q_grd, q_cost, q_diff, q_grp, q_emb, k, tau):
    """Route a single held-out task against the TR lookup table.

    Args:
        TR: The training-fold matrix, used as the kNN lookup table.
        q_res: The query task's per-arm resolved column.
        q_grd: The query task's per-arm graded-score column.
        q_cost: The query task's per-arm cost column.
        q_diff: The query task's difficulty label.
        q_grp: The query task's group (repo) label.
        q_emb: The query task's embedding.
        k: Number of nearest neighbours.
        tau: Minimum predicted P(resolve) to pick an arm.

    Returns:
        The chosen arm's index.
    """
    merged = route.Matrix(arms=TR.arms, qids=TR.qids + ["_q"], resolved=np.c_[TR.resolved, q_res],
                          graded=np.c_[TR.graded, q_grd], cost=np.c_[TR.cost, q_cost],
                          difficulty=TR.difficulty + [q_diff], group=TR.group + [q_grp],
                          emb=np.vstack([TR.emb, q_emb]))
    return route.p_knn_threshold(k, tau)(merged, np.arange(TR.n), TR.n)[0]


def cmd_exp1_holdout9() -> None:
    """CLI (`exp1-holdout9`): EXP1 headline result on the 9-arm pruned-frontier pool.

    Split by repo; (k, tau) chosen inside the 80 only, never touching the holdout;
    reported per seed and aggregated, since a single 20% holdout at ~22 tasks is not
    trustworthy alone -- the 41-arm version of this ranged 0.90x to 5.25x across
    seeds, and that spread is the honest error bar.
    """
    load_env()
    full = build()
    d = datasets.load_deepswe()
    embed(full, d["text"])
    idx = [i for i, a in enumerate(full.arms)
           if a.replace("mini_swe_agent_", "") in NINE]
    assert len(idx) == 9, f"expected 9 arms, resolved {len(idx)}: {[full.arms[i] for i in idx]}"
    m = route.Matrix(arms=[full.arms[i].replace("mini_swe_agent_", "") for i in idx],
                     qids=full.qids, resolved=full.resolved[idx], graded=full.graded[idx],
                     cost=full.cost[idx], difficulty=full.difficulty, group=full.group,
                     emb=full.emb)
    g = m.graded.mean(axis=1)
    tot = m.cost.sum(axis=1)
    print(f"EXP 1 -- 9 arms x {m.n} tasks x {len(set(m.group))} repos\n")
    for i in np.argsort(tot):
        print(f"  {m.arms[i]:24s} graded {g[i]:.3f}  ${tot[i]/m.n:5.2f}/task")

    rows, table = [], []
    for seed in EXP1_SEEDS:
        tr, te = split_by_repo_exp1(m, 0.8, seed)
        TR, TE = sub_exp1(m, tr), sub_exp1(m, te)
        # hyperparameters from inside the 80 only
        itr, ite = split_by_repo_exp1(TR, 0.75, 100 + seed)
        A, V = sub_exp1(TR, itr), sub_exp1(TR, ite)
        vb = int(np.argmax(A.graded.mean(axis=1)))
        vbg, vbc = V.graded[vb].mean(), V.cost[vb].sum()
        cand = []
        for k, t in GRID_EXP1:
            gs = cs = 0.0
            for j in range(V.n):
                i = one_task(A, V.resolved[:, j], V.graded[:, j], V.cost[:, j],
                             V.difficulty[j], V.group[j], V.emb[j], k, t)
                gs += V.graded[i, j]
                cs += V.cost[i, j]
            if cs > 0 and gs / V.n >= vbg - 0.02:
                cand.append((vbc / cs, k, t))
        k, t = (max(cand)[1], max(cand)[2]) if cand else (12, 0.5)
        # evaluate once on the untouched 20
        gb = int(np.argmax(TR.graded.mean(axis=1)))
        gr = np.zeros(TE.n)
        co = np.zeros(TE.n)
        for j in range(TE.n):
            i = one_task(TR, TE.resolved[:, j], TE.graded[:, j], TE.cost[:, j],
                         TE.difficulty[j], TE.group[j], TE.emb[j], k, t)
            gr[j], co[j] = TE.graded[i, j], TE.cost[i, j]
        bg, bc = TE.graded[gb], TE.cost[gb]
        lo, hi = route.boot_ratio(bc, co, TE.group)
        rows.append({"seed": seed, "k": k, "tau": t, "n_test": int(TE.n),
                     "base_graded": float(bg.mean()), "knn_graded": float(gr.mean()),
                     "delta": float(gr.mean() - bg.mean()),
                     "ratio": float(bc.sum() / co.sum()), "ci": [float(lo), float(hi)],
                     "baseline_arm": TR.arms[gb]})
        table.append((seed, f"{TR.n}/{TE.n}", f"{len(set(TR.group))}/{len(set(TE.group))}",
                      f"({k},{t})", f"{bg.mean():.3f}", f"{gr.mean():.3f}",
                      f"{gr.mean()-bg.mean():+.3f}", f"{bc.sum()/co.sum():.2f}",
                      f"[{lo:.2f},{hi:.2f}]"))

    print()
    print(tabulate(table, headers=["seed", "tr/te", "repos", "(k,tau)", "base", "knn",
                                    "delta", "x cheap", "95% CI"], disable_numparse=True))
    R = [r["ratio"] for r in rows]
    Dl = [r["delta"] for r in rows]
    print(f"\n  HEADLINE across {len(EXP1_SEEDS)} seeds:")
    print(f"    cost ratio   median {np.median(R):.2f}x   range {min(R):.2f}-{max(R):.2f}")
    print(f"    graded delta median {np.median(Dl):+.3f}   range {min(Dl):+.3f} to {max(Dl):+.3f}")
    (ROOT / "results" / "exp1_holdout9.json").write_text(json.dumps(
        {"arms": m.arms, "n_tasks": int(m.n), "n_repos": len(set(m.group)),
         "seeds": rows, "median_ratio": float(np.median(R)),
         "median_delta": float(np.median(Dl))}, indent=1))
    print("    -> results/exp1_holdout9.json")


# ============================================================ race-router
def cmd_race_router() -> None:
    """CLI (`race-router`): race every routing policy against the cascade on LiveCodeBench.

    Prints one table under contest-grouped CV. The decision it settles: does any
    learned policy beat the plain cascade at matched accuracy? If not, ship the cascade.
    """
    load_env()
    m = route.load_matrix()
    print(f"matrix: {len(m.arms)} arms x {m.n} problems, "
          f"{len(set(m.group))} contests (CV grouping unit)")

    acc = m.resolved.mean(axis=1)
    med = np.median(m.cost, axis=1)
    best = int(np.argmax(acc))
    cheap = int(np.argmin(med))
    print(f"  best arm   : {m.arms[best]}  {acc[best]*100:.1f}%")
    print(f"  cheapest   : {m.arms[cheap]}  {acc[cheap]*100:.1f}%")

    probs = {p.qid: p for p in sandbox.load()}
    route.attach_embeddings(m, {q: probs[q].statement for q in m.qids if q in probs})
    print(f"  embeddings : {m.emb.shape}")

    folds = route.grouped_folds(m, n_folds=5)
    print(f"  folds      : {[len(f) for f in folds]}\n")

    base_res, base_cost = route.run_policy(m, route.p_always(best), folds)
    B = base_cost.sum()

    policies = [
        ("always-best (baseline)", route.p_always(best)),
        ("always-cheapest", route.p_always(cheap)),
        ("random", route.p_random(0)),
        ("CASCADE cheap->best", route.p_cascade(cheap, best)),
    ]
    for tau in (0.5, 0.7, 0.9):
        policies.append((f"knn-threshold tau={tau}", route.p_knn_threshold(12, tau)))
    for tau in (0.5, 0.7, 0.9):
        policies.append((f"knn-two-sided tau={tau}", route.p_knn_two_sided(12, tau)))
    for tau in (0.7, 0.9):
        policies.append((f"knn+cascade tau={tau}", route.p_knn_cascade(12, tau)))

    rows, table = [], []
    for name, pol in policies:
        r, c = route.run_policy(m, pol, folds)
        ratio = B / c.sum() if c.sum() else float("inf")
        lo, hi = route.boot_ratio(base_cost, c, m.group)
        p = route.mcnemar(base_res, r)
        rows.append((name, r.mean(), c.sum(), ratio, lo, hi, p))
        table.append((name, f"{r.mean()*100:.1f}%", f"{(r.mean()-base_res.mean())*100:+.1f}",
                      f"{c.sum():.3f}", f"{ratio:.2f}", f"[{lo:.2f},{hi:.2f}]", f"{p:.3f}"))

    for label, kw in (("ORACLE", {}), ("PARITY ORACLE", {"parity_to": best})):
        r, c = route.oracle(m, **kw)
        table.append((label, f"{r.mean()*100:.1f}%", f"{(r.mean()-base_res.mean())*100:+.1f}",
                      f"{c.sum():.3f}", f"{B/c.sum():.2f}", "", "(ceiling)"))

    print(tabulate(table, headers=["policy", "acc", "vs base", "cost$", "x cheaper",
                                    "95% CI", "McNemar"], disable_numparse=True))

    casc = next(x for x in rows if x[0].startswith("CASCADE"))
    learned = [x for x in rows if x[0].startswith("knn")]
    # "Matched accuracy" = accuracy not detectably worse than the cascade's.
    ok = [x for x in learned if x[1] >= casc[1] - 0.02 and x[3] > casc[3]]
    print(f"\n  cascade: {casc[3]:.2f}x at {casc[1]*100:.1f}%")
    if ok:
        b = max(ok, key=lambda x: x[3])
        print(f"  BEST LEARNED THAT BEATS IT: {b[0]} -> {b[3]:.2f}x at {b[1]*100:.1f}% "
              f"(CI [{b[4]:.2f},{b[5]:.2f}])")
        print(f"  relative gain over cascade: {b[3]/casc[3]:.2f}x")
    else:
        print("  NO learned policy beats the cascade at matched accuracy on this data.")
        print("  -> Per the stated decision rule, the cascade is what ships.")


# ============================================================ probe-arms
# A prompt that should force exactly one tool call — verifies agentic capability.
PROBE_TASK = ("Use the run_bash tool to list the Python files in the current directory. "
              "Call the tool, do not explain.")
PROBE_TOOL_DESC = "Run a bash command in the repo and return its stdout."
PROBE_SCHEMA = {
    "type": "object",
    "properties": {"command": {"type": "string", "description": "bash command to run"}},
    "required": ["command"],
}

ANTHROPIC_EFFORTS = ["low", "medium", "high", "xhigh", "max"]
OPENAI_EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh"]

# (model, effort|None, thinking_mode) — thinking_mode: "adaptive" | "budget" | "omit" | "off"
ANTHROPIC_ARMS = (
    [("claude-haiku-4-5", None, "off"), ("claude-haiku-4-5", None, "budget"),
     ("claude-haiku-4-5", "high", "off")]  # expect effort to ERROR on haiku
    + [("claude-sonnet-5", e, "adaptive") for e in ANTHROPIC_EFFORTS]
    + [("claude-opus-4-8", e, "adaptive") for e in ANTHROPIC_EFFORTS]
    + [("claude-fable-5", e, "omit") for e in ANTHROPIC_EFFORTS]
)

OPENAI_MODELS = ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4", "gpt-5.5",
                 "gpt-5.3-codex", "gpt-5.6-sol", "gpt-5.1-codex-mini"]
OPENAI_ARMS = [(m, e) for m in OPENAI_MODELS for e in OPENAI_EFFORTS]


def probe_anthropic(model: str, effort: str | None, thinking: str) -> dict:
    """Send one probe request to the Anthropic Messages API and summarize the result.

    Diagnostic dict, shape genuinely varies from `probe_openai`'s (different fields
    per provider) -- see `probe_run`.

    Args:
        model: Anthropic model id.
        effort: `output_config.effort` value, or None/empty to omit it.
        thinking: "adaptive"|"budget"|"off" -- selects the `thinking` request block.

    Returns:
        A summary dict: ok, latency, stop reason, whether a tool was called, token usage.
    """
    cl = anthropic.Anthropic(max_retries=1, timeout=180.0)
    kw: dict = {
        "model": model,
        "max_tokens": 3000,
        "tools": [{"name": "run_bash", "description": PROBE_TOOL_DESC, "input_schema": PROBE_SCHEMA}],
        "messages": [{"role": "user", "content": PROBE_TASK}],
    }
    if thinking == "adaptive":
        kw["thinking"] = {"type": "adaptive"}
    elif thinking == "budget":
        kw["thinking"] = {"type": "enabled", "budget_tokens": 2048}
    elif thinking == "off":
        kw["thinking"] = {"type": "disabled"}
    if effort:
        kw["output_config"] = {"effort": effort}

    t0 = time.time()
    r = cl.messages.create(**kw)
    u = r.usage
    return {
        "ok": True,
        "latency_s": round(time.time() - t0, 2),
        "stop_reason": r.stop_reason,
        "tool_called": any(b.type == "tool_use" for b in r.content),
        "block_types": sorted({b.type for b in r.content}),
        "in_tok": u.input_tokens,
        "out_tok": u.output_tokens,
        "cache_read": getattr(u, "cache_read_input_tokens", None),
    }


def probe_openai(model: str, effort: str) -> dict:
    """Send one probe request to the OpenAI Responses API and summarize the result.

    Args:
        model: OpenAI model id.
        effort: `reasoning.effort` value.

    Returns:
        A summary dict: ok, latency, status, whether a tool was called, token usage.
    """
    cl = openai.OpenAI(max_retries=1, timeout=180.0)
    t0 = time.time()
    r = cl.responses.create(
        model=model,
        input=PROBE_TASK,
        reasoning={"effort": effort},
        max_output_tokens=3000,
        tools=[{"type": "function", "name": "run_bash", "description": PROBE_TOOL_DESC,
                "parameters": PROBE_SCHEMA}],
    )
    types = sorted({getattr(o, "type", "?") for o in r.output})
    u = r.usage
    return {
        "ok": True,
        "latency_s": round(time.time() - t0, 2),
        "status": r.status,
        "tool_called": any(getattr(o, "type", "") == "function_call" for o in r.output),
        "block_types": types,
        "in_tok": u.input_tokens,
        "out_tok": u.output_tokens,
        "reasoning_tok": getattr(u.output_tokens_details, "reasoning_tokens", None),
    }


def probe_run(job):
    """Run one probe job (either provider), catching and recording any failure.

    Args:
        job: A (provider_kind, args) pair, where `provider_kind` is "anthropic" or
            "openai" and `args` are positional args for the matching probe function.

    Returns:
        A (label, result_dict) pair; `result_dict["ok"]` is False on any exception.
    """
    kind, args = job
    label = f"{kind}:{':'.join(str(a) for a in args)}"
    try:
        res = probe_anthropic(*args) if kind == "anthropic" else probe_openai(*args)
    except Exception as e:  # noqa: BLE001 — probing is exactly about capturing failures
        res = {"ok": False, "error_type": type(e).__name__, "error": str(e)[:400]}
    return label, res


def cmd_probe_arms() -> None:
    """CLI (`probe-arms`): pre-flight probe every candidate (model, effort) arm.

    Confirms each combo is accepted by the live API and can emit a tool call, before
    any budget is committed to a sweep. Writes results/arm_probe.json.
    """
    load_env()
    jobs = [("anthropic", a) for a in ANTHROPIC_ARMS] + [("openai", a) for a in OPENAI_ARMS]
    out: dict[str, dict] = {}
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        for label, res in ex.map(probe_run, jobs):
            out[label] = res
            flag = "OK " if res["ok"] else "ERR"
            extra = (f"tool={res['tool_called']} out={res['out_tok']}tok {res['latency_s']}s"
                     if res["ok"] else f"{res['error_type']}: {res['error'][:150]}")
            print(f"{flag} {label:44s} {extra}", flush=True)

    dest = ROOT / "results" / "arm_probe.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    n_ok = sum(1 for v in out.values() if v["ok"])
    n_tool = sum(1 for v in out.values() if v["ok"] and v.get("tool_called"))
    print(f"\n{n_ok}/{len(out)} arms accepted; {n_tool} emitted a tool call -> {dest}")


if __name__ == "__main__":
    import fire

    fire.Fire({
        "holdout-deepswe": cmd_holdout_deepswe,
        "exp1-holdout9": cmd_exp1_holdout9,
        "race-router": cmd_race_router,
        "race-deepswe": cmd_race_deepswe,
        "probe-arms": cmd_probe_arms,
    })
