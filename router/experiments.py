"""Routing-policy experiment CLIs: races, holdout evaluations, and arm probing.

Subcommands:
  holdout-deepswe  -- clean 80/20 repo-split holdout on DeepSWE (41-arm pool).
  exp1-holdout9    -- EXP 1 headline result: 9-arm pruned-frontier pool, 3 seeds.
  race-router      -- race every routing policy against the cascade on LiveCodeBench.
  race-deepswe     -- race routing policies on DeepSWE v1.1 (graded objective).
  probe-arms       -- pre-flight probe of every candidate (model, effort) arm.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import random
import sys
import time

import anthropic
import numpy as np
import openai

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from router import datasets  # noqa: E402
from router import harness as sandbox  # noqa: E402
from router import router_core as route  # noqa: E402
from router.harness import load_env  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

# ============================================================ race-deepswe
"""Race routing policies on DeepSWE v1.1 -- the first non-saturated set we have.

Differences from the LiveCodeBench race, all deliberate:
  * GRADED score is the objective, not binary. A policy's payoff on a task is the best
    graded score among the arms it actually invoked; cost is the sum of those arms.
  * Repo-GROUPED CV (91 repos over 113 tasks), because tasks from one repo share code and
    a random split would leak neighbours into train.
  * A COLLAPSE diagnostic is reported beside accuracy: what share of traffic a policy sends
    to the priciest arm, versus what the oracle sends there. A router that has silently
    degenerated into "always escalate" looks fine on accuracy and is useless.
"""

DEEPSWE_EMB = ROOT / "results" / "deepswe_embeddings.json"


def build() -> route.Matrix:
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


def embed(m: route.Matrix, text: dict[str, str]) -> None:
    cache = json.loads(DEEPSWE_EMB.read_text()) if DEEPSWE_EMB.exists() else {}
    todo = [q for q in m.qids if q not in cache]
    if todo:
        cl = openai.OpenAI()
        for i in range(0, len(todo), 64):
            ch = todo[i:i + 64]
            out = cl.embeddings.create(model="text-embedding-3-large",
                                       input=[(text.get(q) or q)[:8000] for q in ch])
            for q, e in zip(ch, out.data):
                cache[q] = e.embedding
        DEEPSWE_EMB.write_text(json.dumps(cache))
    e = np.array([cache[q] for q in m.qids], dtype=float)
    m.emb = e / np.linalg.norm(e, axis=1, keepdims=True)


def deepswe_run(m: route.Matrix, policy, folds, pricey: int):
    """Graded payoff + cost + share routed to the priciest arm, all out-of-fold."""
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


def cmd_race_deepswe(args: argparse.Namespace) -> None:
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

    print(f"  {'policy':24s} {'graded':>7s} {'vs base':>8s} {'cost$':>8s} {'x cheap':>8s} "
          f"{'95% CI':>14s} {'->pricey':>9s}")
    rows = []
    for name, p in pols:
        g, c, h = deepswe_run(m, p, folds, pricey)
        r = B / c.sum() if c.sum() else float("inf")
        lo, hi = route.boot_ratio(bc, c, m.group)
        rows.append((name, g.mean(), c.sum(), r, lo, hi, h.mean()))
        print(f"  {name:24s} {g.mean():7.3f} {g.mean()-bg.mean():+8.3f} {c.sum():8.1f} "
              f"{r:8.2f} [{lo:5.2f},{hi:5.2f}] {h.mean()*100:8.1f}%")

    og, oc = route.oracle(m)
    # Oracle's own use of the priciest arm is the yardstick for collapse.
    o_pricey = np.mean([bool(m.resolved[pricey, j]
                             and m.cost[pricey, j] <= m.cost[np.where(m.resolved[:, j])[0], j].min())
                        for j in range(m.n)])
    print(f"  {'ORACLE (cheapest win)':24s} {m.graded.max(axis=0).mean():7.3f} "
          f"{'':8s} {oc.sum():8.1f} {B/oc.sum():8.2f} {'':14s} {o_pricey*100:8.1f}%")

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
"""Clean 80/20 repo split on DeepSWE: no CV on the 80, holdout touched once.

Design, deliberately simple so the provenance is legible:
  1. Split the 88 repos 80/20. Tasks follow their repo, so no repo spans both sides.
  2. Inside the 80 ONLY, a single train/val split picks (k, tau). The holdout never
     informs the hyperparameters -- that is the leak nested CV could not fully rule out.
  3. Evaluate once on the 20. Report a repo-clustered bootstrap CI.

Also reports across several split seeds, because a single 80/20 at n~22 can be lucky and
the spread across seeds is the honest picture of how much one split can be trusted.
"""

GRID_HOLDOUT = [(k, t) for k in (6, 12, 20) for t in (0.3, 0.5, 0.7, 0.9)]


def sub_holdout(m: route.Matrix, idx: np.ndarray) -> route.Matrix:
    return route.Matrix(arms=m.arms, qids=[m.qids[j] for j in idx],
                        resolved=m.resolved[:, idx], graded=m.graded[:, idx],
                        cost=m.cost[:, idx], difficulty=[m.difficulty[j] for j in idx],
                        group=[m.group[j] for j in idx], emb=m.emb[idx])


def split_by_repo_holdout(m: route.Matrix, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    repos = sorted(set(m.group))
    random.Random(seed).shuffle(repos)
    n_tr = int(round(len(repos) * frac))
    tr_r = set(repos[:n_tr])
    tr = np.array([j for j in range(m.n) if m.group[j] in tr_r])
    te = np.array([j for j in range(m.n) if m.group[j] not in tr_r])
    return tr, te


def cmd_holdout_deepswe(args: argparse.Namespace) -> None:
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

    print(f"  {'seed':>4s} {'tr/te tasks':>12s} {'tr/te repos':>12s} {'(k,tau)':>10s} "
          f"{'base':>6s} {'knn':>6s} {'delta':>7s} {'x cheap':>8s} {'95% CI':>14s}")
    ratios, deltas = [], []
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
                    route.Matrix(A.arms, A.qids + [V.qids[j]], np.c_[A.resolved, V.resolved[:, j]],
                                 np.c_[A.graded, V.graded[:, j]], np.c_[A.cost, V.cost[:, j]],
                                 A.difficulty + [V.difficulty[j]], A.group + [V.group[j]],
                                 np.vstack([A.emb, V.emb[j]])),
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
            merged = route.Matrix(TR.arms, TR.qids + [TE.qids[j]],
                                  np.c_[TR.resolved, TE.resolved[:, j]],
                                  np.c_[TR.graded, TE.graded[:, j]],
                                  np.c_[TR.cost, TE.cost[:, j]],
                                  TR.difficulty + [TE.difficulty[j]],
                                  TR.group + [TE.group[j]], np.vstack([TR.emb, TE.emb[j]]))
            i = route.p_knn_threshold(k, t)(merged, np.arange(TR.n), TR.n)[0]
            gr[j], co[j] = TE.graded[i, j], TE.cost[i, j]
            bg[j], bc[j] = TE.graded[gbest, j], TE.cost[gbest, j]
        lo, hi = route.boot_ratio(bc, co, TE.group)
        ratios.append(bc.sum() / co.sum())
        deltas.append(gr.mean() - bg.mean())
        print(f"  {seed:4d} {f'{TR.n}/{TE.n}':>12s} "
              f"{f'{len(set(TR.group))}/{len(set(TE.group))}':>12s} {f'({k},{t})':>10s} "
              f"{bg.mean():6.3f} {gr.mean():6.3f} {gr.mean()-bg.mean():+7.3f} "
              f"{bc.sum()/co.sum():8.2f} [{lo:5.2f},{hi:5.2f}]")

    print(f"\n  across 6 splits: cost ratio median {np.median(ratios):.2f} "
          f"(min {min(ratios):.2f}, max {max(ratios):.2f})")
    print(f"                   graded delta median {np.median(deltas):+.3f} "
          f"(min {min(deltas):+.3f}, max {max(deltas):+.3f})")
    print(f"\n  For comparison, nested CV over all {m.n} tasks gave 2.15x, "
          f"delta -0.021, CI [1.87,2.46].")
    print("  The spread above is the cost of a single 20% holdout at this n.")


# ============================================================ exp1-holdout9
"""EXP 1 -- headline result: 9-arm pool, DeepSWE 80/20 clean repo split, 3 seeds.

The 9 arms are the pruned frontier from the 41-arm race (the router selected only 13 of 41,
and dropping the dominated ones improved the ratio), plus claude-fable-5@xhigh, which DeepSWE
marks dominated but which leads coding elsewhere -- kept so a single benchmark does not get to
decide the pool permanently.

Split is by REPO, so no repo appears on both sides. (k, tau) are chosen inside the 80 via one
internal repo split, never touching the holdout. Reported per seed AND aggregated, because a
single 20% holdout at ~22 tasks is not trustworthy alone -- the 41-arm version of this ranged
0.90x to 5.25x across seeds, and that spread is the honest error bar.
"""

NINE = ["gpt_5_6_terra_high", "gpt_5_6_luna_xhigh", "gpt_5_6_luna_max",
        "gpt_5_6_sol_medium", "gpt_5_6_sol_high", "claude_opus_5_low",
        "claude_opus_5_medium", "claude_opus_5_high", "claude_fable_5_xhigh"]
GRID_EXP1 = [(k, t) for k in (6, 12, 20) for t in (0.3, 0.5, 0.7, 0.9)]
EXP1_SEEDS = (0, 1, 2)


def sub_exp1(m, idx):
    return route.Matrix(arms=m.arms, qids=[m.qids[j] for j in idx], resolved=m.resolved[:, idx],
                        graded=m.graded[:, idx], cost=m.cost[:, idx],
                        difficulty=[m.difficulty[j] for j in idx],
                        group=[m.group[j] for j in idx], emb=m.emb[idx])


def split_by_repo_exp1(m, frac, seed):
    repos = sorted(set(m.group))
    random.Random(seed).shuffle(repos)
    tr_r = set(repos[: int(round(len(repos) * frac))])
    return (np.array([j for j in range(m.n) if m.group[j] in tr_r]),
            np.array([j for j in range(m.n) if m.group[j] not in tr_r]))


def one_task(TR, q_res, q_grd, q_cost, q_diff, q_grp, q_emb, k, tau):
    """Route a single held-out task against the TR lookup table."""
    merged = route.Matrix(TR.arms, TR.qids + ["_q"], np.c_[TR.resolved, q_res],
                          np.c_[TR.graded, q_grd], np.c_[TR.cost, q_cost],
                          TR.difficulty + [q_diff], TR.group + [q_grp],
                          np.vstack([TR.emb, q_emb]))
    return route.p_knn_threshold(k, tau)(merged, np.arange(TR.n), TR.n)[0]


def cmd_exp1_holdout9(args: argparse.Namespace) -> None:
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

    rows = []
    print(f"\n  {'seed':>4s} {'tr/te':>9s} {'repos':>8s} {'(k,tau)':>9s} {'base':>6s} "
          f"{'knn':>6s} {'delta':>7s} {'x cheap':>8s} {'95% CI':>13s}")
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
        print(f"  {seed:4d} {f'{TR.n}/{TE.n}':>9s} {f'{len(set(TR.group))}/{len(set(TE.group))}':>8s} "
              f"{f'({k},{t})':>9s} {bg.mean():6.3f} {gr.mean():6.3f} "
              f"{gr.mean()-bg.mean():+7.3f} {bc.sum()/co.sum():8.2f} [{lo:4.2f},{hi:4.2f}]")

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
"""Race every routing policy against the cascade under contest-grouped CV.

Prints one table. The decision it settles: does any learned policy beat the plain cascade at
matched accuracy? If not, ship the cascade.
"""


def cmd_race_router(args: argparse.Namespace) -> None:
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

    print(f"  {'policy':26s} {'acc':>7s} {'vs base':>8s} {'cost$':>8s} {'x cheaper':>10s} "
          f"{'95% CI':>16s} {'McNemar':>8s}")
    rows = []
    for name, pol in policies:
        r, c = route.run_policy(m, pol, folds)
        ratio = B / c.sum() if c.sum() else float("inf")
        lo, hi = route.boot_ratio(base_cost, c, m.group)
        p = route.mcnemar(base_res, r)
        rows.append((name, r.mean(), c.sum(), ratio, lo, hi, p))
        print(f"  {name:26s} {r.mean()*100:6.1f}% {(r.mean()-base_res.mean())*100:+7.1f} "
              f"{c.sum():8.3f} {ratio:10.2f} [{lo:5.2f},{hi:5.2f}] {p:8.3f}")

    for label, kw in (("ORACLE", {}), ("PARITY ORACLE", {"parity_to": best})):
        r, c = route.oracle(m, **kw)
        print(f"  {label:26s} {r.mean()*100:6.1f}% {(r.mean()-base_res.mean())*100:+7.1f} "
              f"{c.sum():8.3f} {B/c.sum():10.2f}   (ceiling)")

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
"""Probe every candidate routing arm with a real tool-calling request.

Cheap pre-flight: confirms each (model, effort) combo is accepted by the live API
and can emit a tool call, before any budget is committed to the sweep.
Writes results/arm_probe.json.
"""

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
    kind, args = job
    label = f"{kind}:{':'.join(str(a) for a in args)}"
    try:
        res = probe_anthropic(*args) if kind == "anthropic" else probe_openai(*args)
    except Exception as e:  # noqa: BLE001 — probing is exactly about capturing failures
        res = {"ok": False, "error_type": type(e).__name__, "error": str(e)[:400]}
    return label, res


def cmd_probe_arms(args: argparse.Namespace) -> None:
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = ap.add_subparsers(dest="command", required=True)

    subparsers.add_parser("holdout-deepswe").set_defaults(func=cmd_holdout_deepswe)
    subparsers.add_parser("exp1-holdout9").set_defaults(func=cmd_exp1_holdout9)
    subparsers.add_parser("race-router").set_defaults(func=cmd_race_router)
    subparsers.add_parser("race-deepswe").set_defaults(func=cmd_race_deepswe)
    subparsers.add_parser("probe-arms").set_defaults(func=cmd_probe_arms)

    ns = ap.parse_args()
    ns.func(ns)
