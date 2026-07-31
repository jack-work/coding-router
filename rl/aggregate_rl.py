"""Aggregate soft-kNN RL results: pooled holdout tables per mode, vs always-best base."""
import json
import pathlib

import numpy as np

d = np.load("/tmp/router_rl_payload.npz")
meta = json.loads(pathlib.Path("/tmp/router_rl_meta.json").read_text())
graded, cost = d["graded"], d["cost"]
arms, splits = meta["arms"], meta["splits"]

res = {}
for f in pathlib.Path("/tmp/rl_results").glob("result_*.json"):
    r = json.loads(f.read_text())
    res[(r["mode"], r["seed"])] = r

for mode in ("frozen", "lora"):
    cells, base, ratios, lams, steps = [], [], [], [], []
    for seed in range(6):
        r = res[(mode, seed)]
        tr = np.array(splits[str(seed)]["tr"])
        te = np.array(r["te"])
        ba = int(graded[:, tr].mean(axis=1).argmax())
        cells += [(a, int(j)) for a, j in zip(r["holdout_picks"], te)]
        base += [(ba, int(j)) for j in te]
        pc = sum(cost[a, int(j)] for a, j in zip(r["holdout_picks"], te))
        bc = sum(cost[ba, int(j)] for j in te)
        ratios.append(bc / pc if pc else float("inf"))
        lams.append(r["lam"])
        steps.append(r["step"])
    n = len(cells)
    total = sum(cost[a, j] for a, j in cells)
    b_tot = sum(cost[a, j] for a, j in base)
    g = np.mean([graded[a, j] for a, j in cells])
    bg = np.mean([graded[a, j] for a, j in base])
    by_arm = {}
    for a, j in cells:
        by_arm.setdefault(a, []).append(j)
    print(f"\n=== softknn-rl | {mode} | pooled over 6 seeds ===")
    print(f"  {'arm':38s} {'% traffic':>10s} {'$/task':>8s} {'$ total':>8s} {'% spend':>8s} {'graded':>7s}")
    for a, js in sorted(by_arm.items(), key=lambda kv: -len(kv[1])):
        c = sum(cost[a, j] for j in js)
        print(f"  {arms[a][15:]:38s} {len(js)/n*100:9.1f}% {c/len(js):8.2f} {c:8.1f} "
              f"{c/total*100:7.1f}% {np.mean([graded[a,j] for j in js]):7.3f}")
    print(f"  {'total':38s} {'100%':>10s} {total/n:8.2f} {total:8.1f} {'100%':>8s} {g:7.3f}")
    print(f"  Baseline ${b_tot/n:.2f}/task, ${b_tot:.1f} total, graded {bg:.3f} -> ratio {b_tot/total:.2f}x, delta {g-bg:+.3f}")
    print(f"  per-seed ratio median {np.median(ratios):.2f} (min {min(ratios):.2f} max {max(ratios):.2f})")
    print(f"  chosen (lam, step) per seed: {list(zip(lams, steps))}")

print("\nbar: kNN frozen-encoder qwen3-emb-0.6b (lr-baseline) = 4.02x, delta -0.027")
