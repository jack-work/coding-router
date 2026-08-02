"""Closed-loop per-turn router training on LIVE DeepSWE. Runs on box 6.

One iteration:
  1. sample N train-repo tasks; launch M rollouts each as separate Pier jobs, every
     rollout carrying its own episode id in the proxy URL (/ep/<task>__it<i>r<m>/v1);
  2. Pier runs the official mini-swe-agent scaffold on Modal; every model call is routed
     per-turn by the Modal proxy, which logs (state, action, probs, usage);
  3. join Pier's verifier f2p to the proxy's decision log; reward = f2p - LAM * cost,
     cost computed from real per-arm token prices;
  4. REINFORCE with a per-task group baseline over the M rollouts (GRPO-style);
  5. push updated weights to the proxy and repeat.

Eval uses held-out repos with exploration disabled, against three controls run in the
same batch: turn0-frozen (the policy's first choice locked for the episode) and the two
static arms that bracket the live frontier.

  python perturn_live.py train --iters 8 --tasks 12 --rollouts 2
  python perturn_live.py eval --tag final
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import re
import subprocess
import sys
import time

import numpy as np

WORK = pathlib.Path("/nvme/work/deepswe-live")
TASKS = WORK / "deep-swe-main" / "tasks"
PIER = os.path.expanduser("~/.local/bin/pier")
BASE = "https://bespoke-ai-agents--coding-router-proxy-router.modal.run"
OUT = WORK / "perturn"
LAM = 0.05          # reward = f2p - LAM * $cost ; $1 of spend costs 0.05 graded
N_ARMS, FEATS = 5, 8
BUDGET_USD = 2500.0

# $/1M tokens (repo pricing table) for the phase-1 OpenAI arm pool
PRICE = {
    "luna_medium": (1.00, 6.00), "luna_high": (1.00, 6.00),
    "terra_high": (2.50, 15.00), "terra_max": (2.50, 15.00),
    "sol_xhigh": (5.00, 30.00),
}


def repo_of(task: str) -> str:
    m = re.match(r"([a-z0-9]+(?:-[a-z0-9]+)?)", task)
    return m.group(1) if m else task


def split_tasks(eval_frac=0.25, seed=0):
    all_t = sorted(p.name for p in TASKS.iterdir() if (p / "task.toml").exists())
    repos = sorted({repo_of(t) for t in all_t})
    rng = np.random.default_rng(seed)
    rng.shuffle(repos)
    ev = set(repos[: max(2, int(len(repos) * eval_frac))])
    return ([t for t in all_t if repo_of(t) not in ev],
            [t for t in all_t if repo_of(t) in ev])


def run_episode(task: str, ep: str) -> dict:
    """One Pier trial with its own proxy episode id. Returns f2p + wall time."""
    jobdir = OUT / "jobs"
    cmd = [PIER, "run", "-p", str(TASKS), "-i", task, "--agent", "mini-swe-agent",
           "--model", "openai/router", "--ak", "model_class=litellm_response",
           "--ae", f"OPENAI_API_BASE={BASE}/ep/{ep}/v1",
           "--ae", "OPENAI_API_KEY=routertoken",
           "--env", "modal", "-o", str(jobdir), "--job-name", ep,
           "-n", "1", "-q", "-y", "--max-retries", "1"]
    t0 = time.time()
    subprocess.run(cmd, capture_output=True, text=True, timeout=5400)
    f2p, steps = None, 0
    for rj in (jobdir / ep).glob("*/verifier/reward.json"):
        try:
            f2p = float(json.loads(rj.read_text()).get("f2p"))
        except Exception:
            pass
    for rs in (jobdir / ep).glob("*/result.json"):
        m = re.search(r'"n_agent_steps": ([0-9]+)', rs.read_text())
        if m:
            steps = int(m.group(1))
    return {"ep": ep, "task": task, "f2p": f2p, "steps": steps,
            "wall": round(time.time() - t0, 1)}


def fetch_decisions(prefix: str) -> dict:
    import modal
    return modal.Function.from_name("coding-router-proxy", "get_episodes").remote(prefix=prefix)


def push_policy(W, temp=1.0, explore=0.05):
    import modal
    return modal.Function.from_name("coding-router-proxy", "put_policy").remote(
        W=[list(map(float, r)) for r in W], temp=temp, explore=explore)


def episode_cost(turns: list[dict]) -> float:
    c = 0.0
    for t in turns:
        u = t.get("usage") or {}
        pin, pout = PRICE.get(t["arm_id"], (2.0, 12.0))
        c += (u.get("input_tokens", u.get("prompt_tokens", 0)) * pin
              + u.get("output_tokens", u.get("completion_tokens", 0)) * pout) / 1e6
    return c


def softmax(z, temp=1.0):
    e = np.exp((z - z.max()) / max(temp, 1e-3))
    return e / e.sum()


def cmd_train(args):
    OUT.mkdir(parents=True, exist_ok=True)
    train_tasks, eval_tasks = split_tasks()
    print(f"{len(train_tasks)} train tasks / {len(eval_tasks)} eval tasks (repo-split)",
          flush=True)
    W = np.zeros((N_ARMS, FEATS))
    W[1, 0] = 1.0                       # cold start: prefer luna_high
    push_policy(W, temp=1.0, explore=args.explore)
    spend = 0.0
    rng = np.random.default_rng(0)
    log = (OUT / "train_log.jsonl").open("a")

    for it in range(args.iters):
        batch = list(rng.choice(train_tasks, min(args.tasks, len(train_tasks)),
                                replace=False))
        jobs = [(t, f"{t}__it{it}r{m}") for t in batch for m in range(args.rollouts)]
        print(f"\n=== iter {it}: {len(jobs)} episodes ===", flush=True)
        recs = []
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_episode, t, ep): (t, ep) for t, ep in jobs}
            for f in cf.as_completed(futs):
                try:
                    recs.append(f.result())
                except Exception as e:  # noqa: BLE001 — infra fault = missing data
                    print(f"  DROPPED {futs[f][1]}: {str(e)[:120]}", flush=True)
        dec = fetch_decisions(prefix="")
        rows = []
        for r in recs:
            turns = dec.get(r["ep"], [])
            if r["f2p"] is None or not turns:
                print(f"  DROPPED {r['ep']}: f2p={r['f2p']} turns={len(turns)}", flush=True)
                continue
            cost = episode_cost(turns)
            spend += cost
            rows.append({**r, "cost": cost, "reward": r["f2p"] - LAM * cost,
                         "turns": turns})
        if not rows:
            print("  no usable rollouts this iteration", flush=True)
            continue

        # REINFORCE with per-task group baseline
        by_task: dict[str, list] = {}
        for r in rows:
            by_task.setdefault(r["task"], []).append(r)
        grad = np.zeros_like(W)
        n_dec = 0
        for _, grp in by_task.items():
            b = float(np.mean([g["reward"] for g in grp]))
            for r in grp:
                adv = r["reward"] - b
                if abs(adv) < 1e-9:
                    continue
                for t in r["turns"]:
                    x = np.array(t["x"], dtype=float)
                    p = softmax(W @ x)
                    onehot = np.zeros(N_ARMS)
                    onehot[t["arm"]] = 1.0
                    grad += adv * np.outer(onehot - p, x)
                    n_dec += 1
        if n_dec:
            W += args.lr * grad / n_dec
        push_policy(W, temp=1.0, explore=args.explore)

        from collections import Counter
        arm_use = Counter(t["arm_id"] for r in rows for t in r["turns"])
        stat = {"it": it, "n": len(rows), "f2p": float(np.mean([r["f2p"] for r in rows])),
                "cost": float(np.mean([r["cost"] for r in rows])),
                "reward": float(np.mean([r["reward"] for r in rows])),
                "spend_total": round(spend, 2), "arms": dict(arm_use),
                "W": W.tolist()}
        log.write(json.dumps(stat) + "\n")
        log.flush()
        print(f"== it{it}: f2p {stat['f2p']:.3f}  ${stat['cost']:.2f}/ep  "
              f"reward {stat['reward']:+.3f}  spend ${spend:.0f}  arms {dict(arm_use)}",
              flush=True)
        if spend > BUDGET_USD:
            print("BUDGET CAP reached — stopping", flush=True)
            break
    log.close()


def cmd_eval(args):
    """Trained policy (greedy) vs turn0-frozen and static controls, held-out repos."""
    _, eval_tasks = split_tasks()
    tasks = eval_tasks[: args.tasks]
    print(f"eval on {len(tasks)} held-out tasks", flush=True)
    variants = {"perturn": {"explore": 0.0, "freeze": False},
                "turn0": {"explore": 0.0, "freeze": True}}
    results = {}
    for name, cfg in variants.items():
        import modal
        modal.Function.from_name("coding-router-proxy", "put_policy")  # weights already set
        jobs = [(t, f"{t}__ev{args.tag}{name}") for t in tasks]
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            recs = [f.result() for f in
                    cf.as_completed([ex.submit(run_episode, t, ep) for t, ep in jobs])]
        dec = fetch_decisions(prefix="")
        ok = [r for r in recs if r["f2p"] is not None and dec.get(r["ep"])]
        g = float(np.mean([r["f2p"] for r in ok]))
        c = float(np.mean([episode_cost(dec[r["ep"]]) for r in ok]))
        results[name] = {"n": len(ok), "f2p": g, "cost": c}
        print(f"  {name:10s} n={len(ok)} f2p {g:.3f}  ${c:.2f}/task", flush=True)
    (OUT / f"eval_{args.tag}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--iters", type=int, default=8)
    t.add_argument("--tasks", type=int, default=12)
    t.add_argument("--rollouts", type=int, default=2)
    t.add_argument("--workers", type=int, default=24)
    t.add_argument("--lr", type=float, default=0.3)
    t.add_argument("--explore", type=float, default=0.12)
    t.set_defaults(func=cmd_train)
    e = sub.add_parser("eval")
    e.add_argument("--tag", default="final")
    e.add_argument("--tasks", type=int, default=20)
    e.add_argument("--workers", type=int, default=20)
    e.set_defaults(func=cmd_eval)
    ns = ap.parse_args()
    sys.exit(ns.func(ns))
