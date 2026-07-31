"""EXP-005: low-rank factor head (arm embeddings) trained with exact expected reward.

Replaces the tabular soft-kNN vote with a parametric bilinear head on FROZEN
Qwen3-Embedding-0.6B embeddings: logit[a](x) = v_a . (P e_x) + b_a, p_hat = sigmoid.
Arms share structure through rank-r factors (EmbedLLM/IRT-style), b_a initialised at
the arm's train-fold base-rate log-odds. Same objective, masking, selection protocol
as train_softknn v1/v2; anchor = KL to the base-rate policy (beta from CLI).

Usage: python train_factor.py <seeds-csv> <rank> <beta> <outdir>
Reads /nvme/work/router-rl/{router_rl_payload.npz,router_rl_meta.json} and the cached
embeddings /nvme/work/router-emb/qwen3-emb-0.6b.json (no encoder forward needed).
"""
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

WORK = pathlib.Path("/nvme/work/router-rl")
LAMBDAS = (0.005, 0.01, 0.02, 0.05, 0.1)
LAMBDA_TRAIN = 0.02
STEPS, EVAL_EVERY, LR = 600, 20, 5e-3

seeds = [int(s) for s in sys.argv[1].split(",")]
rank = int(sys.argv[2])
beta = float(sys.argv[3])
outdir = pathlib.Path(sys.argv[4])
outdir.mkdir(parents=True, exist_ok=True)
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

d = np.load(WORK / "router_rl_payload.npz")
meta = json.loads((WORK / "router_rl_meta.json").read_text())
graded = torch.tensor(d["graded"], dtype=torch.float32, device=dev)
cost = torch.tensor(d["cost"], dtype=torch.float32, device=dev)
qids, groups = meta["qids"], meta["groups"]
A = graded.shape[0]

emb = json.loads(pathlib.Path("/nvme/work/router-emb/qwen3-emb-0.6b.json").read_text())
E = torch.tensor(np.stack([emb[f"dswe:{q}"] for q in qids]), dtype=torch.float32, device=dev)
E = F.normalize(E, dim=1)
D = E.shape[1]


def hard_eval(u, q_idx):
    picks = u.argmax(dim=1)
    g = graded[picks, q_idx].mean().item()
    c = cost[picks, q_idx].sum().item()
    return picks.tolist(), g, c


for seed in seeds:
    sp = meta["splits"][str(seed)]
    tr = torch.tensor(sp["tr"], device=dev)
    te = torch.tensor(sp["te"], device=dev)
    itr = torch.tensor(sp["inner_tr"], device=dev)
    ival = torch.tensor(sp["inner_val"], device=dev)
    vb = int(graded[:, itr].mean(dim=1).argmax().item())
    vb_g = graded[vb, ival].mean().item()
    vb_c = cost[vb, ival].sum().item()

    t0 = time.time()
    tag = f"s{seed}_factor_r{rank}_b{beta}"
    dest = outdir / f"result_{tag}.json"
    if dest.exists():
        print(f"skip {tag}", flush=True)
        continue

    base = graded[:, tr].mean(dim=1).clamp(1e-3, 1 - 1e-3)
    P = torch.nn.Parameter(torch.randn(rank, D, device=dev) * 0.02)
    V = torch.nn.Parameter(torch.randn(A, rank, device=dev) * 0.02)
    b = torch.nn.Parameter(torch.log(base / (1 - base)).clone())
    logtau = torch.nn.Parameter(torch.tensor(np.log(0.05), device=dev))
    opt = torch.optim.AdamW([P, V, b, logtau], lr=LR, weight_decay=1e-3)

    medc_tr = cost[:, tr].median(dim=1).values
    R = (graded[:, tr] - LAMBDA_TRAIN * cost[:, tr]).T
    pi_ref = F.softmax((base[None, :] - LAMBDA_TRAIN * medc_tr[None, :]) / 0.05, dim=1)

    evals, mlog = [], (outdir / f"metrics_{tag}.jsonl").open("w")
    for step in range(1, STEPS + 1):
        tau = logtau.exp().clamp(5e-3, 1.0)
        p_hat = torch.sigmoid(E[tr] @ P.T @ V.T + b[None, :])
        u = p_hat - LAMBDA_TRAIN * medc_tr[None, :]
        pi = F.softmax(u / tau, dim=1)
        loss = -(pi * R).sum(dim=1).mean()
        if beta > 0:
            logpi = F.log_softmax(u / tau, dim=1)
            loss = loss + beta * (pi * (logpi - (pi_ref + 1e-9).log())).sum(dim=1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % EVAL_EVERY == 0:
            with torch.no_grad():
                logits_all = E @ P.T @ V.T + b[None, :]
                ph_all = torch.sigmoid(logits_all)
                medc_i = cost[:, itr].median(dim=1).values
                for lam in LAMBDAS:
                    _, ig, ic = hard_eval(ph_all[ival] - lam * medc_i[None, :], ival)
                    hpicks, hg, hc = hard_eval(ph_all[te] - lam * medc_tr[None, :], te)
                    rec = {"step": step, "lam": lam, "loss": float(loss.detach()),
                           "inner_graded": ig, "inner_cost": ic,
                           "holdout_graded": hg, "holdout_cost": hc}
                    evals.append({**rec, "holdout_picks": hpicks})
                    mlog.write(json.dumps(rec) + "\n")
            mlog.flush()
    mlog.close()

    feas = [e for e in evals if e["inner_graded"] >= vb_g - 0.02 and e["inner_cost"] > 0]
    pick = (max(feas, key=lambda e: vb_c / e["inner_cost"]) if feas
            else max(evals, key=lambda e: e["inner_graded"] - LAMBDA_TRAIN * e["inner_cost"]))
    res = {"seed": seed, "mode": f"factor_r{rank}_b{beta}", "lam": pick["lam"],
           "step": pick["step"], "feasible": bool(feas),
           "inner_graded": pick["inner_graded"], "inner_ratio": vb_c / pick["inner_cost"],
           "vb_g": vb_g, "holdout_picks": pick["holdout_picks"], "te": sp["te"],
           "holdout_graded": pick["holdout_graded"], "holdout_cost": pick["holdout_cost"],
           "wall_s": round(time.time() - t0, 1)}
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    tmp.replace(dest)
    print(f"{tag}: step {pick['step']} lam {pick['lam']} feasible={bool(feas)} "
          f"holdout_g={pick['holdout_graded']:.3f} ({res['wall_s']}s)", flush=True)

print("ALL DONE", flush=True)
