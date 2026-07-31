"""Offline RL on the tabular (soft-kNN) router: exact expected-reward gradients.

Policy: p_hat[a](x) = sum_j softmax(sim(x, x_j)/T)_j * graded[a, j]  (memory = train fold)
        u[a] = p_hat[a] - lam * med_train_cost[a];  train dist pi = softmax(u / tau)
Objective (full-information offline RL, no sampling): maximize
        J = mean_x sum_a pi(a|x) * (graded[a, x] - lam * cost[a, x])
with the query's own REPO masked out of memory (the training-time analogue of
repo-grouped CV). Eval is the HARD rule argmax u, memory = the full 80% split.
Checkpoint + lam selection use ONLY the inner 75/25 repo split, same feasibility
rule as lr-baseline: max cost ratio s.t. graded >= inner-best-arm - 0.02.

Usage: python train_softknn.py <seeds-csv e.g. 0,1,2> <mode: lora|frozen> <outdir>
Reads /nvme/work/router-rl/{router_rl_payload.npz,router_rl_meta.json,texts.json}.
Writes <outdir>/result_s{seed}_{mode}_lam{lam}.json atomically + metrics.jsonl.
"""
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

WORK = pathlib.Path("/nvme/work/router-rl")
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
LAMBDAS = (0.005, 0.01, 0.02, 0.05, 0.1)
STEPS, EVAL_EVERY = 300, 10
LORA_LR, TEMP_LR = 5e-5, 1e-2
MAX_LEN = 1536  # p50 doc is ~500 tokens; 2048 with retained activations OOMs a 94GB H100

LAMBDA_TRAIN = 0.02  # geometry is trained once; lambda is swept in the DECISION RULE at eval
seeds = [int(s) for s in sys.argv[1].split(",")]
mode = sys.argv[2]
outdir = pathlib.Path(sys.argv[3])
outdir.mkdir(parents=True, exist_ok=True)
dev = "cuda:0"
torch.manual_seed(0)

d = np.load(WORK / "router_rl_payload.npz")
meta = json.loads((WORK / "router_rl_meta.json").read_text())
graded = torch.tensor(d["graded"], dtype=torch.float32, device=dev)   # (A, N)
cost = torch.tensor(d["cost"], dtype=torch.float32, device=dev)       # (A, N)
qids, groups = meta["qids"], meta["groups"]
texts = json.loads((WORK / "texts.json").read_text())
docs = [texts[f"dswe:{q}"] for q in qids]
grp_code = torch.tensor([sorted(set(groups)).index(g) for g in groups], device=dev)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
batch = tok(docs, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
ids_all = batch["input_ids"].to(dev)
mask_all = batch["attention_mask"].to(dev)
print(f"tokenised {len(docs)} docs, padded len {ids_all.shape[1]}", flush=True)


def load_model():
    m = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16,
                                  attn_implementation="sdpa").to(dev)
    if mode == "lora":
        from peft import LoraConfig, get_peft_model
        m.gradient_checkpointing_enable()
        m.enable_input_require_grads()  # required for checkpointing + frozen embeddings
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        m = get_peft_model(m, cfg)
        m.train()  # checkpointing is a no-op in eval mode -> silent OOM
    return m


def encode(model, idx, grad):
    """Last-token-pooled, L2-normalised embeddings for doc indices idx."""
    outs = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(idx), 16):
            sl = idx[i:i + 16]
            h = model(input_ids=ids_all[sl], attention_mask=mask_all[sl],
                      use_cache=False).last_hidden_state
            last = mask_all[sl].sum(1) - 1
            outs.append(h[torch.arange(len(sl), device=dev), last])
    e = torch.cat(outs).float()
    return F.normalize(e, dim=1)


def routing_terms(E_q, E_m, mem_idx, T, lam, repo_mask_rows=None):
    """p_hat (Q,A), utility (Q,A) for queries vs memory mem_idx."""
    sims = E_q @ E_m.T / T
    if repo_mask_rows is not None:  # (Q,) query repo codes -> mask same-repo memory
        same = repo_mask_rows[:, None] == grp_code[mem_idx][None, :]
        sims = sims.masked_fill(same, float("-inf"))
    w = F.softmax(sims, dim=1)                                   # (Q, M)
    p_hat = w @ graded[:, mem_idx].T                             # (Q, A)
    medc = cost[:, mem_idx].median(dim=1).values                 # (A,)
    return p_hat, p_hat - lam * medc[None, :]


def hard_eval(E_q, E_m, mem_idx, q_idx, T, lam):
    with torch.no_grad():
        _, u = routing_terms(E_q, E_m, mem_idx, T, lam)
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
    # inner-best arm (on inner_tr) -> feasibility yardstick on inner_val
    vb = int(graded[:, itr].mean(dim=1).argmax().item())
    vb_g = graded[vb, ival].mean().item()
    vb_c = cost[vb, ival].sum().item()

    t0 = time.time()
    tag = f"s{seed}_{mode}"
    dest = outdir / f"result_{tag}.json"
    if dest.exists():
        print(f"skip {tag} (exists)", flush=True)
        continue
    model = load_model()
    params = [{"params": [p for p in model.parameters() if p.requires_grad],
               "lr": LORA_LR}] if mode == "lora" else []
    logT = torch.tensor(np.log(0.05), device=dev, requires_grad=True)
    logtau = torch.tensor(np.log(0.05), device=dev, requires_grad=True)
    opt = torch.optim.AdamW(params + [{"params": [logT, logtau], "lr": TEMP_LR}])

    if mode == "frozen":
        E_all = encode(model, torch.arange(len(qids), device=dev), grad=False)
    evals, mlog = [], (outdir / f"metrics_{tag}.jsonl").open("w")
    for step in range(1, STEPS + 1):
        if mode == "lora":
            E_all = torch.zeros(len(qids), 1024, device=dev)
            E_tr = encode(model, tr, grad=True)
            E_all = E_all.index_put((tr,), E_tr)
        T, tau = logT.exp().clamp(5e-3, 1.0), logtau.exp().clamp(5e-3, 1.0)
        E_trv = E_all[tr]
        p_hat, u = routing_terms(E_trv, E_trv, tr, T, LAMBDA_TRAIN,
                                 repo_mask_rows=grp_code[tr])
        pi = F.softmax(u / tau, dim=1)                            # (Ntr, A)
        R = (graded[:, tr] - LAMBDA_TRAIN * cost[:, tr]).T        # (Ntr, A)
        loss = -(pi * R).sum(dim=1).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g_ in opt.param_groups for p in g_["params"]], 1.0)
        opt.step()

        if step % EVAL_EVERY == 0:
            with torch.no_grad():
                E_eval = (encode(model, torch.arange(len(qids), device=dev), grad=False)
                          if mode == "lora" else E_all)
                Tc = logT.exp().clamp(5e-3, 1.0).detach()
            for lam in LAMBDAS:  # lambda sweep in the decision rule only
                _, ig, ic = hard_eval(E_eval[ival], E_eval[itr], itr, ival, Tc, lam)
                hpicks, hg, hc = hard_eval(E_eval[te], E_eval[tr], tr, te, Tc, lam)
                rec = {"step": step, "lam": lam, "loss": float(loss.detach()),
                       "T": float(Tc), "tau": float(logtau.exp().detach()),
                       "inner_graded": ig, "inner_cost": ic,
                       "holdout_graded": hg, "holdout_cost": hc}
                evals.append({**rec, "holdout_picks": hpicks})
                mlog.write(json.dumps(rec) + "\n")
            mlog.flush()
    mlog.close()

    feas = [e for e in evals if e["inner_graded"] >= vb_g - 0.02 and e["inner_cost"] > 0]
    pick = (max(feas, key=lambda e: vb_c / e["inner_cost"]) if feas
            else max(evals, key=lambda e: e["inner_graded"] - LAMBDA_TRAIN * e["inner_cost"]))
    res = {"seed": seed, "mode": mode, "lam": pick["lam"], "step": pick["step"],
           "feasible": bool(feas), "inner_graded": pick["inner_graded"],
           "inner_ratio": vb_c / pick["inner_cost"], "vb_g": vb_g,
           "holdout_picks": pick["holdout_picks"], "te": sp["te"],
           "holdout_graded": pick["holdout_graded"], "holdout_cost": pick["holdout_cost"],
           "wall_s": round(time.time() - t0, 1)}
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    tmp.replace(dest)
    print(f"{tag}: step {pick['step']} lam {pick['lam']} feasible={bool(feas)} "
          f"inner_ratio={res['inner_ratio']:.2f} holdout_g={pick['holdout_graded']:.3f} "
          f"({res['wall_s']}s)", flush=True)

print("ALL DONE", flush=True)
