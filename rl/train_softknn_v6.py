"""EXP-012: one canonical eval (DeepSWE holdout), training-dataset generalization axis,
quality-first selection (<1% target), tokens+latency measured.

Every run trains the encoder geometry (or just temperatures) on ONE source dataset,
then evaluates the SAME deployable configuration on the DeepSWE holdout: soft-kNN vote
over the DeepSWE-train memory, cheapest-utility argmax, (checkpoint, lam) selected on
the inner DeepSWE-train split ONLY, feasibility margin tightened to inner-best - 0.01
and a lam grid extended down to 0.001 so the tuner can buy quality.

  algo    frozen (temps only) | reward (exact expected reward, REINFORCE-style)
          | grpo (sampled, group-mean baseline)
  source  dswe (in-domain 80% split) | lcb (7x76, graded+cost) | srb (4x1424,
          graded only -> quality-only training reward; minibatched)
  beta    KL(pi || init-geometry pi) anchor weight (0 = off)

Usage: python train_softknn_v6.py <seeds-csv> <algo> <source> <beta> <outdir> [n_heldout_arms]
Slate mode (EXP-013): every training step samples a random arm subset (8..A-heldout);
`n_heldout_arms` arms (cost-stratified) are excluded from ALL slates and evaluated
zero-shot: results carry holdout picks for the full pool AND the unseen-arm pool.
Reads /nvme/work/router-rl/{router_v5_payload.npz,router_v5_meta.json,texts.json
(all namespaced ids)}. Writes result_s{seed}_v5{algo}_{source}_b{beta}.json.
"""
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

WORK = pathlib.Path("/nvme/work/router-rl")
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
LAMBDAS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
FEAS_MARGIN = 0.01
STEPS, EVAL_EVERY = 150, 10
LORA_LR, TEMP_LR = 5e-5, 1e-2
MAX_LEN = 1536
GRPO_S, SRB_BATCH = 8, 160

seeds = [int(s) for s in sys.argv[1].split(",")]
algo, source = sys.argv[2], sys.argv[3]
KL_BETA = float(sys.argv[4])
outdir = pathlib.Path(sys.argv[5])
N_HELDOUT = int(sys.argv[6]) if len(sys.argv) > 6 else 0
assert algo in ("frozen", "reward", "grpo") and source in ("dswe", "lcb", "srb")
outdir.mkdir(parents=True, exist_ok=True)
dev = "cuda:0"
torch.manual_seed(0)

d = np.load(WORK / "router_v5_payload.npz")
meta = json.loads((WORK / "router_v5_meta.json").read_text())
graded = torch.tensor(d["graded"], dtype=torch.float32, device=dev)
cost = torch.tensor(d["cost"], dtype=torch.float32, device=dev)
qids, groups = meta["qids"], meta["groups"]
texts = json.loads((WORK / "texts.json").read_text())
grp_code = torch.tensor([sorted(set(groups)).index(g) for g in groups], device=dev)

# training source: texts, graded matrix, per-cell reward for the TRAINING objective
if source == "dswe":
    src_ids = [f"dswe:{q}" for q in qids]
    src_graded, src_lam = graded, 0.02
    src_cost = cost
    src_grp = grp_code
elif source == "lcb":
    src_ids = [f"lcb:{q}" for q in meta["lcb_qids"]]
    src_graded = torch.tensor(d["lcb_graded"], dtype=torch.float32, device=dev)
    src_cost = torch.tensor(d["lcb_cost"], dtype=torch.float32, device=dev)
    src_lam = 3.0  # lcb costs are cents; scale so cost term ~10% of reward range
    src_grp = torch.tensor([sorted(set(meta["lcb_groups"])).index(g)
                            for g in meta["lcb_groups"]], device=dev)
else:
    src_ids = [f"srb:{q}" for q in meta["srb_qids"]]
    sg = torch.tensor(d["srb_graded"], dtype=torch.float32, device=dev)
    src_graded = torch.nan_to_num(sg, nan=0.0)
    src_nanmask = torch.isfinite(sg)
    src_cost, src_lam = torch.zeros_like(src_graded), 0.0  # no cost field exists
    src_grp = torch.tensor([sorted(set(meta["srb_groups"])).index(g)
                            for g in meta["srb_groups"]], device=dev)

A_SRC = src_graded.shape[0]
if N_HELDOUT:
    order = torch.argsort(src_cost.nanmean(dim=1) if source != "srb"
                          else torch.arange(A_SRC, dtype=torch.float32, device=dev))
    HELD = sorted(order[::max(1, A_SRC // N_HELDOUT)][:N_HELDOUT].tolist())
else:
    HELD = []
TRAINABLE_ARMS = torch.tensor([a for a in range(A_SRC) if a not in HELD], device=dev)
print(f"slate mode: {N_HELDOUT} held-out arms {HELD}", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL_ID)


def tokenize(ids):
    docs = [texts[i][:8000] for i in ids]
    b = tok(docs, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    return b["input_ids"].to(dev), b["attention_mask"].to(dev)


ids_eval, mask_eval = tokenize([f"dswe:{q}" for q in qids])
ids_src, mask_src = tokenize(src_ids)
print(f"{algo}/{source}/b{KL_BETA}: {len(src_ids)} train texts, {len(qids)} eval texts",
      flush=True)


def load_model():
    m = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16,
                                  attn_implementation="sdpa").to(dev)
    if algo != "frozen":
        from peft import LoraConfig, get_peft_model
        m.gradient_checkpointing_enable()
        m.enable_input_require_grads()
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        m = get_peft_model(m, cfg)
        m.train()
    return m


def encode(model, ids_all, mask_all, idx, grad):
    outs = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(idx), 16):
            sl = idx[i:i + 16]
            h = model(input_ids=ids_all[sl], attention_mask=mask_all[sl],
                      use_cache=False).last_hidden_state
            last = mask_all[sl].sum(1) - 1
            outs.append(h[torch.arange(len(sl), device=dev), last])
    return F.normalize(torch.cat(outs).float(), dim=1)


def soft_vote(E_q, E_m, g_m, T, grp_q=None, grp_m=None):
    sims = E_q @ E_m.T / T
    if grp_q is not None:
        sims = sims.masked_fill(grp_q[:, None] == grp_m[None, :], float("-inf"))
    w = F.softmax(sims, dim=1)
    return w @ g_m.T                                     # (Q, A_src)


def hard_eval(E_q, E_m, mem_idx, q_idx, T, lam, pool=None):
    with torch.no_grad():
        g_m = graded[:, mem_idx] if pool is None else graded[pool][:, mem_idx]
        c_m = cost[:, mem_idx] if pool is None else cost[pool][:, mem_idx]
        p = soft_vote(E_q, E_m, g_m, T)
        u = p - lam * c_m.median(dim=1).values[None, :]
        picks = u.argmax(dim=1)
        gsel = graded if pool is None else graded[pool]
        csel = cost if pool is None else cost[pool]
        return (picks.tolist(), gsel[picks, q_idx].mean().item(),
                csel[picks, q_idx].sum().item())


for seed in seeds:
    sp = meta["splits"][str(seed)]
    tr = torch.tensor(sp["tr"], device=dev)
    te = torch.tensor(sp["te"], device=dev)
    itr = torch.tensor(sp["inner_tr"], device=dev)
    ival = torch.tensor(sp["inner_val"], device=dev)
    vb = int(graded[:, itr].mean(dim=1).argmax().item())
    vb_g, vb_c = graded[vb, ival].mean().item(), cost[vb, ival].sum().item()

    t0 = time.time()
    tag = (f"s{seed}_v6{algo}_{source}_b{KL_BETA}_h{N_HELDOUT}" if N_HELDOUT
           else f"s{seed}_v5{algo}_{source}_b{KL_BETA}")
    dest = outdir / f"result_{tag}.json"
    if dest.exists():
        print(f"skip {tag}", flush=True)
        continue
    model = load_model()
    logT = torch.tensor(np.log(0.05), device=dev, requires_grad=True)
    logtau = torch.tensor(np.log(0.05), device=dev, requires_grad=True)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    groups_opt = ([{"params": lora_params, "lr": LORA_LR}] if lora_params else [])
    opt = torch.optim.AdamW(groups_opt + [{"params": [logT, logtau], "lr": TEMP_LR}])

    # for dswe source, train only on this seed's train split; else the whole source
    if source == "dswe":
        src_pool = tr
    else:
        src_pool = torch.arange(len(src_ids), device=dev)
    if algo == "frozen":
        E_src_all = encode(model, ids_src, mask_src, src_pool, grad=False)

    rng = np.random.default_rng(seed)
    pi_ref, evals = None, []
    mlog = (outdir / f"metrics_{tag}.jsonl").open("w")
    for step in range(1, STEPS + 1):
        batch = (src_pool if len(src_pool) <= SRB_BATCH else
                 src_pool[torch.tensor(rng.choice(len(src_pool), SRB_BATCH, replace=False),
                                       device=dev)])
        if algo == "frozen":
            E_b = E_src_all if len(src_pool) <= SRB_BATCH else None
            if E_b is None:
                idxmap = {int(v): i for i, v in enumerate(src_pool.tolist())}
                E_b = E_src_all[[idxmap[int(v)] for v in batch.tolist()]]
        else:
            E_b = encode(model, ids_src, mask_src, batch, grad=True)
        T, tau = logT.exp().clamp(5e-3, 1.0), logtau.exp().clamp(5e-3, 1.0)
        if N_HELDOUT:
            k = int(rng.integers(8, len(TRAINABLE_ARMS) + 1))
            slate = TRAINABLE_ARMS[torch.tensor(
                rng.choice(len(TRAINABLE_ARMS), k, replace=False), device=dev)]
        else:
            slate = torch.arange(src_graded.shape[0], device=dev)
        gm = src_graded[slate][:, batch]
        p_hat = soft_vote(E_b, E_b, gm, T, src_grp[batch], src_grp[batch])
        medc = src_cost[slate][:, batch].median(dim=1).values
        u = p_hat - src_lam * medc[None, :]
        pi = F.softmax(u / tau, dim=1)
        R = (gm - src_lam * src_cost[slate][:, batch]).T
        if pi_ref is None and KL_BETA > 0 and not N_HELDOUT:
            pi_ref = pi.detach()
        if algo == "grpo":
            dist = torch.distributions.Categorical(probs=pi)
            acts = dist.sample((GRPO_S,))
            R_s = R.T.gather(0, acts)
            adv = (R_s - R_s.mean(dim=0, keepdim=True)).detach()
            loss = -(adv * dist.log_prob(acts)).mean()
        else:
            loss = -(pi * R).sum(dim=1).mean()
        if KL_BETA > 0 and pi_ref is not None and pi.shape == pi_ref.shape:
            logpi = F.log_softmax(u / tau, dim=1)
            loss = loss + KL_BETA * (pi * (logpi - (pi_ref + 1e-9).log())).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params + [logT, logtau], 1.0)
        opt.step()

        if step % EVAL_EVERY == 0:
            with torch.no_grad():
                E_dswe = encode(model, ids_eval, mask_eval,
                                torch.arange(len(qids), device=dev), grad=False)
                Tc = logT.exp().clamp(5e-3, 1.0).detach()
            for lam in LAMBDAS:
                _, ig, ic = hard_eval(E_dswe[ival], E_dswe[itr], itr, ival, Tc, lam)
                hpicks, hg, hc = hard_eval(E_dswe[te], E_dswe[tr], tr, te, Tc, lam)
                rec = {"step": step, "lam": lam, "loss": float(loss.detach()),
                       "T": float(Tc), "inner_graded": ig, "inner_cost": ic,
                       "holdout_graded": hg, "holdout_cost": hc}
                if HELD and source == "dswe":
                    hp2, hg2, hc2 = hard_eval(E_dswe[te], E_dswe[tr], tr, te, Tc, lam,
                                              pool=torch.tensor(HELD, device=dev))
                    rec["unseen_pool_graded"], rec["unseen_pool_cost"] = hg2, hc2
                    rec["unseen_pool_picks"] = hp2
                evals.append({**rec, "holdout_picks": hpicks})
                mlog.write(json.dumps(rec) + "\n")
            mlog.flush()
    mlog.close()

    feas = [e for e in evals
            if e["inner_graded"] >= vb_g - FEAS_MARGIN and e["inner_cost"] > 0]
    pick = (max(feas, key=lambda e: vb_c / e["inner_cost"]) if feas
            else max(evals, key=lambda e: e["inner_graded"]))
    res = {"seed": seed, "mode": tag.split("_", 1)[1], "held_arms": HELD, "lam": pick["lam"],
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
