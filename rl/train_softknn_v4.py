"""EXP-009: training-algorithm sweep for the embedding encoder (soft-kNN policy).

Same data/splits/selection protocol as train_softknn v1/v2; what varies is the
TRAINING SIGNAL through the soft-kNN into LoRA(Qwen3-Embedding-0.6B):

  reward  exact expected reward  -E_pi[graded - lam*cost]        (EXP-002/003 objective)
  rank    pairwise utility ranking: for arms a,b with realized reward gap > margin on
          task x, -log sigmoid(u_a(x) - u_b(x))                  (EquiRouter-flavored)
  grpo    sampled policy gradient, S arms/query, group-mean baseline — the REINFORCE
          estimator of `reward`; tests whether estimator noise regularizes
  nca     pure metric learning, no reward: pull tasks with similar arm-outcome
          profiles together (positives = graded-column correlation >= 0.7), NCA loss
          -log(sum_pos w / sum_all w); the decision rule is tuned afterwards as usual

Usage: python train_softknn_v4.py <seeds-csv> <algo> <outdir> [beta]
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
LAMBDAS = (0.005, 0.01, 0.02, 0.05, 0.1)
LAMBDA_TRAIN = 0.02
STEPS, EVAL_EVERY = 150, 10
LORA_LR, TEMP_LR = 5e-5, 1e-2
MAX_LEN = 1536
RANK_MARGIN, GRPO_S = 0.05, 8

seeds = [int(s) for s in sys.argv[1].split(",")]
algo = sys.argv[2]
outdir = pathlib.Path(sys.argv[3])
KL_BETA = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
assert algo in ("reward", "rank", "grpo", "nca")
outdir.mkdir(parents=True, exist_ok=True)
dev = "cuda:0"
torch.manual_seed(0)

d = np.load(WORK / "router_rl_payload.npz")
meta = json.loads((WORK / "router_rl_meta.json").read_text())
graded = torch.tensor(d["graded"], dtype=torch.float32, device=dev)
cost = torch.tensor(d["cost"], dtype=torch.float32, device=dev)
qids, groups = meta["qids"], meta["groups"]
texts = json.loads((WORK / "texts.json").read_text())
docs = [texts[f"dswe:{q}"] for q in qids]
grp_code = torch.tensor([sorted(set(groups)).index(g) for g in groups], device=dev)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
batch = tok(docs, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
ids_all = batch["input_ids"].to(dev)
mask_all = batch["attention_mask"].to(dev)


def load_model():
    from peft import LoraConfig, get_peft_model

    m = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16,
                                  attn_implementation="sdpa").to(dev)
    m.gradient_checkpointing_enable()
    m.enable_input_require_grads()
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    m = get_peft_model(m, cfg)
    m.train()
    return m


def encode(model, idx, grad):
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


def routing_terms(E_q, E_m, mem_idx, T, lam, repo_mask_rows=None):
    sims = E_q @ E_m.T / T
    if repo_mask_rows is not None:
        same = repo_mask_rows[:, None] == grp_code[mem_idx][None, :]
        sims = sims.masked_fill(same, float("-inf"))
    w = F.softmax(sims, dim=1)
    p_hat = w @ graded[:, mem_idx].T
    medc = cost[:, mem_idx].median(dim=1).values
    return w, p_hat, p_hat - lam * medc[None, :]


def hard_eval(E_q, E_m, mem_idx, q_idx, T, lam):
    with torch.no_grad():
        _, _, u = routing_terms(E_q, E_m, mem_idx, T, lam)
        picks = u.argmax(dim=1)
        return (picks.tolist(), graded[picks, q_idx].mean().item(),
                cost[picks, q_idx].sum().item())


for seed in seeds:
    sp = meta["splits"][str(seed)]
    tr = torch.tensor(sp["tr"], device=dev)
    te = torch.tensor(sp["te"], device=dev)
    itr = torch.tensor(sp["inner_tr"], device=dev)
    ival = torch.tensor(sp["inner_val"], device=dev)
    vb = int(graded[:, itr].mean(dim=1).argmax().item())
    vb_g, vb_c = graded[vb, ival].mean().item(), cost[vb, ival].sum().item()

    t0 = time.time()
    tag = f"s{seed}_v4{algo}_b{KL_BETA}"
    dest = outdir / f"result_{tag}.json"
    if dest.exists():
        print(f"skip {tag}", flush=True)
        continue
    model = load_model()
    logT = torch.tensor(np.log(0.05), device=dev, requires_grad=True)
    logtau = torch.tensor(np.log(0.05), device=dev, requires_grad=True)
    opt = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad], "lr": LORA_LR},
         {"params": [logT, logtau], "lr": TEMP_LR}])

    # rank targets: realized per-cell reward on train tasks
    Y = (graded[:, tr] - LAMBDA_TRAIN * cost[:, tr]).T          # (Ntr, A)
    # nca positives: outcome-profile correlation between train tasks
    G = graded[:, tr]
    Gc = G - G.mean(dim=0, keepdim=True)
    corr = (Gc.T @ Gc) / (Gc.norm(dim=0)[:, None] * Gc.norm(dim=0)[None, :] + 1e-6)
    pos_mask = (corr >= 0.7)
    pos_mask.fill_diagonal_(False)

    pi_ref, rng = None, np.random.default_rng(seed)
    evals, mlog = [], (outdir / f"metrics_{tag}.jsonl").open("w")
    for step in range(1, STEPS + 1):
        E_tr = encode(model, tr, grad=True)
        T, tau = logT.exp().clamp(5e-3, 1.0), logtau.exp().clamp(5e-3, 1.0)
        w, p_hat, u = routing_terms(E_tr, E_tr, tr, T, LAMBDA_TRAIN,
                                    repo_mask_rows=grp_code[tr])
        if algo == "reward" or algo == "grpo":
            pi = F.softmax(u / tau, dim=1)
            if pi_ref is None and KL_BETA > 0:
                pi_ref = pi.detach()
            if algo == "reward":
                loss = -(pi * Y).sum(dim=1).mean()
            else:
                dist = torch.distributions.Categorical(probs=pi)
                acts = dist.sample((GRPO_S,))                      # (S, Ntr)
                R_s = Y.T.gather(0, acts)                          # (S, Ntr)
                adv = (R_s - R_s.mean(dim=0, keepdim=True)).detach()
                loss = -(adv * dist.log_prob(acts)).mean()
            if KL_BETA > 0 and pi_ref is not None:
                logpi = F.log_softmax(u / tau, dim=1)
                loss = loss + KL_BETA * (pi * (logpi - (pi_ref + 1e-9).log())).sum(1).mean()
        elif algo == "rank":
            du = u[:, :, None] - u[:, None, :]                     # (Ntr, A, A)
            dy = Y[:, :, None] - Y[:, None, :]
            mask = (dy > RANK_MARGIN).float()
            loss = (F.softplus(-du / 0.05) * mask).sum() / mask.sum().clamp(min=1)
        else:  # nca
            wm = w.clamp(min=1e-9)
            pos = (wm * pos_mask.float()).sum(dim=1)
            has_pos = pos_mask.any(dim=1)
            loss = -(pos[has_pos].log()).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g_ in opt.param_groups for p in g_["params"]], 1.0)
        opt.step()

        if step % EVAL_EVERY == 0:
            with torch.no_grad():
                E_all = encode(model, torch.arange(len(qids), device=dev), grad=False)
                Tc = logT.exp().clamp(5e-3, 1.0).detach()
            for lam in LAMBDAS:
                _, ig, ic = hard_eval(E_all[ival], E_all[itr], itr, ival, Tc, lam)
                hpicks, hg, hc = hard_eval(E_all[te], E_all[tr], tr, te, Tc, lam)
                rec = {"step": step, "lam": lam, "loss": float(loss.detach()),
                       "T": float(Tc), "inner_graded": ig, "inner_cost": ic,
                       "holdout_graded": hg, "holdout_cost": hc}
                evals.append({**rec, "holdout_picks": hpicks})
                mlog.write(json.dumps(rec) + "\n")
            mlog.flush()
    mlog.close()

    feas = [e for e in evals if e["inner_graded"] >= vb_g - 0.02 and e["inner_cost"] > 0]
    pick = (max(feas, key=lambda e: vb_c / e["inner_cost"]) if feas
            else max(evals, key=lambda e: e["inner_graded"] - LAMBDA_TRAIN * e["inner_cost"]))
    res = {"seed": seed, "mode": f"v4{algo}_b{KL_BETA}", "lam": pick["lam"],
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
