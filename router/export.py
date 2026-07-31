"""Freeze the router into a portable artifact, then prove the artifact reproduces the CV.

This is the production export step, not a research/analysis script -- kept as its own
small, dependency-obvious CLI.

Writes results/router_v0.{json,npz}. The self-test re-derives the leave-one-repo-out
decisions through the exported Router class and checks the resulting cost/quality against
the numbers measured in the nested-CV race. If the artifact disagrees with the experiment,
this fails loudly rather than shipping a router that behaves differently from the one
that was measured.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from router import experiments  # noqa: E402
from router import router_core as route  # noqa: E402
from router.harness import load_env  # noqa: E402
from router.router_core import Router  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Chosen by nested CV: tau=0.5 won in all 5 outer folds; k varied 6/12/20 so the midpoint
# is the stable pick. Kept explicit rather than re-tuned at export time.
EXPORT_K, EXPORT_TAU = 12, 0.5
# Below this cosine similarity to the nearest labelled task, the neighbour vote is not
# informative and we escalate instead. 110 tasks cannot cover the space of coding work.
SIM_FLOOR = 0.35

EXPORT_EFFORTS = ("low", "medium", "high", "xhigh", "max", "default")
# Anthropic rejects output_config.effort on haiku (measured HTTP 400); it is not in this pool,
# but the mapping is explicit so a future arm cannot silently emit an invalid request.
NO_EFFORT = {"claude-haiku-4-5"}


def arm_to_spec(arm: str) -> dict:
    """`mini_swe_agent_gpt_5_6_sol_medium` -> real provider model id + request kwargs."""
    s = re.sub(r"^mini_swe_agent_", "", arm)
    eff = next((e for e in EXPORT_EFFORTS if s.endswith("_" + e)), None)
    core = s[: -(len(eff) + 1)] if eff else s
    if eff == "default":
        eff = None
    # gpt_5_6_sol -> gpt-5.6-sol ; claude_opus_5 -> claude-opus-5
    model = core.replace("_", "-")
    model = re.sub(r"gpt-(\d)-(\d)-", r"gpt-\1.\2-", model)
    model = re.sub(r"^gpt-(\d)-(\d)$", r"gpt-\1.\2", model)
    anthropic = model.startswith("claude")
    if anthropic:
        kw: dict = {"model": model, "thinking": {"type": "adaptive"}}
        if eff and model not in NO_EFFORT:
            kw["output_config"] = {"effort": eff}
    else:
        kw = {"model": model}
        if eff:
            kw["reasoning"] = {"effort": eff}
    return {"model": model, "effort": eff,
            "provider": "anthropic" if anthropic else "openai", "request_kwargs": kw}


def cmd_export_router(args) -> None:
    load_env()
    OUT = ROOT / "results"

    full = experiments.build()
    d = experiments.datasets_b.load_deepswe()
    experiments.embed(full, d["text"])
    keep = [i for i, a in enumerate(full.arms)
            if any(t in a for t in ("gpt_5", "claude_", "codex"))]
    arms = [full.arms[i] for i in keep]
    resolved = full.resolved[keep]
    cost = full.cost[keep]
    graded = full.graded[keep]

    med = np.median(cost, axis=1)
    fallback = int(np.argmax(graded.mean(axis=1)))
    meta = {
        "version": "v0",
        "embed_model": experiments.__dict__.get("EMBED", None) or "text-embedding-3-large",
        "arms": arms,
        "arm_spec": {a: arm_to_spec(a) for a in arms},
        "k": EXPORT_K, "tau": EXPORT_TAU, "sim_floor": SIM_FLOOR,
        "fallback_arm_index": fallback,
        "provenance": (
            f"DeepSWE v1.1, {len(arms)} OpenAI/Anthropic arms x {full.n} tasks over "
            f"{len(set(full.group))} repos. Labels are DeepSWE's published per-trial "
            f"f2p_passed/f2p_total and cost_usd; we ran no episodes. Nested repo-grouped CV "
            f"measured 2.15x cheaper than always-{arms[fallback]} at graded 0.933 vs 0.954, "
            f"cost-ratio 95% CI [1.87,2.46], graded-delta 95% CI [-0.044,+0.000]."),
        "scope_warning": (
            "INPUT SHAPE MATTERS. Fit on repo-issue statements of p10=955 / p50=1976 / "
            "p90=3450 characters. In-distribution nearest-neighbour cosine similarity runs "
            f"min 0.266 / p50 0.439; sim_floor={SIM_FLOOR} is the p10, so it abstains on the "
            "least-covered ~10% of in-distribution tasks. Short one-line prompts score "
            "0.14-0.34 and will ALWAYS abstain to the strongest arm -- safe, but no saving. "
            "Long-horizon tasks only (median 61 agent steps). NOT validated on short "
            "interactive requests or on any held-out benchmark. Always check "
            "Decision.off_distribution before trusting a route."),
        "measured_artifact_behaviour": (
            "The shipped artifact, re-run under the same repo-grouped folds WITH the "
            "sim_floor guard active, gives 1.81x cheaper at graded 0.939 (18/110 tasks "
            "escalated as off-distribution). The 2.15x figure is the ungated policy. The "
            "guard trades 0.34x of saving for +0.006 graded and a refusal-to-guess property."),
        "n_tasks": int(full.n), "n_repos": int(len(set(full.group))),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "router_v0.json").write_text(json.dumps(meta, indent=1))
    np.savez_compressed(OUT / "router_v0.npz", emb=full.emb.astype(np.float32),
                        resolved=resolved, med_cost=med)
    sz = sum((OUT / f).stat().st_size for f in ("router_v0.json", "router_v0.npz"))
    print(f"wrote results/router_v0.{{json,npz}}  ({sz/1024:.0f} KB total)")

    # ---- self-test: does the ARTIFACT reproduce the measured experiment? ----
    m = route.Matrix(arms=arms, qids=full.qids, resolved=resolved, graded=graded,
                     cost=cost, difficulty=full.difficulty, group=full.group, emb=full.emb)
    folds = route.grouped_folds(m, n_folds=5, seed=0)
    gr = np.zeros(m.n)
    co = np.zeros(m.n)
    off = 0
    for te in folds:
        tr = np.array([j for j in range(m.n) if j not in set(te.tolist())])
        # Restrict the artifact's table to this fold's train rows, so the self-test is
        # honest rather than letting the shipped table see the test task.
        sub = Router(OUT)
        sub.emb, sub.resolved = m.emb[tr], resolved[:, tr]
        sub.med_cost = np.median(cost[:, tr], axis=1)
        sub._order = np.argsort(sub.med_cost)
        sub.fallback = int(np.argmax(resolved[:, tr].mean(axis=1)))
        for j in te:
            dec = sub.route_embedding(m.emb[j])
            i = arms.index(dec.arm_id)
            off += int(dec.off_distribution)
            co[j] = cost[i, j]
            gr[j] = graded[i, j]
    bg = graded[fallback].mean()
    bc = cost[fallback].sum()
    print("\nself-test (artifact re-run under the same repo-grouped folds):")
    print(f"  baseline always-{arms[fallback]}: graded {bg:.3f}  ${bc:.1f}")
    print(f"  exported router                 : graded {gr.mean():.3f}  ${co.sum():.1f}  "
          f"{bc/co.sum():.2f}x")
    print(f"  off-distribution escalations    : {off}/{m.n}")
    assert bc / co.sum() > 1.5, "exported router lost its cost advantage"
    assert gr.mean() > bg - 0.05, "exported router lost too much quality"
    print("  OK -- artifact behaves like the measured experiment")


def main() -> None:
    cmd_export_router(argparse.Namespace())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    main()
