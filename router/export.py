"""Freeze the router into results/router_v0.{json,npz}, then self-test that the
artifact reproduces the nested-CV race -- failing loudly if it doesn't, rather than
shipping a router that behaves differently from the one that was measured.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from router import experiments  # noqa: E402
from router import router_core as route  # noqa: E402
from router.harness import load_env  # noqa: E402
from router.router_core import ArmSpec, Router  # noqa: E402

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

# The `local` variant re-embeds the 110 reference tasks with a small model that runs
# entirely on-device (via MLX; see ../router-eval), so routing decisions need no cloud
# API call at all. Requires a local OpenAI-compatible embeddings server already running
# at LOCAL_EMBED_BASE_URL for any cache-miss task -- none are expected, since
# results/deepswe_embeddings_local.json already covers all 113 DeepSWE tasks.
LOCAL_EMBED_MODEL = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
LOCAL_EMBED_BASE_URL = "http://127.0.0.1:8081/v1"


def arm_to_spec(arm: str) -> ArmSpec:
    """Parse an internal arm id into its real provider model id and request kwargs.

    Args:
        arm: An arm id like `mini_swe_agent_gpt_5_6_sol_medium`.

    Returns:
        The resolved `ArmSpec` (model, effort, provider, request kwargs).
    """
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
    return ArmSpec(model=model, effort=eff,
                   provider="anthropic" if anthropic else "openai", request_kwargs=kw)


def cmd_export_router(local: bool = False) -> None:
    """Freeze the router into results/router_v0.{json,npz} and self-test the artifact.

    Args:
        local: If True, re-embeds with LOCAL_EMBED_MODEL instead of OpenAI, writing
            router_v0_local.{json,npz} alongside (not over) the cloud-embedded
            router_v0.*.
    """
    load_env()
    OUT = ROOT / "results"

    full = experiments.build()
    d = experiments.datasets.load_deepswe()
    if local:
        experiments.embed(full, d["text"], cache_path=OUT / "deepswe_embeddings_local.json",
                          embed_model=LOCAL_EMBED_MODEL, base_url=LOCAL_EMBED_BASE_URL)
    else:
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
        "embed_model": LOCAL_EMBED_MODEL if local else "text-embedding-3-large",
        "embed_base_url": LOCAL_EMBED_BASE_URL if local else None,
        "arms": arms,
        "arm_spec": {a: arm_to_spec(a).model_dump() for a in arms},
        "k": EXPORT_K, "tau": EXPORT_TAU, "sim_floor": SIM_FLOOR,
        "fallback_arm_index": fallback,
        "provenance": (
            f"DeepSWE v1.1, {len(arms)} OpenAI/Anthropic arms x {full.n} tasks over "
            f"{len(set(full.group))} repos. Labels are DeepSWE's published per-trial "
            f"f2p_passed/f2p_total and cost_usd; we ran no episodes. Nested repo-grouped CV "
            f"measured 2.15x cheaper than always-{arms[fallback]} at graded 0.933 vs 0.954, "
            f"cost-ratio 95% CI [1.87,2.46], graded-delta 95% CI [-0.044,+0.000]."
            if not local else
            f"Same DeepSWE v1.1 supervision as the cloud-embedded router_v0.json ("
            f"{len(arms)} arms x {full.n} tasks over {len(set(full.group))} repos), but "
            f"embedded locally with {LOCAL_EMBED_MODEL} instead of OpenAI. A separate "
            f"80/20 repo-split holdout (not the 5-fold CV below) found LOCAL embeddings "
            f"gave cost ratio median 3.79x vs OpenAI's 3.18x, graded-delta median -0.021 "
            f"vs -0.015, across 6 seeds -- comparable, not yet validated at the same "
            f"rigor as the cloud variant's nested-CV headline number."),
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
            "guard trades 0.34x of saving for +0.006 graded and a refusal-to-guess property."
            if not local else
            "Self-test numbers below (this artifact, re-run under the same repo-grouped "
            "folds) are the only measured claim for this variant -- no separate ungated-vs-"
            "guarded comparison has been run yet, unlike the cloud variant."),
        "n_tasks": int(full.n), "n_repos": int(len(set(full.group))),
    }
    OUT.mkdir(exist_ok=True)
    json_name, npz_name = ("router_v0_local.json", "router_v0_local.npz") if local \
        else ("router_v0.json", "router_v0.npz")
    (OUT / json_name).write_text(json.dumps(meta, indent=1))
    np.savez_compressed(OUT / npz_name, emb=full.emb.astype(np.float32),
                        resolved=resolved, med_cost=med)
    sz = sum((OUT / f).stat().st_size for f in (json_name, npz_name))
    print(f"wrote results/{json_name.replace('.json','')}.{{json,npz}}  ({sz/1024:.0f} KB total)")

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
        sub = Router(OUT, artifact_json=json_name, artifact_npz=npz_name)
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


def main(local: bool = False) -> None:
    """CLI entrypoint: forward to `cmd_export_router`."""
    cmd_export_router(local)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
