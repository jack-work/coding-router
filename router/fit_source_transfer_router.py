"""Fit a zero-shot difficulty router from SWE-rebench for a DeepSWE eval.

This file intentionally has no DeepSWE outcome or cost input.  The source labels are the
free, graded SWE-rebench trajectories.  DeepSWE task text is only used as the query at
deployment time, exactly as it would be for an incoming task.

The output is a policy consumable by the WMO direct DeepSWE runner.  It contains a
source-kNN difficulty score for every DeepSWE task and a fixed three-rung policy:

  easy source prediction   -> Luna low
  normal source prediction -> Luna max
  hard source prediction   -> Opus max

The last two thresholds are source-independent quantiles.  They are deliberately not
selected on DeepSWE.  The online runner can escalate one rung after a failed tool/test
turn, subject to the cache-aware prefill guard.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from router import datasets

ROOT = pathlib.Path(__file__).resolve().parent.parent
EMBED_PATH = ROOT / "results" / "emb" / "te3-large.json"
TEXT_PATH = ROOT / "results" / "router_texts.jsonl"

CHEAP = "gpt-5.6-luna__low"
BASELINE = "gpt-5.6-luna__max"
STRONG = "claude-opus-5__max"

POOL = [
    {"name": CHEAP, "input_per_mtok": 1.0, "cached_input_per_mtok": 0.1,
     "output_per_mtok": 6.0, "cache_write_per_mtok": 1.25},
    {"name": BASELINE, "input_per_mtok": 1.0, "cached_input_per_mtok": 0.1,
     "output_per_mtok": 6.0, "cache_write_per_mtok": 1.25},
    {"name": STRONG, "input_per_mtok": 5.0, "cached_input_per_mtok": 0.5,
     "output_per_mtok": 25.0, "cache_write_per_mtok": 6.25},
]


def _load_embeddings() -> dict[str, np.ndarray]:
    raw = json.loads(EMBED_PATH.read_text(encoding="utf-8"))
    out: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        e = np.asarray(value, dtype=np.float32)
        out[key] = e / (np.linalg.norm(e) or 1.0)
    return out


def _load_text_ids() -> tuple[dict[str, str], set[str]]:
    texts: dict[str, str] = {}
    for line in TEXT_PATH.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        texts[str(row["id"])] = str(row["text"])
    return texts, {key.split(":", 1)[1] for key in texts if key.startswith("dswe:")}


def _source_ease(score: np.ndarray) -> np.ndarray:
    """Normalize each source arm before averaging, so model tier is not the label.

    SWE-rebench has four different scaffolds/model rows with uneven coverage.  A raw
    average would treat the high-base-rate OpenHands row as a task label and would treat
    missing values as failures.  Per-arm robust normalization makes this a difficulty
    target instead of a source-model leaderboard target.
    """
    rows: list[np.ndarray] = []
    for row in score:
        valid = row[np.isfinite(row)]
        if len(valid) == 0:
            continue
        lo, hi = np.percentile(valid, [10, 90])
        scale = max(float(hi - lo), 0.05)
        rows.append(np.clip((row - lo) / scale, 0.0, 1.0))
    normalized = np.asarray(rows, dtype=float)
    with np.errstate(invalid="ignore"):
        ease = np.nanmean(normalized, axis=0)
    return np.nan_to_num(ease, nan=float(np.nanmean(ease)))


def _knn_predict(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray,
                 k: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.empty(len(query_x), dtype=float)
    nearest = np.empty(len(query_x), dtype=float)
    k = max(1, min(k, len(train_x)))
    for i, q in enumerate(query_x):
        sims = train_x @ q
        nn = np.argsort(-sims)[:k]
        weights = np.clip(sims[nn], 0.0, None) + 1e-6
        values[i] = float(np.dot(weights, train_y[nn]) / weights.sum())
        nearest[i] = float(sims[nn[0]])
    return values, nearest


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    def rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="stable")
        result = np.empty(len(x), dtype=float)
        result[order] = np.arange(len(x), dtype=float)
        return result

    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def _source_cv(x: np.ndarray, y: np.ndarray, groups: list[str]) -> dict[str, Any]:
    unique = sorted(set(groups))
    random.Random(31).shuffle(unique)
    folds = [unique[i::5] for i in range(5)]
    rows: list[dict[str, float]] = []
    for k in (8, 16, 32, 64):
        pred = np.full(len(y), np.nan)
        for held_out in folds:
            test_groups = set(held_out)
            te = np.array([i for i, group in enumerate(groups) if group in test_groups])
            tr = np.array([i for i, group in enumerate(groups) if group not in test_groups])
            pred[te], _ = _knn_predict(x[tr], y[tr], x[te], k)
        rows.append({
            "k": k,
            "spearman": _rank_corr(y, pred),
            "mae": float(np.mean(np.abs(y - pred))),
        })
    best = max(rows, key=lambda row: (row["spearman"], -row["mae"]))
    return {"folds": 5, "groups": len(unique), "grid": rows, "selected": best}


def fit(output: pathlib.Path, *, easy_quantile: float = 0.55,
        hard_quantile: float = 0.05, use_strong: bool = True) -> dict[str, Any]:
    if not 0.0 < hard_quantile < easy_quantile < 1.0:
        raise ValueError("require 0 < hard_quantile < easy_quantile < 1")
    embeddings = _load_embeddings()
    texts, deepswe_ids = _load_text_ids()
    source = datasets.load_swe_rebench("free")
    source_ids = [f"srb:{task}" for task in source["tasks"]]
    missing = [key for key in source_ids if key not in embeddings]
    if missing:
        raise RuntimeError(f"missing source embeddings: {missing[:3]}")
    target_ids = [f"dswe:{task}" for task in sorted(deepswe_ids)]
    if any(key not in embeddings for key in target_ids):
        missing = [key for key in target_ids if key not in embeddings]
        raise RuntimeError(f"missing DeepSWE query embeddings: {missing[:3]}")

    source_x = np.stack([embeddings[key] for key in source_ids])
    target_x = np.stack([embeddings[key] for key in target_ids])
    score = np.asarray(source["score"], dtype=float)
    ease = _source_ease(score)
    groups = [str(source["group"][task]) for task in source["tasks"]]
    cv = _source_cv(source_x, ease, groups)
    k = int(cv["selected"]["k"])
    target_ease, nearest = _knn_predict(source_x, ease, target_x, k)
    easy_cut = float(np.quantile(target_ease, easy_quantile))
    hard_cut = float(np.quantile(target_ease, hard_quantile))

    routes: dict[str, str] = {}
    route_meta: dict[str, dict[str, Any]] = {}
    for key, value, sim in zip(target_ids, target_ease, nearest):
        task_id = key.split(":", 1)[1]
        if value >= easy_cut:
            tier, model = "easy", CHEAP
        elif value <= hard_cut and use_strong:
            tier, model = "hard", STRONG
        else:
            tier, model = "normal", BASELINE
        routes[task_id] = model
        route_meta[task_id] = {
            "tier": tier, "base_model": model, "source_ease": round(float(value), 6),
            "nearest_source_similarity": round(float(sim), 6),
        }

    task_ids = sorted(routes)
    artifact = {
        "schema": "source-transfer-per-turn-v1",
        "benchmark_eval_only": "DeepSWE v1.1",
        "fit_source": "nebius/SWE-rebench free graded coding trajectories",
        "fit_source_arms": source["arms"],
        "fit_source_tasks": len(source["tasks"]),
        "fit_labels": "per-task graded pass rates, robust-normalized by source arm",
        "fit_embedding": "cached text-embedding-3-large",
        "deep_swe_outcomes_used_for_fit": False,
        "deep_swe_costs_used_for_fit": False,
        "source_cv": cv,
        "knn_k": k,
        "target_query_count": len(task_ids),
        "route_quantiles": {"easy": easy_quantile, "hard": hard_quantile},
        "strong_arm_enabled": use_strong,
        "route_cutoffs": {"easy_source_ease": easy_cut, "hard_source_ease": hard_cut},
        "task_routes": routes,
        "task_route_metadata": route_meta,
        "escalation_order": [CHEAP, BASELINE, STRONG] if use_strong else [CHEAP, BASELINE],
        "escalate_on_failure": True,
        "default_model": BASELINE,
        "pool": POOL if use_strong else POOL[:2],
        "profile_bins": [0.0],
        "profile_models": [BASELINE, BASELINE],
        "fit_source_text_sha256": __import__("hashlib").sha256(
            "\n".join(texts[f"srb:{task}"] for task in source["tasks"]).encode()
        ).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--easy-quantile", type=float, default=0.55)
    parser.add_argument("--hard-quantile", type=float, default=0.05)
    parser.add_argument("--no-strong", action="store_true",
                        help="fit a two-rung Luna-only policy; keep the strong tier out of routing")
    args = parser.parse_args()
    artifact = fit(args.output, easy_quantile=args.easy_quantile,
                   hard_quantile=args.hard_quantile, use_strong=not args.no_strong)
    counts: dict[str, int] = {}
    for model in artifact["task_routes"].values():
        counts[model] = counts.get(model, 0) + 1
    print(json.dumps({
        "source": artifact["fit_source"],
        "source_tasks": artifact["fit_source_tasks"],
        "target_eval_tasks": artifact["target_query_count"],
        "source_cv": artifact["source_cv"]["selected"],
        "route_counts": counts,
        "cutoffs": artifact["route_cutoffs"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
