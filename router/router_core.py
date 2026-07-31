"""The deployable router: price table, kNN inference, and the policies/race harness.

Three layers: pricing (`Price`/`Arm`, the canonical price table), predict (`Router`/
`Decision`: load an exported artifact and route a task to an arm), and route (`Matrix`
plus the CV policies and stats used to compare a learned router against a cascade).
"""
from __future__ import annotations

import collections
import dataclasses
import glob
import json
import logging
import math
import pathlib
import random
import re
import sys
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict
from tabulate import tabulate

ROOT = pathlib.Path(__file__).resolve().parent.parent
EMB_CACHE = ROOT / "results" / "lcb_embeddings.json"

PRICE_FETCH_DATE = "2026-07-28"
ARTIFACT_JSON = "router_v0.json"
ARTIFACT_NPZ = "router_v0.npz"
EMBED_MODEL = "text-embedding-3-large"

logger = logging.getLogger(__name__)


# ============================================================================ pricing
@dataclasses.dataclass(frozen=True)
class Price:
    """One model's price row. USD per 1M tokens; reasoning/thinking bills as output."""

    inp: float          # uncached input, $/1M
    cache_read: float   # cached input read, $/1M
    out: float          # output (incl. reasoning/thinking), $/1M
    cache_write: float  # 5-min cache write, $/1M -- 1.25x inp where a provider prices
                         # the premium at all; equal to inp where it doesn't (see below)


# provider -> model -> Price. Fetched live 2026-07-28 from OpenAI
# (developers.openai.com/api/docs/pricing) and Anthropic
# (platform.claude.com/docs/en/about-claude/pricing).
STANDARD: dict[str, dict[str, Price]] = {
    "openai": {
        "gpt-5.4-nano":  Price(0.20, 0.02,  1.25, 0.20),
        "gpt-5.4-mini":  Price(0.75, 0.075, 4.50, 0.75),
        "gpt-5.4":       Price(2.50, 0.25, 15.00, 2.50),
        "gpt-5.5":       Price(5.00, 0.50, 30.00, 5.00),
        "gpt-5.3-codex": Price(1.75, 0.175, 14.00, 1.75),
        # GPT-5.6 family is the first to charge for cache writes (1.25x input).
        "gpt-5.6-luna":  Price(1.00, 0.10,  6.00, 1.25),
        "gpt-5.6-terra": Price(2.50, 0.25, 15.00, 3.125),
        "gpt-5.6-sol":   Price(5.00, 0.50, 30.00, 6.25),
    },
    "anthropic": {
        # Anthropic rows re-verified against the live pricing page 2026-07-31, when
        # sonnet-4-6 and opus-5 were added (opus-5 prices identically to opus-4-8;
        # sonnet-4-6 identically to post-intro sonnet-5).
        "claude-haiku-4-5":  Price(1.00, 0.10,  5.00, 1.25),
        "claude-sonnet-4-6": Price(3.00, 0.30, 15.00, 3.75),
        "claude-sonnet-5":   Price(3.00, 0.30, 15.00, 3.75),   # post-intro
        "claude-opus-4-8":   Price(5.00, 0.50, 25.00, 6.25),
        "claude-opus-5":     Price(5.00, 0.50, 25.00, 6.25),
        "claude-fable-5":    Price(10.00, 1.00, 50.00, 12.50),
    },
}

# Sonnet 5 introductory rate, active through 2026-08-31 only; STANDARD above already
# uses the post-intro $3/$15, so use this override only to reconcile a live invoice.
INTRO_OVERRIDES = {("anthropic", "claude-sonnet-5"): Price(2.00, 0.20, 10.00, 2.50)}

# Anthropic models that reject output_config.effort with HTTP 400 (measured).
NO_EFFORT_PARAM = {"claude-haiku-4-5"}
# OpenAI effort values accepted today; 'minimal' was removed (measured: 400).
OPENAI_EFFORTS = ("none", "low", "medium", "high", "xhigh")
ANTHROPIC_EFFORTS = ("low", "medium", "high", "xhigh", "max")


class Arm(BaseModel):
    """One routing candidate: a model plus its reasoning configuration."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    effort: str | None = None
    thinking: str = "adaptive"  # anthropic: adaptive|budget|off|omit ; openai: unused

    @property
    def id(self) -> str:
        """This arm's internal key: model name plus effort or thinking mode."""
        return f"{self.model}@{self.effort or self.thinking}"

    @property
    def price(self) -> Price:
        """This arm's row in the standard price table."""
        return STANDARD[self.provider][self.model]

    def cost(self, *, inp: int = 0, cache_read: int = 0, cache_write: int = 0,
             out: int = 0, intro: bool = False) -> float:
        """Compute USD cost for one request's measured token usage.

        Args:
            inp: Uncached input tokens.
            cache_read: Cached input tokens read.
            cache_write: Input tokens written to cache (5-minute rate).
            out: Output tokens, including reasoning/thinking.
            intro: If True, price at the introductory rate where one exists.

        Returns:
            The USD cost of this token usage under this arm's price.
        """
        p = INTRO_OVERRIDES.get((self.provider, self.model), self.price) if intro else self.price
        return (inp * p.inp + cache_read * p.cache_read
                + cache_write * p.cache_write + out * p.out) / 1_000_000

    def request_kwargs(self) -> dict[str, Any]:
        """Build this arm's provider-native request kwargs.

        Returns:
            Keyword arguments ready to splat into the provider SDK call.

        Raises:
            AssertionError: If this arm's (model, effort) combination is known to be
                rejected by the provider (e.g. output_config.effort on claude-haiku-4-5).
        """
        if self.provider == "anthropic":
            kw: dict[str, Any] = {"model": self.model}
            if self.thinking == "adaptive":
                kw["thinking"] = {"type": "adaptive"}
            elif self.thinking == "budget":
                kw["thinking"] = {"type": "enabled", "budget_tokens": 4096}
            elif self.thinking == "off":
                kw["thinking"] = {"type": "disabled"}
            # 'omit' -> send nothing (required for fable-5, whose thinking is always on)
            if self.effort:
                assert self.model not in NO_EFFORT_PARAM, (
                    f"{self.model} rejects output_config.effort (measured HTTP 400)")
                kw["output_config"] = {"effort": self.effort}
            return kw
        assert self.effort in OPENAI_EFFORTS, f"bad openai effort {self.effort!r}"
        return {"model": self.model, "reasoning": {"effort": self.effort}}


def all_validated_arms() -> list[Arm]:
    """List the 47 arms confirmed live and tool-calling on 2026-07-28.

    Returns:
        Every routing candidate this router is allowed to pick from.
    """
    arms = [
        Arm(provider="anthropic", model="claude-haiku-4-5", effort=None, thinking="off"),
        Arm(provider="anthropic", model="claude-haiku-4-5", effort=None, thinking="budget"),
    ]
    for m in ("claude-sonnet-5", "claude-opus-4-8"):
        arms += [Arm(provider="anthropic", model=m, effort=e, thinking="adaptive")
                 for e in ANTHROPIC_EFFORTS]
    arms += [Arm(provider="anthropic", model="claude-fable-5", effort=e, thinking="omit")
             for e in ANTHROPIC_EFFORTS]
    for m in ("gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4", "gpt-5.5", "gpt-5.3-codex", "gpt-5.6-sol"):
        arms += [Arm(provider="openai", model=m, effort=e) for e in OPENAI_EFFORTS]
    return arms


def cost_span() -> tuple[float, float]:
    """Return the (cheapest, priciest) blended $/1M price at a 1:1 in:out mix."""
    blended = [(p.inp + p.out) / 2 for prov in STANDARD.values() for p in prov.values()]
    return min(blended), max(blended)


def main_pricing() -> None:
    """CLI (`arms`): print the validated arm count, cost span, and per-episode cost ladder."""
    arms = all_validated_arms()
    lo, hi = cost_span()
    logger.info(f"{len(arms)} validated arms | blended $/1M span: {lo:.2f} -> {hi:.2f} ({hi/lo:.0f}x)")
    # Cost of a representative agentic episode: 40 turns, heavy cache reuse.
    ep = dict(inp=30_000, cache_read=1_200_000, cache_write=40_000, out=25_000)
    rows = sorted({a.model: a for a in arms}.values(), key=lambda a: a.cost(**ep))
    logger.info(f"\nper-episode cost @ {ep}:")
    logger.info(tabulate([(a.model, f"${a.cost(**ep):.3f}") for a in rows], headers=["model", "cost"],
                         disable_numparse=True))
    c = rows[0].cost(**ep)
    logger.info(f"\nspread cheapest->priciest per episode: {rows[-1].cost(**ep)/c:.1f}x")


# ============================================================================ predict
class ArmSpec(BaseModel):
    """One arm's resolved routing spec, as stored in an exported artifact's `arm_spec`."""

    model: str
    effort: str | None = None
    provider: str
    # Ready to splat into the provider SDK call. Shape genuinely varies by provider
    # (openai: model+reasoning; anthropic: model+thinking+output_config) and is never
    # inspected key-by-key downstream, only forwarded -- see Decision.request_kwargs.
    request_kwargs: dict[str, Any]


class Decision(BaseModel):
    """One routing decision: the chosen arm, its predicted odds, and request kwargs."""

    model: str                  # provider model id, e.g. "gpt-5.6-sol"
    effort: str | None          # reasoning effort / thinking level
    arm_id: str                 # internal arm key
    p_solve: float              # predicted probability this arm resolves the task
    est_cost_usd: float         # median observed cost of this arm per task
    nearest_sim: float          # cosine similarity to the closest labelled task
    off_distribution: bool      # True when no neighbour is close enough to trust
    fallback_used: bool         # True when no arm cleared the threshold
    # Provider-varying shape, same reasoning as ArmSpec.request_kwargs above.
    request_kwargs: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return this decision as a plain dict, e.g. for JSON logging."""
        return self.model_dump()


class Router:
    """kNN router: an exported lookup table of task embeddings, per-arm outcomes, and
    per-arm median cost. `route()`/`route_embedding()` embed a task, find its nearest
    labelled neighbours, and return the cheapest arm they predict will solve it.

    Fit on 110 long-horizon SWE tasks from DeepSWE v1.1 (median 61 agent steps), where
    nested CV measured it at 2.15x cheaper than always using the strongest arm, with the
    accuracy delta's 95% CI containing zero. NOT validated on short interactive tasks,
    non-Python repos beyond DeepSWE's mix, or any held-out benchmark -- always check
    `Decision.off_distribution` before trusting a route outside that scope.
    """

    def __init__(self, artifact_dir: str | pathlib.Path, *,
                artifact_json: str = ARTIFACT_JSON, artifact_npz: str = ARTIFACT_NPZ,
                providers: set[str] | None = None):
        """Load an exported router artifact.

        Args:
            artifact_dir: Directory containing the artifact's `.json`/`.npz` pair.
            artifact_json: Filename of the metadata JSON within `artifact_dir`.
            artifact_npz: Filename of the numpy archive within `artifact_dir`.
            providers: If given (e.g. `{"openai"}`), restricts the arm pool to just
                those providers -- filtered once here so `route_embedding()`/`route()`
                need no changes and can't pick a disallowed arm. Cheapest-first order
                and the fallback arm are both recomputed within the restricted pool.

        Raises:
            ValueError: If restricting to `providers` would leave no arms at all.
        """
        p = pathlib.Path(artifact_dir)
        meta = json.loads((p / artifact_json).read_text())
        arr = np.load(p / artifact_npz)
        self.meta = meta
        arms: list[str] = meta["arms"]
        arm_spec = {k: ArmSpec.model_validate(v) for k, v in meta["arm_spec"].items()}
        resolved: np.ndarray = arr["resolved"]  # (n_arms, n_tasks) bool
        med_cost: np.ndarray = arr["med_cost"]  # (n_arms,)
        fallback = int(meta["fallback_arm_index"])
        if providers is not None:
            keep = [i for i, a in enumerate(arms) if arm_spec[a].provider in providers]
            if not keep:
                raise ValueError(f"no arms left after filtering to providers={providers}")
            arms = [arms[i] for i in keep]
            resolved = resolved[keep]
            med_cost = med_cost[keep]
            fallback = int(np.argmax(resolved.mean(axis=1)))
        self.arms = arms
        self.arm_spec: dict[str, ArmSpec] = arm_spec
        self.k: int = meta["k"]
        self.tau: float = meta["tau"]
        self.sim_floor: float = meta["sim_floor"]
        self.emb: np.ndarray = arr["emb"]            # (n_tasks, dim) L2-normalised
        self.resolved = resolved
        self.med_cost = med_cost
        self.fallback = fallback
        self._order = np.argsort(self.med_cost)      # cheapest arm first

    # ---------------------------------------------------------------- internals
    def _probs(self, v: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute each arm's weighted-neighbour P(resolve) for embedding `v`.

        Args:
            v: An L2-normalised task embedding.

        Returns:
            A (per-arm probabilities, nearest-neighbour similarity) pair.
        """
        sims = self.emb @ v
        k = min(self.k, len(sims))
        nn = np.argsort(-sims)[:k]
        w = np.clip(sims[nn], 0, None) + 1e-6
        return (self.resolved[:, nn] * w).sum(axis=1) / w.sum(), float(sims[nn[0]])

    def route_embedding(self, v: np.ndarray) -> Decision:
        """Route from a pre-computed, L2-normalised embedding of the task text.

        Args:
            v: The task embedding. Normalised internally if not already unit-length.

        Returns:
            The routing `Decision`: chosen arm, predicted odds, and request kwargs.

        Raises:
            ValueError: If `v` is the zero vector and cannot be normalised.
        """
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        if n == 0:
            raise ValueError("zero embedding")
        v = v / n
        p, nearest = self._probs(v)
        pick, fb = None, False
        for i in self._order:
            if p[i] >= self.tau:
                pick = int(i)
                break
        if pick is None:                       # nothing clears the bar -> strongest arm
            pick, fb = self.fallback, True
        # Off-distribution: no labelled task is close enough for the vote to mean much.
        # Escalate rather than trust it -- escalation can only cost money, not accuracy.
        off = nearest < self.sim_floor
        if off:
            pick, fb = self.fallback, True
        spec = self.arm_spec[self.arms[pick]]
        return Decision(model=spec.model, effort=spec.effort,
                        arm_id=self.arms[pick], p_solve=float(p[pick]),
                        est_cost_usd=float(self.med_cost[pick]), nearest_sim=nearest,
                        off_distribution=off, fallback_used=fb,
                        request_kwargs=spec.request_kwargs)

    def route(self, task_text: str, *, api_key: str | None = None,
             base_url: str | None = None) -> Decision:
        """Embed `task_text` with the artifact's embedding model, then route.

        Args:
            task_text: The raw task/issue text to route (truncated to 8000 chars).
            api_key: OpenAI API key; falls back to the client default, or "not-needed"
                when `base_url` points at a local server that ignores auth.
            base_url: Overrides the artifact's own `embed_base_url` (present when
                `embed_model` isn't actually hosted by OpenAI -- e.g. a local server).

        Returns:
            The routing `Decision` for this task.
        """
        import openai  # deferred: keeps this module import-light enough to drop into

        base_url = base_url or self.meta.get("embed_base_url")
        cl = openai.OpenAI(api_key=api_key or ("not-needed" if base_url else None),
                           base_url=base_url)
        e = cl.embeddings.create(model=self.meta["embed_model"],
                                 input=task_text[:8000]).data[0].embedding
        return self.route_embedding(np.array(e, dtype=float))


def main_predict(artifact_dir: str = "results") -> None:
    """CLI (`demo`): load the router artifact and route one demo task, printing the decision.

    Args:
        artifact_dir: Directory holding the router_v0.{json,npz} artifact.
    """
    import os as _os

    # Demo convenience only. The module itself never reads a .env, so it stays droppable
    # into a service that manages its own credentials.
    for cand in (pathlib.Path.cwd() / ".env.local",
                 pathlib.Path(__file__).resolve().parent.parent / ".env.local"):
        if cand.exists():
            for line in cand.read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    _os.environ.setdefault(k.strip(), v.strip())
            break

    r = Router(artifact_dir)
    logger.info(f"loaded: {len(r.arms)} arms, {r.emb.shape[0]} labelled tasks, "
                f"k={r.k} tau={r.tau} sim_floor={r.sim_floor}")
    logger.info(f"provenance: {r.meta['provenance']}")
    demo = ("Fix a race condition in the connection pool so concurrent checkouts "
            "cannot hand the same connection to two callers.")
    d = r.route(demo)
    logger.info(f"\nrouted -> {d.model} effort={d.effort}  p_solve={d.p_solve:.2f} "
                f"est ${d.est_cost_usd:.2f}  nearest_sim={d.nearest_sim:.3f} "
                f"off_dist={d.off_distribution} fallback={d.fallback_used}")
    logger.info(f"request_kwargs = {d.request_kwargs}")


# ============================================================================ route
class Matrix(BaseModel):
    """A dense (arm x problem) outcome+cost matrix with per-problem features."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    arms: list[str]
    qids: list[str]
    resolved: np.ndarray           # (n_arms, n_probs) bool -- all tests passed
    # Fraction of tests passed, in [0,1]. This is the PRIMARY objective, not `resolved`:
    # binary pass/fail overstates the model-tier gap ~4x (measured 9.2pp binary vs 2.3pp
    # graded on identical episodes), and it makes abandoning a 39/40 task look free.
    graded: np.ndarray             # (n_arms, n_probs) float
    cost: np.ndarray               # (n_arms, n_probs) float USD
    difficulty: list[str]
    group: list[str]               # contest id -- the CV grouping key
    emb: np.ndarray | None = None  # (n_probs, d) L2-normalised; set by embed()/attach_embeddings()

    @property
    def n(self) -> int:
        """Number of problems in this matrix."""
        return len(self.qids)

    def arm_idx(self, a: str) -> int:
        """Return the row index of arm `a`."""
        return self.arms.index(a)


def contest_of(qid: str) -> str:
    """abc387_b -> abc387. Problems in one contest share setters, so they group together."""
    m = re.match(r"([a-z]+\d+)", qid)
    return m.group(1) if m else qid


def load_matrix(min_coverage: float = 0.9) -> Matrix:
    """Load the LiveCodeBench outcome matrix from `results/episodes/*.json`.

    Args:
        min_coverage: Minimum fraction of problems an arm must have been run on to
            be kept; drops partially-swept arms that would otherwise shrink the
            complete-matrix intersection to a handful of problems.

    Returns:
        The dense (arm x problem) `Matrix`, restricted to problems every surviving
        arm was run on.
    """
    recs = []
    for f in glob.glob(str(ROOT / "results" / "episodes" / "*.json")):
        r = json.loads(pathlib.Path(f).read_text())
        if "arm" not in r:
            continue
        # Three ways an episode is MISSING DATA rather than a model failure: a provider
        # error (429/529), a harness fault, or a turn/token limit reached with nothing
        # written. Scoring any of them 0/N is what faked the "expensive frontier failure"
        # result. They are dropped, not zeroed.
        if r.get("harness_error") or r.get("stop") == "error":
            continue
        if r.get("outcome") == "harness_limit":
            continue
        if r.get("outcome") is None and not r.get("wrote_solution", True) \
                and r.get("stop") in ("max_turns", "max_tokens"):
            continue  # pre-`outcome` records: reconstruct the same judgement
        recs.append(r)
    qids = sorted({r["qid"] for r in recs})
    counts = collections.Counter(r["arm"] for r in recs)
    arms = sorted(a for a, c in counts.items() if c >= min_coverage * len(qids))
    cell = {(r["arm"], r["qid"]): r for r in recs if r["arm"] in arms}
    # Keep only problems with every surviving arm present, so cost/accuracy are comparable.
    keep = [q for q in qids if all((a, q) in cell for a in arms)]
    res = np.zeros((len(arms), len(keep)), dtype=bool)
    grd = np.zeros((len(arms), len(keep)), dtype=float)
    cost = np.zeros((len(arms), len(keep)), dtype=float)
    for i, a in enumerate(arms):
        for j, q in enumerate(keep):
            r = cell[(a, q)]
            res[i, j] = bool(r.get("resolved"))
            g = r.get("graded")
            if g is None:  # pre-`graded` records: recover it from passed/total
                g = (r["passed"] / r["total"]) if r.get("total") else 0.0
            grd[i, j] = g
            cost[i, j] = r.get("cost_usd") or 0.0
    diff = [cell[(arms[0], q)]["difficulty"] for q in keep]
    return Matrix(arms=arms, qids=keep, resolved=res, graded=grd, cost=cost,
                 difficulty=diff, group=[contest_of(q) for q in keep])


def attach_embeddings(m: Matrix, statements: dict[str, str]) -> None:
    """Embed problem statements with text-embedding-3-large, cached on disk.

    Args:
        m: The matrix to attach embeddings to; mutates `m.emb` in place.
        statements: Problem id -> statement text, for ids in `m.qids` missing from cache.
    """
    cache: dict[str, list[float]] = {}
    if EMB_CACHE.exists():
        cache = json.loads(EMB_CACHE.read_text())
    todo = [q for q in m.qids if q not in cache]
    if todo:
        import openai

        cl = openai.OpenAI()
        for i in range(0, len(todo), 64):
            chunk = todo[i : i + 64]
            out = cl.embeddings.create(
                model="text-embedding-3-large",
                input=[statements[q][:8000] for q in chunk])
            for q, d in zip(chunk, out.data):
                cache[q] = d.embedding
        EMB_CACHE.parent.mkdir(exist_ok=True)
        EMB_CACHE.write_text(json.dumps(cache))
    e = np.array([cache[q] for q in m.qids], dtype=float)
    m.emb = e / np.linalg.norm(e, axis=1, keepdims=True)


# --------------------------------------------------------------------------- policies
# A policy sees the TRAIN slice and one test problem, and returns an ordered list of arms to
# try. Cost is the sum over arms actually invoked; the episode resolves if any of them did.
Plan = list[int]


def p_always(idx: int):
    """Build a policy that always routes to arm `idx`."""
    return lambda m, tr, j: [idx]


def p_random(seed: int = 0):
    """Build a policy that routes to a uniformly random arm, seeded for reproducibility."""
    rng = random.Random(seed)
    return lambda m, tr, j: [rng.randrange(len(m.arms))]


def p_cascade(cheap: int, strong: int):
    """Build the FrugalGPT cascade policy: run `cheap`, escalate to `strong` on failure.

    No predictor, no training -- this is THE BAR a learned router must beat. Measured
    on LiveCodeBench the cascade sits at 1.81x cheaper than the strongest arm (parity
    oracle: 6.54x), so the cascade, not the priciest arm, is what "beating the baseline" means.

    Args:
        cheap: Index of the first (cheap) arm to try.
        strong: Index of the arm to escalate to on failure.

    Returns:
        A policy callable of `(matrix, train_indices, test_index) -> Plan`.
    """
    return lambda m, tr, j: [cheap, strong]


def _knn_probs(m: Matrix, tr: np.ndarray, j: int, k: int) -> np.ndarray:
    """Compute each arm's P(resolve) for problem j from its k nearest TRAIN neighbours.

    Args:
        m: The full matrix (train + test).
        tr: Indices of problems eligible as neighbours (the training fold).
        j: Index of the problem to predict for.
        k: Number of nearest neighbours to weight by similarity.

    Returns:
        Per-arm predicted probability of resolving problem `j`.
    """
    sims = m.emb[tr] @ m.emb[j]
    k = min(k, len(tr))
    nn = tr[np.argsort(-sims)[:k]]
    w = np.clip(m.emb[nn] @ m.emb[j], 0, None) + 1e-6
    return (m.resolved[:, nn] * w).sum(axis=1) / w.sum()


def p_knn_threshold(k: int = 12, tau: float = 0.6):
    """Build a policy: cheapest arm whose predicted P(resolve) clears tau.

    Args:
        k: Number of nearest neighbours to weight by similarity.
        tau: Minimum predicted P(resolve) required to pick an arm.

    Returns:
        A policy callable of `(matrix, train_indices, test_index) -> Plan`; falls
        back to the best-observed arm if nothing clears `tau`.
    """
    def go(m: Matrix, tr: np.ndarray, j: int) -> Plan:
        """Return the cheapest arm clearing `tau`, or the best-observed arm otherwise."""
        p = _knn_probs(m, tr, j, k)
        med = np.median(m.cost[:, tr], axis=1)
        order = np.argsort(med)
        for i in order:
            if p[i] >= tau:
                return [int(i)]
        return [int(np.argmax(m.resolved[:, tr].mean(axis=1)))]
    return go


def p_knn_two_sided(k: int = 12, tau: float = 0.6, hopeless: float = 0.15):
    """Build a policy adding the abandon-down rule: if no arm looks likely, spend the least.

    Measured on the free SWE-bench matrix this is worth 1.40x -> 1.97x, because a naive
    `argmin cost s.t. p>=tau` router sends the hopeless stratum to the priciest arm.

    Args:
        k: Number of nearest neighbours to weight by similarity.
        tau: Minimum predicted P(resolve) required to pick an arm normally.
        hopeless: If the best predicted P(resolve) is below this, route to the
            cheapest arm instead of escalating.

    Returns:
        A policy callable of `(matrix, train_indices, test_index) -> Plan`.
    """
    def go(m: Matrix, tr: np.ndarray, j: int) -> Plan:
        """Return the cheapest arm, the cheapest clearing `tau`, or the best-observed arm."""
        p = _knn_probs(m, tr, j, k)
        med = np.median(m.cost[:, tr], axis=1)
        order = np.argsort(med)
        if p.max() < hopeless:
            return [int(order[0])]
        for i in order:
            if p[i] >= tau:
                return [int(i)]
        return [int(np.argmax(m.resolved[:, tr].mean(axis=1)))]
    return go


def p_knn_cascade(k: int = 12, tau: float = 0.6):
    """Build a policy: predict with kNN, then still escalate on failure.

    The hybrid of `p_knn_threshold` and `p_cascade`.

    Args:
        k: Number of nearest neighbours to weight by similarity.
        tau: Minimum predicted P(resolve) required to pick an arm without also
            trying the best-observed arm as a second try.

    Returns:
        A policy callable of `(matrix, train_indices, test_index) -> Plan`.
    """
    def go(m: Matrix, tr: np.ndarray, j: int) -> Plan:
        """Return the cheapest arm clearing `tau` (plus the best arm as a fallback try)."""
        p = _knn_probs(m, tr, j, k)
        med = np.median(m.cost[:, tr], axis=1)
        order = np.argsort(med)
        best = int(np.argmax(m.resolved[:, tr].mean(axis=1)))
        for i in order:
            if p[i] >= tau:
                return [int(i)] if int(i) == best else [int(i), best]
        return [best]
    return go


# --------------------------------------------------------------------------- evaluation
def grouped_folds(m: Matrix, n_folds: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Split problems into folds by group (e.g. contest), so no group spans folds.

    Args:
        m: The matrix whose `group` labels define the split.
        n_folds: Number of folds to produce.
        seed: Shuffle seed, for reproducibility.

    Returns:
        One array of problem indices per fold.
    """
    groups = sorted(set(m.group))
    random.Random(seed).shuffle(groups)
    buckets: list[list[str]] = [[] for _ in range(n_folds)]
    for i, g in enumerate(groups):
        buckets[i % n_folds].append(g)
    return [np.array([j for j, g in enumerate(m.group) if g in set(b)]) for b in buckets]


def run_policy(m: Matrix, policy, folds: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Run a policy under grouped cross-validation.

    Args:
        m: The matrix to evaluate on.
        policy: A callable of `(matrix, train_indices, test_index) -> Plan` (an
            ordered list of arm indices to try, stopping at the first resolve).
        folds: Problem-index groups from `grouped_folds`; each fold is held out in turn.

    Returns:
        Per-problem `(resolved, cost)` arrays, each problem judged out-of-fold.
    """
    res = np.zeros(m.n, dtype=bool)
    cost = np.zeros(m.n, dtype=float)
    for te in folds:
        tr = np.array([j for j in range(m.n) if j not in set(te.tolist())])
        for j in te:
            plan = policy(m, tr, int(j))
            got = False
            for i in plan:
                cost[j] += m.cost[i, j]
                if m.resolved[i, j]:
                    got = True
                    break
            res[j] = got
    return res, cost


def oracle(m: Matrix, parity_to: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute the cheapest-arm-that-solves-it oracle, optionally at matched accuracy.

    Args:
        m: The matrix to evaluate on.
        parity_to: If given, an arm index; problems that arm fails are scored as
            failed (at that arm's cost) rather than solved by a cheaper alternative,
            so the oracle's accuracy exactly matches that arm's instead of exceeding it.

    Returns:
        Per-problem `(resolved, cost)` arrays.
    """
    res = np.zeros(m.n, dtype=bool)
    cost = np.zeros(m.n, dtype=float)
    for j in range(m.n):
        winners = np.where(m.resolved[:, j])[0]
        if parity_to is not None and not m.resolved[parity_to, j]:
            res[j], cost[j] = False, m.cost[parity_to, j]
        elif len(winners):
            res[j] = True
            cost[j] = m.cost[winners, j].min()
        else:
            res[j], cost[j] = False, m.cost[:, j].min()
    return res, cost


def boot_ratio(base_cost: np.ndarray, pol_cost: np.ndarray, groups: list[str],
               n: int = 10000, seed: int = 0) -> tuple[float, float]:
    """Cluster-bootstrap (resample groups) a 95% CI on the cost ratio base/policy.

    Args:
        base_cost: Per-problem cost under the baseline policy.
        pol_cost: Per-problem cost under the policy being compared.
        groups: Per-problem group label (e.g. contest); resampling is by group, not
            by problem, since problems in one group are not independent.
        n: Number of bootstrap resamples.
        seed: RNG seed, for reproducibility.

    Returns:
        The (2.5th percentile, 97.5th percentile) of the resampled cost ratio.
    """
    rng = np.random.default_rng(seed)
    gs = sorted(set(groups))
    idx = {g: np.array([i for i, x in enumerate(groups) if x == g]) for g in gs}
    out = []
    for _ in range(n):
        pick = rng.choice(len(gs), len(gs), replace=True)
        sel = np.concatenate([idx[gs[p]] for p in pick])
        d = pol_cost[sel].sum()
        if d > 0:
            out.append(base_cost[sel].sum() / d)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def mcnemar(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the two-sided exact McNemar p-value on paired resolve outcomes.

    Args:
        a: Baseline per-problem resolved outcomes.
        b: Comparison per-problem resolved outcomes, paired with `a`.

    Returns:
        The two-sided exact McNemar p-value for whether `a` and `b` differ.
    """
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return min(1.0, p)


if __name__ == "__main__":
    import fire

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    # Root at INFO unmutes httpx's per-request "HTTP Request: ..." records (the
    # OpenAI/Anthropic SDKs' HTTP layer); print never showed them, so gate them out.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    fire.Fire({"arms": main_pricing, "demo": main_predict})
