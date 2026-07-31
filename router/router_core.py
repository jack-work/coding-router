"""The deployable router: price table, kNN inference, and the policies/race harness.

Three layers that belong together:
  * pricing  -- canonical price table and `Arm` (provider + model + reasoning config).
  * predict  -- `Router`/`Decision`: load an exported artifact, route a task to an arm.
  * route    -- `Matrix`, CV policies (cascade/kNN/oracle), and the stats used to prove
                a learned router beats a plain FrugalGPT-style cascade.

======================================================================== pricing
Prices fetched live 2026-07-28 from:
  OpenAI    https://developers.openai.com/api/docs/pricing (+ per-model pages)
  Anthropic https://platform.claude.com/docs/en/about-claude/pricing
USD per 1M tokens. Reasoning/thinking tokens bill as OUTPUT on both providers.

Two deliberate conservatism choices, so the headline cost-saving claim cannot be
accused of leaning on a temporary discount or an optimistic cache assumption:
  * Sonnet 5's $2/$10 is INTRODUCTORY and expires 2026-08-31. `STANDARD` uses the
    post-intro $3/$15. Use `INTRO` only to reconcile against a live invoice.
  * Cache-write premiums are charged at the 5-minute rate (1.25x input).

======================================================================== predict
Standalone coding-task router. Load an exported artifact, get an arm back.

Deliberately dependency-light so it can be dropped into another service: numpy plus
whatever already calls the OpenAI embeddings endpoint. No sklearn, no torch, no repo imports.

kNN has no fitted weights, so the "model" IS the lookup table: task embeddings, each arm's
outcome on each of those tasks, and each arm's median cost. Routing embeds the incoming task,
finds its nearest labelled neighbours, and picks the cheapest arm those neighbours say will
probably solve it.

SCOPE -- read before trusting a decision. Fit on 110 long-horizon SWE tasks from DeepSWE v1.1
(median 61 agent steps, ~15 min/episode). Measured there at 2.15x cheaper than always using
the strongest arm, with the accuracy delta's 95% CI containing zero. It has NOT been validated
on short interactive tasks, on non-Python repos beyond DeepSWE's mix, or on any held-out
benchmark. `route()` returns its own confidence and neighbour distance so a caller can refuse
to trust an off-distribution decision rather than silently getting a bad arm.

======================================================================== route
The router: policies, and an honest race against the baselines that must be beaten.

Decision rule this file exists to settle: a learned router is only worth shipping if it beats
a plain FrugalGPT-style cascade at matched accuracy. On our measured LiveCodeBench matrix the
cascade is at 1.81x and the parity oracle at 6.54x, so the cascade -- not the most expensive
arm -- is the bar.

Everything is evaluated under CONTEST-grouped cross-validation. Problems from one AtCoder
contest (abc387_a, abc387_b, ...) share setters and style, so a random split leaks: neighbours
from the same contest would sit in both train and test and flatter KNN. Grouping by contest
removes that.
"""
from __future__ import annotations

import collections
import dataclasses
import glob
import json
import math
import pathlib
import random
import re
from dataclasses import dataclass

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
EMB_CACHE = ROOT / "results" / "lcb_embeddings.json"

PRICE_FETCH_DATE = "2026-07-28"
ARTIFACT_JSON = "router_v0.json"
ARTIFACT_NPZ = "router_v0.npz"
EMBED_MODEL = "text-embedding-3-large"


# ============================================================================ pricing
@dataclass(frozen=True)
class Price:
    inp: float          # uncached input, $/1M
    cache_read: float   # cached input read, $/1M
    out: float          # output (incl. reasoning/thinking), $/1M
    cache_write: float  # 5-min cache write, $/1M


# provider -> model -> Price
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
        "claude-haiku-4-5": Price(1.00, 0.10,  5.00, 1.25),
        "claude-sonnet-5":  Price(3.00, 0.30, 15.00, 3.75),   # post-intro
        "claude-opus-4-8":  Price(5.00, 0.50, 25.00, 6.25),
        "claude-fable-5":   Price(10.00, 1.00, 50.00, 12.50),
    },
}

# Sonnet 5 introductory rate, active through 2026-08-31 only.
INTRO_OVERRIDES = {("anthropic", "claude-sonnet-5"): Price(2.00, 0.20, 10.00, 2.50)}

# Anthropic models that reject output_config.effort with HTTP 400 (measured).
NO_EFFORT_PARAM = {"claude-haiku-4-5"}
# OpenAI effort values accepted today; 'minimal' was removed (measured: 400).
OPENAI_EFFORTS = ("none", "low", "medium", "high", "xhigh")
ANTHROPIC_EFFORTS = ("low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class Arm:
    """One routing candidate: a model plus its reasoning configuration."""

    provider: str
    model: str
    effort: str | None = None
    thinking: str = "adaptive"  # anthropic: adaptive|budget|off|omit ; openai: unused

    @property
    def id(self) -> str:
        return f"{self.model}@{self.effort or self.thinking}"

    @property
    def price(self) -> Price:
        return STANDARD[self.provider][self.model]

    def cost(self, *, inp: int = 0, cache_read: int = 0, cache_write: int = 0,
             out: int = 0, intro: bool = False) -> float:
        """USD for one request's measured token usage."""
        p = INTRO_OVERRIDES.get((self.provider, self.model), self.price) if intro else self.price
        return (inp * p.inp + cache_read * p.cache_read
                + cache_write * p.cache_write + out * p.out) / 1_000_000

    def request_kwargs(self) -> dict:
        """Provider-native params for this arm, honouring the measured constraints."""
        if self.provider == "anthropic":
            kw: dict = {"model": self.model}
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
    """The 47 arms confirmed live + tool-calling on 2026-07-28."""
    arms = [
        Arm("anthropic", "claude-haiku-4-5", None, "off"),
        Arm("anthropic", "claude-haiku-4-5", None, "budget"),
    ]
    for m in ("claude-sonnet-5", "claude-opus-4-8"):
        arms += [Arm("anthropic", m, e, "adaptive") for e in ANTHROPIC_EFFORTS]
    arms += [Arm("anthropic", "claude-fable-5", e, "omit") for e in ANTHROPIC_EFFORTS]
    for m in ("gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4", "gpt-5.5", "gpt-5.3-codex", "gpt-5.6-sol"):
        arms += [Arm("openai", m, e) for e in OPENAI_EFFORTS]
    return arms


def cost_span() -> tuple[float, float]:
    """(cheapest, priciest) blended $/1M at a 1:1 in:out mix, for sanity checks."""
    blended = [(p.inp + p.out) / 2 for prov in STANDARD.values() for p in prov.values()]
    return min(blended), max(blended)


def main_pricing() -> None:
    arms = all_validated_arms()
    lo, hi = cost_span()
    print(f"{len(arms)} validated arms | blended $/1M span: {lo:.2f} -> {hi:.2f} ({hi/lo:.0f}x)")
    # Cost of a representative agentic episode: 40 turns, heavy cache reuse.
    ep = dict(inp=30_000, cache_read=1_200_000, cache_write=40_000, out=25_000)
    rows = sorted({a.model: a for a in arms}.values(), key=lambda a: a.cost(**ep))
    print(f"\nper-episode cost @ {ep}:")
    for a in rows:
        print(f"  {a.model:20s} ${a.cost(**ep):6.3f}")
    c = rows[0].cost(**ep)
    print(f"\nspread cheapest->priciest per episode: {rows[-1].cost(**ep)/c:.1f}x")


# ============================================================================ predict
@dataclasses.dataclass
class Decision:
    model: str                  # provider model id, e.g. "gpt-5.6-sol"
    effort: str | None          # reasoning effort / thinking level
    arm_id: str                 # internal arm key
    p_solve: float              # predicted probability this arm resolves the task
    est_cost_usd: float         # median observed cost of this arm per task
    nearest_sim: float          # cosine similarity to the closest labelled task
    off_distribution: bool      # True when no neighbour is close enough to trust
    fallback_used: bool         # True when no arm cleared the threshold
    request_kwargs: dict        # ready to splat into the provider SDK call

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


class Router:
    def __init__(self, artifact_dir: str | pathlib.Path):
        p = pathlib.Path(artifact_dir)
        meta = json.loads((p / ARTIFACT_JSON).read_text())
        arr = np.load(p / ARTIFACT_NPZ)
        self.meta = meta
        self.arms: list[str] = meta["arms"]
        self.arm_spec: dict = meta["arm_spec"]
        self.k: int = meta["k"]
        self.tau: float = meta["tau"]
        self.sim_floor: float = meta["sim_floor"]
        self.emb: np.ndarray = arr["emb"]            # (n_tasks, dim) L2-normalised
        self.resolved: np.ndarray = arr["resolved"]  # (n_arms, n_tasks) bool
        self.med_cost: np.ndarray = arr["med_cost"]  # (n_arms,)
        self.fallback: int = int(meta["fallback_arm_index"])
        self._order = np.argsort(self.med_cost)      # cheapest arm first

    # ---------------------------------------------------------------- internals
    def _probs(self, v: np.ndarray) -> tuple[np.ndarray, float]:
        sims = self.emb @ v
        k = min(self.k, len(sims))
        nn = np.argsort(-sims)[:k]
        w = np.clip(sims[nn], 0, None) + 1e-6
        return (self.resolved[:, nn] * w).sum(axis=1) / w.sum(), float(sims[nn[0]])

    def route_embedding(self, v: np.ndarray) -> Decision:
        """Route from a pre-computed, L2-normalised embedding of the task text."""
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
        return Decision(model=spec["model"], effort=spec.get("effort"),
                        arm_id=self.arms[pick], p_solve=float(p[pick]),
                        est_cost_usd=float(self.med_cost[pick]), nearest_sim=nearest,
                        off_distribution=off, fallback_used=fb,
                        request_kwargs=spec["request_kwargs"])

    def route(self, task_text: str, *, api_key: str | None = None) -> Decision:
        """Embed `task_text` with the artifact's embedding model, then route."""
        import openai

        cl = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
        e = cl.embeddings.create(model=self.meta["embed_model"],
                                 input=task_text[:8000]).data[0].embedding
        return self.route_embedding(np.array(e, dtype=float))


def main_predict(artifact_dir: str) -> None:
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
    print(f"loaded: {len(r.arms)} arms, {r.emb.shape[0]} labelled tasks, "
          f"k={r.k} tau={r.tau} sim_floor={r.sim_floor}")
    print(f"provenance: {r.meta['provenance']}")
    demo = ("Fix a race condition in the connection pool so concurrent checkouts "
            "cannot hand the same connection to two callers.")
    d = r.route(demo)
    print(f"\nrouted -> {d.model} effort={d.effort}  p_solve={d.p_solve:.2f} "
          f"est ${d.est_cost_usd:.2f}  nearest_sim={d.nearest_sim:.3f} "
          f"off_dist={d.off_distribution} fallback={d.fallback_used}")
    print(f"request_kwargs = {d.request_kwargs}")


# ============================================================================ route
@dataclasses.dataclass
class Matrix:
    """A dense (arm x problem) outcome+cost matrix with per-problem features."""

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
    emb: np.ndarray | None = None  # (n_probs, d) L2-normalised

    @property
    def n(self) -> int:
        return len(self.qids)

    def arm_idx(self, a: str) -> int:
        return self.arms.index(a)


def contest_of(qid: str) -> str:
    """abc387_b -> abc387. Problems in one contest share setters, so they group together."""
    m = re.match(r"([a-z]+\d+)", qid)
    return m.group(1) if m else qid


def load_matrix(min_coverage: float = 0.9) -> Matrix:
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
    return Matrix(arms, keep, res, grd, cost, diff, [contest_of(q) for q in keep])


def attach_embeddings(m: Matrix, statements: dict[str, str]) -> None:
    """Embed problem statements with text-embedding-3-large, cached on disk."""
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
    return lambda m, tr, j: [idx]


def p_random(seed: int = 0):
    rng = random.Random(seed)
    return lambda m, tr, j: [rng.randrange(len(m.arms))]


def p_cascade(cheap: int, strong: int):
    """FrugalGPT: run cheap, escalate on failure. No predictor, no training. THE BAR."""
    return lambda m, tr, j: [cheap, strong]


def _knn_probs(m: Matrix, tr: np.ndarray, j: int, k: int) -> np.ndarray:
    """Per-arm P(resolve) for problem j, from its k nearest TRAIN neighbours."""
    sims = m.emb[tr] @ m.emb[j]
    k = min(k, len(tr))
    nn = tr[np.argsort(-sims)[:k]]
    w = np.clip(m.emb[nn] @ m.emb[j], 0, None) + 1e-6
    return (m.resolved[:, nn] * w).sum(axis=1) / w.sum()


def p_knn_threshold(k: int = 12, tau: float = 0.6):
    """Cheapest arm whose predicted P(resolve) clears tau; fall back to the best arm."""
    def go(m: Matrix, tr: np.ndarray, j: int) -> Plan:
        p = _knn_probs(m, tr, j, k)
        med = np.median(m.cost[:, tr], axis=1)
        order = np.argsort(med)
        for i in order:
            if p[i] >= tau:
                return [int(i)]
        return [int(np.argmax(m.resolved[:, tr].mean(axis=1)))]
    return go


def p_knn_two_sided(k: int = 12, tau: float = 0.6, hopeless: float = 0.15):
    """Adds the abandon-down rule: if NO arm is predicted to solve it, spend the least.

    Measured on the free SWE-bench matrix this is worth 1.40x -> 1.97x, because a naive
    `argmin cost s.t. p>=tau` router sends the hopeless stratum to the priciest arm.
    """
    def go(m: Matrix, tr: np.ndarray, j: int) -> Plan:
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
    """Predict, then still escalate on failure -- the hybrid of KNN and cascade."""
    def go(m: Matrix, tr: np.ndarray, j: int) -> Plan:
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
    groups = sorted(set(m.group))
    random.Random(seed).shuffle(groups)
    buckets: list[list[str]] = [[] for _ in range(n_folds)]
    for i, g in enumerate(groups):
        buckets[i % n_folds].append(g)
    return [np.array([j for j, g in enumerate(m.group) if g in set(b)]) for b in buckets]


def run_policy(m: Matrix, policy, folds: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Returns per-problem (resolved, cost) under CV -- each problem judged out-of-fold."""
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
    """Cluster bootstrap (resample contests) on the cost ratio base/policy."""
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
    """Two-sided exact McNemar p-value on paired resolve outcomes."""
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return min(1.0, p)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = ap.add_subparsers(dest="command", required=True)

    subparsers.add_parser("arms").set_defaults(func=lambda ns: main_pricing())

    sp = subparsers.add_parser("demo")
    sp.add_argument("artifact_dir", nargs="?", default="results")
    sp.set_defaults(func=lambda ns: main_predict(ns.artifact_dir))

    ns = ap.parse_args()
    ns.func(ns)
