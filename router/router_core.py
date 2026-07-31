"""The deployable router: the canonical price table (`Price`/`Arm`) and artifact
inference (`Router`/`Decision`: load an exported artifact, route a task to an arm).

The research half that built and validated this — dataset loaders, the measurement
harness, CV policies, benchmarks — lives in world-model-optimizer's
`packages/router-lab`, which depends on this repo.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import sys
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

ROOT = pathlib.Path(__file__).resolve().parent.parent

PRICE_FETCH_DATE = "2026-07-28"
ARTIFACT_JSON = "router.json"
ARTIFACT_NPZ = "router.npz"

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

def main_predict(artifact_dir: str = "results") -> None:
    """CLI (`demo`): load the router artifact and route one demo task, printing the decision.

    Args:
        artifact_dir: Directory holding the router.{json,npz} artifact (auto-downloaded
            from Hugging Face if missing).
    """
    # Deferred: serve owns env loading, artifact download, and the local embedding
    # backend; importing it here keeps this module import-light for library users
    # (who call route_embedding with their own vectors and never pay these imports).
    from router.serve import ensure_artifact, load_env, load_local_embedder

    load_env()
    ensure_artifact(pathlib.Path(artifact_dir))
    r = Router(artifact_dir)
    logger.info(f"loaded: {len(r.arms)} arms, {r.emb.shape[0]} labelled tasks, "
                f"k={r.k} tau={r.tau} sim_floor={r.sim_floor}")
    logger.info(f"provenance: {r.meta['provenance']}")
    demo = ("Fix a race condition in the connection pool so concurrent checkouts "
            "cannot hand the same connection to two callers.")
    d = r.route_embedding(load_local_embedder()(demo))
    logger.info(f"\nrouted -> {d.model} effort={d.effort}  p_solve={d.p_solve:.2f} "
                f"est ${d.est_cost_usd:.2f}  nearest_sim={d.nearest_sim:.3f} "
                f"off_dist={d.off_distribution} fallback={d.fallback_used}")
    logger.info(f"request_kwargs = {d.request_kwargs}")


if __name__ == "__main__":
    import fire

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    # Root at INFO unmutes httpx's per-request "HTTP Request: ..." records (the
    # OpenAI/Anthropic SDKs' HTTP layer); print never showed them, so gate them out.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    fire.Fire({"demo": main_predict})
