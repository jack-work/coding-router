"""EXP-008: on-policy PER-TURN routing on LiveCodeBench with the models in the loop.

The router picks an arm for EVERY agent turn, mid-episode switches included. To make
that provider-safe, each turn is a STATELESS call: the conversation so far is kept as
canonical text (tool calls and their outputs flattened, mini-swe-agent-style), and the
chosen arm gets native tool *emission* for its one turn only — no provider-native
history ever crosses turns, so switching gpt->claude->gpt is trivial and every arm
sees byte-identical context.

Policy: linear softmax over the 7 LCB-matrix arms on a small state vector
(train-matrix kNN p_solve priors + live episode signals). Trained with REINFORCE on
sampled episodes, task-mean baseline (M rollouts per task, GRPO-style), reward =
graded - LAM * episode_cost. Contest-grouped split: rollouts on train contests, eval
argmax on held-out contests vs best-static-arm and turn0-frozen (per-task) controls.

Run from the repo root (needs .env keys + E2B):
  uv run python rl/perturn.py smoke          # 2 cheap episodes, no training
  uv run python rl/perturn.py train          # full run (budget-capped)
  uv run python rl/perturn.py eval POLICY.json
Artifacts: results/rl/perturn/{metrics.jsonl,policy_it*.json,episodes/*.json}
"""
from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import json
import pathlib
import re
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from router import harness as sandbox  # noqa: E402
from router import router_core as route  # noqa: E402
from router.harness import load_env  # noqa: E402
from router.router_core import Arm  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "rl" / "perturn"
EP_DIR = OUT / "episodes"

# Handicapped episodes (EXP-011) burn 4 weak turns before recovering; the default
# 600s sandbox end-of-life killed one mid-grade and took the whole run down.
sandbox.SANDBOX_TIMEOUT_S = 1800

ARMS = [Arm("anthropic", "claude-haiku-4-5", None, "budget"),
        Arm("anthropic", "claude-opus-4-8", "medium", "adaptive"),
        Arm("openai", "gpt-5.3-codex", "high"),
        Arm("openai", "gpt-5.4-mini", "medium"),
        Arm("openai", "gpt-5.4-nano", "high"),
        Arm("openai", "gpt-5.4-nano", "low"),
        Arm("openai", "gpt-5.4", "medium")]
ARM_IDS = [a.id for a in ARMS]

LAM = 8.0                # reward = graded - LAM * cost_usd (LCB costs are cents)
MAX_TURNS = 25
MAX_TOKENS = 32_000      # generous on purpose: reasoning depth must not decide outcomes
BUDGET_USD = 40.0        # hard abort for the whole run
EVAL_CONTEST_FRAC = 0.3
SEED = 0

_spend_lock = threading.Lock()
_spend = {"usd": 0.0}

SYSTEM = (
    "You are a competitive programming agent working in a scratch directory. "
    "Write your solution to solution.py. It must read from stdin and write to stdout. "
    "Use the bash tool to create the file and to run `python3 check.py`, which runs the "
    "PUBLIC sample tests and prints results. Iterate until all public tests pass, then "
    "reply with exactly DONE. Earlier turns may have been taken by a different "
    "assistant; the transcript of commands and outputs above is accurate."
)

BASH_SCHEMA = {"type": "object",
               "properties": {"command": {"type": "string", "description": "command to run"}},
               "required": ["command"]}


# --------------------------------------------------------------------- one stateless turn
def flatten(history: list[dict]) -> str:
    parts = []
    for h in history:
        if h["kind"] == "assistant":
            parts.append(f"[assistant said]\n{h['text']}")
        elif h["kind"] == "tool":
            parts.append(f"[ran bash] {h['cmd']}\n[output]\n{h['out']}")
    return "\n\n".join(parts)


def one_turn(arm: Arm, task: str, history: list[dict]) -> tuple[str, list[str], float, str]:
    """One stateless model call. Returns (text, bash_cmds, cost_usd, stop_kind)."""
    ctx = task if not history else (
        f"{task}\n\n=== transcript so far ===\n{flatten(history)}\n=== end transcript ===\n"
        "Continue from here. Run commands or reply DONE when all public tests pass.")
    kw = arm.request_kwargs()
    if arm.provider == "anthropic":
        import anthropic

        cl = anthropic.Anthropic(max_retries=3, timeout=600.0)
        r = cl.messages.create(
            max_tokens=MAX_TOKENS,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            tools=[{"name": "bash", "description": "Run a bash command in the scratch dir.",
                    "input_schema": BASH_SCHEMA}],
            messages=[{"role": "user", "content": ctx}], **kw)
        u = r.usage
        cost = arm.cost(inp=u.input_tokens, out=u.output_tokens,
                        cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
                        cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0)
        text = " ".join(b.text for b in r.content if b.type == "text")
        cmds = [b.input.get("command", "") for b in r.content if b.type == "tool_use"]
        stop = r.stop_reason or "?"
    else:
        import openai

        cl = openai.OpenAI(max_retries=3, timeout=600.0)
        r = cl.responses.create(
            instructions=SYSTEM, input=ctx, max_output_tokens=MAX_TOKENS,
            tools=[{"type": "function", "name": "bash",
                    "description": "Run a bash command in the scratch dir.",
                    "parameters": BASH_SCHEMA}], **kw)
        u = r.usage
        det = getattr(u, "input_tokens_details", None)
        cached = getattr(det, "cached_tokens", 0) or 0
        cost = arm.cost(inp=max(0, u.input_tokens - cached), cache_read=cached,
                        out=u.output_tokens)
        text = r.output_text or ""
        cmds = []
        for o in r.output:
            if getattr(o, "type", "") == "function_call":
                try:
                    cmds.append(json.loads(o.arguments or "{}").get("command", ""))
                except json.JSONDecodeError:
                    pass
        stop = r.status or "?"
    with _spend_lock:
        _spend["usd"] += cost
    return text, [c for c in cmds if c], cost, stop


# --------------------------------------------------------------------------- features
@dataclasses.dataclass
class Feats:
    """State featurizer. Task priors come from TRAIN contests only (no eval leakage).

    sticky=True appends a one-hot of the PREVIOUS turn's arm, letting the policy learn
    switch economics (a switch pays the candidate arm's cold prefill; staying can reuse
    provider cache) — the divergence-cost point from the exp/per-turn-routing lane.
    """

    prior: np.ndarray            # (n_arms,) kNN p_solve for this task
    sticky: bool = False

    def vec(self, turn: int, cost_so_far: float, passed_frac: float,
            wrote: bool, last_ok: bool, prev_arm: int = -1) -> np.ndarray:
        base = np.concatenate([
            self.prior,
            [turn / MAX_TURNS, np.log10(cost_so_far + 1e-4) / 4.0 + 1.0,
             passed_frac, float(wrote), float(last_ok), 1.0]])
        if not self.sticky:
            return base
        onehot = np.zeros(len(ARMS))
        if prev_arm >= 0:
            onehot[prev_arm] = 1.0
        return np.concatenate([base, onehot])


DIM = len(ARMS) + 6


class Policy:
    def __init__(self, W: np.ndarray | None = None, dim: int = DIM):
        self.W = W if W is not None else np.zeros((len(ARMS), dim))

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self.W @ x

    def sample(self, x: np.ndarray, rng: np.random.Generator) -> tuple[int, np.ndarray]:
        z = self.logits(x)
        p = np.exp(z - z.max())
        p = 0.97 * p / p.sum() + 0.03 / len(p)   # exploration floor
        return int(rng.choice(len(p), p=p)), p

    def save(self, path: pathlib.Path) -> None:
        path.write_text(json.dumps({"W": self.W.tolist(), "arms": ARM_IDS}))

    @classmethod
    def load(cls, path: pathlib.Path) -> Policy:
        d = json.loads(path.read_text())
        assert d["arms"] == ARM_IDS
        return cls(np.array(d["W"]))


class Turn0Policy:
    """Per-task control: the wrapped policy's turn-0 argmax, frozen for the episode."""

    def __init__(self, inner: Policy):
        self.inner, self.choice = inner, None

    def logits(self, x: np.ndarray) -> np.ndarray:
        if self.choice is None:
            self.choice = int(np.argmax(self.inner.logits(x[: self.inner.W.shape[1]])))
        z = np.full(len(ARMS), -1e3)
        z[self.choice] = 1e3
        return z


class StaticPolicy:
    def __init__(self, arm_idx: int):
        self.arm_idx = arm_idx

    def logits(self, x: np.ndarray) -> np.ndarray:
        z = np.full(len(ARMS), -1e3)
        z[self.arm_idx] = 1e3
        return z


LADDER_ORDER = [5, 4, 3, 6, 2, 1, 0]  # ARMS indices sorted by LCB median cost, ascending


class LadderPolicy:
    """Hand-built per-turn cascade: start cheap, escalate while public tests fail."""

    def logits(self, x: np.ndarray) -> np.ndarray:
        turn = x[len(ARMS)] * MAX_TURNS
        passed = x[len(ARMS) + 2]
        rung = 0 if turn < 4 else (2 if turn < 8 else (4 if turn < 12 else 6))
        if passed >= 1.0:
            rung = min(rung, 2)
        z = np.full(len(ARMS), -1e3)
        z[LADDER_ORDER[rung]] = 1e3
        return z


# --------------------------------------------------------------------------- episode
def run_episode(prob, feats: Feats, policy: Policy, rng: np.random.Generator,
                greedy: bool, tag: str, handicap: tuple[int, int] | None = None) -> dict:
    """handicap=(arm_idx, k): FORCE that arm for the first k turns (EXP-011: injected
    difficulty so mid-episode escalation has value). Forced turns are excluded from
    the policy gradient."""
    key = f"{prob.qid}__{tag}"
    dest = EP_DIR / f"{key}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    if _spend["usd"] > BUDGET_USD:
        raise RuntimeError(f"budget cap ${BUDGET_USD} reached")

    t0 = time.time()
    history: list[dict] = []
    turns, cost = [], 0.0
    passed_frac, wrote, last_ok, prev_arm = 0.0, False, True, -1
    task = sandbox.task_prompt(prob)
    with sandbox.SandboxSession({"lane": "coding-router-perturn", "qid": prob.qid}) as s:
        s.write("public_tests.json", json.dumps(prob.public_tests))
        s.write("check.py", sandbox.CHECKER)
        for turn in range(MAX_TURNS):
            x = feats.vec(turn, cost, passed_frac, wrote, last_ok, prev_arm)
            forced = bool(handicap and turn < handicap[1])
            if forced:
                a = handicap[0]
            elif greedy:
                a = int(np.argmax(policy.logits(x)))
            else:
                a, _ = policy.sample(x, rng)
            prev_arm = a
            try:
                text, cmds, c, stop = one_turn(ARMS[a], task, history)
            except Exception as e:  # noqa: BLE001 — arm/API failure ends the episode
                turns.append({"arm": a, "x": x.tolist(), "error": str(e)[:300]})
                break
            cost += c
            turns.append({"arm": a, "x": x.tolist(), "cost": c, "n_cmds": len(cmds),
                          "stop": stop, "forced": forced})
            history.append({"kind": "assistant", "text": text[:3000]})
            if not cmds:
                break  # model stopped (DONE or gave up)
            for cmd in cmds[:4]:
                try:
                    rc, out, err = s.run(cmd, timeout=90.0)
                    last_ok = rc == 0
                    obs = f"exit={rc}\n{out[-4000:]}\n{err[-1000:]}"
                except Exception as e:  # noqa: BLE001
                    obs, last_ok = f"sandbox error: {e}"[:400], False
                history.append({"kind": "tool", "cmd": cmd[:500], "out": obs})
            code = s.read("solution.py") or ""
            wrote = bool(code.strip())
            if wrote:
                _, chk, _ = s.run("python3 check.py", timeout=120.0)
                m = re.search(r"(\d+)/(\d+) public tests passed", chk)
                if m:
                    passed_frac = int(m.group(1)) / max(1, int(m.group(2)))
        code = s.read("solution.py") or ""
        passed, total = s.grade(prob.private_tests) if code.strip() else (0, prob.n_tests)

    rec = {"qid": prob.qid, "difficulty": prob.difficulty, "tag": tag,
           "graded": passed / total if total else 0.0,
           "resolved": bool(total and passed == total),
           "cost_usd": round(cost, 6), "n_turns": len(turns), "turns": turns,
           "reward": (passed / total if total else 0.0) - LAM * cost,
           "wall_s": round(time.time() - t0, 1)}
    EP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.replace(dest)
    return rec


# --------------------------------------------------------------------------- data plumbing
def contest_split() -> tuple[list, list]:
    m = route.load_matrix()
    probs = {p.qid: p for p in sandbox.load()}
    contests = sorted(set(m.group))
    rng = np.random.default_rng(SEED)
    rng.shuffle(contests)
    n_ev = max(2, int(len(contests) * EVAL_CONTEST_FRAC))
    ev = set(contests[:n_ev])
    train = [probs[q] for q, g in zip(m.qids, m.group) if g not in ev and q in probs]
    evalp = [probs[q] for q, g in zip(m.qids, m.group) if g in ev and q in probs]
    return train, evalp


def make_feats(qid: str, train_qids: set[str]) -> Feats:
    """kNN per-arm p_solve prior from TRAIN-contest columns of the LCB matrix only."""
    m = route.load_matrix()
    probs = {p.qid: p for p in sandbox.load()}
    route.attach_embeddings(m, {q: probs[q].statement for q in m.qids if q in probs})
    tr = np.array([j for j, q in enumerate(m.qids) if q in train_qids and q != qid])
    j = m.qids.index(qid)
    sims = m.emb[tr] @ m.emb[j]
    nn = tr[np.argsort(-sims)[:12]]
    w = np.clip(m.emb[nn] @ m.emb[j], 0, None) + 1e-6
    knn = (m.resolved[:, nn] * w).sum(axis=1) / w.sum()
    order = [m.arms.index(a) for a in ARM_IDS]
    return Feats(prior=knn[order])


# --------------------------------------------------------------------------- train / eval
def cmd_train(iters: int = 5, tasks_per_iter: int = 20, rollouts: int = 2,
              workers: int = 10, lr: float = 0.05, sticky: bool = False,
              handicap_k: int = 0) -> None:
    handicap = (0, handicap_k) if handicap_k else None  # arm 0 = haiku@budget, weakest
    load_env()
    sandbox.semaphore(workers)
    train, evalp = contest_split()
    train_qids = {p.qid for p in train}
    print(f"{len(train)} train tasks / {len(evalp)} eval tasks; arms: {ARM_IDS}; "
          f"sticky={sticky}")
    feats = {p.qid: dataclasses.replace(make_feats(p.qid, train_qids), sticky=sticky)
             for p in train + evalp}
    policy = Policy(dim=DIM + len(ARMS) if sticky else DIM)
    OUT.mkdir(parents=True, exist_ok=True)
    mlog = (OUT / "metrics.jsonl").open("a")
    rng0 = np.random.default_rng(SEED)

    pref = ("st-" if sticky else "") + (f"h{handicap_k}-" if handicap_k else "")
    for it in range(iters):
        batch = list(rng0.choice(len(train), min(tasks_per_iter, len(train)), replace=False))
        jobs = [(train[i], r) for i in batch for r in range(rollouts)]
        recs = []
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_episode, p, feats[p.qid], policy,
                              np.random.default_rng(hash((it, p.qid, r)) % 2**32),
                              False, f"{pref}it{it}r{r}", handicap): (p, r)
                    for p, r in jobs}
            for f in cf.as_completed(futs):
                try:
                    rec = f.result()
                except Exception as e:  # noqa: BLE001 — infra fault = missing data, not 0
                    print(f"  it{it} EPISODE DROPPED (harness): {str(e)[:150]}", flush=True)
                    continue
                recs.append(rec)
                print(f"  it{it} {rec['qid']:14s} graded={rec['graded']:.2f} "
                      f"${rec['cost_usd']:.4f} turns={rec['n_turns']} R={rec['reward']:+.3f}",
                      flush=True)
        # REINFORCE with task-mean baseline over the M rollouts
        by_task: dict[str, list[dict]] = {}
        for rec in recs:
            by_task.setdefault(rec["qid"], []).append(rec)
        grad = np.zeros_like(policy.W)
        n_dec = 0
        for _, group in by_task.items():
            b = float(np.mean([r["reward"] for r in group]))
            for rec in group:
                adv = rec["reward"] - b
                for t in rec["turns"]:
                    if "x" not in t or "cost" not in t or t.get("forced"):
                        continue
                    x = np.array(t["x"])
                    z = policy.logits(x)
                    p = np.exp(z - z.max())
                    p /= p.sum()
                    onehot = np.zeros(len(ARMS))
                    onehot[t["arm"]] = 1.0
                    grad += adv * np.outer(onehot - p, x)
                    n_dec += 1
        if n_dec:
            policy.W += lr * grad / n_dec
        stats = {"it": it, "n_eps": len(recs),
                 "graded": float(np.mean([r["graded"] for r in recs])),
                 "cost": float(np.sum([r["cost_usd"] for r in recs])),
                 "reward": float(np.mean([r["reward"] for r in recs])),
                 "spend_total": round(_spend["usd"], 2),
                 "arm_use": {ARM_IDS[a]: int(sum(t["arm"] == a for r in recs
                                                 for t in r["turns"]))
                             for a in range(len(ARMS))}}
        mlog.write(json.dumps(stats) + "\n")
        mlog.flush()
        policy.save(OUT / f"policy_{pref}it{it}.json")
        print(f"== it{it}: graded {stats['graded']:.3f}, ${stats['cost']:.2f}, "
              f"mean R {stats['reward']:+.3f}, total spend ${stats['spend_total']:.2f}",
              flush=True)
    mlog.close()
    print(f"train done; final policy {OUT / f'policy_it{iters-1}.json'}")


def cmd_eval(policy_path: str, workers: int = 10, handicap_k: int = 0) -> None:
    """EXP-010 comparison: trained per-turn policy vs current bests, all LIVE episodes
    on the held-out contests: turn0-frozen (per-task control), hand ladder cascade,
    and the two strongest static arms from the offline matrix."""
    load_env()
    sandbox.semaphore(workers)
    train, evalp = contest_split()
    train_qids = {p.qid for p in train}
    feats = {p.qid: make_feats(p.qid, train_qids) for p in evalp}
    policy = Policy.load(pathlib.Path(policy_path))
    sticky = policy.W.shape[1] > DIM
    handicap = (0, handicap_k) if handicap_k else None
    factories = {
        "perturn": lambda: policy,
        "turn0-frozen": lambda: Turn0Policy(policy),
        "ladder": lambda: LadderPolicy(),
        "static-nano@high": lambda: StaticPolicy(ARM_IDS.index("gpt-5.4-nano@high")),
        "static-opus@medium": lambda: StaticPolicy(ARM_IDS.index("claude-opus-4-8@medium")),
    }
    if handicap:  # EXP-011 controls: never escalate / always escalate to strongest
        factories["continue-weak"] = lambda: StaticPolicy(0)
        factories["escalate-opus"] = lambda: StaticPolicy(
            ARM_IDS.index("claude-opus-4-8@medium"))
    stem = pathlib.Path(policy_path).stem
    hpre = f"h{handicap_k}-" if handicap_k else ""
    rows = {}
    for name, mk in factories.items():
        # ladder/static are policy-independent (cache across evals); the rest are not
        tag = f"eval-{hpre}{name}" if name.startswith(("ladder", "static", "continue",
                                                       "escalate")) \
            else f"eval-{hpre}{name}-{stem}"
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_episode, p,
                              dataclasses.replace(
                                  feats[p.qid],
                                  sticky=sticky and name in ("perturn", "turn0-frozen")),
                              mk(), np.random.default_rng(0), True, tag, handicap)
                    for p in evalp]
            recs = []
            for f in cf.as_completed(futs):
                try:
                    recs.append(f.result())
                except Exception as e:  # noqa: BLE001 — report dropped, never zeroed
                    print(f"  {name}: EPISODE DROPPED (harness): {str(e)[:150]}",
                          flush=True)
        rows[name] = recs
        g = np.mean([r["graded"] for r in recs])
        c = np.sum([r["cost_usd"] for r in recs])
        print(f"{name:20s}: graded {g:.3f}, ${c:.3f} total, "
              f"${c/len(recs):.4f}/task over {len(recs)} eval tasks", flush=True)
    (OUT / "eval.json").write_text(json.dumps(
        {k: [{kk: r[kk] for kk in ("qid", "graded", "cost_usd", "n_turns")}
             for r in v] for k, v in rows.items()}))


def cmd_smoke() -> None:
    load_env()
    sandbox.semaphore(4)
    train, _ = contest_split()
    train_qids = {p.qid for p in train}
    for p in train[:2]:
        rec = run_episode(p, make_feats(p.qid, train_qids), Policy(),
                          np.random.default_rng(0), False, "smoke")
        print(json.dumps({k: rec[k] for k in
                          ("qid", "graded", "cost_usd", "n_turns", "reward")}))
        print("  arms used:", [ARM_IDS[t["arm"]] for t in rec["turns"] if "arm" in t])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    def _flag(name: str) -> int:
        return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else 0

    if cmd == "smoke":
        cmd_smoke()
    elif cmd == "train":
        cmd_train(sticky="--sticky" in sys.argv, handicap_k=_flag("--handicap"))
    elif cmd == "eval":
        cmd_eval(sys.argv[2], handicap_k=_flag("--handicap"))
    else:
        raise SystemExit(f"unknown command {cmd}")
