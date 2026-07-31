"""Benchmark CLIs: fetching published outcome matrices, transfer tests, and live sweeps.

Subcommands:
  fetch-swebench-matrix  -- download the published bash-only SWE-bench outcome matrix.
  transfer-swebench      -- transfer test: does the DeepSWE kNN policy also beat
                             always-best on the SWE-bench bash-only matrix?
  run-lcb                -- run agentic LiveCodeBench episodes in E2B sandboxes.
  smoke-agent            -- end-to-end correctness check of the agent loop.
"""
from __future__ import annotations

import base64
import concurrent.futures as cf
import json
import pathlib
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np
from tabulate import tabulate

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from router import harness as sandbox  # noqa: E402
from router import router_core as route  # noqa: E402
from router.harness import AgentRunner, Tool, load_env  # noqa: E402
from router.router_core import Arm  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

# ============================================================ fetch-swebench-matrix
# Every submission under evaluation/bash-only ran the SAME mini-swe-agent bash-only
# scaffold on the SAME 500 SWE-bench Verified instances, varying only the model. Each
# ships per_instance_details.json with {resolved, cost, api_calls} per instance -- a
# free, published, controlled (model x instance) outcome matrix WITH cost. Provenance
# is public, so every number here is published-not-measured and must be labelled as such
# downstream. Writes results/swebench_matrix.json.

SWEBENCH_REPO = "SWE-bench/experiments"
SWEBENCH_BASE = "evaluation/bash-only"


def gh_json(path: str):
    """Call `gh api <path>` and parse the JSON response.

    Args:
        path: A GitHub API path, relative to the API root.

    Returns:
        The parsed JSON response.

    Raises:
        RuntimeError: If the `gh` invocation fails.
    """
    out = subprocess.run(["gh", "api", path], capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[:300])
    return json.loads(out.stdout)


def gh_file(path: str) -> bytes:
    """Fetch one file's raw bytes via the GitHub contents API."""
    d = gh_json(path)
    return base64.b64decode(d["content"])


def fetch_submission(name: str) -> tuple[str, dict | None]:
    """Fetch one SWE-bench/experiments submission's per-instance details and metadata.

    Args:
        name: Submission directory name under `SWEBENCH_BASE`.

    Returns:
        A (name, payload) pair. `payload` is `{"error": ...}` on failure, otherwise
        `{"meta": ..., "details": ...}` -- a raw parsed copy of that submission's own
        published files, shape owned by the publisher, not by us.
    """
    try:
        details = json.loads(gh_file(
            f"repos/{SWEBENCH_REPO}/contents/{SWEBENCH_BASE}/{name}/per_instance_details.json"))
    except Exception as e:  # noqa: BLE001
        return name, {"error": str(e)[:200]}
    meta_txt = ""
    try:
        meta_txt = gh_file(
            f"repos/{SWEBENCH_REPO}/contents/{SWEBENCH_BASE}/{name}/metadata.yaml").decode()
    except Exception:  # noqa: BLE001, S110
        pass
    # metadata.yaml is tiny and flat; avoid a yaml dependency.
    meta = {}
    for line in meta_txt.splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return name, {"meta": meta, "details": details}


def cmd_fetch_swebench_matrix() -> None:
    """CLI (`fetch-swebench-matrix`): download every bash-only submission and write the matrix."""
    dirs = [d["name"] for d in gh_json(f"repos/{SWEBENCH_REPO}/contents/{SWEBENCH_BASE}")
            if d["type"] == "dir"]
    print(f"{len(dirs)} submissions found")

    out: dict[str, dict] = {}
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for name, payload in ex.map(fetch_submission, dirs):
            if payload is None or "error" in payload:
                print(f"  SKIP {name}: {payload and payload.get('error')}")
                continue
            det = payload["details"]
            n = len(det)
            res = sum(1 for v in det.values() if v.get("resolved"))
            costs = [v.get("cost") or 0 for v in det.values()]
            out[name] = payload
            print(f"  {name:52s} n={n:4d} resolved={res:3d} ({res/n*100:5.1f}%) "
                  f"cost=${sum(costs):7.2f} mean=${sum(costs)/n:.3f}")

    dest = ROOT / "results" / "swebench_matrix.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out))
    inst = {i for p in out.values() for i in p["details"]}
    print(f"\n{len(out)} submissions x {len(inst)} instances -> {dest}")
    print(f"total (model, instance) cells available FREE: {sum(len(p['details']) for p in out.values()):,}")


# ============================================================ transfer-swebench
# Transfer test: does the SAME kNN policy beat always-best on a different dataset?
# DeepSWE gave 2.15x, but that is one dataset. This runs the identical policy on the only
# other free matrix that carries per-instance cost: the SWE-bench bash-only archive (40
# same-scaffold submissions x 500 SWE-bench Verified instances, each with resolved + cost).
# Different tasks, different scaffold, different arm pool, and BINARY labels rather than
# graded -- if kNN wins here too, the method transfers rather than being a property of
# the DeepSWE dataset specifically.
#
# Repo-grouped CV: SWE-bench instance ids are `<org>__<repo>-<number>`, and instances from
# one repo share code, so the repo is the grouping unit.

TRANSFER_EMB = ROOT / "results" / "swebench_embeddings.json"


def build() -> route.Matrix:
    """Load the published SWE-bench bash-only matrix into a `Matrix`.

    Returns:
        The (arm x instance) matrix, restricted to instances every kept submission
        has a result for, with broken (0-resolved, non-trivial-spend) submissions excluded.
    """
    raw = json.loads((ROOT / "results" / "swebench_matrix.json").read_text())
    subs = sorted(raw)
    # Drop the broken submission: 0 resolved on $480 of spend is a failed run, not a result.
    keep = []
    for s in subs:
        det = raw[s]["details"]
        res = sum(1 for v in det.values() if v.get("resolved"))
        spend = sum(v.get("cost") or 0 for v in det.values())
        if res == 0 and spend > 1.0:
            print(f"  excluded broken submission {s}")
            continue
        keep.append(s)
    inst = sorted(set.intersection(*(set(raw[s]["details"]) for s in keep)))
    r = np.zeros((len(keep), len(inst)), dtype=bool)
    c = np.zeros((len(keep), len(inst)), dtype=float)
    for i, s in enumerate(keep):
        for j, q in enumerate(inst):
            d = raw[s]["details"][q]
            r[i, j] = bool(d.get("resolved"))
            c[i, j] = d.get("cost") or 0.0
    grp = [q.split("-")[0] for q in inst]   # astropy__astropy-12907 -> astropy__astropy
    return route.Matrix(arms=[s.split("_", 1)[1] for s in keep], qids=inst,
                        resolved=r, graded=r.astype(float), cost=c,
                        difficulty=["?"] * len(inst), group=grp)


def embed(m: route.Matrix) -> None:
    """Embed the problem statements from SWE-bench Verified, cached on disk.

    Args:
        m: The matrix to attach embeddings to; mutates `m.emb` in place.
    """
    cache = json.loads(TRANSFER_EMB.read_text()) if TRANSFER_EMB.exists() else {}
    missing = [q for q in m.qids if q not in cache]
    if missing:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        text = {r["instance_id"]: r["problem_statement"] for r in ds}
        import openai
        cl = openai.OpenAI()
        for i in range(0, len(missing), 64):
            ch = missing[i:i + 64]
            out = cl.embeddings.create(model="text-embedding-3-large",
                                       input=[(text.get(q) or q)[:8000] for q in ch])
            for q, e in zip(ch, out.data):
                cache[q] = e.embedding
        TRANSFER_EMB.write_text(json.dumps(cache))
    e = np.array([cache[q] for q in m.qids], dtype=float)
    m.emb = e / np.linalg.norm(e, axis=1, keepdims=True)


def cmd_transfer_swebench() -> None:
    """CLI (`transfer-swebench`): race routing policies on the SWE-bench bash-only matrix."""
    load_env()
    m = build()
    embed(m)
    print(f"SWE-bench bash-only: {len(m.arms)} arms x {m.n} instances, "
          f"{len(set(m.group))} repos")
    acc = m.resolved.mean(axis=1)
    tot = m.cost.sum(axis=1)
    best = int(np.argmax(acc))
    print(f"  best arm: {m.arms[best]}  {acc[best]*100:.1f}%  ${tot[best]:.0f}")
    # The honest bar is the best STATIC arm on the cost-quality frontier, not the priciest.
    front = [i for i in range(len(m.arms))
             if not any(tot[o] < tot[i] and acc[o] >= acc[i] for o in range(len(m.arms)))]
    print(f"  static frontier has {len(front)} arms; cheapest-at->=90%-of-best: ", end="")
    cand = [i for i in front if acc[i] >= 0.9 * acc[best]]
    ref = min(cand, key=lambda i: tot[i]) if cand else best
    print(f"{m.arms[ref]} {acc[ref]*100:.1f}% ${tot[ref]:.0f}")

    folds = route.grouped_folds(m, n_folds=5, seed=0)
    br, bc = route.run_policy(m, route.p_always(best), folds)
    B = bc.sum()
    rows = [("always-best", f"{br.mean()*100:.1f}%", f"{0.0:+.1f}", f"{B:.1f}", f"{1.00:.2f}", "")]
    for tau in (0.3, 0.5, 0.7, 0.9):
        r, c = route.run_policy(m, route.p_knn_threshold(12, tau), folds)
        lo, hi = route.boot_ratio(bc, c, m.group)
        rows.append((f"knn-threshold t={tau:.1f}", f"{r.mean()*100:.1f}%",
                     f"{(r.mean()-br.mean())*100:+.1f}", f"{c.sum():.1f}",
                     f"{B/c.sum():.2f}", f"[{lo:.2f},{hi:.2f}]"))
    rc, cc = route.run_policy(m, route.p_cascade(int(np.argmin(tot)), best), folds)
    lo, hi = route.boot_ratio(bc, cc, m.group)
    rows.append(("cascade cheapest->best", f"{rc.mean()*100:.1f}%",
                 f"{(rc.mean()-br.mean())*100:+.1f}", f"{cc.sum():.1f}",
                 f"{B/cc.sum():.2f}", f"[{lo:.2f},{hi:.2f}]"))
    orr, oc = route.oracle(m)
    rows.append(("ORACLE", f"{orr.mean()*100:.1f}%", f"{(orr.mean()-br.mean())*100:+.1f}",
                 f"{oc.sum():.1f}", f"{B/oc.sum():.2f}", "(ceiling)"))
    print()
    print(tabulate(rows, headers=["policy", "acc", "delta", "cost$", "x cheap", "95% CI"],
                   disable_numparse=True))


# ============================================================ run-lcb
# Run agentic LiveCodeBench episodes and report wall-clock, cost, and resolve. Purpose is
# to establish the inner-loop speed empirically. Episodes run concurrently; per-episode
# results are checkpointed atomically so nothing is ever paid for twice.
#
# Usage:
#   uv run python -m router.benchmarks run-lcb --arms cheap --n 6 --workers 6

LCB_OUT = ROOT / "results" / "episodes"

ARM_SETS = {
    # Two extremes plus a mid arm: enough to see a gradient without spending.
    "cheap": [Arm(provider="openai", model="gpt-5.4-nano", effort="low"),
              Arm(provider="anthropic", model="claude-haiku-4-5", effort=None, thinking="off")],
    # Isolates the one variable that made haiku look broken: reasoning on vs off.
    # nano@low reasons; haiku@off does not. Comparing them conflates model with config.
    "haiku-thinking": [Arm(provider="anthropic", model="claude-haiku-4-5", effort=None, thinking="off"),
                       Arm(provider="anthropic", model="claude-haiku-4-5", effort=None, thinking="budget")],
    "ladder": [Arm(provider="openai", model="gpt-5.4-nano", effort="low"),
               Arm(provider="openai", model="gpt-5.4-mini", effort="medium"),
               Arm(provider="anthropic", model="claude-haiku-4-5", effort=None, thinking="budget"),
               Arm(provider="anthropic", model="claude-sonnet-5", effort="medium", thinking="adaptive")],
    # Phase 1: the candidate ladder, spanning ~50x list price. Two efforts on each cheap
    # model because the nano-beats-haiku result suggests reasoning config may matter more
    # than model tier on this task family -- that is the hypothesis this run tests.
    "full": [Arm(provider="openai", model="gpt-5.4-nano", effort="low"),
             Arm(provider="openai", model="gpt-5.4-nano", effort="high"),
             Arm(provider="openai", model="gpt-5.4-mini", effort="medium"),
             Arm(provider="openai", model="gpt-5.4-mini", effort="xhigh"),
             Arm(provider="anthropic", model="claude-haiku-4-5", effort=None, thinking="budget"),
             Arm(provider="openai", model="gpt-5.3-codex", effort="high"),
             Arm(provider="openai", model="gpt-5.4", effort="medium"),
             Arm(provider="anthropic", model="claude-sonnet-5", effort="medium", thinking="adaptive"),
             Arm(provider="anthropic", model="claude-opus-4-8", effort="medium", thinking="adaptive")],
}


def bash_tool(session: sandbox.SandboxSession) -> Tool:
    """Build the agent's bash tool, executed in the E2B sandbox -- never on this machine.

    Args:
        session: The sandbox session to run commands in.

    Returns:
        A `Tool` that runs a shell command and returns exit code, stdout, and stderr.
    """
    def run(args: dict) -> str:
        """Run `args["command"]` in the sandbox and format its exit code/stdout/stderr."""
        cmd = args.get("command", "")
        if not cmd:
            return "error: empty command"
        try:
            rc, out, err = session.run(cmd, timeout=90.0)
        except Exception as e:  # noqa: BLE001 — surface sandbox faults to the agent as output
            return f"sandbox error: {type(e).__name__}: {e}"[:600]
        return f"exit={rc}\n--- stdout ---\n{out[-6000:]}\n--- stderr ---\n{err[-1500:]}"

    return Tool(
        name="bash",
        description="Run a bash command in the scratch directory; returns exit code, stdout, stderr.",
        schema={"type": "object",
                "properties": {"command": {"type": "string", "description": "command to run"}},
                "required": ["command"]},
        run=run,
    )


def episode(job) -> dict:
    """Run (or load a cached) LiveCodeBench episode for one (problem, arm) pair.

    Args:
        job: A (problem, arm, max_turns) tuple.

    Returns:
        A record dict describing the outcome. Shape is deliberately sparse rather
        than fixed: a successful run has ~20 fields (outcome/graded/turns/...), while
        a harness-level exception (caught below) writes only a handful (qid/arm/
        resolved/harness_error/...) -- downstream readers use `.get(key, default)`
        precisely because a key can be genuinely absent, not just falsy. This is a
        raw episode-log record, not a fixed reused shape.
    """
    prob, arm, max_turns = job
    key = f"{prob.qid}__{arm.id.replace('/', '-')}"
    dest = LCB_OUT / f"{key}.json"
    if dest.exists():
        rec = json.loads(dest.read_text())
        rec["cached"] = True
        return rec

    t0 = time.time()
    try:
        tag = {"lane": "coding-router", "qid": prob.qid, "arm": arm.id}
        with sandbox.SandboxSession(tag) as session:
            session.write("public_tests.json", json.dumps(prob.public_tests))
            session.write("check.py", sandbox.CHECKER)
            # Generous cap on purpose: an unused cap is free, and a tight one silently
            # converts reasoning depth into scored task failures (see AgentRunner docstring).
            runner = AgentRunner(arm, [bash_tool(session)], sandbox.SYSTEM,
                                 max_turns=max_turns, max_tokens=32_000)
            res = runner.run(sandbox.task_prompt(prob))
            code = session.read("solution.py") or ""
            # Ground truth. Private tests are written only NOW, after the agent stopped,
            # so it never had access to what it is graded on. Grading executes the
            # model's code, so it stays in the sandbox too -- never on this machine.
            if code.strip():
                passed, total = session.grade(prob.private_tests)
            else:
                passed, total = 0, prob.n_tests

        # Classify the outcome instead of collapsing everything to a boolean. An episode
        # that never wrote a solution because it ran out of turns or tokens is a HARNESS
        # LIMIT, not a wrong answer, and scoring it 0/N is what produced $11 of fake
        # "expensive frontier failures" and a retracted 1.85x result.
        if total and passed == total:
            outcome = "solved"
        elif not code.strip() and res.stop in ("max_turns", "max_tokens"):
            outcome = "harness_limit"
        elif not code.strip():
            outcome = "no_solution"
        else:
            outcome = "wrong"
        rec = {
            "outcome": outcome,
            # Graded score is the primary objective. Binary pass/fail overstates the
            # model-tier gap ~4x (measured: 9.2pp binary vs 2.3pp graded, same episodes).
            "graded": (passed / total) if total else None,
            "qid": prob.qid, "difficulty": prob.difficulty, "arm": arm.id,
            "resolved": bool(total and passed == total),
            "passed": passed, "total": total,
            "turns": res.turns, "stop": res.stop, "error": res.error,
            "cost_usd": round(res.cost_usd, 6), "wall_s": res.wall_s,
            "in_tok": res.usage.inp, "out_tok": res.usage.out,
            "cache_read": res.usage.cache_read, "reasoning_tok": res.usage.reasoning,
            "requests": res.usage.requests, "wrote_solution": bool(code.strip()),
            "total_wall_s": round(time.time() - t0, 2), "cached": False,
        }
    except Exception as e:  # noqa: BLE001 — never let one episode kill the sweep
        rec = {"qid": prob.qid, "difficulty": prob.difficulty, "arm": arm.id,
               "resolved": False, "harness_error": f"{type(e).__name__}: {e}"[:400],
               "total_wall_s": round(time.time() - t0, 2), "cached": False}
    # No local cleanup needed: SandboxSession.__exit__ always kills its sandbox.

    LCB_OUT.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.replace(dest)  # atomic: a killed run never leaves a half-written record
    return rec


def cmd_run_lcb(arms: str = "cheap", n: int = 6, workers: int = 8, max_turns: int = 60,
                max_concurrent: int = sandbox.DEFAULT_MAX_CONCURRENT) -> None:
    """CLI (`run-lcb`): run agentic LiveCodeBench episodes and print per-arm results.

    Args:
        arms: Which entry of `ARM_SETS` to sweep.
        n: Number of problems per difficulty band (easy/medium/hard) to sample.
        workers: Number of episodes to run concurrently.
        max_turns: Maximum agent turns per episode. 16 was tried and was indefensible:
            opus hit it on 3 hard problems having never written a solution, burning
            $2.41-$4.41 each and scoring 0. The reference harness (mini-swe-agent) uses
            a 250-step limit with a $3/instance cost cap; 60 is the compromise here for
            single-file tasks.
        max_concurrent: Cap on live E2B sandboxes; account cap is 1100 and this lane
            must never starve another run.

    Raises:
        ValueError: If `arms` is not a key of `ARM_SETS`.
    """
    if arms not in ARM_SETS:
        raise ValueError(f"arms must be one of {sorted(ARM_SETS)}, got {arms!r}")
    load_env()
    # Bind the sandbox cap before any worker starts, so the limit actually applies.
    sandbox.semaphore(max_concurrent)
    probs = sandbox.load()
    # Stratify so the gradient is visible even on a tiny dev slice.
    picked = []
    for band in ("easy", "medium", "hard"):
        picked += [p for p in probs if p.difficulty == band][:n]
    arm_list = ARM_SETS[arms]
    jobs = [(p, arm, max_turns) for p in picked for arm in arm_list]
    print(f"{len(picked)} problems x {len(arm_list)} arms = {len(jobs)} episodes, "
          f"{workers} workers")

    t0 = time.time()
    recs = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, rec in enumerate(ex.map(episode, jobs), 1):
            recs.append(rec)
            flag = "cache" if rec.get("cached") else ("ERR " if rec.get("harness_error") else
                                                     ("PASS" if rec["resolved"] else "fail"))
            print(f"[{i:3d}/{len(jobs)}] {flag} {rec['qid']:12s} {rec['difficulty']:6s} "
                  f"{rec['arm']:26s} {rec.get('passed','?')}/{rec.get('total','?')} "
                  f"turns={rec.get('turns','?'):>3} ${rec.get('cost_usd',0):.4f} "
                  f"{rec.get('total_wall_s',0):5.1f}s", flush=True)
    wall = time.time() - t0

    fresh = [r for r in recs if not r.get("cached")]
    print(f"\n=== {len(jobs)} episodes in {wall:.1f}s wall "
          f"({len(fresh)} fresh) ===")
    for arm in arm_list:
        rs = [r for r in recs if r["arm"] == arm.id]
        if not rs:
            continue
        res = sum(1 for r in rs if r["resolved"])
        errs = sum(1 for r in rs if r.get("harness_error"))
        cost = sum(r.get("cost_usd", 0) for r in rs)
        eps = [r["total_wall_s"] for r in rs if not r.get("cached")]
        med = statistics.median(eps) if eps else float("nan")
        # Cap-hit and no-solution rates go BESIDE accuracy: both are infrastructure
        # failures that otherwise get silently counted as the model being wrong.
        cap = sum(1 for r in rs if r.get("stop") == "max_tokens")
        turnlim = sum(1 for r in rs if r.get("stop") == "max_turns")
        nosol = sum(1 for r in rs if not r.get("wrote_solution", True))
        print(f"  {arm.id:26s} {res}/{len(rs)} resolved  ${cost:.4f}  "
              f"median {med:5.1f}s/episode")
        print(f"      cap_hit={cap} turn_limit={turnlim} no_solution={nosol} "
              f"harness_errors={errs}")
        for band in ("easy", "medium", "hard"):
            b = [r for r in rs if r["difficulty"] == band]
            if b:
                print(f"      {band:6s} {sum(1 for r in b if r['resolved'])}/{len(b)}")
    if fresh:
        allw = sorted(r["total_wall_s"] for r in fresh)
        print(f"\n  episode wall-clock: p50={allw[len(allw)//2]:.1f}s "
              f"p90={allw[int(len(allw)*.9)]:.1f}s max={allw[-1]:.1f}s")


# ============================================================ smoke-agent
# End-to-end correctness check of the agent loop on a real, verifiable task. Not a
# science probe -- an assert that the tool loop, history round-tripping, and usage
# accounting work identically on both providers. Costs a few cents.
#
# The task is deliberately one that cannot be solved in a single turn without running
# anything: a buggy function plus a failing test the agent must actually execute to see
# the failure.

SMOKE_BUGGY = '''\
def rolling_max(xs):
    """Return a list where element i is the max of xs[:i+1]."""
    out = []
    for x in xs:
        out.append(x)          # BUG: ignores the running maximum
    return out
'''

SMOKE_TEST = '''\
from solution import rolling_max

def test_basic():
    assert rolling_max([1, 3, 2, 5, 4]) == [1, 3, 3, 5, 5]

def test_negatives():
    assert rolling_max([-5, -7, -2]) == [-5, -5, -2]

def test_empty():
    assert rolling_max([]) == []
'''

SMOKE_SYSTEM = (
    "You are a coding agent working in a sandboxed repo. Use the bash tool to inspect "
    "and modify files and to run tests. Keep going until the tests pass. "
    "When they pass, reply with exactly DONE and nothing else."
)
SMOKE_TASK = (
    "The test suite in this directory fails. Run `python -m pytest -q` to see the failure, "
    "find the bug in solution.py, fix it, and re-run the tests until they all pass."
)


def make_bash_tool(workdir: pathlib.Path) -> Tool:
    """Build a bash tool that runs commands directly in `workdir` (no sandbox).

    Args:
        workdir: The repo working directory to run commands in.

    Returns:
        A `Tool` that runs a shell command and returns exit code, stdout, and stderr.
    """
    def run(args: dict) -> str:
        """Run `args["command"]` in `workdir` and format its exit code/stdout/stderr."""
        cmd = args.get("command", "")
        try:
            p = subprocess.run(cmd, shell=True, cwd=workdir, capture_output=True,
                               text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return "TIMEOUT after 60s"
        return (f"exit={p.returncode}\n--- stdout ---\n{p.stdout[-4000:]}"
                f"\n--- stderr ---\n{p.stderr[-2000:]}")

    return Tool(
        name="bash",
        description="Run a bash command in the repo working directory and return exit code, "
                    "stdout and stderr.",
        schema={"type": "object",
                "properties": {"command": {"type": "string", "description": "the command to run"}},
                "required": ["command"]},
        run=run,
    )


def verify(workdir: pathlib.Path) -> bool:
    """Ground truth: do the ORIGINAL tests pass? Re-written to defeat test tampering."""
    (workdir / "test_solution.py").write_text(SMOKE_TEST)
    p = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=workdir,
                       capture_output=True, text=True, timeout=120)
    return p.returncode == 0


def cmd_smoke_agent() -> None:
    """CLI (`smoke-agent`): run the buggy-function task on both providers and assert success."""
    arms = [
        Arm(provider="anthropic", model="claude-haiku-4-5", effort=None, thinking="off"),
        Arm(provider="openai", model="gpt-5.4-nano", effort="low"),
    ]
    for arm in arms:
        work = pathlib.Path(tempfile.mkdtemp(prefix="smoke-"))
        try:
            (work / "solution.py").write_text(SMOKE_BUGGY)
            (work / "test_solution.py").write_text(SMOKE_TEST)
            runner = AgentRunner(arm, [make_bash_tool(work)], SMOKE_SYSTEM, max_turns=12,
                                 max_tokens=4000)
            res = runner.run(SMOKE_TASK)
            passed = verify(work)
            print(f"\n=== {arm.id}")
            print(f"  resolved      : {passed}")
            print(f"  stop / turns  : {res.stop} / {res.turns}")
            print(f"  usage         : in={res.usage.inp} cache_r={res.usage.cache_read} "
                  f"cache_w={res.usage.cache_write} out={res.usage.out} "
                  f"reasoning={res.usage.reasoning} reqs={res.usage.requests}")
            print(f"  cost / wall   : ${res.cost_usd:.5f} / {res.wall_s}s")
            if res.error:
                print(f"  ERROR         : {res.error}")
            tools_used = sum(len(t.calls) for t in res.transcript)
            print(f"  tool calls    : {tools_used}")
            assert res.usage.requests >= 1, "no requests recorded"
            assert tools_used >= 1, "agent never called a tool"
        finally:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    import fire

    fire.Fire({
        "fetch-swebench-matrix": cmd_fetch_swebench_matrix,
        "transfer-swebench": cmd_transfer_swebench,
        "run-lcb": cmd_run_lcb,
        "smoke-agent": cmd_smoke_agent,
    })
