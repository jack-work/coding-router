"""Analysis CLIs over measured episodes, published matrices, and production traces.

Subcommands:
  analyze-phase1  -- Phase 1 analysis: measured arm ladder, data-quality flags, headroom.
  headroom        -- routing headroom in the published SWE-bench bash-only matrix.
  price-traces    -- price real Claude Code sessions to get the measured baseline spend.
  extract-traces  -- extract the real production query distribution from coding-agent traces.

The production export step (freezing the router artifact) is a separate CLI: router/export.py.
"""
from __future__ import annotations

import collections
import glob
import itertools
import json
import pathlib
import re
import statistics
import sys
from collections import Counter

from tabulate import tabulate

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from router.router_core import STANDARD  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

# ============================================================ analyze-phase1
def load_episodes() -> list[dict]:
    """Load every raw episode record under results/episodes/*.json."""
    return [json.loads(pathlib.Path(f).read_text())
            for f in glob.glob(str(ROOT / "results" / "episodes" / "*.json"))]


def cmd_analyze_phase1() -> None:
    """CLI (`analyze-phase1`): the measured arm ladder, data-quality flags, and headroom.

    Everything here is MEASURED by us on LiveCodeBench AtCoder via E2B sandboxes, not
    published. Reports the three things that decide the next phase: (1) the
    cost-quality frontier over our own arms (which arms are even worth carrying),
    (2) data-quality flags (items where the label is probably wrong, and errored
    requests), and (3) oracle + cascade headroom, i.e. whether routing can pay at all
    on this task family.
    """
    recs = [r for r in load_episodes() if "arm" in r]
    qids = sorted({r["qid"] for r in recs})
    # Drop arms that were never swept over the full problem set (e.g. a leftover smoke-test
    # arm with 6 episodes). Keeping them silently collapses the complete-matrix intersection
    # to a handful of problems and makes the headroom numbers meaningless.
    counts = collections.Counter(r["arm"] for r in recs)
    arms = sorted(a for a, n in counts.items() if n >= 0.5 * len(qids))
    dropped = {a: n for a, n in counts.items() if a not in arms}
    if dropped:
        print(f"NOTE dropping under-swept arms {dropped} (need >= {int(0.5*len(qids))} episodes)")
    recs = [r for r in recs if r["arm"] in arms]
    # An episode killed by a provider 429/529 or a harness fault is NOT evidence the model
    # failed the task. Exclude those cells and report the count, rather than letting
    # provider overload depress an arm's measured accuracy.
    def valid(r: dict) -> bool:
        """An episode counts as valid evidence only if it wasn't killed by infra."""
        return not r.get("harness_error") and r.get("stop") != "error"

    n_bad = sum(1 for r in recs if not valid(r))
    recs = [r for r in recs if valid(r)]
    print(f"NOTE excluded {n_bad} episodes killed by provider/harness errors "
          f"(not model failures)")
    cell = {(r["arm"], r["qid"]): r for r in recs}
    print(f"{len(recs)} episodes | {len(arms)} arms x {len(qids)} problems "
          f"({len(arms)*len(qids)} full matrix)\n")

    # ---------------------------------------------------------------- data quality first
    print("=== DATA QUALITY (check before believing any accuracy) ===")
    errored = [r for r in recs if r.get("harness_error")
               or (r.get("turns") == 1 and not r.get("cost_usd"))]
    print(f"errored/zero-cost single-turn episodes: {len(errored)}")
    for r in errored[:6]:
        print(f"    {r['qid']:12s} {r['arm']:26s} stop={r.get('stop')} "
              f"err={str(r.get('error') or r.get('harness_error'))[:80]}")
    caps = collections.Counter(r["arm"] for r in recs if r.get("stop") == "max_tokens")
    print(f"cap_hits (must be ~0 after the max_tokens fix): {dict(caps) or 'none'}")

    # An item where nearly every arm lands on the SAME near-miss score is a broken test,
    # not N models making an identical mistake. These are label noise and get excluded.
    suspect = []
    for q in qids:
        rs = [cell[(a, q)] for a in arms if (a, q) in cell]
        rs = [r for r in rs if r.get("total")]
        if len(rs) < 5:
            continue
        scores = [r["passed"] / r["total"] for r in rs]
        near = [s for s in scores if 0.9 <= s < 1.0]
        if len(near) >= len(rs) - 1 and len(near) >= 5:
            suspect.append((q, len(near), len(rs), round(statistics.median(near), 3)))
    print(f"\nsuspect items (>=5 arms stuck at 90-99% -- probably a bad test case): {len(suspect)}")
    for q, n, tot, med in suspect[:8]:
        print(f"    {q:12s} {n}/{tot} arms at median {med:.3f}")
    bad = {q for q, *_ in suspect}

    # ---------------------------------------------------------------- the ladder
    print("\n=== MEASURED ARM LADDER (suspect items excluded) ===")
    ok_q = [q for q in qids if q not in bad]
    rows = []
    for a in arms:
        rs = [cell[(a, q)] for q in ok_q if (a, q) in cell]
        if not rs:
            continue
        res = sum(1 for r in rs if r.get("resolved"))
        cost = sum(r.get("cost_usd") or 0 for r in rs)
        rows.append({"arm": a, "n": len(rs), "acc": res / len(rs), "cost": cost,
                     "per": cost / len(rs),
                     "med_s": statistics.median([r.get("total_wall_s", 0) for r in rs])})
    rows.sort(key=lambda r: r["per"])
    print(tabulate(
        [(r["arm"], r["n"], f"{r['acc']*100:.1f}%", f"{r['per']:.4f}", f"{r['cost']:.3f}",
          f"{r['med_s']:.1f}") for r in rows],
        headers=["arm", "n", "acc", "$/prob", "total$", "med s"], disable_numparse=True))

    # Pareto frontier: an arm is dominated if something cheaper is at least as accurate.
    front = [r for r in rows if not any(o["per"] < r["per"] and o["acc"] >= r["acc"] for o in rows)]
    print(f"\n  ON THE COST-QUALITY FRONTIER: {', '.join(r['arm'] for r in front)}")
    print(f"  DOMINATED (a cheaper arm is >= as accurate): "
          f"{', '.join(r['arm'] for r in rows if r not in front) or 'none'}")

    # ---------------------------------------------------------------- difficulty gradient
    print("\n=== accuracy by difficulty ===")
    diff = {r["qid"]: r["difficulty"] for r in recs}
    bands = ("easy", "medium", "hard")
    diff_table = []
    for r in rows:
        out = []
        for b in bands:
            g = [cell[(r["arm"], q)] for q in ok_q
                 if (r["arm"], q) in cell and diff.get(q) == b]
            out.append(f"{sum(1 for x in g if x.get('resolved'))/len(g)*100:.1f}%" if g else "-")
        diff_table.append((r["arm"], *out))
    print(tabulate(diff_table, headers=["arm", *bands], disable_numparse=True))

    # ---------------------------------------------------------------- routing headroom
    print("\n=== ROUTING HEADROOM on our measured arms ===")
    full = [q for q in ok_q if all((a, q) in cell for a in arms)]
    solved = {q: [a for a in arms if cell[(a, q)].get("resolved")] for q in full}
    hist = collections.Counter(len(v) for v in solved.values())
    print(f"complete matrix on {len(full)} problems; solved-by-k histogram: "
          f"{ {k: hist.get(k, 0) for k in range(len(arms)+1)} }")
    none_ = sum(1 for v in solved.values() if not v)
    allv = sum(1 for v in solved.values() if len(v) == len(arms))
    print(f"  solved by NO arm: {none_} ({none_/len(full)*100:.1f}%) | "
          f"by ALL arms: {allv} ({allv/len(full)*100:.1f}%) | "
          f"routable: {len(full)-none_-allv} ({(len(full)-none_-allv)/len(full)*100:.1f}%)")

    best = max(rows, key=lambda r: r["acc"])
    bcost = sum(cell[(best["arm"], q)].get("cost_usd") or 0 for q in full)
    bacc = sum(1 for q in full if cell[(best["arm"], q)].get("resolved")) / len(full)
    print(f"\n  best single arm: {best['arm']} @ {bacc*100:.1f}%  ${bcost:.3f}")

    # Oracle: cheapest arm that solves it; unsolvable -> cheapest arm overall.
    cheapest = rows[0]["arm"]
    o_cost = o_res = 0.0
    for q in full:
        w = [(cell[(a, q)].get("cost_usd") or 0, a) for a in solved[q]]
        if w:
            o_cost += min(w)[0]
            o_res += 1
        else:
            o_cost += cell[(cheapest, q)].get("cost_usd") or 0
    print(f"  ORACLE          : {o_res/len(full)*100:.1f}%  ${o_cost:.3f}  "
          f"-> {bcost/o_cost:.2f}x cheaper than best arm, {(o_res/len(full)-bacc)*100:+.1f}pp")

    # Parity oracle: only downgrade problems the best arm already solves.
    p_cost = 0.0
    for q in full:
        if cell[(best["arm"], q)].get("resolved"):
            p_cost += min(cell[(a, q)].get("cost_usd") or 0 for a in solved[q])
        else:
            p_cost += cell[(best["arm"], q)].get("cost_usd") or 0
    print(f"  PARITY ORACLE   : {bacc*100:.1f}% (identical)  ${p_cost:.3f}  "
          f"-> {bcost/p_cost:.2f}x cheaper")

    # Cascade: cheapest arm first, escalate to best on failure. Needs no predictor.
    c_cost = c_res = 0.0
    for q in full:
        c_cost += cell[(cheapest, q)].get("cost_usd") or 0
        if cell[(cheapest, q)].get("resolved"):
            c_res += 1
        else:
            c_cost += cell[(best["arm"], q)].get("cost_usd") or 0
            c_res += int(bool(cell[(best["arm"], q)].get("resolved")))
    print(f"  CASCADE ({cheapest} -> {best['arm']}): {c_res/len(full)*100:.1f}%  "
          f"${c_cost:.3f}  -> {bcost/c_cost:.2f}x cheaper  <-- THE BASELINE TO BEAT")

    print("\n=== complementarity (cheap arm solves what a pricier arm misses) ===")
    pairs = []
    for a, b in itertools.permutations(arms, 2):
        ra = next((r for r in rows if r["arm"] == a), None)
        rb = next((r for r in rows if r["arm"] == b), None)
        if not ra or not rb or ra["per"] >= rb["per"]:
            continue
        only = sum(1 for q in full
                   if cell[(a, q)].get("resolved") and not cell[(b, q)].get("resolved"))
        if only:
            pairs.append((only, a, b))
    for only, a, b in sorted(pairs, reverse=True)[:8]:
        print(f"    {a:26s} solves {only:3d} that {b:26s} misses")
    if not pairs:
        print("    none -- arms are perfectly nested, so there is nothing for a router to learn")


# ============================================================ headroom
# The 2026-02-17 mini-v2.0.0 cohort: 11 models, same scaffold, same day.
# Holding the scaffold fixed is what makes cross-model comparison legitimate.
CLEAN_COHORT_PREFIX = "20260217_mini-v2.0.0_"


def load_headroom_matrix(cohort_only: bool) -> tuple[list[str], list[str], dict]:
    """Load the published SWE-bench bash-only matrix, optionally restricted to one cohort.

    Args:
        cohort_only: If True, keep only the `CLEAN_COHORT_PREFIX` submissions (same
            scaffold, same day); otherwise keep every submission (scaffold varies).

    Returns:
        A (kept_submission_names, shared_instance_ids, raw_parsed_json) tuple. Broken
        submissions (0 resolved with non-trivial spend -- an infra failure recorded as
        a model failure) are excluded from the first element.
    """
    raw = json.loads((ROOT / "results" / "swebench_matrix.json").read_text())
    subs = sorted(raw)
    if cohort_only:
        subs = [s for s in subs if s.startswith(CLEAN_COHORT_PREFIX)]
    # Drop degenerate submissions: 0% resolved with non-trivial spend is a broken
    # run (infra failure recorded as model failure), not a real result.
    keep = []
    for s in subs:
        det = raw[s]["details"]
        res = sum(1 for v in det.values() if v.get("resolved"))
        spend = sum(v.get("cost") or 0 for v in det.values())
        if res == 0 and spend > 1.0:
            print(f"  EXCLUDED {s}: 0 resolved but ${spend:.0f} spent -> broken run")
            continue
        keep.append(s)
    inst = sorted(set.intersection(*(set(raw[s]["details"]) for s in keep)))
    return keep, inst, raw


def arm_stats(raw, subs, inst):
    """Compute each submission's accuracy and total cost over the shared instances.

    Args:
        raw: The raw parsed swebench_matrix.json.
        subs: Submission names to compute stats for.
        inst: Shared instance ids to score over.

    Returns:
        Per-submission `{"arm", "sub", "acc", "cost"}` dicts, sorted by cost ascending.
    """
    rows = []
    for s in subs:
        det = raw[s]["details"]
        acc = sum(1 for i in inst if det[i].get("resolved")) / len(inst)
        cost = sum(det[i].get("cost") or 0 for i in inst)
        rows.append({"arm": s.split("_", 1)[1], "sub": s, "acc": acc, "cost": cost})
    return sorted(rows, key=lambda r: r["cost"])


def cmd_headroom() -> None:
    """CLI (`headroom`): routing headroom in the published SWE-bench bash-only matrix.

    Costs nothing -- this is the bail-early gate: if an ORACLE router (perfect
    foresight, always picks the cheapest arm that solves the instance) cannot beat
    the best single arm on cost at equal accuracy, no learned router can either, and
    the project should stop. All numbers here are PUBLISHED-NOT-MEASURED: they come
    from third-party submissions to the SWE-bench bash-only leaderboard, not our own runs.
    """
    for cohort_only in (True, False):
        tag = ("CLEAN v2.0.0 COHORT (same scaffold, same day)" if cohort_only else
               "ALL SUBMISSIONS (scaffold varies - confounded)")
        print(f"\n{'='*76}\n{tag}\n{'='*76}")
        subs, inst, raw = load_headroom_matrix(cohort_only)
        rows = arm_stats(raw, subs, inst)
        n = len(inst)
        print(f"{len(subs)} arms x {n} instances\n")
        print(tabulate(
            [(r["arm"][:38], f"{r['acc']*100:.1f}%", f"{r['cost']:.2f}", f"{r['cost']/n:.3f}")
             for r in rows],
            headers=["arm", "acc", "total$", "$/inst"], disable_numparse=True))

        best = max(rows, key=lambda r: r["acc"])
        cheapest = rows[0]
        print(f"\n  best single arm : {best['arm']} @ {best['acc']*100:.1f}%  ${best['cost']:.2f}")
        print(f"  cheapest arm    : {cheapest['arm']} @ {cheapest['acc']*100:.1f}%  ${cheapest['cost']:.2f}")

        # ORACLE: per instance, among arms that resolve it, take the cheapest.
        # Unsolvable instances fall back to the cheapest arm (cost still paid).
        det = {s: raw[s]["details"] for s in subs}
        oracle_cost = 0.0
        oracle_solved = 0
        solved_by_any = 0
        for i in inst:
            winners = [(det[s][i].get("cost") or 0, s) for s in subs if det[s][i].get("resolved")]
            if winners:
                solved_by_any += 1
                oracle_solved += 1
                oracle_cost += min(winners)[0]
            else:
                oracle_cost += min((det[s][i].get("cost") or 0) for s in subs)
        print(f"\n  ORACLE routing  : {oracle_solved/n*100:.1f}%  ${oracle_cost:.2f}  "
              f"({oracle_cost/n:.3f}/inst)")
        print(f"  union solvable  : {solved_by_any/n*100:.1f}%  <- ceiling for ANY router")

        # The headline: oracle vs best-single-arm.
        print(f"\n  vs best single arm: accuracy {(oracle_solved/n - best['acc'])*100:+.1f}pp, "
              f"cost {(1 - oracle_cost/best['cost'])*100:+.1f}% cheaper "
              f"({best['cost']/oracle_cost:.1f}x)")

        # ORACLE CONSTRAINED TO PARITY: cheapest routing that matches best-arm accuracy
        # exactly, by only downgrading instances the best arm already solves.
        bs = best["sub"]
        per_inst = []
        for i in inst:
            if det[bs][i].get("resolved"):
                w = [(det[s][i].get("cost") or 0) for s in subs if det[s][i].get("resolved")]
                per_inst.append(min(w))
            else:
                per_inst.append(det[bs][i].get("cost") or 0)
        parity_cost = sum(per_inst)
        print(f"  PARITY oracle   : {best['acc']*100:.1f}% (identical to best arm)  "
              f"${parity_cost:.2f} -> {(1-parity_cost/best['cost'])*100:.1f}% cheaper "
              f"({best['cost']/parity_cost:.1f}x)")

        if cohort_only:
            # Disagreement drives routing: if a cheap arm solves what an expensive one
            # misses, there is signal to learn. Pure dominance means nothing to learn.
            print("\n  pairwise complementarity (cheap solves what expensive misses):")
            for a, b in itertools.combinations(rows, 2):
                if a["cost"] >= b["cost"]:
                    a, b = b, a
                only_cheap = sum(1 for i in inst
                                 if det[a["sub"]][i].get("resolved") and not det[b["sub"]][i].get("resolved"))
                if only_cheap >= 25:
                    print(f"    {a['arm'][:30]:30s} solves {only_cheap:3d} that "
                          f"{b['arm'][:28]:28s} misses")


# ============================================================ price-traces
# Models absent from the price table, mapped to their price-equivalent tier.
PRICE_ALIAS = {"claude-opus-5": "claude-opus-4-8", "claude-sonnet-4-6": "claude-sonnet-5"}


def cmd_price_traces() -> None:
    """CLI (`price-traces`): price real Claude Code sessions to get the measured baseline spend.

    Only the Claude traces carry a `usage` block, so they are the only ones priced
    from observed tokens rather than assumption. This total is the denominator for
    any cost-saving claim on real traffic.
    """
    rows = [json.loads(line) for line in (ROOT / "results" / "trace_tasks.jsonl").open()]
    sessions = [r for r in rows
                if r["trace_type"] == "claude-code" and r["cache_read"] > 0]

    priced, skipped = [], collections.Counter()
    for r in sessions:
        model = PRICE_ALIAS.get(r["primary_model"], r["primary_model"])
        p = STANDARD["anthropic"].get(model)
        if p is None:
            skipped[r["primary_model"]] += 1
            continue
        cost = (r["in_tok"] * p.inp + r["cache_read"] * p.cache_read
                + r["cache_write"] * p.cache_write + r["out_tok"] * p.out) / 1e6
        priced.append((cost, r))

    priced.sort(key=lambda x: -x[0])
    n = len(priced)
    total = sum(c for c, _ in priced)
    costs = sorted(c for c, _ in priced)

    print(f"claude sessions priced: {n}  (skipped: {dict(skipped) or 'none'})")
    print(f"TOTAL realized cost: ${total:,.2f}   mean ${total/n:.3f}/session")
    print(f"per-session $: p50={costs[n//2]:.3f}  p90={costs[int(n*.9)]:.3f}  max={costs[-1]:.3f}")
    head = sum(c for c, _ in priced[:max(1, int(n * 0.2))])
    print(f"top 20% of sessions = {head/total*100:.0f}% of all spend "
          f"<-- routing leverage concentrates here")

    # Correlate the free difficulty proxy against realized cost: if turn_count
    # predicts cost, it is a usable label for router supervision.
    turns = [r["max_turns"] for _, r in priced]
    cc = [c for c, _ in priced]
    mt, mc = statistics.mean(turns), statistics.mean(cc)
    num = sum((t - mt) * (c - mc) for t, c in zip(turns, cc))
    den = (sum((t - mt) ** 2 for t in turns) * sum((c - mc) ** 2 for c in cc)) ** 0.5
    print(f"\npearson r(turn_count, realized_cost) = {num/den:.3f}  (n={n})")

    print("\nmost expensive sessions:")
    for c, r in priced[:6]:
        print(f"  ${c:7.3f} turns={r['max_turns']:3d} out={r['out_tok']:6d} "
              f"cache_read={r['cache_read']:9d} {r['primary_model']:18s} {r['ask'][:58]!r}")
    print("\ncheapest sessions:")
    for c, r in priced[-5:]:
        print(f"  ${c:7.3f} turns={r['max_turns']:3d} out={r['out_tok']:6d} "
              f"{r['primary_model']:18s} {r['ask'][:58]!r}")


# ============================================================ extract-traces
TRACE_DATA = ROOT.parent / "data" / "peter-coding-router"

# Wrappers that mean "this record is machinery, not a human task statement".
MACHINE_PREFIXES = (
    "<task-notification>", "<system-reminder>", "<local-command-",
    "<command-message>", "<command-name>", "Caveat:",
)
CODEX_ASK = re.compile(r"##\s*My request for Codex:\s*(.*)", re.S)
IDE_CTX = re.compile(r"^#\s*Context from my IDE setup:.*?(?=##\s*My request for Codex:)", re.S)


def clean_ask(prompt: str, trace_type: str) -> str:
    """Pull the human task statement out of the raw prompt envelope."""
    if not prompt:
        return ""
    if trace_type == "codex":
        m = CODEX_ASK.search(prompt)
        if m:
            return m.group(1).strip()
        return IDE_CTX.sub("", prompt).strip()
    # Claude Code: strip harness envelopes but keep the human text.
    p = re.sub(r"<(task-notification|system-reminder|local-command-[a-z]+)>.*?</\1>", " ", prompt, flags=re.S)
    p = re.sub(r"</?(command-message|command-name|command-args)>", " ", p)
    return p.strip()


def is_machine(prompt: str) -> bool:
    """Return whether `prompt` is harness machinery rather than a human task statement."""
    s = (prompt or "").lstrip()
    return any(s.startswith(x) for x in MACHINE_PREFIXES)


def cmd_extract_traces() -> None:
    """CLI (`extract-traces`): extract the production query distribution from agent traces.

    Each trace record is one API request; several records share a session. The
    routing decision happens once, when a task ARRIVES, so this aggregates per
    session and keeps the initial user ask plus realized-effort outcomes (turns,
    tokens) as difficulty proxies. Writes results/trace_tasks.jsonl and
    results/trace_summary.json.
    """
    sessions: dict[str, dict] = {}
    files = sorted(TRACE_DATA.glob("*.jsonl"))
    assert files, f"no trace files under {TRACE_DATA}"

    for path in files:
        with path.open() as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                m = d.get("metadata") or {}
                sid = m.get("session_id")
                if not sid:
                    continue
                tt = m.get("trace_type") or "?"
                nmsg = len(d.get("messages") or [])
                prompt = d.get("prompt") or ""
                u = m.get("usage") if isinstance(m.get("usage"), dict) else {}

                s = sessions.setdefault(sid, {
                    "session_id": sid, "trace_type": tt, "n_records": 0,
                    "repo": (m.get("cwd") or "").rstrip("/").split("/")[-1],
                    "models": Counter(), "max_turns": 0, "max_msgs": 0,
                    "out_tok": 0, "in_tok": 0, "cache_read": 0, "cache_write": 0,
                    "_best_nmsg": 10**9, "ask": "", "raw_prompt": "",
                    "cli_version": m.get("cli_version"),
                })
                s["n_records"] += 1
                if m.get("model"):
                    s["models"][m["model"]] += 1
                s["max_turns"] = max(s["max_turns"], m.get("turn_count") or 0)
                s["max_msgs"] = max(s["max_msgs"], nmsg)
                for k, dst in (("output_tokens", "out_tok"), ("input_tokens", "in_tok"),
                               ("cache_read_input_tokens", "cache_read"),
                               ("cache_creation_input_tokens", "cache_write")):
                    v = u.get(k)
                    if isinstance(v, (int, float)):
                        s[dst] += v

                # The initial ask = earliest request in the session (fewest messages)
                # that is a human statement rather than harness machinery.
                if not is_machine(prompt) and prompt.strip() and nmsg < s["_best_nmsg"]:
                    ask = clean_ask(prompt, tt)
                    if len(ask) >= 12:
                        s["_best_nmsg"] = nmsg
                        s["ask"] = ask
                        s["raw_prompt"] = prompt[:20000]

    rows = []
    for s in sessions.values():
        s.pop("_best_nmsg", None)
        s["primary_model"] = s["models"].most_common(1)[0][0] if s["models"] else None
        s["models"] = dict(s["models"])
        s["ask_chars"] = len(s["ask"])
        if s["ask"]:
            rows.append(s)

    rows.sort(key=lambda r: (r["trace_type"], -r["max_turns"]))
    out = ROOT / "results" / "trace_tasks.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    def q(a, ks=(0.5, 0.9, 0.99)):
        """Summarize a numeric list as n/min/max/mean plus the requested percentiles."""
        a = sorted(a)
        if not a:
            return {}
        r = {"n": len(a), "min": a[0], "max": a[-1], "mean": round(sum(a) / len(a), 1)}
        for k in ks:
            r[f"p{int(k*100)}"] = a[min(len(a) - 1, int(len(a) * k))]
        return r

    summary = {"total_sessions_seen": len(sessions), "sessions_with_usable_ask": len(rows)}
    for tt in sorted({r["trace_type"] for r in rows}):
        g = [r for r in rows if r["trace_type"] == tt]
        summary[tt] = {
            "sessions": len(g),
            "ask_chars": q([r["ask_chars"] for r in g]),
            "max_turns": q([r["max_turns"] for r in g]),
            "out_tok": q([r["out_tok"] for r in g]),
            "cache_read": q([r["cache_read"] for r in g]),
            "repos": Counter(r["repo"] for r in g).most_common(10),
            "models": Counter(r["primary_model"] for r in g).most_common(10),
            # Turn count is the difficulty proxy the router must learn to predict.
            "turn_buckets": {
                "1 (trivial)": sum(1 for r in g if r["max_turns"] <= 1),
                "2-5": sum(1 for r in g if 2 <= r["max_turns"] <= 5),
                "6-20": sum(1 for r in g if 6 <= r["max_turns"] <= 20),
                "21-100": sum(1 for r in g if 21 <= r["max_turns"] <= 100),
                ">100 (epic)": sum(1 for r in g if r["max_turns"] > 100),
            },
        }
    (ROOT / "results" / "trace_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"\nwrote {len(rows)} sessions -> {out}")


if __name__ == "__main__":
    import fire

    fire.Fire({
        "analyze-phase1": cmd_analyze_phase1,
        "headroom": cmd_headroom,
        "price-traces": cmd_price_traces,
        "extract-traces": cmd_extract_traces,
    })
