"""Dataset loaders: SWE-rebench family + DeepSWE v1.1 -- FREE routing supervision.

Two pure loaders with no cross-references to each other. `load_deepswe()` is used by
experiments.py/analysis.py.

======================================================================== swe_rebench
SWE-rebench family loader: task pools for our own sweep + the only FREE labels that exist.

WHY this file is shaped the way it is:

The three nebius task datasets (V2 / v1 / leaderboard) contain ZERO model outcomes. Every
column in all three is a task-definition field, so on their own they cannot supervise a
router at all -- they are a *sweep plan*, not a matrix. That is verified here, not assumed:
`python router/datasets_b.py swe-rebench` prints the column list it actually read.

But outcomes for these instance ids do exist in two adjacent nebius dumps, and they are
free, per-task, and -- because both dumps ship MANY rollouts per instance -- GRADED, not
binary. A per-task pass rate over ~10-21 rollouts is exactly the continuous label the
binary-overstates-the-tier-gap lesson says we need:

  * nebius/SWE-agent-trajectories   80,036 rollouts, `model_name` in {llama-8b, -70b, -405b}
                                    -> a same-scaffold 3-tier ladder (the routable part)
  * nebius/SWE-rebench-openhands-trajectories  67,074 rollouts, resolved 0/1, and NO model
                                    column at all. The repo's own config.toml sets
                                    [llm] model = "qwen3-coder-480b-a35b-instruct", but the
                                    parquet does not name a model per row, so the arm is
                                    labelled `openhands:model-unrecorded` -- UNVERIFIED.

Two things are deliberately NOT smoothed over:
  * The two dumps use DIFFERENT scaffolds (SWE-agent vs OpenHands). Comparing across them
    breaks the "only the model differs" invariant router_core.py depends on, so the
    scaffold is part of every arm name and `notes` says so. The clean routing sub-matrix is
    the three swe-agent arms.
  * Infrastructure stops are dropped from the denominator instead of being scored as model
    failures (this has bitten us four times). `exit_status` drives that, and the summary
    prints how many rollouts each rule removed.

Cost: NOT AVAILABLE. No cost/token field exists in any of the five datasets, and the public
swe-rebench.com leaderboard only publishes per-model averages, so `cost` is None by design.

Pools:
  free        -- instances that have >=`min_arms` free-label arms. Runs today, real numbers.
  v2_python   -- the 7,243 parse_log_pytest rows of SWE-rebench-V2 (sweep target).
  v2_all      -- all 32,079 V2 rows (30+ non-python log parsers; the hidden work).
  leaderboard -- the 860-row post-cutoff pool (15 monthly splits, union == `test`).
The three sweep pools have no labels, so they read our own episodes from
results/episodes_swerebench/*.json and RAISE if that directory is empty rather than
inventing a matrix.

======================================================================== deepswe
DeepSWE v1.1 (Datacurve) -> router supervision matrix, from FREE published labels.

Why this exists: this is the only SWE-shaped source we have found that already ships a dense
(arm x task) outcome table, so we get routing supervision without spending a GPU-hour or an API
dollar. 50 configs (18 models x reasoning effort, all on ONE harness -- mini-swe-agent -- so the
arms are comparable) x 113 long-horizon tasks, backed by 22,586 individual rollouts, ~4 per cell.

Worth loading over SWE-bench Verified because it is not saturated and every task is routable:
main_deepswe() recomputes best single arm, the any-arm oracle, and the all-arms/no-arms counts.

Three project lessons are handled explicitly here rather than assumed away:
  * GRADED, not binary. `f2p_passed/f2p_total` is the fail-to-pass test fraction and is strictly
    intermediate on 41% of scored rows, so the default metric="f2p" is a real [0,1] score.
    `partial` also exists but is diluted by the p2p regression suite -- prefer f2p.
  * Infra failure != model failure. Rows with included_in_score=False (model_routing_404,
    provider_timeout, verifier_timeout, ...) are MISSING DATA and are dropped, which leaves 2 of
    5,650 cells empty; those come back as None, never 0.0. Rows the publisher deliberately scored
    as failures (agent_timeout, context_window_exceeded) have included_in_score=True and are kept
    as failures -- that is what reproduces the published leaderboard, verified by _crosscheck().
  * No invented numbers. There is NO explicit difficulty/rating field anywhere (checked
    tasks.json, task.toml, manifest.json), so `difficulty` is DERIVED: the task's mean binary pass
    rate across arms. `cost` is None on the 5 cells with no priced trial (Fable-5 on one task).

Sources, fetched on first call and cached under data/deepswe/:
  * labels      https://deepswe.datacurve.ai/artifacts/v1.1/trials.json    (no auth, 37.3 MB)
  * tasks       https://deepswe.datacurve.ai/artifacts/v1/tasks.json       (no auth)
  * leaderboard https://deepswe.datacurve.ai/artifacts/v1.1/leaderboard-live.json (cross-check)
  * prompts     github.com/datacurve-ai/deep-swe tarball (Apache-2.0) -- tasks.json carries only a
    one-line description, so the real agent prompt (instruction.md) has to come from the repo.
The HF mirrors (datacurve/deep-swe, datacurve/deep-swe-leaderboard) are gated and are NOT used.
"""
from __future__ import annotations

import collections
import glob
import json
import math
import os
import pathlib
import statistics
import tarfile
import urllib.request
from collections.abc import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parent.parent
SWE_REBENCH_CACHE = ROOT / "data" / "swe_rebench"
DEEPSWE_CACHE = ROOT / "data" / "deepswe"
EPISODE_DIR = pathlib.Path(
    os.environ.get("SWEREBENCH_EPISODES", str(ROOT / "results" / "episodes_swerebench"))
)

# --------------------------------------------------------------------------- swe_rebench constants
HF = "https://huggingface.co/datasets"
# refs/convert/parquet is the datasets-server's normalised copy: one shard, stable path.
V2_URL = f"{HF}/nebius/SWE-rebench-V2/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
LB_URL = (f"{HF}/nebius/SWE-rebench-leaderboard/resolve/refs%2Fconvert%2Fparquet/default/"
          f"{{split}}/0000.parquet")
LB_MONTHS = (
    "2025_01", "2025_02", "2025_03", "2025_04", "2025_05", "2025_06", "2025_07", "2025_08",
    "2025_09", "2025_10", "2025_11", "2025_12", "2026_01", "2026_02", "2026_03",
)
V1_FILES = (
    "datasets/nebius/SWE-rebench/data/filtered-00000-of-00001.parquet",
    "datasets/nebius/SWE-rebench/data/test-00000-of-00002.parquet",
    "datasets/nebius/SWE-rebench/data/test-00001-of-00002.parquet",
)
V1_DEF_COLS = ("instance_id", "repo", "created_at", "problem_statement", "meta",
               "FAIL_TO_PASS", "PASS_TO_PASS", "image_name", "install_config")
OPENHANDS_FILE = "datasets/nebius/SWE-rebench-openhands-trajectories/trajectories.parquet"
SWEAGENT_FILES = tuple(
    f"datasets/nebius/SWE-agent-trajectories/data/train-{i:05d}-of-00012.parquet"
    for i in range(12)
)

# Rollouts killed by the harness, not by the model. Dropping them from the denominator is
# the whole point; scoring them 0 is the mistake we have made four times.
OPENHANDS_INFRA = ("Timeout:", "ServiceUnavailable", "litellm", "unexpected error")
SWEAGENT_INFRA = ("early_exit", "exit_cost")
# Kept as genuine model failures, but counted separately so the choice stays visible:
# openhands "reached maximum iteration" (100 turns) and swe-agent exit_context/exit_format.
OPENHANDS_LIMIT = ("maximum iteration", "StuckInLoop")
SWEAGENT_LIMIT = ("exit_context", "exit_format")


# ------------------------------------------------------------------ swe_rebench: download / cache
def _fetch_swe_rebench(url: str, dest: pathlib.Path) -> pathlib.Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    part.rename(dest)
    return dest


def _hf_project(files: Iterable[str], columns: Iterable[str], dest: pathlib.Path) -> pa.Table:
    """Read only `columns` from HF-hosted parquet and cache the projection locally.

    The two trajectory dumps are GBs of conversation text; we want a handful of scalar
    columns. Column projection over range requests avoids downloading the rest.
    """
    if dest.exists():
        return pq.read_table(dest)
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    parts = []
    for f in files:
        with fs.open(f, "rb") as fh:
            parts.append(pq.ParquetFile(fh).read(columns=list(columns)))
    table = pa.concat_tables(parts)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dest)
    return table


def _v2_table() -> pa.Table:
    path = _fetch_swe_rebench(V2_URL, SWE_REBENCH_CACHE / "v2_train.parquet")
    return pq.read_table(path, columns=["instance_id", "repo", "language", "created_at",
                                        "problem_statement", "install_config", "meta",
                                        "FAIL_TO_PASS", "PASS_TO_PASS", "image_name"])


def _leaderboard_table() -> tuple[pa.Table, dict[str, str]]:
    """Returns the 860-row union plus instance_id -> monthly split (the post-cutoff key)."""
    month_of: dict[str, str] = {}
    for m in LB_MONTHS:
        t = pq.read_table(_fetch_swe_rebench(LB_URL.format(split=m), SWE_REBENCH_CACHE / f"lb_{m}.parquet"),
                          columns=["instance_id"])
        for i in t.column("instance_id").to_pylist():
            month_of[i] = m
    t = pq.read_table(_fetch_swe_rebench(LB_URL.format(split="test"), SWE_REBENCH_CACHE / "lb_test.parquet"),
                      columns=["instance_id", "repo", "created_at", "problem_statement",
                               "meta", "FAIL_TO_PASS", "PASS_TO_PASS", "docker_image",
                               "install_config"])
    return t, month_of


def _v1_defs() -> pa.Table:
    return _hf_project(V1_FILES, V1_DEF_COLS, SWE_REBENCH_CACHE / "v1_defs.parquet")


# ------------------------------------------------------------------ swe_rebench: free outcome labels
class Cell:
    """One (arm, task) cell: passes over kept rollouts, plus what was thrown away."""

    __slots__ = ("passed", "kept", "infra", "limit")

    def __init__(self) -> None:
        self.passed = self.kept = self.infra = self.limit = 0

    @property
    def score(self) -> float | None:
        return self.passed / self.kept if self.kept else None


def _classify(exit_status: str | None, infra: tuple[str, ...], limit: tuple[str, ...]) -> str:
    s = exit_status or ""
    if any(k in s for k in infra):
        return "infra"
    return "limit" if any(k in s for k in limit) else "ok"


def _openhands_cells() -> dict[str, dict[str, Cell]]:
    t = _hf_project([OPENHANDS_FILE], ["instance_id", "resolved", "exit_status"],
                    SWE_REBENCH_CACHE / "outcomes_openhands.parquet")
    out: dict[str, Cell] = collections.defaultdict(Cell)
    for iid, res, ex in zip(t.column("instance_id").to_pylist(),
                            t.column("resolved").to_pylist(),
                            t.column("exit_status").to_pylist()):
        c = out[iid]
        kind = _classify(ex, OPENHANDS_INFRA, OPENHANDS_LIMIT)
        if kind == "infra":
            c.infra += 1
            continue
        c.limit += kind == "limit"
        c.kept += 1
        c.passed += int(res or 0)
    return {"openhands:model-unrecorded": dict(out)}


def _sweagent_cells() -> dict[str, dict[str, Cell]]:
    t = _hf_project(SWEAGENT_FILES, ["instance_id", "model_name", "target", "exit_status"],
                    SWE_REBENCH_CACHE / "outcomes_sweagent.parquet")
    out: dict[str, dict[str, Cell]] = collections.defaultdict(lambda: collections.defaultdict(Cell))
    for iid, mdl, tgt, ex in zip(t.column("instance_id").to_pylist(),
                                 t.column("model_name").to_pylist(),
                                 t.column("target").to_pylist(),
                                 t.column("exit_status").to_pylist()):
        c = out[f"swe-agent:{mdl.removeprefix('swe-agent-')}"][iid]
        kind = _classify(ex, SWEAGENT_INFRA, SWEAGENT_LIMIT)
        if kind == "infra":
            c.infra += 1
            continue
        c.limit += kind == "limit"
        c.kept += 1
        c.passed += int(bool(tgt))
    return {a: dict(v) for a, v in out.items()}


# ------------------------------------------------------------------ swe_rebench: our own sweep episodes
def _sweep_cells(pool: str) -> dict[str, dict[str, float]]:
    """Read results/episodes_swerebench/*.json written by a future sweep script.

    Expected per-episode fields (this is the contract a sweep must honour):
      arm, instance_id, harness_error | None, cost_usd | None,
      and EITHER graded (float in [0,1]) OR f2p_passed/f2p_total/p2p_passed/p2p_total.
    graded = |(F2P u P2P) n passed| / |F2P u P2P|, i.e. the fraction of curated tests that
    pass -- boolean `resolved` is recoverable as (all F2P and all P2P pass).
    """
    files = sorted(glob.glob(str(EPISODE_DIR / "*.json")))
    if not files:
        raise RuntimeError(
            f"pool={pool!r} has NO outcome labels: SWE-rebench ships task definitions only, "
            f"and no episodes were found in {EPISODE_DIR}. Run a sweep first "
            f"(docker-in-E2B, one image per instance) or use pool='free'."
        )
    out: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for f in files:
        r = json.loads(pathlib.Path(f).read_text())
        if r.get("harness_error"):  # infra stop -> missing data, never a 0
            continue
        g = r.get("graded")
        if g is None:
            den = (r.get("f2p_total") or 0) + (r.get("p2p_total") or 0)
            if not den:
                continue
            g = ((r.get("f2p_passed") or 0) + (r.get("p2p_passed") or 0)) / den
        out[r["arm"]][r["instance_id"]] = float(g)
    return dict(out)


# ------------------------------------------------------------------ swe_rebench: assembly
def _v2_fields(t: pa.Table) -> tuple[dict, dict, dict, dict]:
    ids = t.column("instance_id").to_pylist()
    text = dict(zip(ids, t.column("problem_statement").to_pylist()))
    group = dict(zip(ids, t.column("repo").to_pylist()))
    meta = t.column("meta").to_pylist()
    # 119 of 32,079 rows have no llm_metadata -> difficulty stays None, never guessed.
    diff = {i: ((m.get("llm_metadata") or {}).get("difficulty")) for i, m in zip(ids, meta)}
    parser = {i: (ic or {}).get("log_parser") for i, ic in
              zip(ids, t.column("install_config").to_pylist())}
    return text, group, diff, parser


def _v1_fields(t: pa.Table) -> tuple[dict, dict, dict]:
    ids = t.column("instance_id").to_pylist()
    text = dict(zip(ids, t.column("problem_statement").to_pylist()))
    group = dict(zip(ids, t.column("repo").to_pylist()))
    meta = t.column("meta").to_pylist()
    # difficulty_score is an int 0..4 with -1 as the model's "cannot score" sentinel;
    # kept raw as a float because we have no verified mapping for -1.
    diff: dict[str, float | None] = {}
    for i, m in zip(ids, meta):
        s = (m.get("llm_score") or {}).get("difficulty_score")
        diff[i] = None if s is None else float(s)
    return text, group, diff


def load_swe_rebench(pool: str = "free", min_arms: int = 2, min_rollouts: int = 1) -> dict:
    """Build a router matrix. See the module docstring for the pool semantics.

    score[i][j] is a FLOAT in [0,1] (pass rate over kept rollouts for `free`, graded
    test-pass fraction for a sweep) and float('nan') where the cell was never run.
    """
    if pool == "free":
        cells = {**_sweagent_cells(), **_openhands_cells()}
        defs = _v1_defs()
        text, group, diff = _v1_fields(defs)
        arms = sorted(cells)
        per_task = collections.Counter()
        for a in arms:
            for iid, c in cells[a].items():
                if iid in text and c.kept >= min_rollouts:
                    per_task[iid] += 1
        tasks = sorted(i for i, n in per_task.items() if n >= min_arms)
        score = [[float(cells[a].get(t).score) if (cells[a].get(t) and cells[a][t].kept >= min_rollouts)
                  else math.nan for t in tasks] for a in arms]
        rollouts = [[cells[a][t].kept if t in cells[a] else 0 for t in tasks] for a in arms]
        dropped = {a: {"infra": sum(c.infra for c in cells[a].values()),
                       "turn_or_ctx_limit_kept_as_model_failure":
                           sum(c.limit for c in cells[a].values())} for a in arms}
        notes = ("free labels; score = resolved rate over kept rollouts (GRADED). "
                 "MIXED SCAFFOLDS: swe-agent:* arms share one scaffold, "
                 "openhands:model-unrecorded does not and its model is UNVERIFIED. "
                 "cost=None: no cost/token field exists in any SWE-rebench dataset.")
    elif pool in ("v2_python", "v2_all", "leaderboard"):
        if pool == "leaderboard":
            defs, month_of = _leaderboard_table()
            ids = defs.column("instance_id").to_pylist()
            text = dict(zip(ids, defs.column("problem_statement").to_pylist()))
            group = {i: month_of.get(i, "unknown") for i in ids}  # month == contamination key
            diff = {i: None for i in ids}  # leaderboard splits carry NO difficulty field
            pool_ids = set(ids)
        else:
            t = _v2_table()
            text, group, diff, parser = _v2_fields(t)
            pool_ids = ({i for i, p in parser.items() if p == "parse_log_pytest"}
                        if pool == "v2_python" else set(text))
        cells_sweep = _sweep_cells(pool)  # raises when no sweep has been run
        arms = sorted(cells_sweep)
        tasks = sorted(i for i in pool_ids
                       if sum(i in cells_sweep[a] for a in arms) >= min_arms)
        score = [[cells_sweep[a].get(t, math.nan) for t in tasks] for a in arms]
        rollouts = [[1 if t in cells_sweep[a] else 0 for t in tasks] for a in arms]
        dropped = {}
        notes = f"pool={pool}; score = graded curated-test pass fraction from our own sweep."
    else:
        raise ValueError(f"unknown pool {pool!r}")

    return {
        "arms": arms,
        "tasks": tasks,
        "score": score,
        "cost": None,  # nothing in these datasets carries per-task cost or tokens
        "text": {t: text[t] for t in tasks},
        "difficulty": {t: diff.get(t) for t in tasks},
        "group": {t: group.get(t) for t in tasks},
        "rollouts": rollouts,
        "dropped": dropped,
        "notes": notes,
    }


# ------------------------------------------------------------------ swe_rebench: summary
def _pool_report() -> None:
    t = _v2_table()
    text, group, diff, parser = _v2_fields(t)
    print(f"V2 task pool: {t.num_rows:,} rows, {len(set(group.values())):,} repos, "
          f"created_at {min(t.column('created_at').to_pylist())} .. "
          f"{max(t.column('created_at').to_pylist())}")
    print(f"  language     {collections.Counter(t.column('language').to_pylist()).most_common(6)}")
    print(f"  log_parser   {collections.Counter(parser.values()).most_common(4)}")
    print(f"  difficulty   {dict(collections.Counter(diff.values()))}")
    # Medians, not means: |F2P| runs to 117,906 on one row, so the mean is meaningless.
    f2p = sorted(len(x) for x in t.column("FAIL_TO_PASS").to_pylist())
    p2p = sorted(len(x) for x in t.column("PASS_TO_PASS").to_pylist())
    print(f"  |F2P| median {statistics.median(f2p)} max {f2p[-1]:,} ==1 for "
          f"{sum(v == 1 for v in f2p):,}; |P2P| median {statistics.median(p2p)} "
          f"max {p2p[-1]:,} ==0 for {sum(v == 0 for v in p2p):,} "
          "(this is the graded-score denominator)")
    lb, month_of = _leaderboard_table()
    print(f"leaderboard pool: {lb.num_rows} rows, months {len(set(month_of.values()))}, "
          f"created_at {min(lb.column('created_at').to_pylist())} .. "
          f"{max(lb.column('created_at').to_pylist())}")
    for pool in ("v2_python", "leaderboard"):
        try:
            load_swe_rebench(pool)
        except RuntimeError as e:
            print(f"  pool={pool}: NO LABELS -> {e}")


def main_swe_rebench() -> None:
    _pool_report()
    d = load_swe_rebench("free", min_arms=2)
    arms, tasks, score = d["arms"], d["tasks"], d["score"]
    n_cells = len(arms) * len(tasks)
    filled = sum(not math.isnan(v) for row in score for v in row)
    print(f"\npool=free  n_arms={len(arms)}  n_tasks={len(tasks)}  "
          f"cells={n_cells:,} filled={filled:,} sparsity={1 - filled / n_cells:.3f}")
    print(f"{'arm':34s} {'tasks':>6s} {'mean score':>11s} {'rollouts/task':>14s} "
          f"{'infra dropped':>14s}")
    for i, a in enumerate(arms):
        vals = [v for v in score[i] if not math.isnan(v)]
        roll = [r for r in d["rollouts"][i] if r]
        print(f"{a:34s} {len(vals):6d} {sum(vals)/len(vals):11.4f} "
              f"{sum(roll)/len(roll):14.1f} {d['dropped'][a]['infra']:14d}")
    graded = sum(1 for i in range(len(arms)) for j in range(len(tasks))
                 if d["rollouts"][i][j] > 1 and not math.isnan(score[i][j]))
    frac01 = sum(1 for row in score for v in row if not math.isnan(v) and v in (0.0, 1.0))
    print(f"cells with >1 rollout (i.e. genuinely graded): {graded:,} / {filled:,}; "
          f"cells still exactly 0.0 or 1.0: {frac01:,}")
    print(f"difficulty (v1 llm_score.difficulty_score): "
          f"{dict(collections.Counter(d['difficulty'].values()))}")
    print(f"groups (repos): {len(set(d['group'].values()))}  cost: {d['cost']}")
    print(f"notes: {d['notes']}")


# --------------------------------------------------------------------------- deepswe constants
TRIALS_URL = "https://deepswe.datacurve.ai/artifacts/v1.1/trials.json"
DEEPSWE_TASKS_URL = "https://deepswe.datacurve.ai/artifacts/v1/tasks.json"
LEADERBOARD_URL = "https://deepswe.datacurve.ai/artifacts/v1.1/leaderboard-live.json"
TARBALL_URL = "https://codeload.github.com/datacurve-ai/deep-swe/tar.gz/refs/heads/main"

# Every instruction.md ends with this harness boilerplate. It is identical on all 113 tasks and
# carries no task signal, so strip it before the text reaches an embedder.
PROMPT_BOILERPLATE = (
    "\nIMPORTANT: Please work on this in a new branch from main and "
    "commit everything when you are done.\n"
)

METRICS = ("f2p", "partial", "passed", "reward")


def _fetch_deepswe(url: str, name: str) -> pathlib.Path:
    DEEPSWE_CACHE.mkdir(parents=True, exist_ok=True)
    dest = DEEPSWE_CACHE / name
    if not dest.exists() or dest.stat().st_size == 0:
        tmp = dest.with_name(dest.name + ".part")  # atomic: never leave a half file cached
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    return dest


def _task_dir() -> pathlib.Path:
    """Extract only instruction.md/task.toml/manifest.json from the 3.8 MB GitHub tarball."""
    out = DEEPSWE_CACHE / "deep-swe-main" / "tasks"
    if not out.is_dir() or not any(out.glob("*/instruction.md")):
        tar_path = _fetch_deepswe(TARBALL_URL, "deep-swe-main.tar.gz")
        wanted = ("instruction.md", "task.toml", "manifest.json")
        with tarfile.open(tar_path) as tf:
            members = [m for m in tf.getmembers() if m.name.endswith(wanted)]
            tf.extractall(DEEPSWE_CACHE, members=members, filter="data")
    return out


def _prompt(task_id: str) -> str:
    raw = (_task_dir() / task_id / "instruction.md").read_text()
    return raw[: -len(PROMPT_BOILERPLATE)] if raw.endswith(PROMPT_BOILERPLATE) else raw


def load_deepswe(metric: str = "f2p") -> dict:
    """Dense (arm x task) matrix from the published DeepSWE v1.1 per-trial table.

    metric: "f2p" (graded fail-to-pass fraction, DEFAULT), "partial" (graded, p2p-diluted),
            "passed"/"reward" (binary -- overstates the tier gap ~6x on this very data).
    Cells with no scored trial are None in both `score` and `cost`, never 0.0.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")

    trials = json.loads(_fetch_deepswe(TRIALS_URL, "trials.json").read_text())
    tasks = json.loads(_fetch_deepswe(DEEPSWE_TASKS_URL, "tasks.json").read_text())
    rows, task_rows = trials["rows"], tasks["rows"]
    if len(rows) != trials["n_trials"] or len(task_rows) != tasks["n_tasks"]:
        raise ValueError("artifact row count disagrees with its own header; refusing to load")

    scored = [r for r in rows if r["included_in_score"]]
    arms = sorted({r["config"] for r in scored})
    task_meta = {t["id"]: t for t in task_rows}
    task_ids = sorted(task_meta)
    if {r["task_name"] for r in scored} - set(task_ids):
        raise ValueError("trials reference task ids absent from tasks.json")

    cells: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for r in scored:
        cells[(r["config"], r["task_name"])].append(r)

    score: list[list[float | None]] = []
    cost: list[list[float | None]] = []
    for a in arms:
        srow: list[float | None] = []
        crow: list[float | None] = []
        for t in task_ids:
            trs = cells.get((a, t), [])
            vals = [float(r[metric]) for r in trs if r.get(metric) is not None]
            priced = [r["cost_usd"] for r in trs if r.get("cost_usd") is not None]
            srow.append(statistics.fmean(vals) if vals else None)
            crow.append(statistics.fmean(priced) if priced else None)
        score.append(srow)
        cost.append(crow)

    # Derived difficulty: mean binary pass rate over all arms. Deliberately independent of
    # `metric` so the difficulty axis does not move when the score definition changes.
    difficulty: dict[str, float] = {}
    for t in task_ids:
        p = [float(r["passed"]) for a in arms for r in cells.get((a, t), [])]
        if not p:
            raise ValueError(f"task {t} has no scored trial on any arm")
        difficulty[t] = statistics.fmean(p)

    return {
        "arms": arms,
        "tasks": task_ids,
        "score": score,
        "cost": cost,
        "text": {t: _prompt(t) for t in task_ids},
        "difficulty": difficulty,
        # 91 repos over 113 tasks: 38 tasks share a repo with another, so grouping by repository
        # is the contest_of() analogue that stops a same-repo neighbour leaking across a CV split.
        "group": {t: task_meta[t]["repository"] for t in task_ids},
    }


def _crosscheck(arms: list[str]) -> str:
    """Recompute pass@1 per arm from raw trials and diff against the published leaderboard."""
    lb = json.loads(_fetch_deepswe(LEADERBOARD_URL, "leaderboard-live.json").read_text())
    published = {r["config"]: r["pass_at_1"] for r in lb["rows"]}
    rows = json.loads(_fetch_deepswe(TRIALS_URL, "trials.json").read_text())["rows"]
    mine: dict[str, list[float]] = collections.defaultdict(list)
    for r in rows:
        if r["included_in_score"]:
            mine[r["config"]].append(float(r["passed"]))
    bad = [a for a in arms
           if a not in published or abs(statistics.fmean(mine[a]) - published[a]) > 1e-9]
    return f"{len(arms) - len(bad)}/{len(arms)} arms reproduce published pass@1, {len(bad)} off"


def _mean(row: list[float | None]) -> float:
    return statistics.fmean(v for v in row if v is not None)


def main_deepswe() -> None:
    d = load_deepswe()
    arms, task_ids, score, cost = d["arms"], d["tasks"], d["score"], d["cost"]
    n_a, n_t = len(arms), len(task_ids)
    filled = sum(v is not None for row in score for v in row)
    n_cost = sum(v is not None for row in cost for v in row)
    print(f"DeepSWE v1.1  n_arms={n_a}  n_tasks={n_t}  metric=f2p (graded)")
    print(f"score cells {filled}/{n_a * n_t} filled ({100 * filled / (n_a * n_t):.2f}%), "
          f"{n_a * n_t - filled} None (provider_timeout = missing label, not model failure)")
    print(f"cost cells {n_cost}/{n_a * n_t} filled ({100 * n_cost / (n_a * n_t):.2f}%)")
    print(f"crosscheck: {_crosscheck(arms)}")
    print(f"text: instruction.md chars min={min(len(v) for v in d['text'].values())} "
          f"median={statistics.median(len(v) for v in d['text'].values()):.0f} "
          f"max={max(len(v) for v in d['text'].values())}")
    print(f"groups: {len(set(d['group'].values()))} repos over {n_t} tasks; "
          f"difficulty (DERIVED mean pass rate) min={min(d['difficulty'].values()):.3f} "
          f"median={statistics.median(d['difficulty'].values()):.3f} "
          f"max={max(d['difficulty'].values()):.3f}")

    binary = load_deepswe("passed")["score"]
    print(f"\n{'arm':<44}{'f2p':>7}{'pass@1':>8}{'$/task':>8}{'n':>5}")
    for i in sorted(range(n_a), key=lambda i: -_mean(score[i])):
        print(f"{arms[i]:<44}{_mean(score[i]):>7.3f}{_mean(binary[i]):>8.3f}"
              f"{_mean(cost[i]):>8.2f}{sum(v is not None for v in score[i]):>5}")

    # Routing headroom, computed binary so it is comparable to a published pass@1. "solves" means
    # the arm's majority of trials passed; unpriced cells are excluded from the cost sums rather
    # than counted as free, so both sums are over the same 113 tasks only where a price exists.
    best = max(range(n_a), key=lambda i: _mean(binary[i]))
    solved = [[i for i in range(n_a) if (binary[i][j] or 0.0) > 0.5] for j in range(n_t)]
    always = sum(cost[best][j] for j in range(n_t) if cost[best][j] is not None)
    cheapest = [min((cost[i][j] for i in s if cost[i][j] is not None), default=None)
                for j, s in enumerate(solved)]
    cheap = sum(c for c in cheapest if c is not None)
    print(f"\nbest single arm {arms[best]} pass@1={_mean(binary[best]):.3f}, "
          f"${always:.0f} to run all {n_t} tasks")
    print(f"oracle any-arm-solves={statistics.fmean(float(bool(s)) for s in solved):.3f}; "
          f"cheapest-arm-that-solves ${cheap:.0f} over "
          f"{sum(c is not None for c in cheapest)} priced tasks ({always / cheap:.1f}x cheaper)")
    print(f"tasks solved by ALL arms: {sum(len(s) == n_a for s in solved)}; "
          f"by NO arm: {sum(not s for s in solved)}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = ap.add_subparsers(dest="command", required=True)

    subparsers.add_parser("swe-rebench").set_defaults(func=lambda ns: main_swe_rebench())
    subparsers.add_parser("deepswe").set_defaults(func=lambda ns: main_deepswe())

    ns = ap.parse_args()
    ns.func(ns)
