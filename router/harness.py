"""Execution/eval harness: unified OpenAI+Anthropic agent client, E2B sandbox exec, and
the LiveCodeBench problem loader/grader. Sandbox exec and grading live alongside the
agent client because both run untrusted, model-generated code and must stay together.
"""
from __future__ import annotations

import base64
import dataclasses
import json
import os
import pathlib
import pickle
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from collections.abc import Callable
from typing import Any

import anthropic
import openai
from pydantic import BaseModel, ConfigDict

# Needed so `python router/harness.py` (run standalone, e.g. for its LiveCodeBench-loader
# demo below) can resolve `router.router_core` even without the repo root pre-set on
# sys.path. Harmless / a no-op when imported normally as `router.harness`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from router.router_core import Arm  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "data" / "lcb_test6.jsonl"


# ============================================================================ llm
def load_env(path: pathlib.Path | None = None) -> None:
    """Load KEY=VALUE lines from `.env.local` into `os.environ`, without overwriting.

    Args:
        path: Path to the env file; defaults to `.env.local` at the repo root.
    """
    p = path or pathlib.Path(__file__).resolve().parent.parent / ".env.local"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


class Usage(BaseModel):
    """Accumulated token usage. Reasoning/thinking bills as output on both providers."""

    inp: int = 0
    cache_read: int = 0
    cache_write: int = 0
    out: int = 0
    reasoning: int = 0
    requests: int = 0

    def add(self, **kw: int) -> None:
        """Accumulate token counts in place, treating a missing/None count as zero."""
        for k, v in kw.items():
            setattr(self, k, getattr(self, k) + (v or 0))

    def cost(self, arm: Arm, *, intro: bool = False) -> float:
        """Price this accumulated usage under `arm`'s rates.

        Args:
            arm: The arm whose price table row to use.
            intro: If True, price at the introductory rate where one exists.

        Returns:
            The USD cost of all usage accumulated so far.
        """
        return arm.cost(inp=self.inp, cache_read=self.cache_read,
                        cache_write=self.cache_write, out=self.out, intro=intro)


class Tool(BaseModel):
    """One agent tool: name, description, JSON schema, and the function that runs it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    schema: dict  # arbitrary per-tool JSON schema -- shape genuinely varies per tool
    run: Callable[[dict], str]


class ToolCallRecord(BaseModel):
    """One tool call captured in a transcript turn."""

    name: str
    # Anthropic content blocks give already-parsed args (dict); OpenAI's function_call
    # items give the raw JSON-encoded arguments string, parsed separately just before
    # the tool actually runs. Recorded as received, so this stays a union of both.
    input: dict[str, Any] | str


class TranscriptEntry(BaseModel):
    """One agent turn: assistant text plus any tool calls made and their results."""

    turn: int
    text: str
    calls: list[ToolCallRecord]
    results: list[str] | None = None


class EpisodeResult(BaseModel):
    """The outcome of one completed agent episode."""

    arm_id: str
    turns: int
    usage: Usage
    cost_usd: float
    wall_s: float
    stop: str            # end_turn | max_turns | max_tokens | error | refusal
    error: str | None
    transcript: list[TranscriptEntry]


class AgentRunner:
    """Runs one tool-calling episode to completion under a single arm.

    The scaffold, tool set, prompts, and stopping rule are identical across every arm;
    only the model and its reasoning config (`Arm`) differ -- that invariant is what
    makes cross-arm comparisons in this project meaningful, and it is enforced here.
    """

    # Reasoning/thinking tokens are billed as output AND count against max_tokens. If they
    # exhaust the cap the model spends turn 1 reasoning, never emits a tool call, and the
    # episode is scored as a model failure. This bit us twice, and the second time it faked
    # a whole result: at max_tokens=8000, 69 of 139 failures were stop=max_tokens, and
    # higher effort looked *worse* than lower purely because it hit the wall more often
    # (gpt-5.4-nano low 0 cap-hits/87.3% vs high 16/76.6%; mini medium 14 vs xhigh 23).
    #
    # The floor must therefore be generous enough that reasoning depth never decides the
    # outcome. There is no reason to be stingy: an unused cap costs nothing, only tokens
    # actually generated are billed. Anything above this is honoured as-is.
    # 32k was still not enough: claude-opus-4-8 with adaptive thinking returned
    # stop=max_tokens, out_tok=32000, turns=1, no solution written -- burning $0.81 to
    # produce nothing, then being scored as a wrong answer. Adaptive thinking has no
    # declared budget to size against, so the only safe move is a cap high enough that
    # reasoning depth cannot decide the outcome.
    MIN_MAX_TOKENS = 64_000

    def __init__(self, arm: Arm, tools: list[Tool], system: str,
                 max_turns: int = 60, max_tokens: int = 64_000, timeout_s: float = 900.0):
        """Configure one episode runner for a single arm.

        Args:
            arm: The routing arm (provider, model, reasoning config) to run under.
            tools: The tools available to the agent this episode.
            system: The system prompt.
            max_turns: Maximum number of agent turns before giving up.
            max_tokens: Requested max output tokens per turn; raised to at least
                `MIN_MAX_TOKENS` (and 2x any Anthropic thinking budget) so reasoning
                depth never decides the outcome -- see MIN_MAX_TOKENS above.
            timeout_s: Per-request HTTP timeout, in seconds.
        """
        load_env()
        self.arm, self.tools, self.system = arm, tools, system
        self.max_turns = max_turns
        budget = arm.request_kwargs().get("thinking", {}).get("budget_tokens", 0)
        self.max_tokens = max(max_tokens, budget * 2, self.MIN_MAX_TOKENS)
        self.by_name = {t.name: t for t in tools}
        if arm.provider == "anthropic":
            self._client = anthropic.Anthropic(max_retries=4, timeout=timeout_s)
        else:
            self._client = openai.OpenAI(max_retries=4, timeout=timeout_s)

    # ---------------------------------------------------------------- anthropic
    def _tools_anthropic(self) -> list[dict]:
        """Build the Anthropic-shaped tool list from `self.tools`."""
        return [{"name": t.name, "description": t.description, "input_schema": t.schema}
                for t in self.tools]

    def _run_anthropic(self, task: str) -> EpisodeResult:
        """Run one episode against the Anthropic Messages API.

        Args:
            task: The task prompt, sent as the initial user turn.

        Returns:
            The completed episode's usage, cost, transcript, and stop reason.
        """
        u: Usage = Usage()
        transcript: list[TranscriptEntry] = []
        messages: list[dict] = [{"role": "user", "content": task}]
        # Cache the system prompt + tool list: the prefix is identical every turn,
        # and Anthropic cache reads do not count against ITPM.
        system = [{"type": "text", "text": self.system,
                   "cache_control": {"type": "ephemeral"}}]
        stop, err, turns = "max_turns", None, 0

        for turns in range(1, self.max_turns + 1):
            kw = self.arm.request_kwargs()
            try:
                r = self._client.messages.create(
                    max_tokens=self.max_tokens, system=system, tools=self._tools_anthropic(),
                    messages=messages, **kw)
            except Exception as e:  # noqa: BLE001
                stop, err = "error", f"{type(e).__name__}: {e}"[:600]
                break

            ru = r.usage
            u.add(inp=ru.input_tokens, out=ru.output_tokens,
                  cache_read=getattr(ru, "cache_read_input_tokens", 0) or 0,
                  cache_write=getattr(ru, "cache_creation_input_tokens", 0) or 0,
                  requests=1)

            if r.stop_reason == "refusal":
                stop = "refusal"
                break
            # Echo assistant content back verbatim — thinking blocks must not be edited.
            messages.append({"role": "assistant", "content": r.content})
            calls = [b for b in r.content if b.type == "tool_use"]
            entry = TranscriptEntry(
                turn=turns,
                text=" ".join(b.text for b in r.content if b.type == "text")[:4000],
                calls=[ToolCallRecord(name=b.name, input=b.input) for b in calls],
            )
            transcript.append(entry)
            if not calls:
                stop = "max_tokens" if r.stop_reason == "max_tokens" else "end_turn"
                break

            results = []
            for b in calls:
                tool = self.by_name.get(b.name)
                if tool is None:
                    out, is_err = f"No such tool: {b.name}", True
                else:
                    try:
                        out, is_err = tool.run(b.input), False
                    except Exception as e:  # noqa: BLE001
                        out, is_err = f"{type(e).__name__}: {e}", True
                if entry.results is None:
                    entry.results = []
                entry.results.append(out[:2000])
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": out[:30000], "is_error": is_err})
            # All results for one assistant turn go back in a SINGLE user message,
            # otherwise the model learns to stop making parallel calls.
            messages.append({"role": "user", "content": results})

        return EpisodeResult(arm_id=self.arm.id, turns=turns, usage=u, cost_usd=u.cost(self.arm),
                             wall_s=0.0, stop=stop, error=err, transcript=transcript)

    # ------------------------------------------------------------------ openai
    def _tools_openai(self) -> list[dict]:
        """Build the OpenAI Responses-shaped tool list from `self.tools`."""
        return [{"type": "function", "name": t.name, "description": t.description,
                 "parameters": t.schema} for t in self.tools]

    def _run_openai(self, task: str) -> EpisodeResult:
        """Run one episode against the OpenAI Responses API.

        Args:
            task: The task prompt, sent as the initial user turn.

        Returns:
            The completed episode's usage, cost, transcript, and stop reason.
        """
        u: Usage = Usage()
        transcript: list[TranscriptEntry] = []
        history: list[Any] = [{"role": "user", "content": task}]
        stop, err, turns = "max_turns", None, 0

        for turns in range(1, self.max_turns + 1):
            kw = self.arm.request_kwargs()
            try:
                r = self._client.responses.create(
                    instructions=self.system, input=history,
                    max_output_tokens=self.max_tokens, tools=self._tools_openai(), **kw)
            except Exception as e:  # noqa: BLE001
                stop, err = "error", f"{type(e).__name__}: {e}"[:600]
                break

            ru = r.usage
            det = getattr(ru, "input_tokens_details", None)
            cached = getattr(det, "cached_tokens", 0) or 0
            odet = getattr(ru, "output_tokens_details", None)
            u.add(inp=max(0, ru.input_tokens - cached), cache_read=cached,
                  out=ru.output_tokens,
                  reasoning=getattr(odet, "reasoning_tokens", 0) or 0, requests=1)

            # Preserve reasoning items verbatim; dropping them breaks the next turn.
            history += [item.model_dump(exclude_none=True) for item in r.output]
            calls = [o for o in r.output if getattr(o, "type", "") == "function_call"]
            entry = TranscriptEntry(
                turn=turns,
                text=(r.output_text or "")[:4000],
                calls=[ToolCallRecord(name=c.name, input=c.arguments) for c in calls],
            )
            transcript.append(entry)
            if not calls:
                stop = "max_tokens" if r.status == "incomplete" else "end_turn"
                break

            for c in calls:
                tool = self.by_name.get(c.name)
                try:
                    args = json.loads(c.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if tool is None:
                    out = f"No such tool: {c.name}"
                else:
                    try:
                        out = tool.run(args)
                    except Exception as e:  # noqa: BLE001
                        out = f"{type(e).__name__}: {e}"
                if entry.results is None:
                    entry.results = []
                entry.results.append(out[:2000])
                history.append({"type": "function_call_output", "call_id": c.call_id,
                                "output": out[:30000]})

        return EpisodeResult(arm_id=self.arm.id, turns=turns, usage=u, cost_usd=u.cost(self.arm),
                             wall_s=0.0, stop=stop, error=err, transcript=transcript)

    # ------------------------------------------------------------------ public
    def run(self, task: str) -> EpisodeResult:
        """Run one episode end to end, dispatching to the arm's provider.

        Args:
            task: The task prompt.

        Returns:
            The completed episode, with `wall_s` set to the measured wall-clock time.
        """
        t0 = time.time()
        res = (self._run_anthropic(task) if self.arm.provider == "anthropic"
               else self._run_openai(task))
        res.wall_s = round(time.time() - t0, 2)
        return res


# ============================================================================ sandbox
# ---------------------------------------------------------------- E2B sandbox execution
# Why E2B, not local exec: running unreviewed model-generated code locally, with the
# caller's own permissions, previously saturated the machine (measured load 16.85/18 at
# 20 workers), and LiveCodeBench's own harness says its `reliability_guard` "is NOT a
# security sandbox" -- ours had none at all before this.

# The account cap is 1100 concurrent sandboxes (WMH_E2B_SANDBOX_CAP). Stay well under it:
# this lane must never starve another run, and orphaned sandboxes have previously blocked
# later work for 14-23h. Bounded here, and every sandbox is killed in a finally.
DEFAULT_MAX_CONCURRENT = 64
_CREATE_DELAYS = (1.0, 3.0, 9.0)
SANDBOX_TIMEOUT_S = 600

_sem: threading.Semaphore | None = None
_sem_lock = threading.Lock()


def semaphore(limit: int = DEFAULT_MAX_CONCURRENT) -> threading.Semaphore:
    """Get the process-wide sandbox concurrency semaphore, creating it on first use.

    Args:
        limit: Concurrency limit used only the first time this is called.

    Returns:
        The shared semaphore.
    """
    global _sem
    with _sem_lock:
        if _sem is None:
            _sem = threading.Semaphore(limit)
    return _sem


def create_sandbox(metadata: dict[str, str]) -> Any:
    """Open one E2B sandbox, retrying capacity errors with fixed backoff.

    Args:
        metadata: Tags attached at create time so an orphaned sandbox (owning
            process died) can still be found and reaped via `Sandbox.list`.

    Returns:
        The created `e2b.Sandbox`.

    Raises:
        RuntimeError: If `$E2B_API_KEY` is unset, or every retry attempt fails.
    """
    from e2b import Sandbox

    key = os.environ.get("E2B_API_KEY")
    if not key:
        raise RuntimeError("set $E2B_API_KEY")
    template = os.environ.get("WMH_E2B_TEMPLATE") or None
    last: Exception | None = None
    for delay in (*_CREATE_DELAYS, None):
        try:
            kw: dict[str, Any] = {"timeout": SANDBOX_TIMEOUT_S, "api_key": key,
                                  "metadata": metadata}
            if template:
                kw["template"] = template
            return Sandbox.create(**kw)
        except Exception as e:  # noqa: BLE001 — capacity/transient errors are the point
            last = e
            if delay is None:
                raise
            time.sleep(delay)
    raise RuntimeError(f"sandbox create failed: {last}")


GRADER = '''\
import json, subprocess, sys
tests = json.load(open("private_tests.json"))
ok = 0
for t in tests:
    try:
        p = subprocess.run([sys.executable, "solution.py"], input=t["input"],
                           capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        continue
    if p.returncode == 0 and p.stdout.strip() == t["output"].strip():
        ok += 1
print(f"GRADE {ok} {len(tests)}")
'''


class SandboxSession:
    """One sandbox for one episode. Always kills itself, even on error."""

    def __init__(self, tag: dict[str, str], limit: int = DEFAULT_MAX_CONCURRENT):
        """Configure a session; the sandbox itself opens in `__enter__`.

        Args:
            tag: Metadata tags for the underlying sandbox (see `create_sandbox`).
            limit: Concurrency limit to acquire against.
        """
        self.tag, self.limit = tag, limit
        self.sb: Any = None
        self._held = False

    def __enter__(self) -> SandboxSession:
        """Acquire a concurrency slot and open the sandbox.

        Returns:
            This session, ready for `write`/`run`/`read`/`grade`.
        """
        semaphore(self.limit).acquire()
        self._held = True
        try:
            self.sb = create_sandbox(self.tag)
        except Exception:
            semaphore(self.limit).release()
            self._held = False
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        """Kill the sandbox and release the concurrency slot, unconditionally."""
        try:
            if self.sb is not None:
                self.sb.kill()  # never leak: orphans starve later runs
        except Exception:  # noqa: BLE001, S110 — teardown must not mask the real error
            pass
        finally:
            if self._held:
                semaphore(self.limit).release()
                self._held = False

    def write(self, path: str, data: str) -> None:
        """Write `data` to `path` inside the sandbox."""
        self.sb.files.write(path, data)

    def run(self, cmd: str, timeout: float = 90.0) -> tuple[int, str, str]:
        """Run `cmd` inside the sandbox.

        Args:
            cmd: Shell command to execute.
            timeout: Maximum seconds to wait for completion.

        Returns:
            A (exit_code, stdout, stderr) tuple.
        """
        r = self.sb.commands.run(cmd, timeout=timeout)
        code = getattr(r, "exit_code", None)
        return (0 if code is None else int(code),
                getattr(r, "stdout", "") or "", getattr(r, "stderr", "") or "")

    def read(self, path: str) -> str:
        """Read `path` from the sandbox, returning "" if it does not exist."""
        try:
            return self.sb.files.read(path)
        except Exception:  # noqa: BLE001 — a missing file is a real outcome, not an error
            return ""

    def grade(self, private_tests: list[dict]) -> tuple[int, int]:
        """Write private tests only now, so the agent never saw them, then grade in-sandbox.

        Args:
            private_tests: Held-out test cases, written to the sandbox for the first time.

        Returns:
            A (passed, total) pair.
        """
        self.write("private_tests.json", json.dumps(private_tests))
        self.write("grade.py", GRADER)
        _, out, _ = self.run("python3 grade.py", timeout=300.0)
        for line in reversed(out.splitlines()):
            if line.startswith("GRADE "):
                _, ok, tot = line.split()
                return int(ok), int(tot)
        return 0, len(private_tests)


# ---------------------------------------------------------------- LiveCodeBench loader
# Why LiveCodeBench, not SWE-bench Pro: SWE-bench Pro measured 7-17 min/episode behind
# amd64-only Docker, a useless inner loop. LiveCodeBench's AtCoder subset is pure
# stdin/stdout, needs no Docker, and still gives a real multi-turn agentic loop (write ->
# run public tests -> fix -> repeat), graded on held-out private tests.

# github.com/LiveCodeBench/LiveCodeBench/blob/main/ERRATA.md -- multiple-solutions,
# interactive, and erroneous-test items. Excluding these removes known label noise.
ERRATA = {
    "abc311_c", "abc326_d", "abc327_b", "abc333_e", "abc343_e", "abc362_c", "arc185_c",
    "abc343_a", "find-words-containing-character", "find-the-peaks",
    "generate-binary-strings-without-adjacent-zeros",
    "abc337_e", "abc355_e",  # interactive: unsolvable by this harness
    "abc350_c", "apply-operations-to-make-string-empty", "most-frequent-ids", "arc189_a",
}


@dataclasses.dataclass
class Problem:
    """One LiveCodeBench problem: statement plus public/private stdin-stdout test cases."""

    qid: str
    platform: str
    difficulty: str  # easy|medium|hard; question_id joins 1:1 to real AtCoder IRT ratings
    title: str
    statement: str
    public_tests: list[dict]
    private_tests: list[dict]

    @property
    def n_tests(self) -> int:
        """Number of private (held-out) test cases for this problem."""
        return len(self.private_tests)


def _decode_tests(raw) -> list[dict]:
    """Decode one problem's test cases from either a JSON list or a packed blob.

    LiveCodeBench's private_test_cases ship as base64(zlib(pickle(json))) -- an
    UNPICKLE of data downloaded from HF, so this trusts the repo or doesn't load it.
    Only unpickling happens here, never exec; the unpickled payload is itself a JSON
    string, which is parsed, not executed.

    Args:
        raw: Either an already-decoded list of test-case dicts, or the packed blob.

    Returns:
        The list of test-case dicts.
    """
    if isinstance(raw, list):
        return raw
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode()))))


def load(path: pathlib.Path = DEFAULT_PATH, *, platform: str = "atcoder",
         drop_errata: bool = True) -> list[Problem]:
    """Load stdin/stdout problems. Defaults to AtCoder: no starter_code, no func_name.

    Args:
        path: Path to the LiveCodeBench jsonl file.
        platform: If set, keep only problems from this platform.
        drop_errata: If True, drop problems listed in `ERRATA` as known label noise.

    Returns:
        The loaded problems, restricted to pure stdin/stdout items (this harness
        only drives stdin/stdout; functional items would need a call driver).
    """
    out: list[Problem] = []
    with path.open() as fh:
        for line in fh:
            d = json.loads(line)
            if platform and d.get("platform") != platform:
                continue
            qid = d["question_id"]
            if drop_errata and qid in ERRATA:
                continue
            pub = _decode_tests(d.get("public_test_cases"))
            prv = _decode_tests(d.get("private_test_cases"))
            # This harness only handles stdin/stdout; functional items need a call driver.
            if any(t.get("testtype") != "stdin" for t in pub + prv):
                continue
            out.append(Problem(qid, d["platform"], d.get("difficulty", "?"),
                               d.get("question_title", ""), d.get("question_content", ""),
                               pub, prv))
    return out


def grade(code: str, tests: list[dict], *, timeout: float = 6.0,
          max_fail_report: int = 2) -> tuple[int, int, list[str]]:
    """Run `code` against `tests` via stdin, outside any sandbox.

    Args:
        code: Python source to execute as a script.
        tests: Test cases with "input"/"output" keys.
        timeout: Per-test-case timeout, in seconds.
        max_fail_report: Maximum number of failure snippets to collect.

    Returns:
        A (passed, total, failure_snippets) tuple.
    """
    passed, fails = 0, []
    for t in tests:
        try:
            p = subprocess.run([sys.executable, "-c", code], input=t["input"],
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            if len(fails) < max_fail_report:
                fails.append(f"TIMEOUT (>{timeout}s) on input:\n{t['input'][:300]}")
            continue
        got, want = (p.stdout or "").strip(), (t["output"] or "").strip()
        if p.returncode == 0 and got == want:
            passed += 1
        elif len(fails) < max_fail_report:
            fails.append(
                f"input:\n{t['input'][:300]}\nexpected:\n{want[:200]}\n"
                f"got:\n{got[:200]}\nstderr:\n{(p.stderr or '')[:300]}")
    return passed, len(tests), fails


SYSTEM = (
    "You are a competitive programming agent working in a scratch directory. "
    "Write your solution to solution.py. It must read from stdin and write to stdout. "
    "Use the bash tool to create the file and to run `python3 check.py`, which runs the "
    "PUBLIC sample tests and prints results. Iterate until all public tests pass, then "
    "reply with exactly DONE."
)

CHECKER = '''\
import json, subprocess, sys
tests = json.load(open("public_tests.json"))
bad = 0
for i, t in enumerate(tests):
    try:
        p = subprocess.run([sys.executable, "solution.py"], input=t["input"],
                           capture_output=True, text=True, timeout=6)
    except subprocess.TimeoutExpired:
        print(f"test {i}: TIMEOUT"); bad += 1; continue
    got, want = p.stdout.strip(), t["output"].strip()
    if p.returncode != 0:
        print(f"test {i}: CRASH\\n{p.stderr[-500:]}"); bad += 1
    elif got != want:
        print(f"test {i}: WRONG\\n input={t['input'][:200]}\\n want={want[:200]}\\n got={got[:200]}")
        bad += 1
    else:
        print(f"test {i}: PASS")
print(f"\\n{len(tests)-bad}/{len(tests)} public tests passed")
'''


def make_workdir(p: Problem) -> pathlib.Path:
    """Create a scratch directory with `p`'s public tests and the checker script.

    Args:
        p: The problem to prepare a workdir for.

    Returns:
        The created temporary directory's path.
    """
    d = pathlib.Path(tempfile.mkdtemp(prefix=f"lcb-{p.qid}-"))
    (d / "public_tests.json").write_text(json.dumps(p.public_tests))
    (d / "check.py").write_text(CHECKER)
    return d


def task_prompt(p: Problem) -> str:
    """Render the agent-facing task prompt for problem `p`, including sample I/O."""
    ex = "\n\n".join(f"Sample input:\n{t['input']}\nSample output:\n{t['output']}"
                     for t in p.public_tests[:2])
    return (f"# {p.title}\n\n{p.statement}\n\n{ex}\n\n"
            "Write solution.py, then run `python3 check.py` until all public tests pass.")


if __name__ == "__main__":
    t0 = time.time()
    probs = load()
    by_diff: dict[str, int] = {}
    for p in probs:
        by_diff[p.difficulty] = by_diff.get(p.difficulty, 0) + 1
    print(f"loaded {len(probs)} stdin problems in {time.time()-t0:.1f}s  {by_diff}")
    tests = sorted(p.n_tests for p in probs)
    print(f"private tests/problem: min={tests[0]} p50={tests[len(tests)//2]} max={tests[-1]}")

    # Timing floor: how long does grading a KNOWN-GOOD solution take?
    p = probs[0]
    print(f"\nsample: {p.qid} [{p.difficulty}] {p.title!r} "
          f"pub={len(p.public_tests)} prv={len(p.private_tests)}")
    t0 = time.time()
    ok, tot, _ = grade("import sys\nprint(sys.stdin.read().strip())", p.private_tests[:10])
    print(f"grading 10 private tests took {time.time()-t0:.2f}s ({ok}/{tot} passed by a stub)")
