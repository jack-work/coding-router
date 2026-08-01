"""Measure the production defect directly: how much does the deployed router's pick move
between the RAW task statement (the format the bank/eval used) and the SERVE-SHAPED text
(what build_trajectory actually embeds), and how degenerate is the pick distribution?

Scratch probe -- run, read the numbers, then fold into the lane's real experiment code.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

ART = pathlib.Path("/Users/admin/Documents/experientiallabs/hf-staging-trained")
TEXTS = pathlib.Path(
    "/Users/admin/Documents/experientiallabs/coding-router-lrbase/results/router_texts.jsonl")

# serve.py:538 -- the exact renderer, replicated verbatim.
def message_preview(role: str, content: str) -> str:
    return f"[{role}] {json.dumps(content)[:2000]}"


# A realistic opencode client preamble. Not opencode's literal bytes, but the right shape and
# order of magnitude (~12k chars of constant instructions + tool schemas) per the field report.
SYSTEM_PREAMBLE = (
    "You are opencode, an interactive CLI agent specializing in software engineering tasks. "
    "Use the instructions below and the tools available to you to assist the user.\n\n"
    "# Core Mandates\n"
    "- Rigorously adhere to existing project conventions when reading or modifying code.\n"
    "- Never assume a library is available; verify it is already used in the project.\n"
    "- Mimic the style, structure, framework choice, typing, and architectural patterns of "
    "existing code in the project.\n"
    "- Do not add comments that explain what the code does; explain WHY only where non-obvious.\n"
    "- Do not revert changes unless asked or unless they caused an error.\n\n"
    "# Primary Workflows\n"
    "## Software Engineering Tasks\n"
    "1. Understand: think about the request and the codebase context. Use search tools "
    "extensively in parallel to understand file structures and existing conventions.\n"
    "2. Plan: build a coherent, grounded plan. Share an extremely concise plan with the user.\n"
    "3. Implement: use the available tools to act on the plan, strictly adhering to conventions.\n"
    "4. Verify: run the project's own tests and build/lint commands to confirm correctness.\n\n"
    "# Operational Guidelines\n"
    "## Tone and Style\n"
    "- Concise and direct. Aim for fewer than 3 lines of text output per response.\n"
    "- No chitchat, no preamble, no postamble. Answer directly.\n"
    "- Use GitHub-flavored markdown. Output is rendered in a monospace terminal.\n"
    "## Security\n"
    "- Always explain critical commands before executing them.\n"
    "- Never introduce code that exposes, logs, or commits secrets or keys.\n\n"
    "# Tools\n"
) + "\n".join(
    json.dumps({
        "name": n,
        "description": d,
        "input_schema": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Absolute path to the target file."},
            "pattern": {"type": "string", "description": "Regular expression to match."},
            "content": {"type": "string", "description": "Content to write."},
        }, "required": ["path"]},
    })
    for n, d in [
        ("bash", "Executes a bash command in a persistent shell session with timeout."),
        ("read", "Reads a file from the local filesystem with line numbers."),
        ("write", "Writes content to a file, overwriting if it exists."),
        ("edit", "Performs exact string replacement in a file."),
        ("glob", "Fast file pattern matching against any codebase size."),
        ("grep", "Fast content search using regular expressions across the codebase."),
        ("todowrite", "Create and manage a structured task list for the session."),
        ("webfetch", "Fetches content from a URL and processes it with a prompt."),
        ("task", "Launch a new sub-agent to handle a complex multi-step task."),
        ("patch", "Apply a unified diff patch to one or more files."),
    ]
)

# Realistic short/utility asks -- the traffic the report says always abstains.
SHORT_ASKS = [
    "fix the failing test",
    "add a docstring to this function",
    "what does this file do?",
    "rename the variable `x` to `count` everywhere in this module",
    "why is the build broken?",
    "run the tests",
    "add type hints to parse_config",
    "how many callers does `resolve_path` have?",
    "bump the version to 2.1.0",
    "make this function async",
    "remove the unused imports",
    "explain this stack trace",
    "commit these changes",
    "what's the difference between these two branches?",
    "add a test for the empty-input case",
    "reformat this file with ruff",
    "is there a race condition here?",
    "convert this to use pathlib",
    "add error handling around the network call",
    "what version of numpy does this project pin?",
]


def entropy(counts: dict) -> float:
    tot = sum(counts.values())
    return -sum((c / tot) * math.log2(c / tot) for c in counts.values() if c)


def main() -> None:
    sys.path.insert(0, "/Users/admin/Documents/experientiallabs/coding-router")
    from router.router_core import TrainedRouter

    r = TrainedRouter(ART)
    print(f"artifact: {r.meta['version']} kind={r.meta['kind']} "
          f"T={r.T:.4f} lam={r.lam} sim_floor={r.sim_floor} "
          f"arms={len(r.arms)} bank={r.emb.shape} fallback={r.arms[r.fallback]}")

    import mlx.core as mx
    from mlx_embeddings import generate, load

    model, tokenizer = load(str(ART / r.meta["embed_model_mlx"]))

    def embed(text: str) -> np.ndarray:
        out = generate(model, tokenizer, texts=[text])
        v = np.array(out.text_embeds.astype(mx.float32))[0].astype(float)
        return v / np.linalg.norm(v)

    dswe = [json.loads(x) for x in TEXTS.read_text().splitlines()]
    dswe = [d for d in dswe if d["id"].startswith("dswe:")]
    lens = sorted(len(d["text"]) for d in dswe)
    print(f"\nDeepSWE task texts: n={len(dswe)}  chars min={lens[0]} "
          f"p10={lens[len(lens)//10]} p50={lens[len(lens)//2]} "
          f"p90={lens[9*len(lens)//10]} max={lens[-1]}")

    variants = {
        # what the bank + every eval used
        "raw": lambda t: t,
        # what serve embeds today, at turn 0, after the system-strip fix
        "serve_turn0": lambda t: message_preview("user", t),
        # what serve embedded BEFORE the fix (system prompt included)
        "serve_with_system": lambda t: "\n".join([
            message_preview("system", SYSTEM_PREAMBLE), message_preview("user", t)]),
    }

    rows = {k: [] for k in variants}
    for d in dswe:
        for k, fn in variants.items():
            rows[k].append(embed(fn(d["text"])[:24_000]))
    vecs = {k: np.stack(v) for k, v in rows.items()}

    print(f"\nsystem preamble is {len(SYSTEM_PREAMBLE):,} chars; "
          f"a serve_with_system turn-0 text is "
          f"{len(variants['serve_with_system'](dswe[0]['text'])):,} chars "
          f"({100 * 2007 / len(variants['serve_with_system'](dswe[0]['text'])):.0f}% of it "
          f"is the constant preamble)")

    def decide(v: np.ndarray) -> tuple[str, bool, float]:
        sims = r.emb @ v
        nearest = float(sims.max())
        w = np.exp((sims - sims.max()) / r.T)
        p = r.graded @ (w / w.sum())
        pick = int(np.argmax(p - r.lam * r.med_cost))
        off = nearest < r.sim_floor
        return (r.arms[r.fallback] if off else r.arms[pick]), off, nearest

    print("\n=== DeepSWE tasks (n=%d): format -> pick distribution ===" % len(dswe))
    print(f"{'variant':<20} {'off%':>6} {'nearest p50':>12} {'modal arm share':>16} "
          f"{'n arms':>7} {'entropy':>8}")
    decisions = {}
    for k in variants:
        picks, offs, nears = [], [], []
        for v in vecs[k]:
            a, o, n = decide(v)
            picks.append(a)
            offs.append(o)
            nears.append(n)
        decisions[k] = picks
        c: dict = {}
        for a in picks:
            c[a] = c.get(a, 0) + 1
        top = max(c.values()) / len(picks)
        print(f"{k:<20} {100*sum(offs)/len(offs):>5.0f}% {np.median(nears):>12.3f} "
              f"{top:>15.0%} {len(c):>7} {entropy(c):>8.2f}")
        for a, n in sorted(c.items(), key=lambda kv: -kv[1])[:4]:
            print(f"{'':<20}   {n:>4}  {a}")

    print("\n=== pairwise: same task, two formats ===")
    base = "raw"
    for k in variants:
        if k == base:
            continue
        cos = float(np.mean(np.sum(vecs[base] * vecs[k], axis=1)))
        agree = np.mean([a == b for a, b in zip(decisions[base], decisions[k])])
        print(f"  {base:>18} vs {k:<20} cos(same task)={cos:.3f}   pick agreement={agree:.0%}")
    cross = float(np.mean(vecs["serve_with_system"] @ vecs["serve_with_system"].T))
    craw = float(np.mean(vecs["raw"] @ vecs["raw"].T))
    cs0 = float(np.mean(vecs["serve_turn0"] @ vecs["serve_turn0"].T))
    print(f"\n  mean BETWEEN-TASK cosine (the variance the router needs to see):")
    print(f"    raw               {craw:.3f}")
    print(f"    serve_turn0       {cs0:.3f}")
    print(f"    serve_with_system {cross:.3f}   <- 1.0 means every task looks identical")

    print(f"\n=== short/utility asks (n={len(SHORT_ASKS)}) ===")
    for k in ("raw", "serve_turn0"):
        picks, offs, nears = [], [], []
        for t in SHORT_ASKS:
            a, o, n = decide(embed(variants[k](t)))
            picks.append(a)
            offs.append(o)
            nears.append(n)
        c = {}
        for a in picks:
            c[a] = c.get(a, 0) + 1
        print(f"  {k:<14} off={100*sum(offs)/len(offs):.0f}%  "
              f"nearest min/p50/max={min(nears):.3f}/{np.median(nears):.3f}/{max(nears):.3f}  "
              f"picks={dict(sorted(c.items(), key=lambda kv: -kv[1]))}")

    out = {
        "n_dswe": len(dswe),
        "char_pctiles": {"min": lens[0], "p50": lens[len(lens) // 2], "max": lens[-1]},
        "between_task_cos": {"raw": craw, "serve_turn0": cs0, "serve_with_system": cross},
        "pick_agreement_raw_vs_serve": float(
            np.mean([a == b for a, b in zip(decisions["raw"], decisions["serve_turn0"])])),
    }
    (pathlib.Path(__file__).parent / "_probe_format_sensitivity.json").write_text(
        json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
