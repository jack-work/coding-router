# AGENTS.md

Instructions for any agent (or human) making changes in this repo.

## Layout

All source lives flat in `router/`, one package, eight files:

| file | contents |
|---|---|
| `router_core.py` | pricing table, `Router`/`Decision` (inference), `Matrix`/policies (CV, races) |
| `harness.py` | unified OpenAI+Anthropic agent client, E2B sandbox exec, LiveCodeBench eval |
| `serve.py` | `python -m router.serve` — one command, one OpenAI-compatible endpoint |
| `export.py` | freezes the router into `results/router_v0.{json,npz}`, self-tests the artifact |
| `datasets.py` | SWE-rebench + DeepSWE dataset loaders |
| `benchmarks.py` | CLI: fetch-swebench-matrix, transfer-swebench, run-lcb, smoke-agent |
| `experiments.py` | CLI: holdout-deepswe, exp1-holdout9, race-router, race-deepswe, probe-arms |
| `analysis.py` | CLI: analyze-phase1, headroom, price-traces, extract-traces |

## Hard rules

- **1000 lines per source file, max.** Applies to `.py` files under `router/`. Does not apply to
  data/config files (`results/`, `data/`, `*.json`, `*.jsonl`, `*.npz`, `*.log`, `uv.lock`).
- **8 files max.** This repo was deliberately consolidated from 24 files down to 7, then 8 when
  `serve.py` was added (previously also had separate `loaders/`, `harness/`, `scripts/`
  directories that no longer exist). Before creating a new file, put the code in the existing
  file it belongs to per the table above. A new CLI subcommand goes in whichever of
  benchmarks/experiments/analysis it's closest to in kind. If nothing above fits and you're
  certain a new file is warranted, say so explicitly and explain why consolidation doesn't
  work, rather than defaulting to a new file.
- If consolidating would push a file over 1000 lines, that's a signal to split by genuine topic
  (dataset loaders were briefly two files for exactly this reason) — not to abandon the limit.

## Style

- **No underdefined types.** Don't type a parameter or return value as bare `dict` or `Any` when
  its shape is actually known and reused (a routing decision, a request body, a tool call). Define
  a `pydantic.BaseModel` for it once and reuse that type everywhere it appears, instead of every
  call site re-deriving the shape from how it's used. Reach for plain `dict`/`Any` only at genuine
  boundaries where the shape isn't fixed (arbitrary JSON-schema `parameters` blobs, a raw parsed
  artifact file) — not as a default.
- **Every function gets a docstring.** Non-trivial functions (real logic, non-obvious behavior,
  more than a couple of parameters) get a full Google-style docstring — one-line summary, then
  `Args:`/`Returns:`/`Raises:` sections as needed. A genuinely trivial function (a one-line
  wrapper, an obvious getter) gets a single-sentence docstring instead — not a full section
  breakdown for something with nothing non-obvious to say.
- **No essay-length file headers.** A module docstring states what the file contains in a
  sentence or two. Non-obvious rationale (a specific measured number, a hard-won lesson, a
  constraint that shaped the design) belongs as a short comment right next to the code it
  explains, not as a standalone prose section at the top of the file that a reader has to hold
  in their head while reading everything below it.
- **No `print()` — use `logging` everywhere.** Every module gets `logger =
  logging.getLogger(__name__)` and emits via `logger.info(...)` (or the appropriate level).
  Each CLI entry point (`main*()` / the `__main__` block) configures
  `logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")` so CLI
  output — tables, results, progress — is byte-identical to what `print` produced, while
  staying filterable, redirectable, and silenceable by anyone importing this as a library.
  Interactive prompts via `getpass`/`input` are not prints and stay as they are.

## Telemetry

- **STRICTLY metadata. WE NEVER UPLOAD TRACES OR PII FROM A USER.** Telemetry (PostHog, in
  `serve.py`) may carry only: counts (tokens, messages, tools), durations/rates (latency, tps),
  booleans/enums about OUR system (model picked, effort, off_distribution, via-mode), and
  estimated costs. Never — under any future change — message content, prompts, completions,
  code, diffs, tool names or arguments, file paths, repo names, URLs, API keys, usernames,
  hostnames, or anything else user-authored or user-identifying. The distinct id is a random
  UUID generated locally; it maps to nothing.
- Any new telemetry property must be defensible as non-PII metadata under the list above; when
  in doubt, leave it out.
- Opt-out must always work: `ROUTER_TELEMETRY_DISABLED=1` (and the industry-standard
  `DO_NOT_TRACK=1`) disable all capture, and the server announces telemetry status + the
  opt-out variable at startup. The PostHog key in source is a public write-only project key —
  that's standard PostHog practice, not a leaked secret.

## Tooling

Ruff (lint) and ty (type check) are dev dependencies. Before considering any change complete, run:

```
uv run ruff check router/
uv run ty check router/
```

Ruff must pass clean. `pyproject.toml` selects `E, F, I, UP` (pycodestyle, pyflakes, import
sort, pyupgrade) — deliberately not `B` (flake8-bugbear) or `SIM` (flake8-simplify), since
several of their suggestions (e.g. adding `zip(..., strict=True)`) change runtime behavior
and need per-call-site judgment, not a blanket rule.

**Line-length limit is not the same as the file-length limit above** — `line-length = 110` in
`[tool.ruff]` caps individual lines, it does not check file size. Neither ruff nor ty has a
built-in "max lines per file" rule, so the 1000-line rule is enforced by this check, run by hand:

```
find router -name "*.py" | xargs wc -l | awk '$1 > 1000 && $2 != "total" {print; exit 1}'
```

**ty baseline: 55 known diagnostics, not bugs, don't chase them to zero without asking first.**
Per-file: analysis.py 3, datasets.py 2, experiments.py 9, export.py 3, harness.py 4,
router_core.py 12, serve.py 27 (the rest have none). (Was 56 until the print->logging pass
removed serve.py's `sys.stdout.reconfigure` line, which carried one diagnostic.) They trace to a small number of
intentional dynamic-typing patterns a static checker can't see through, e.g.:
  1. `Matrix.emb: np.ndarray | None = None` in `router_core.py` — `Matrix` is built once via
     `build()` and then mutated in place by `embed()`; `emb` is genuinely `None` between those
     two calls but always set by the time downstream code reads it. Fixing this "properly" means
     restructuring the two-phase construction, not adding a type: ignore.
  2. `AgentRunner._client: Anthropic | OpenAI` in `harness.py` — the concrete type is picked by
     `arm.provider` at construction time and used correctly downstream via that same provider
     check, but ty can't link the two attributes to narrow the union.
  3. `serve.py` carries most of the count: `ChatMessage.content: str | None` participates in
     string concatenation (`system + "\n\n" + m.content`) in the Chat Completions translation
     functions — a real latent gap (a `None` content would crash there) that predates this file's
     Pydantic types (the same crash risk existed, invisibly, behind an untyped `dict` access
     before), not something introduced by typing it properly. Fixing it means deciding what an
     empty/absent message content should mean, which is a product decision, not a type fix.
If this count grows for a new, different reason, treat that as a real signal worth looking at.
Before changing this number, verify a claimed baseline shift the same way it was established
here: diff `ty check` output against a stash of the prior state, not just eyeball the total.

## Config files

- `[tool.ruff]` / `[tool.ruff.lint]` and `[tool.ty.src]` live in `pyproject.toml`.
- `ty`'s exclude (`data/`, `results/`) only matters if non-`.py` files under those paths were
  ever imported as modules, which they aren't — it's there for clarity, not because ty would
  otherwise choke on a 128MB `.jsonl` file.
