# AGENTS.md

Instructions for any agent (or human) making changes in this repo.

## Layout

This repo is the PRODUCT: the deployable router and nothing else. Two code files:

| file | contents |
|---|---|
| `router_core.py` | canonical price table (`Price`/`Arm`) + artifact inference (`Router`/`Decision`) |
| `serve.py` | `python -m router.serve` — one command, one OpenAI-compatible endpoint |

The research half that built and validates this — dataset loaders, the measurement
harness, CV policies, benchmarks, artifact export — lives in world-model-optimizer
(`wmo optimize route` / `wmo research`). New research/eval code goes THERE, never here.

Two invariants of the product itself:

- **Routing runs fully locally.** Embeddings are computed in-process by a small local
  model (Qwen3-Embedding-0.6B: MLX on Apple Silicon, sentence-transformers elsewhere)
  and the kNN decision is pure numpy. API keys exist only to dispatch the chosen model
  and summarize long trajectories — never to route.
- **Exactly one default artifact, and it lives on Hugging Face, never in git.**
  `serve` downloads `router.{json,npz}` from `experiential-labs/coding-router` on first
  run (then runs offline); new router versions overwrite that repo in place. Users fit
  their own artifacts from their own traces via wmo.

## Hard rules

- **1000 lines per source file, max.** Applies to `.py` files under `router/`. Does not apply to
  data/config files (`results/`, `data/`, `*.json`, `*.jsonl`, `*.npz`, `*.log`, `uv.lock`).
- **2 code files max.** This repo went from 24 files to 8 by consolidation, then to 2 when the
  research half moved to wmo's `packages/router-lab`. Before creating a new file here, ask
  whether the code is product (belongs in one of the two files above) or research (belongs in
  router-lab). If you're certain a new product file is warranted, say so explicitly and explain
  why, rather than defaulting to a new file.
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

**ty baseline: 27 known diagnostics (all in serve.py), not bugs, don't chase them to zero
without asking first.** Two of them are `unresolved-import` on the platform-conditional
embedding backends (`mlx.core`, `sentence_transformers`) — only one of the two packages is
ever installed on a given platform, so a static checker on any single machine cannot
resolve the other. They trace to intentional dynamic-typing patterns a static checker
can't see through — chiefly `ChatMessage.content: str | None` participating in string
concatenation in the Chat Completions translation functions (a real but pre-existing latent
gap: the same crash risk existed invisibly behind untyped dict access before Pydantic typing
made it visible; fixing it means deciding what absent content should mean, a product decision).
If this count grows for a new, different reason, treat that as a real signal worth looking at.
Before changing this number, verify a claimed baseline shift the same way it was established
here: diff `ty check` output against a stash of the prior state, not just eyeball the total.

## Config files

- `[tool.ruff]` / `[tool.ruff.lint]` and `[tool.ty.src]` live in `pyproject.toml`.
- `ty`'s exclude (`data/`, `results/`) only matters if non-`.py` files under those paths were
  ever imported as modules, which they aren't — it's there for clarity, not because ty would
  otherwise choke on a 128MB `.jsonl` file.
