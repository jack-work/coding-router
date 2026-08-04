# AGENTS.md

Instructions for any agent (or human) making changes in this repo.

## Layout

This repo is the PRODUCT: the deployable router and nothing else. Three code files:

| file | contents |
|---|---|
| `router_core.py` | canonical price table (`Price`/`Arm`), prompt-cache capability tables (`CachePolicy`), and artifact inference (`Router`/`Decision`, incl. arm stickiness) |
| `wire.py` | the public Chat Completions contract (request/response models) and its translation into each provider's native request, incl. cache-breakpoint placement |
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
- **3 code files max.** This repo went from 24 files to 8 by consolidation, then to 2 when the
  research half moved to wmo's `packages/router-lab`. It became 3 when prompt caching pushed
  `serve.py` past 1000 lines and the wire (models + provider translation) came out as its own
  topic. Before creating a fourth, ask whether the code is product (belongs in one of the files
  above) or research (belongs in router-lab). If you're certain a new product file is warranted,
  say so explicitly and explain why, rather than defaulting to a new file.
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
uv run pytest -q
```

Tests live in `tests/`, outside the `router/` file-count and line-length rules. They run
offline in under a second: no artifact, no encoder, no network. The router kinds are driven
from synthetic artifacts written to a tmp dir, and `tests/test_live_cache.py` skips itself
unless a provider key is present in the environment. `tests/bench_wire.py` is a standalone
micro-benchmark (`.venv/bin/python tests/bench_wire.py`) for the per-request shaping path;
run it on both sides of a change that touches translation or the trajectory.

`nix develop` gives the same python 3.12 + uv without installing either.

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

**ty baseline: 7 known diagnostics, not bugs, don't chase them to zero without asking
first.** Two of them are `unresolved-import` on the Apple-Silicon-only embedding backend
(`mlx.core`, `mlx_embeddings`) — those packages are never installed on Linux, so a static
checker there cannot resolve them. The rest are SDK-overload mismatches on
`Messages.stream`, `Responses.create` and `Completions.create`, plus one `Tensor.astype`.

This number was 28 before structured content blocks were typed (`ChatMessage.content` is
now `str | list[ContentPart] | None`, so the string-concatenation gap in the translation
functions is closed). Note that AGENTS.md previously claimed 27 while `ty` reported 28 at
the very commit that wrote the claim, and named `sentence_transformers` as an unresolved
import when both unresolved imports were in fact the mlx pair — measure, don't inherit.
If this count grows for a new, different reason, treat that as a real signal worth looking
at. Before changing this number, verify a claimed baseline shift the same way it was
established here: diff `ty check` output against a stash of the prior state, not just
eyeball the total.

## Config files

- `[tool.ruff]` / `[tool.ruff.lint]` and `[tool.ty.src]` live in `pyproject.toml`.
- `ty`'s exclude (`data/`, `results/`) only matters if non-`.py` files under those paths were
  ever imported as modules, which they aren't — it's there for clarity, not because ty would
  otherwise choke on a 128MB `.jsonl` file.
