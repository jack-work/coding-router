# AGENTS.md

Instructions for any agent (or human) making changes in this repo.

## Layout

All source lives flat in `router/`, one package, seven files:

| file | contents |
|---|---|
| `router_core.py` | pricing table, `Router`/`Decision` (inference), `Matrix`/policies (CV, races) |
| `harness.py` | unified OpenAI+Anthropic agent client, E2B sandbox exec, LiveCodeBench eval |
| `export.py` | freezes the router into `results/router_v0.{json,npz}`, self-tests the artifact |
| `datasets.py` | SWE-rebench + DeepSWE dataset loaders |
| `benchmarks.py` | CLI: fetch-swebench-matrix, transfer-swebench, run-lcb, smoke-agent |
| `experiments.py` | CLI: holdout-deepswe, exp1-holdout9, race-router, race-deepswe, probe-arms |
| `analysis.py` | CLI: analyze-phase1, headroom, price-traces, extract-traces |

## Hard rules

- **1000 lines per source file, max.** Applies to `.py` files under `router/`. Does not apply to
  data/config files (`results/`, `data/`, `*.json`, `*.jsonl`, `*.npz`, `*.log`, `uv.lock`).
- **Adding new files is HIGHLY discouraged.** This repo was deliberately consolidated from 24
  files down to 7 (previously also had separate `loaders/`, `harness/`, `scripts/` directories
  that no longer exist). Before creating a new file, put the code in the existing file it belongs
  to per the table above. A new CLI subcommand goes in whichever of benchmarks/experiments/analysis
  it's closest to in kind. If nothing above fits and you're certain a new file is warranted, say so
  explicitly and explain why consolidation doesn't work, rather than defaulting to a new file.
- If consolidating would push a file over 1000 lines, that's a signal to split by genuine topic
  (dataset loaders were briefly two files for exactly this reason) — not to abandon the limit.

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

**ty baseline: ~29 known diagnostics, not bugs, don't chase them to zero without asking first.**
They trace to two intentional dynamic-typing patterns that a static checker can't see through:
  1. `Matrix.emb: np.ndarray | None = None` in `router_core.py` — `Matrix` is built once via
     `build()` and then mutated in place by `embed()`; `emb` is genuinely `None` between those
     two calls but always set by the time downstream code reads it. Fixing this "properly" means
     restructuring the two-phase construction, not adding a type: ignore.
  2. `AgentRunner._client: Anthropic | OpenAI` in `harness.py` — the concrete type is picked by
     `arm.provider` at construction time and used correctly downstream via that same provider
     check, but ty can't link the two attributes to narrow the union.
If this count grows for a new, different reason, treat that as a real signal worth looking at.

## Config files

- `[tool.ruff]` / `[tool.ruff.lint]` and `[tool.ty.src]` live in `pyproject.toml`.
- `ty`'s exclude (`data/`, `results/`) only matters if non-`.py` files under those paths were
  ever imported as modules, which they aren't — it's there for clarity, not because ty would
  otherwise choke on a 128MB `.jsonl` file.
