# coding-router

Routes every coding-agent request to the cheapest OpenAI/Anthropic model likely to solve it,
via kNN over measured outcomes. One local OpenAI-compatible endpoint; point opencode (or any
openai-compatible client) at it. This repo is the product only — the research lab that builds
and validates the routing artifact lives in world-model-optimizer's `packages/router-lab`.

## Run

```
uv run python -m router.serve
```

Serves `http://127.0.0.1:61890/v1` and prints a ready-to-paste opencode provider config on
startup. First run prompts once for API keys (hidden input, validated, saved to `.env.local`).

## Telemetry

Anonymous, **metadata-only** stats (tokens, tps/ttft, model picked, est. savings) to verify
and improve the router — never prompts, code, or anything user-authored ([policy](./AGENTS.md)).

```
export ROUTER_TELEMETRY_DISABLED=1   # opt out (DO_NOT_TRACK=1 works too)
```
