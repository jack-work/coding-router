# coding-router

Routes every coding-agent request to the cheapest OpenAI/Anthropic model likely to solve it,
via kNN over measured outcomes. One local OpenAI-compatible endpoint; point opencode (or any
openai-compatible client) at it.

## Run

```
uv run python -m router.serve
```

Serves `http://127.0.0.1:61890/v1` and prints a ready-to-paste opencode provider config on
startup. First run prompts once for API keys (hidden input, validated, saved to `.env.local`).

## Telemetry

We collect anonymous, **metadata-only** telemetry to make sure the router is actually working
in the wild and to improve it: token counts, latency (tps/ttft), which model was picked, and
estimated cost savings — that's how we find out whether routing decisions hold up outside our
own benchmarks. Never message content, prompts, code, file paths, or anything user-authored —
the full policy is in [AGENTS.md](./AGENTS.md), and the entire implementation is one small
block in `router/serve.py` you can read.

Opt out:

```
export ROUTER_TELEMETRY_DISABLED=1
```

(`DO_NOT_TRACK=1` is honored too.)
