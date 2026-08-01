# coding-router

One local endpoint that optimizes coding requests between big and small models, so you get
more usage at the same cost. Point opencode (or any OpenAI-compatible client) at it. The
routing decision is made by a small model running on your machine. Routers are hosted on
[Hugging Face](https://huggingface.co/experiential-labs/coding-router).

## Run

```
uv run python -m router.serve
```

Serves `http://127.0.0.1:61890/v1` and prints a ready-to-paste opencode provider config on
startup. First run downloads the routing artifact from Hugging Face, then prompts once for
API keys (hidden input, validated, saved to `.env.local`).

## Telemetry

Anonymous, **metadata-only** stats (tokens, tps/ttft, model picked, est. savings) to verify
and improve the router — never prompts, code, or anything user-authored ([policy](./AGENTS.md)).

```
export ROUTER_TELEMETRY_DISABLED=1   # opt out (DO_NOT_TRACK=1 works too)
```
