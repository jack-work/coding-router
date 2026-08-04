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

## Prompt caching

On by default. The router places cache breakpoints for you — it is the only party that knows
which model a request was routed to, and therefore that model's minimum cacheable length and
breakpoint budget. Clients that mark their own blocks keep them: inbound `cache_control`
markers are preserved and topped up to Anthropic's limit of four, never past it.

```
--nocache          send no cache directives at all
--cache-ttl 1h     ask for the 1-hour cache (2x write price, 5m is the default)
--nosticky         re-decide the arm every turn (restores the artifact's certified rule)
```

`--nocache` governs what the router *sends*. It cannot switch off a provider that caches
implicitly — OpenAI-family backends cache any stable prefix with no request-side signal, so
you will still see `cached_tokens` come back.

Stickiness is what makes caching pay: a conversation stays on the arm it started with while
that arm still clears the artifact's own bar, because a cache lives per model and exact
prefix, so switching arms mid-conversation throws it away. Escalation always overrides it.
Cache reads and writes are reported back in `usage.prompt_tokens_details`.

## Telemetry

Anonymous, **metadata-only** stats (tokens, tps/ttft, model picked, est. savings) to verify
and improve the router — never prompts, code, or anything user-authored ([policy](./AGENTS.md)).

```
export ROUTER_TELEMETRY_DISABLED=1   # opt out (DO_NOT_TRACK=1 works too)
```
