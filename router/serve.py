"""One command, one OpenAI-compatible `/v1/chat/completions` endpoint that re-routes
every request to a real OpenAI or Anthropic model, chosen from the full conversation
so far.

Usage:
    uv run python -m router.serve                  # serves on 127.0.0.1:61890
    uv run python -m router.serve --port 9000
"""
from __future__ import annotations

import asyncio
import collections
import getpass
import hashlib
import json
import logging
import os
import pathlib
import queue
import sys
import threading
import time
import urllib.request
import uuid
from collections.abc import Callable, Iterator
from typing import Any, cast

import anthropic
import numpy as np
import openai
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from router.router_core import (
    ARTIFACT_JSON,
    ARTIFACT_NPZ,
    Decision,
    Router,
    TrainedRouter,
    cache_policy,
    est_cost_usd,
    load_router,
    remember_arm,
    sticky_arm,
)
from router.wire import (
    SESSION_KEY_MAX,
    ChatCompletionResponse,
    ChatMessage,
    ChatTool,
    Choice,
    FunctionCall,
    ModelCard,
    ModelList,
    PromptTokensDetails,
    StreamDelta,
    TokenUsage,
    ToolCall,
    anthropic_response_to_message,
    collect,
    mark_anthropic_cache,
    messages_to_anthropic,
    messages_to_responses_input,
    parse_chat_tool,
    responses_output_to_message,
    session_key,
    sse_passthrough,
    tools_to_anthropic,
    tools_to_responses,
    usage_from_chat,
    usage_from_responses,
    wants_usage,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env.local"
DISPATCH_MAX_TOKENS = 32_000

# The one default artifact lives on Hugging Face (heavy files never in git); new router
# versions overwrite it in place. `serve` downloads it on first run, then runs offline.
HF_ARTIFACT_REPO = "experiential-labs/coding-router"


def load_env(path: pathlib.Path | None = None) -> None:
    """Load KEY=VALUE lines from `.env.local` into the environment (existing vars win)."""
    p = path or ENV_FILE
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

logger = logging.getLogger(__name__)


def prompt_chars(system: str | None, messages: list[ChatMessage]) -> int:
    """Approximate the outbound prompt's size in characters, for the minimum check."""
    return len(system or "") + sum(len(m.text()) for m in messages)


_implicit_cache_noticed = False


def note_implicit_cache() -> None:
    """Say once that a provider cached even though directives were switched off.

    `--nocache` governs what this router SENDS. A provider in the implicit-caching
    family takes no request-side signal at all, so it keeps caching a stable
    prefix regardless, and a user watching cost expects to be told rather than to
    discover it in an invoice.
    """
    global _implicit_cache_noticed
    if _implicit_cache_noticed:
        return
    _implicit_cache_noticed = True
    logger.info("note: --nocache stops the cache directives THIS ROUTER sends; the "
                "provider just served a cache read anyway, because it caches "
                "implicitly and takes no request-side signal")


# Only standard API-key auth is used here, never OAuth/subscription-plan login
# (ChatGPT Plus/Pro, Claude Pro/Max): as of 2026 both providers restrict those
# consumer tokens to their own first-party clients (Anthropic made this an explicit
# Consumer Terms violation for third-party tools in Feb 2026).
def ensure_key(env_var: str, dashboard_url: str, validate) -> str:
    """Get a credential from the environment, .env.local, or an interactive prompt.

    Args:
        env_var: Name of the environment variable holding the key.
        dashboard_url: Where to create a key, shown if none is found.
        validate: Called with the raw key string; must raise if the key is invalid.

    Returns:
        The validated key.
    """
    load_env(ENV_FILE)
    key = os.environ.get(env_var)
    if key:
        return key

    logger.info(f"No {env_var} found (checked environment and .env.local).")
    logger.info(f"Get one at {dashboard_url}")
    key = getpass.getpass(f"Paste your {env_var} (input hidden): ").strip()
    if not key:
        raise SystemExit("no key provided, exiting")

    logger.info("validating key ...")
    try:
        validate(key)
    except Exception as e:  # noqa: BLE001 — surface the real reason a pasted key failed
        raise SystemExit(f"key rejected: {type(e).__name__}: {e}") from e

    os.environ[env_var] = key
    existing = ENV_FILE.read_text() if ENV_FILE.exists() else ""
    lines = [ln for ln in existing.splitlines() if not ln.startswith(f"{env_var}=")]
    lines.append(f"{env_var}={key}")
    ENV_FILE.write_text("\n".join(lines) + "\n")
    logger.info(f"saved to {ENV_FILE}\n")
    return key


# ---------------------------------------------------------------- telemetry
# STRICTLY metadata. WE NEVER UPLOAD TRACES OR PII FROM A USER -- see AGENTS.md's
# Telemetry section for the binding property allowlist (counts, durations, model ids,
# booleans, estimated costs; never anything user-authored). Deliberately hand-rolled
# on stdlib (queue + daemon thread + urllib) instead of the PostHog SDK: zero added
# dependencies, zero hot-path latency (capture() only enqueues), and the entire
# privacy surface is this one auditable block. The key below is a PostHog PUBLIC
# write-only project key -- standard practice to ship in source, not a secret.
POSTHOG_KEY = "phc_BKPc6suQaTaWWDftyThB3pVPHfK7bEMmpo3UNbyMcXdm"
POSTHOG_HOST = "https://us.i.posthog.com"

_telemetry_q: queue.Queue[dict[str, Any]] | None = None


def telemetry_enabled() -> bool:
    """True unless ROUTER_TELEMETRY_DISABLED or the standard DO_NOT_TRACK is set."""
    off = ("1", "true", "yes", "on")
    return not (os.environ.get("ROUTER_TELEMETRY_DISABLED", "").lower() in off
                or os.environ.get("DO_NOT_TRACK", "").lower() in off)


def _anon_id() -> str:
    """Get-or-create the random install id (a UUID mapping to nothing) in .env.local."""
    load_env(ENV_FILE)
    existing = os.environ.get("ROUTER_ANALYTICS_ID")
    if existing:
        return existing
    anon = uuid.uuid4().hex
    os.environ["ROUTER_ANALYTICS_ID"] = anon
    prior = ENV_FILE.read_text() if ENV_FILE.exists() else ""
    ENV_FILE.write_text(prior.rstrip("\n") + f"\nROUTER_ANALYTICS_ID={anon}\n" if prior
                        else f"ROUTER_ANALYTICS_ID={anon}\n")
    return anon


def _post_event(payload: dict[str, Any]) -> int:
    """POST one event to PostHog; returns the HTTP status. Raises on network failure."""
    req = urllib.request.Request(
        f"{POSTHOG_HOST}/i/v0/e/", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status


def _telemetry_worker() -> None:
    """Drain the queue forever, swallowing failures -- telemetry may never break serving."""
    assert _telemetry_q is not None
    while True:
        payload = _telemetry_q.get()
        try:
            _post_event(payload)
        except Exception as e:  # noqa: BLE001 -- drop the event, never disturb the server
            logger.debug(f"telemetry: dropped event ({type(e).__name__})")


def start_telemetry() -> None:
    """Start the background sender (call once at startup, only when enabled)."""
    global _telemetry_q
    _telemetry_q = queue.Queue(maxsize=256)
    threading.Thread(target=_telemetry_worker, daemon=True, name="telemetry").start()


def capture(event: str, properties: dict[str, Any]) -> None:
    """Enqueue one metadata-only event; never blocks, drops silently when full/disabled."""
    if _telemetry_q is None:
        return
    payload = {"api_key": POSTHOG_KEY, "event": event, "distinct_id": _anon_id(),
               "properties": properties}
    try:
        _telemetry_q.put_nowait(payload)
    except queue.Full:
        pass


def dispatch(decision: Decision, messages: list[ChatMessage], tools: list[ChatTool] | None,
            openai_client: openai.OpenAI, anthropic_client: anthropic.Anthropic,
            *, cache: bool = True, ttl: str | None = None, session: str | None = None,
            stream: bool = True) -> Iterator[StreamDelta]:
    """Stream one routed decision from its real provider, delta by delta.

    Dispatch is streaming-native in both directions: the provider is always asked
    to stream, and a non-streaming client is served by draining this with
    `wire.collect()`. One code path to the provider means the streaming and
    buffered responses can never disagree about what was said.

    Args:
        decision: A `router_core.Decision` naming the chosen arm.
        messages: The incoming Chat Completions-style conversation.
        tools: The incoming Chat Completions-style tool definitions, if any.
        openai_client: Client used when the decision picked an OpenAI arm.
        anthropic_client: Client used when the decision picked an Anthropic arm.
        cache: Whether to drive prompt caching at all (`--nocache` turns it off).
        ttl: Cache lifetime to request, e.g. "1h"; None means the provider default.
        session: Stable session key, sent to OpenAI as `prompt_cache_key` so repeat
            turns land on the machine holding the warm prefix.
        stream: Ask the provider to stream. Only set when the CLIENT asked to
            stream: some gateways omit `prompt_tokens_details` from streamed usage
            (measured: LiteLLM over Copilot), so a buffered client must not pay
            for a transport it did not request. When it is missing, only the
            ATTRIBUTION is lost -- `prompt_tokens` is still whole, so a client
            summing input + cached + written + output still gets the right
            context size, it just cannot tell how much of the prompt was cached.

    Yields:
        `StreamDelta`s carrying text, tool calls, and finally token usage (whose
        prompt count includes cache reads and writes, so counts stay comparable
        across providers -- Anthropic reports those separately, OpenAI folds them in).
    """
    provider = "anthropic" if decision.model.startswith("claude") else "openai"
    policy = cache_policy(provider, decision.model, ttl=ttl)
    if provider == "anthropic":
        system, anthropic_messages = messages_to_anthropic(messages)
        anthropic_tools = tools_to_anthropic(tools)
        system_blocks = ([{"type": "text", "text": system}] if system else [])
        if cache and policy.worth_marking(prompt_chars(system, messages)):
            mark_anthropic_cache(system_blocks, anthropic_tools, anthropic_messages, policy)
        kwargs = {"system": system_blocks} if system_blocks else {}
        with anthropic_client.messages.stream(
                messages=anthropic_messages, max_tokens=DISPATCH_MAX_TOKENS,
                tools=anthropic_tools, **kwargs, **decision.request_kwargs) as events:
            if stream:
                for event in events:
                    if (getattr(event, "type", None) == "content_block_delta"
                            and getattr(event.delta, "type", None) == "text_delta"):
                        yield StreamDelta(text=event.delta.text)
            # Tool calls arrive as input_json_delta fragments that are only valid
            # JSON once complete, so they are taken from the assembled message
            # rather than forwarded piecemeal.
            r = events.get_final_message()
        message = anthropic_response_to_message(r.content)
        if not stream and message.content:
            yield StreamDelta(text=message.content)
        read = getattr(r.usage, "cache_read_input_tokens", 0) or 0
        write = getattr(r.usage, "cache_creation_input_tokens", 0) or 0
        prompt = r.usage.input_tokens + read + write
        if message.tool_calls:
            yield StreamDelta(tool_calls=message.tool_calls)
        yield StreamDelta(usage=TokenUsage(
            prompt_tokens=prompt, completion_tokens=r.usage.output_tokens,
            total_tokens=prompt + r.usage.output_tokens,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=read,
                                                      cache_write_tokens=write)))
        return

    system, input_items = messages_to_responses_input(messages, policy)
    # OpenAI caches implicitly; the only lever is routing repeat turns to the machine
    # that holds the prefix, which is what prompt_cache_key pins.
    extra = {"prompt_cache_key": session} if (cache and session) else {}
    call = dict(instructions=system, input=input_items,
                max_output_tokens=DISPATCH_MAX_TOKENS, tools=tools_to_responses(tools),
                **extra, **decision.request_kwargs)
    if not stream:
        r = openai_client.responses.create(**call)
        buffered = responses_output_to_message(r.output)
        if buffered.content:
            yield StreamDelta(text=buffered.content)
        if buffered.tool_calls:
            yield StreamDelta(tool_calls=buffered.tool_calls)
        yield StreamDelta(usage=usage_from_responses(getattr(r, "usage", None)))
        return
    for event in openai_client.responses.create(stream=True, **call):
        etype = getattr(event, "type", "")
        if etype == "response.output_text.delta":
            yield StreamDelta(text=event.delta)
        elif etype == "response.output_item.done":
            item = getattr(event, "item", None)
            if item is not None and getattr(item, "type", None) == "function_call":
                yield StreamDelta(tool_calls=[ToolCall(
                    id=item.call_id,
                    function=FunctionCall(name=item.name, arguments=item.arguments))])
        elif etype == "response.completed":
            yield StreamDelta(usage=usage_from_responses(
                getattr(event.response, "usage", None)))


def dispatch_via_openrouter(decision: Decision, messages: list[ChatMessage],
                            tools: list[ChatTool] | None, client: openai.OpenAI,
                            *, cache: bool = True, ttl: str | None = None,
                            session: str | None = None, stream: bool = True,
                            ) -> Iterator[StreamDelta]:
    """Stream the routed model through OpenRouter instead of the provider directly.

    OpenRouter speaks the same Chat Completions dialect the incoming request already
    uses, so no format translation is needed -- only the model id gets a provider
    prefix and the request goes to a different base_url. Messages/tools are
    re-serialized back to plain dicts for the SDK call, keeping any fields our own
    models don't declare (via `extra="allow"`), which is what carries a client's own
    `cache_control` markers through untouched.

    Caching is driven with OpenRouter's request-level directive rather than by
    placing breakpoints here: it advances the breakpoint through the conversation
    itself, and it is the form OpenRouter documents for multi-turn chat. It applies
    to explicit-cache providers only; implicit ones ignore it. `session_id` pins
    sticky routing so later turns reach the endpoint holding the warm prefix.

    Args:
        decision: A `router_core.Decision` naming the chosen arm.
        messages: The incoming Chat Completions-style conversation, passed through as-is.
        tools: The incoming Chat Completions-style tool definitions, if any.
        client: An `openai.OpenAI` client pointed at OpenRouter's base_url.
        cache: Whether to drive prompt caching at all.
        ttl: Cache lifetime to request, e.g. "1h"; None means the provider default.
        session: Stable session key for sticky routing (<=256 chars).
        stream: Ask the gateway to stream; see `dispatch` for why this follows the
            client rather than always being on.

    Yields:
        `StreamDelta`s carrying text, tool calls, and finally token usage.
    """
    family = "anthropic" if decision.model.startswith("claude") else "openai"
    raw_messages = [m.model_dump(exclude_none=True) for m in messages]
    raw_tools = [t.model_dump(exclude_none=True) for t in tools] if tools else None
    body: dict[str, Any] = {}
    if cache:
        policy = cache_policy(family, decision.model, ttl=ttl)
        if policy.style == "explicit":
            cc: dict[str, str] = {"type": "ephemeral"}
            if policy.ttl:
                cc["ttl"] = policy.ttl
            body["cache_control"] = cc
        if session:
            body["session_id"] = session[:SESSION_KEY_MAX]
    call = dict(model=f"{family}/{decision.model}", messages=raw_messages,
                tools=raw_tools, extra_body=body or None)
    if not stream:
        r = client.chat.completions.create(**call)
        buffered = ChatMessage.model_validate(
            r.choices[0].message.model_dump(exclude_none=True))
        if buffered.content:
            yield StreamDelta(text=str(buffered.content))
        if buffered.tool_calls:
            yield StreamDelta(tool_calls=buffered.tool_calls)
        yield StreamDelta(usage=usage_from_chat(r.usage))
        return
    # Usage only rides along a stream when it is asked for, and without it every
    # bucket reads zero however much was really spent.
    chunks = client.chat.completions.create(
        stream=True, stream_options={"include_usage": True}, **call)
    pending: dict[int, ToolCall] = {}
    for chunk in chunks:
        u = getattr(chunk, "usage", None)
        if u is not None:
            yield StreamDelta(usage=usage_from_chat(u))
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            yield StreamDelta(text=delta.content)
        # Tool-call arguments arrive as fragments keyed by index, so they are
        # accumulated here and emitted whole -- a partial argument string is not
        # valid JSON and a client cannot act on it.
        for fragment in getattr(delta, "tool_calls", None) or []:
            slot = pending.setdefault(
                fragment.index, ToolCall(id=fragment.id or "",
                                         function=FunctionCall(name="", arguments="")))
            if fragment.id:
                slot.id = fragment.id
            fn = getattr(fragment, "function", None)
            if fn is not None:
                slot.function.name += fn.name or ""
                slot.function.arguments += fn.arguments or ""
    if pending:
        yield StreamDelta(tool_calls=[pending[i] for i in sorted(pending)])


def message_preview(m: ChatMessage) -> str:
    """Render one message for the routing-embedding trajectory.

    User content renders BARE: the evidence bank's rows are embedded from raw task
    texts, and any wrapper ("[user] \"...\"" was measured) inflates short-text
    similarity across the board -- short non-coding asks stopped abstaining because
    they matched the wrapper, not the task. Other roles keep a role tag so multi-turn
    structure survives.
    """
    text = m.text()
    if m.role == "user" and text:
        return text[:2000]
    payload = text or ([tc.model_dump() for tc in m.tool_calls] if m.tool_calls else None)
    return f"[{m.role}] {json.dumps(payload)[:2000]}"


# ---------------------------------------------------------------- trajectory budget
# The routing embedding's input budget. text-embedding-3-large accepts 8,191 TOKENS;
# these are char budgets (~4 chars/token) with headroom, and embed_text() halves and
# retries on a token-limit rejection, so the heuristic can never hard-fail a request.
#
# Layout once a conversation outgrows the budget:
#   anchor  -- the first user message, verbatim. The kNN reference set is task-
#              description-shaped (repo-issue statements, p50 ~2,000 chars), so this
#              is what carries in-distribution similarity; lose it and off_distribution
#              abstention routes every long session to the expensive fallback arm.
#   middle  -- older messages, replaced by cached small-model summaries.
#   recent  -- the newest messages, verbatim: the per-turn "struggling vs cruising"
#              signal that per-turn routing exists to detect.
EMBED_BUDGET = 24_000
ANCHOR_BUDGET = 6_000
MIDDLE_BUDGET = 8_000
RECENT_MSGS = 8       # newest messages kept verbatim
CHUNK_MSGS = 6        # ~3 tool round-trips per summarized chunk
# ---------------------------------------------------------------- local routing infra
# Routing never touches an API: the artifact is fetched once, and every embedding is
# computed in-process by a small local model in the SAME vector space the artifact's
# reference tasks were embedded in (kNN cosine across two embedding models' spaces is
# meaningless). MLX runs the exact quantized model the shipped bank was built with.
EMBED_MODEL_MLX = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
EMBED_MODEL_TORCH = "Qwen/Qwen3-Embedding-0.6B"

_local_embed: Callable[[str], np.ndarray] | None = None


def load_local_embedder(mlx_model: str = EMBED_MODEL_MLX,
                        torch_model: str = EMBED_MODEL_TORCH) -> Callable[[str], np.ndarray]:
    """Load the local Qwen3 embedding backend once and return the embed function.

    Backend selection: MLX on Apple Silicon (the quantized model the shipped artifact
    was embedded with), otherwise sentence-transformers on CUDA when available, CPU if
    not. `mlx_model`/`torch_model` may be local paths (a trained artifact's tuned
    encoder) or hub ids; weights download from Hugging Face on first ever use, then it
    is fully offline -- no API key, no network, no per-request cost.

    Returns:
        A function embedding one text into a unit-norm vector in the artifact's space.
    """
    global _local_embed
    if _local_embed is not None:
        return _local_embed
    import platform as _platform

    if _platform.system() == "Darwin" and _platform.machine() == "arm64":
        import mlx.core as mx
        from mlx_embeddings import generate, load

        model, tokenizer = load(mlx_model)

        def embed(text: str) -> np.ndarray:
            """Embed one text via mlx_embeddings (pooling/normalization are the package's)."""
            out = generate(model, tokenizer, texts=[text])
            return np.array(out.text_embeds.astype(mx.float32))[0].astype(float)

        logger.info(f"embedding: local mlx {mlx_model}")
    else:
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(torch_model)

        def embed(text: str) -> np.ndarray:
            """Embed one text via sentence-transformers, unit-normalized."""
            return st.encode([text], normalize_embeddings=True)[0].astype(float)

        logger.info(f"embedding: local {st.device} {torch_model}")
    _local_embed = embed
    return embed


def ensure_artifact(artifact_dir: pathlib.Path) -> None:
    """Download the default router artifact from Hugging Face if not already on disk.

    Delegates to huggingface_hub (the same library the encoder download uses), which
    handles redirects, tokens, and retries -- hand-rolled urllib redirect handling
    broke the day HF started answering with relative Location headers.

    Args:
        artifact_dir: Directory the router.{json,npz} pair should live in.
    """
    missing = [n for n in (ARTIFACT_JSON, ARTIFACT_NPZ) if not (artifact_dir / n).exists()]
    if not missing:
        return
    from huggingface_hub import hf_hub_download  # deferred: startup-only

    load_env(ENV_FILE)  # HF_TOKEN, if the artifact repo is private
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for name in missing:
        logger.info(f"artifact: downloading {name} from {HF_ARTIFACT_REPO} ...")
        hf_hub_download(HF_ARTIFACT_REPO, name, local_dir=artifact_dir)
        logger.info(f"artifact: saved {artifact_dir / name}")


SUMMARY_MODEL = "gpt-5.4-nano"
SUMMARY_CACHE_CAP = 512

# Chat Completions is stateless (no session id), but conversations are append-only, so
# summaries are keyed by a hash of the segment's own content: the same segment hashes
# the same on every later turn, across concurrent sessions, and after client restarts.
# Each segment is therefore summarized exactly once per process. In-memory LRU only --
# losing it on restart just re-summarizes each segment once.
_summary_cache: collections.OrderedDict[str, str] = collections.OrderedDict()


def summarize_segment(text: str, client: openai.OpenAI, model: str) -> str:
    """Summarize one trajectory segment for the routing embedding, cached by content hash.

    Failures are cached too (as a head-slice of the segment), so a broken or slow
    summarizer is paid at most once per segment and can never block dispatch.

    Args:
        text: The rendered message previews making up this segment.
        client: OpenAI-protocol client for the summarization call (OpenAI direct, or
            OpenRouter in `--via=openrouter` mode -- Chat Completions works on both).
        model: Model to summarize with (cheap and fast matters more than eloquence).

    Returns:
        A dense summary of the segment, or a head-slice fallback on failure.
    """
    key = hashlib.sha256(text.encode()).hexdigest()[:16]
    if key in _summary_cache:
        _summary_cache.move_to_end(key)
        return _summary_cache[key]
    try:
        # Generous cap on purpose: reasoning tokens count against it, and an unused
        # cap is free (see AgentRunner.MIN_MAX_TOKENS for the measured version of
        # this lesson).
        r = client.chat.completions.create(
            model=model, reasoning_effort="low", max_completion_tokens=4_000, timeout=20.0,
            messages=[{"role": "user", "content":
                       "Summarize this segment of a coding-agent conversation in under 120 "
                       "words. Keep only what matters for judging task difficulty and "
                       "progress: what was attempted, key files/commands/errors, and current "
                       "blockers. No preamble.\n\n" + text}])
        out = (r.choices[0].message.content or "").strip() or text[:600]
        logger.info(f"trajectory: summarized segment {key} ({len(text)} -> {len(out)} chars)")
    except Exception as e:  # noqa: BLE001 -- routing must degrade, never block dispatch
        out = text[:600]
        logger.warning(f"trajectory: summarizer failed on {key} ({type(e).__name__}); using head slice")
    _summary_cache[key] = out
    if len(_summary_cache) > SUMMARY_CACHE_CAP:
        _summary_cache.popitem(last=False)
    return out


def build_trajectory(messages: list[ChatMessage], client: openai.OpenAI,
                     summary_model: str, summarize_middle: bool = True) -> str:
    """Render the conversation as the routing-embedding input, within EMBED_BUDGET.

    System-role messages are excluded entirely: a client's system prompt (opencode
    ships ~12k chars of constant preamble + tool schemas) is near-identical bytes on
    EVERY request, so embedding it homogenizes all trajectories toward one vector and
    the router degenerates to a constant arm (measured in production). The task and
    difficulty signals live in user/assistant/tool messages, which all stay.

    Conversations that fit are rendered whole -- the fast path, no model calls,
    byte-identical to pre-budget behavior. Past the budget: anchor + summarized
    middle + verbatim recent (see the budget comment above for why each part
    exists). The middle is chunked on a fixed index grid so a completed chunk's
    content -- and its summary cache key -- never changes as the conversation
    grows; only complete chunks are summarized, so no per-turn re-summarization
    of a still-moving partial chunk ever happens. If the summaries themselves
    overflow MIDDLE_BUDGET they are compacted once more (summary-of-summaries,
    also cached); a final hard slice is the invariant of last resort.

    Args:
        messages: The full incoming conversation.
        client: OpenAI client used for summarization calls.
        summary_model: Model to summarize middle chunks with.
        summarize_middle: If False, skip summaries entirely and fall back to
            anchor + recent -- the zero-LLM-call baseline (`--nosummarize`).

    Returns:
        The trajectory text to embed, at most EMBED_BUDGET chars.
    """
    routable = [m for m in messages if m.role != "system"]
    previews = [message_preview(m) for m in routable]
    full = "\n".join(previews)
    if len(full) <= EMBED_BUDGET:
        return full

    anchor_i = next((i for i, m in enumerate(routable) if m.role == "user"), 0)
    anchor = previews[anchor_i][:ANCHOR_BUDGET]

    recent_start = max(anchor_i + 1, len(previews) - RECENT_MSGS)
    mid_previews = previews[anchor_i + 1:recent_start]
    n_complete = (len(mid_previews) // CHUNK_MSGS) * CHUNK_MSGS
    chunks = [mid_previews[i:i + CHUNK_MSGS] for i in range(0, n_complete, CHUNK_MSGS)]
    # The partial chunk next to the recent window is still growing -- summarizing it
    # would mean a fresh (uncacheable) summary call on every turn, so it stays verbatim.
    leftover = mid_previews[n_complete:]

    mid = ""
    if summarize_middle and chunks:
        summaries = [summarize_segment("\n".join(c), client, summary_model) for c in chunks]
        mid = "\n".join(f"[earlier] {s}" for s in summaries)
        if len(mid) > MIDDLE_BUDGET:
            mid = "[earlier] " + summarize_segment("\n".join(summaries), client, summary_model)
        mid = mid[:MIDDLE_BUDGET]

    recent = leftover + previews[recent_start:]
    parts = [anchor] + ([mid] if mid else []) + recent
    while len("\n".join(parts)) > EMBED_BUDGET and len(recent) > 1:
        recent.pop(0)  # trim the oldest of the verbatim recent messages first
        parts = [anchor] + ([mid] if mid else []) + recent
    return "\n".join(parts)[:EMBED_BUDGET]


def make_app(router: Router | TrainedRouter, openai_client: openai.OpenAI | None,
            anthropic_client: anthropic.Anthropic | None,
            openrouter_client: openai.OpenAI | None = None,
            summary_model: str = SUMMARY_MODEL, summarize_middle: bool = True,
            cache: bool = True, cache_ttl: str | None = None,
            sticky: bool = True) -> FastAPI:
    """Build the FastAPI app exposing /v1/chat/completions and /v1/models.

    Routing itself (embedding + kNN) runs fully locally -- the clients below exist
    only to dispatch the chosen arm and to summarize long trajectories.

    Args:
        router: A loaded `Router` used for every routing decision.
        openai_client: Dispatches OpenAI arms and summarizes trajectories in direct
            mode; None in `--via=openrouter` mode (OpenRouter covers both).
        anthropic_client: Dispatches Anthropic arms directly; None when
            `openrouter_client` is given.
        openrouter_client: If given, every dispatch (and summarization) goes through
            OpenRouter instead of calling OpenAI/Anthropic directly.
        summary_model: Model used to summarize older trajectory chunks.
        summarize_middle: If False, long conversations embed anchor + recent only
            (no summarization calls at all).
        cache: Drive provider prompt caching (breakpoints, directives, cache keys).
        cache_ttl: Cache lifetime to request, e.g. "1h" (2x write on Anthropic).
        sticky: Keep a session on the arm it used last, so its prefix stays warm.

    Returns:
        A FastAPI application ready to serve.
    """
    app = FastAPI()
    if openrouter_client is None and (openai_client is None or anthropic_client is None):
        raise ValueError("direct mode needs both an OpenAI and an Anthropic client")
    local_embed = load_local_embedder(*(router.embed_models()
                                        or (EMBED_MODEL_MLX, EMBED_MODEL_TORCH)))
    # The guard above makes these casts honest: exactly one dispatch mode is fully wired.
    summary_client = cast(openai.OpenAI, openrouter_client or openai_client)

    embed_lock = threading.Lock()  # mlx/torch encoders aren't promised thread-safe

    def embed_text(text: str) -> np.ndarray:
        """Embed `text` locally in the artifact's own vector space, in-process."""
        with embed_lock:
            return local_embed(text[:EMBED_BUDGET])

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """Route one Chat Completions request to a real model and return its reply.

        The whole pipeline (summaries, local embedding, provider dispatch) is
        blocking work, so it runs in a worker thread: one slow request -- a long
        dispatch, a stuck summarizer -- must never stall the event loop and every
        other session with it.

        Args:
            request: The raw incoming HTTP request; its JSON body is Chat
                Completions shaped (`messages`, optional `tools`, optional `stream`).

        Returns:
            A `ChatCompletionResponse` JSON body for a normal request, or a
            `StreamingResponse` emitting one SSE chunk plus a `[DONE]` sentinel
            when `stream` is set.
        """
        return await asyncio.to_thread(complete, await request.json(), request.headers)

    def complete(body: dict, headers=None) -> JSONResponse | StreamingResponse:
        """Route, dispatch, and shape one parsed Chat Completions request (sync)."""
        messages = [ChatMessage.model_validate(m) for m in body["messages"]]
        stream = bool(body.get("stream"))
        session = session_key(body, headers)

        t0 = time.perf_counter()
        trajectory = build_trajectory(messages, summary_client, summary_model, summarize_middle)
        decision = router.route_embedding(
            embed_text(trajectory), prefer=sticky_arm(session) if sticky else None)
        remember_arm(session, decision.arm_id)
        t_routed = time.perf_counter()
        tools = [parse_chat_tool(t) for t in (body.get("tools") or [])] or None
        deltas = (dispatch_via_openrouter(decision, messages, tools, openrouter_client,
                                          cache=cache, ttl=cache_ttl, session=session,
                                          stream=stream)
                  if openrouter_client
                  else dispatch(decision, messages, tools,
                                cast(openai.OpenAI, openai_client),
                                cast(anthropic.Anthropic, anthropic_client),
                                cache=cache, ttl=cache_ttl, session=session,
                                stream=stream))

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        # Time to FIRST token, not to the last one. It only means anything now that
        # the provider's output is forwarded as it arrives.
        timing: dict[str, float] = {}

        def timed(source: Iterator[StreamDelta]) -> Iterator[StreamDelta]:
            """Pass deltas through, recording when the first and last one arrived."""
            for delta in source:
                if delta.text or delta.tool_calls:
                    timing.setdefault("first", time.perf_counter())
                yield delta
            timing["last"] = time.perf_counter()

        def report(usage: TokenUsage) -> None:
            """Log the routed line and capture metadata-only telemetry for this request.

            Runs once the reply is complete: token counts and timings are not known
            until the provider's stream ends.
            """
            t_done = timing.get("last", time.perf_counter())
            if not cache and usage.cache_read:
                note_implicit_cache()
            provider_s = t_done - t_routed
            ttft_s = timing.get("first", t_done) - t0
            tps = (round(usage.completion_tokens / provider_s, 1)
                   if provider_s > 0 and usage.completion_tokens else None)
            logger.info(f"routed -> {decision.model}@{decision.effort or 'default'}  "
                        f"p_solve={decision.p_solve:.2f} off_dist={decision.off_distribution} "
                        f"sticky={decision.sticky_held} "
                        f"({len(messages)} messages in, traj={len(trajectory)}ch) | "
                        f"cache r/w {usage.cache_read}/{usage.cache_write} | "
                        f"routing {t_routed - t0:.2f}s, ttft {ttft_s:.2f}s, "
                        f"provider {provider_s:.1f}s, {tps or '-'} tps")

            # Metadata-only telemetry (see the telemetry section + AGENTS.md): counts,
            # durations, model ids, and cost estimates -- never any request content.
            def price(model: str) -> float | None:
                """Cost of this request's measured tokens under `model`'s rates."""
                return est_cost_usd(model, inp=usage.uncached_prompt,
                                    cache_read=usage.cache_read,
                                    cache_write=usage.cache_write,
                                    out=usage.completion_tokens)

            cost = price(decision.model)
            # Savings vs the always-strongest-arm baseline, priced on THIS request's
            # token counts -- the standard counterfactual (the baseline model would
            # produce somewhat different output lengths). Both sides now price cache
            # reads and writes from the reported split, so the figure is smaller, and
            # honest: the baseline arm would have been cached too.
            baseline = price(router.arm_spec[router.arms[router.fallback]].model)
            capture("request_routed", {
                "model": decision.model, "effort": decision.effort,
                "p_solve": round(decision.p_solve, 3),
                "off_distribution": decision.off_distribution,
                "fallback_used": decision.fallback_used,
                "sticky_held": decision.sticky_held,
                "via": "openrouter" if openrouter_client else "direct",
                "stream": stream, "n_messages": len(messages),
                "trajectory_chars": len(trajectory),
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                # Counts only. The session key they belong to is user-derived and never
                # leaves this process.
                "cache_read_tokens": usage.cache_read,
                "cache_write_tokens": usage.cache_write,
                "routing_s": round(t_routed - t0, 3), "provider_s": round(provider_s, 3),
                "ttft_s": round(ttft_s, 3), "total_s": round(t_done - t0, 3),
                "tps": tps,
                "cost_usd": cost, "baseline_cost_usd": baseline,
                "est_savings_usd": (round(baseline - cost, 6)
                                    if cost is not None and baseline is not None else None),
            })

        if not stream:
            message, usage = collect(timed(deltas))
            report(usage)
            resp = ChatCompletionResponse(
                id=completion_id, created=created, model=decision.model,
                choices=[Choice(index=0, message=message,
                                finish_reason="tool_calls" if message.tool_calls else "stop")],
                usage=usage)
            return JSONResponse(resp.model_dump(exclude_none=True))

        def streamed() -> Iterator[str]:
            """Forward the provider's output as SSE, then report once it has ended."""
            seen = TokenUsage()

            def tap(source: Iterator[StreamDelta]) -> Iterator[StreamDelta]:
                """Remember the usage the provider reported on its way past."""
                nonlocal seen
                for delta in source:
                    if delta.usage is not None:
                        seen = delta.usage
                    yield delta

            yield from sse_passthrough(completion_id, created, decision.model,
                                       tap(timed(deltas)), wants_usage(body))
            report(seen)

        return StreamingResponse(streamed(), media_type="text/event-stream")

    @app.get("/v1/models")
    def models() -> ModelList:
        """List the one synthetic "auto" model this router exposes."""
        return ModelList(data=[ModelCard(id="auto", created=int(time.time()))])

    return app


def print_opencode_config(port: int) -> None:
    """Print a ready-to-paste opencode.jsonc provider block for this server."""
    logger.info("opencode config (opencode.jsonc):")
    logger.info(json.dumps({
        "$schema": "https://opencode.ai/config.json",
        "provider": {"local-router": {
            "npm": "@ai-sdk/openai-compatible", "name": "Local Router (per-turn)",
            "options": {"baseURL": f"http://127.0.0.1:{port}/v1", "apiKey": "not-needed"},
            "models": {"auto": {"name": "Router (auto per-turn)",
                                "limit": {"context": 200000, "output": DISPATCH_MAX_TOKENS}}},
        }},
        "model": "local-router/auto",
    }, indent=2))


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def openrouter_base_url() -> str:
    """Where `--via=openrouter` should send requests.

    OpenRouter's dialect is spoken by every self-hosted gateway (LiteLLM and
    friends), and pointing at one is how you route through credentials you
    already hold, or test without an OpenRouter account at all. Hardcoding the
    URL made that impossible, so `OPENROUTER_BASE_URL` in the environment or in
    `.env.local` overrides it.

    Returns:
        The configured base URL, or OpenRouter's own.
    """
    load_env(ENV_FILE)
    return os.environ.get("OPENROUTER_BASE_URL") or OPENROUTER_BASE_URL


def main(port: int = 61890, artifact_dir: str | None = None, via: str = "direct",
         summarize: bool = True, summary_model: str = SUMMARY_MODEL,
         cache: bool = True, cache_ttl: str | None = None, sticky: bool = True) -> None:
    """Start the router-proxy server.

    Routing runs fully locally (in-process embeddings + kNN over the artifact); API
    keys are only for dispatching the chosen model and summarizing long trajectories.

    Args:
        port: Local TCP port to serve on.
        artifact_dir: Directory holding router.{json,npz}; defaults to ./results, and
            the default artifact is downloaded from Hugging Face if missing.
        via: "direct" dispatches to OpenAI/Anthropic with their own keys (default).
            "openrouter" dispatches everything through one OpenRouter key instead --
            no other key needed.
        summarize: Summarize older messages (cached, one cheap call per ~6 messages)
            when a conversation outgrows the embedding budget. `--nosummarize` falls
            back to embedding just the task anchor + most recent messages.
        summary_model: Model used for those summaries.
        cache: Drive provider prompt caching -- breakpoints on Anthropic, the
            request-level directive via OpenRouter, `prompt_cache_key` on OpenAI.
            `--nocache` sends none of it. It does NOT switch caching off: providers
            that cache implicitly take no request-side signal and will keep caching
            a stable prefix regardless.
        cache_ttl: Ask for a longer cache lifetime, e.g. "1h". Anthropic bills a 1h
            write at 2x input instead of 1.25x, so it pays off only across a session
            longer than the 5-minute default.
        sticky: Keep a session on the arm it chose first, while that arm still meets
            the artifact's own bar, so its cached prefix survives. `--nosticky`
            restores per-turn re-decision (and reproduces the artifact's certified
            selection rule exactly).
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    # Root at INFO unmutes httpx's per-request "HTTP Request: ..." records (the
    # OpenAI/Anthropic SDKs' HTTP layer); print never showed them, so gate them out.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    openai_client = anthropic_client = openrouter_client = None
    if via == "openrouter":
        base_url = openrouter_base_url()
        if base_url != OPENROUTER_BASE_URL:
            logger.info(f"openrouter dialect via gateway: {base_url}")
        openrouter_key = ensure_key("OPENROUTER_API_KEY", "https://openrouter.ai/keys",
                                    lambda k: openai.OpenAI(api_key=k, base_url=base_url)
                                    .chat.completions.create(
                                        model="openai/gpt-4o-mini",
                                        messages=[{"role": "user", "content": "hi"}], max_tokens=1))
        openrouter_client = openai.OpenAI(api_key=openrouter_key, base_url=base_url)
        if "/" not in summary_model:
            summary_model = f"openai/{summary_model}"  # OpenRouter namespaces models
    else:
        openai_key = ensure_key("OPENAI_API_KEY", "https://platform.openai.com/api-keys",
                                lambda k: openai.OpenAI(api_key=k).models.list())
        openai_client = openai.OpenAI(api_key=openai_key)
        anthropic_key = ensure_key("ANTHROPIC_API_KEY", "https://console.anthropic.com/settings/keys",
                                   lambda k: anthropic.Anthropic(api_key=k).models.list())
        anthropic_client = anthropic.Anthropic(api_key=anthropic_key)

    art_dir = pathlib.Path(artifact_dir) if artifact_dir else ROOT / "results"
    ensure_artifact(art_dir)
    router = load_router(art_dir)
    app = make_app(router, openai_client, anthropic_client, openrouter_client,
                   summary_model=summary_model, summarize_middle=summarize,
                   cache=cache, cache_ttl=cache_ttl, sticky=sticky)
    logger.info(f"ready: {len(router.arms)} arms, {router.rule}, via={via}, "
                f"summarize={summarize}, cache={cache}"
                f"{'/' + cache_ttl if cache_ttl else ''}, sticky={sticky}")
    if not cache:
        logger.info("cache: directives OFF -- this stops what the router sends, not "
                    "caching itself; implicit-caching providers still cache")

    if telemetry_enabled():
        start_telemetry()
        capture("server_started", {"n_arms": len(router.arms), "via": via,
                                   "summarize": summarize})
        logger.info("telemetry: anonymous metadata-only usage stats ON "
                    "(opt out: ROUTER_TELEMETRY_DISABLED=1; policy in AGENTS.md/README)")
    else:
        logger.info("telemetry: disabled")

    logger.info(f"\nrouter serving at http://127.0.0.1:{port}/v1")
    print_opencode_config(port)
    logger.info("")

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
