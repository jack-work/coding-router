"""One command, one endpoint: a local OpenAI-compatible /v1/chat/completions server
that re-routes to a real OpenAI or Anthropic model on every request, based on the full
conversation so far, not just the first message.

Point any tool that speaks the standard OpenAI-compatible Chat Completions API at this
(opencode, or anything else using an openai-compatible client) and every turn gets
routed independently across the router's full arm pool.

No OAuth/subscription-plan login (ChatGPT Plus/Pro, Claude Pro/Max) is used or supported
here: as of 2026 both OpenAI and Anthropic restrict those consumer OAuth tokens to their
own first-party clients (Anthropic made this an explicit Consumer Terms violation for
third-party tools in Feb 2026). Only standard API-key auth is used.

Usage:
    uv run python -m router.serve                  # serves on 127.0.0.1:61890
    uv run python -m router.serve --port 9000

First run with a missing key prompts once (hidden input), validates it with a real
minimal call, and saves it to .env.local.
"""
from __future__ import annotations

import getpass
import json
import os
import pathlib
import sys
import time
import uuid

import anthropic
import numpy as np
import openai
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from router.harness import load_env
from router.router_core import Router

sys.stdout.reconfigure(line_buffering=True)  # status/routing logs show up live, not on exit

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env.local"
DISPATCH_MAX_TOKENS = 32_000


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

    print(f"No {env_var} found (checked environment and .env.local).")
    print(f"Get one at {dashboard_url}")
    key = getpass.getpass(f"Paste your {env_var} (input hidden): ").strip()
    if not key:
        raise SystemExit("no key provided, exiting")

    print("validating key ...")
    try:
        validate(key)
    except Exception as e:  # noqa: BLE001 — surface the real reason a pasted key failed
        raise SystemExit(f"key rejected: {type(e).__name__}: {e}") from e

    os.environ[env_var] = key
    existing = ENV_FILE.read_text() if ENV_FILE.exists() else ""
    lines = [ln for ln in existing.splitlines() if not ln.startswith(f"{env_var}=")]
    lines.append(f"{env_var}={key}")
    ENV_FILE.write_text("\n".join(lines) + "\n")
    print(f"saved to {ENV_FILE}\n")
    return key


# ---------------------------------------------------------------- Chat Completions -> OpenAI Responses

def messages_to_responses_input(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Translate Chat Completions messages into a Responses API instructions/input pair.

    Args:
        messages: Chat Completions-style message list.

    Returns:
        A (system_instructions, input_items) pair for `client.responses.create()`.
    """
    system, items = None, []
    for m in messages:
        role = m.get("role")
        if role == "system":
            system = (system + "\n\n" + m["content"]) if system else m["content"]
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": m["tool_call_id"],
                          "output": m.get("content") or ""})
        elif role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                items.append({"type": "function_call", "call_id": tc["id"],
                              "name": tc["function"]["name"],
                              "arguments": tc["function"]["arguments"]})
            if m.get("content"):
                items.append({"role": "assistant", "content": m["content"]})
        else:
            items.append({"role": role, "content": m.get("content") or ""})
    return system, items


def tools_to_responses(tools: list[dict] | None) -> list[dict]:
    """Flatten Chat Completions' nested tool schema into the Responses API's flat shape."""
    if not tools:
        return []
    out = []
    for t in tools:
        f = t.get("function", t)
        out.append({"type": "function", "name": f["name"],
                    "description": f.get("description", ""),
                    "parameters": f.get("parameters", {"type": "object", "properties": {}})})
    return out


def responses_output_to_message(output: list) -> dict:
    """Collapse Responses API output items into one Chat Completions message object."""
    text_parts, tool_calls = [], []
    for item in output:
        itype = getattr(item, "type", None)
        if itype == "message":
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", "") == "output_text":
                    text_parts.append(c.text)
        elif itype == "function_call":
            tool_calls.append({"id": item.call_id, "type": "function",
                               "function": {"name": item.name, "arguments": item.arguments}})
    msg = {"role": "assistant", "content": "\n".join(text_parts) or None}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


# ---------------------------------------------------------------- Chat Completions -> Anthropic Messages

def messages_to_anthropic(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Translate Chat Completions messages into an Anthropic system/messages pair.

    Consecutive tool-result messages are merged into one user turn with multiple
    tool_result blocks -- Anthropic requires all results for one assistant turn to
    arrive together, or the model learns to stop making parallel tool calls.

    Args:
        messages: Chat Completions-style message list.

    Returns:
        A (system_prompt, messages) pair for `client.messages.create()`.
    """
    system, out = None, []
    for m in messages:
        role = m.get("role")
        if role == "system":
            system = (system + "\n\n" + m["content"]) if system else m["content"]
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"],
                     "content": m.get("content") or ""}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif role == "assistant" and m.get("tool_calls"):
            content = []
            if m.get("content"):
                content.append({"type": "text", "text": m["content"]})
            for tc in m["tool_calls"]:
                content.append({"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"],
                                "input": json.loads(tc["function"]["arguments"] or "{}")})
            out.append({"role": "assistant", "content": content})
        else:
            out.append({"role": role, "content": m.get("content") or ""})
    return system, out


def tools_to_anthropic(tools: list[dict] | None) -> list[dict]:
    """Flatten Chat Completions' nested tool schema into Anthropic's flat shape."""
    if not tools:
        return []
    out = []
    for t in tools:
        f = t.get("function", t)
        out.append({"name": f["name"], "description": f.get("description", ""),
                    "input_schema": f.get("parameters", {"type": "object", "properties": {}})})
    return out


def anthropic_response_to_message(content: list) -> dict:
    """Collapse an Anthropic response's content blocks into one Chat Completions message."""
    text_parts, tool_calls = [], []
    for block in content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append({"id": block.id, "type": "function",
                               "function": {"name": block.name, "arguments": json.dumps(block.input)}})
    msg = {"role": "assistant", "content": "\n".join(text_parts) or None}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def dispatch(decision, messages: list[dict], tools: list[dict] | None,
            openai_client: openai.OpenAI, anthropic_client: anthropic.Anthropic) -> dict:
    """Call the real provider for a routed decision and return a Chat Completions message.

    Args:
        decision: A `router_core.Decision` naming the chosen arm.
        messages: The incoming Chat Completions-style conversation.
        tools: The incoming Chat Completions-style tool definitions, if any.
        openai_client: Client used when the decision picked an OpenAI arm.
        anthropic_client: Client used when the decision picked an Anthropic arm.

    Returns:
        A Chat Completions `message` object (`role`, `content`, optional `tool_calls`).
    """
    if decision.model.startswith("claude"):
        system, anthropic_messages = messages_to_anthropic(messages)
        kwargs = {}
        if system:
            kwargs["system"] = [{"type": "text", "text": system,
                                 "cache_control": {"type": "ephemeral"}}]
        # Anthropic requires streaming for requests that might run long at this
        # max_tokens; .stream() avoids that restriction and still returns one
        # complete message via get_final_message(), same as .create() would.
        with anthropic_client.messages.stream(
                messages=anthropic_messages, max_tokens=DISPATCH_MAX_TOKENS,
                tools=tools_to_anthropic(tools), **kwargs, **decision.request_kwargs) as stream:
            r = stream.get_final_message()
        return anthropic_response_to_message(r.content)

    system, input_items = messages_to_responses_input(messages)
    r = openai_client.responses.create(
        instructions=system, input=input_items, max_output_tokens=DISPATCH_MAX_TOKENS,
        tools=tools_to_responses(tools), **decision.request_kwargs)
    return responses_output_to_message(r.output)


def dispatch_via_openrouter(decision, messages: list[dict], tools: list[dict] | None,
                            client: openai.OpenAI) -> dict:
    """Call the routed model through OpenRouter instead of the provider directly.

    OpenRouter speaks the same Chat Completions dialect the incoming request already
    uses, so no format translation is needed -- only the model id gets a provider
    prefix and the request goes to a different base_url.

    Args:
        decision: A `router_core.Decision` naming the chosen arm.
        messages: The incoming Chat Completions-style conversation, passed through as-is.
        tools: The incoming Chat Completions-style tool definitions, if any.
        client: An `openai.OpenAI` client pointed at OpenRouter's base_url.

    Returns:
        A Chat Completions `message` object (`role`, `content`, optional `tool_calls`).
    """
    family = "anthropic" if decision.model.startswith("claude") else "openai"
    r = client.chat.completions.create(
        model=f"{family}/{decision.model}", messages=messages, tools=tools or None)
    return r.choices[0].message.model_dump(exclude_none=True)


def make_app(router: Router, openai_client: openai.OpenAI, anthropic_client: anthropic.Anthropic,
            openrouter_client: openai.OpenAI | None = None):
    """Build the FastAPI app exposing /v1/chat/completions and /v1/models.

    Args:
        router: A loaded `Router` used for every routing decision.
        openai_client: Used to embed every request (always), and to dispatch OpenAI
            arms directly when `openrouter_client` is not given.
        anthropic_client: Used to dispatch Anthropic arms directly; unused when
            `openrouter_client` is given.
        openrouter_client: If given, every dispatch goes through OpenRouter instead
            of calling OpenAI/Anthropic directly. Embeddings still use `openai_client`
            regardless, since routing decisions are only valid in the embedding space
            the shipped artifact was built with.

    Returns:
        A FastAPI application ready to serve.
    """
    app = FastAPI()

    def embed_text(text: str):
        e = openai_client.embeddings.create(model=router.meta["embed_model"],
                                            input=text[:8000]).data[0].embedding
        return np.array(e, dtype=float)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages: list[dict] = body["messages"]
        stream = bool(body.get("stream"))

        trajectory = "\n".join(
            f"[{m.get('role')}] {json.dumps(m.get('content') or m.get('tool_calls'))[:2000]}"
            for m in messages)
        decision = router.route_embedding(embed_text(trajectory))
        tools = body.get("tools")
        message = (dispatch_via_openrouter(decision, messages, tools, openrouter_client)
                   if openrouter_client
                   else dispatch(decision, messages, tools, openai_client, anthropic_client))

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        print(f"routed -> {decision.model}@{decision.effort or 'default'}  "
              f"p_solve={decision.p_solve:.2f} off_dist={decision.off_distribution} "
              f"({len(messages)} messages in)")
        finish_reason = "tool_calls" if message.get("tool_calls") else "stop"

        if not stream:
            return JSONResponse({
                "id": completion_id, "object": "chat.completion", "created": created,
                "model": decision.model,
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            })

        def sse():
            chunk = {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                     "model": decision.model,
                     "choices": [{"index": 0, "delta": message, "finish_reason": None}]}
            yield f"data: {json.dumps(chunk)}\n\n"
            done = {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                   "model": decision.model,
                   "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]}
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list",
               "data": [{"id": "auto", "object": "model", "created": int(time.time())}]}

    return app


def print_opencode_config(port: int) -> None:
    """Print a ready-to-paste opencode.jsonc provider block for this server."""
    print("opencode config (opencode.jsonc):")
    print(json.dumps({
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


def main(port: int = 61890, artifact_dir: str | None = None, via: str = "direct") -> None:
    """Start the router-proxy server.

    Args:
        port: Local TCP port to serve on.
        artifact_dir: Directory holding router_v0.{json,npz}; defaults to ./results.
        via: "direct" dispatches to OpenAI/Anthropic with their own keys (default).
            "openrouter" dispatches everything through one OpenRouter key instead.
            Embeddings always use a direct OpenAI key either way -- the shipped
            artifact's routing decisions are only valid in that embedding space.
    """
    openai_key = ensure_key("OPENAI_API_KEY", "https://platform.openai.com/api-keys",
                            lambda k: openai.OpenAI(api_key=k).models.list())
    openai_client = openai.OpenAI(api_key=openai_key)

    anthropic_client = openrouter_client = None
    if via == "openrouter":
        openrouter_key = ensure_key("OPENROUTER_API_KEY", "https://openrouter.ai/keys",
                                    lambda k: openai.OpenAI(api_key=k, base_url=OPENROUTER_BASE_URL)
                                    .chat.completions.create(
                                        model="openai/gpt-4o-mini",
                                        messages=[{"role": "user", "content": "hi"}], max_tokens=1))
        openrouter_client = openai.OpenAI(api_key=openrouter_key, base_url=OPENROUTER_BASE_URL)
    else:
        anthropic_key = ensure_key("ANTHROPIC_API_KEY", "https://console.anthropic.com/settings/keys",
                                   lambda k: anthropic.Anthropic(api_key=k).models.list())
        anthropic_client = anthropic.Anthropic(api_key=anthropic_key)

    router = Router(artifact_dir or str(ROOT / "results"))
    app = make_app(router, openai_client, anthropic_client, openrouter_client)
    print(f"ready: {len(router.arms)} arms, k={router.k} tau={router.tau}, via={via}")

    print(f"\nrouter serving at http://127.0.0.1:{port}/v1")
    print_opencode_config(port)
    print()

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
