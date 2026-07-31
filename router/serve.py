"""One command, one OpenAI-compatible `/v1/chat/completions` endpoint that re-routes
every request to a real OpenAI or Anthropic model, chosen from the full conversation
so far.

Usage:
    uv run python -m router.serve                  # serves on 127.0.0.1:61890
    uv run python -m router.serve --port 9000
"""
from __future__ import annotations

import getpass
import json
import os
import pathlib
import sys
import time
import uuid
from typing import Any

import anthropic
import numpy as np
import openai
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from router.harness import load_env
from router.router_core import Decision, Router

sys.stdout.reconfigure(line_buffering=True)  # status/routing logs show up live, not on exit

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env.local"
DISPATCH_MAX_TOKENS = 32_000


# ---------------------------------------------------------------- Chat Completions shapes
# These mirror the OpenAI Chat Completions wire format, which is this server's public
# contract (what opencode et al. actually send/receive). `extra="allow"` lets an
# unrecognized field survive a parse -> re-serialize round trip (e.g. the OpenRouter
# passthrough path below), instead of silently dropping it.
class FunctionCall(BaseModel):
    """Function name and JSON-encoded arguments for one tool call, Chat Completions style."""

    model_config = ConfigDict(extra="allow")

    name: str
    arguments: str


class ToolCall(BaseModel):
    """One assistant-issued tool call, Chat Completions style."""

    model_config = ConfigDict(extra="allow")

    id: str
    type: str = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    """One Chat Completions conversation message (system/user/assistant/tool)."""

    model_config = ConfigDict(extra="allow")

    role: str | None = None
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ChatFunctionDef(BaseModel):
    """The `function` block of a Chat Completions tool definition."""

    model_config = ConfigDict(extra="allow")

    name: str
    description: str = ""
    # Arbitrary JSON-schema blob describing the tool's parameters -- shape is fixed
    # only by the tool author, never by us.
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class ChatTool(BaseModel):
    """One Chat Completions tool definition."""

    model_config = ConfigDict(extra="allow")

    type: str = "function"
    function: ChatFunctionDef


def parse_chat_tool(raw: dict) -> ChatTool:
    """Parse one incoming tool definition, tolerating a flat (non-nested) shape.

    Args:
        raw: A raw tool dict, either `{"type": "function", "function": {...}}` or
            (defensively) just the function block itself with no wrapping.

    Returns:
        The parsed `ChatTool`.
    """
    fn = raw.get("function", raw)
    return ChatTool(type=raw.get("type", "function"), function=ChatFunctionDef.model_validate(fn))


# ---------------------------------------------------------------- response bodies
class Choice(BaseModel):
    """One choice in a non-streaming Chat Completions response."""

    index: int
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    """A non-streaming Chat Completions response body."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]


class ChunkChoice(BaseModel):
    """One choice within a streaming `chat.completion.chunk` payload."""

    index: int
    delta: ChatMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """A single SSE `chat.completion.chunk` payload."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]


class ModelCard(BaseModel):
    """One entry in the `/v1/models` listing."""

    id: str
    object: str = "model"
    created: int


class ModelList(BaseModel):
    """The `/v1/models` response body."""

    object: str = "list"
    data: list[ModelCard]


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

def messages_to_responses_input(messages: list[ChatMessage]) -> tuple[str | None, list[dict]]:
    """Translate Chat Completions messages into a Responses API instructions/input pair.

    Args:
        messages: Chat Completions-style message list.

    Returns:
        A (system_instructions, input_items) pair for `client.responses.create()`.
    """
    system, items = None, []
    for m in messages:
        role = m.role
        if role == "system":
            system = (system + "\n\n" + m.content) if system else m.content
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": m.tool_call_id,
                          "output": m.content or ""})
        elif role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                items.append({"type": "function_call", "call_id": tc.id,
                              "name": tc.function.name,
                              "arguments": tc.function.arguments})
            if m.content:
                items.append({"role": "assistant", "content": m.content})
        else:
            items.append({"role": role, "content": m.content or ""})
    return system, items


def tools_to_responses(tools: list[ChatTool] | None) -> list[dict]:
    """Flatten Chat Completions' nested tool schema into the Responses API's flat shape."""
    if not tools:
        return []
    return [{"type": "function", "name": t.function.name,
             "description": t.function.description,
             "parameters": t.function.parameters} for t in tools]


def responses_output_to_message(output: list) -> ChatMessage:
    """Collapse Responses API output items into one Chat Completions message.

    Args:
        output: The `response.output` list from an OpenAI Responses API call.

    Returns:
        The equivalent Chat Completions `ChatMessage`.
    """
    text_parts, tool_calls = [], []
    for item in output:
        itype = getattr(item, "type", None)
        if itype == "message":
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", "") == "output_text":
                    text_parts.append(c.text)
        elif itype == "function_call":
            tool_calls.append(ToolCall(id=item.call_id,
                                       function=FunctionCall(name=item.name, arguments=item.arguments)))
    return ChatMessage(role="assistant", content="\n".join(text_parts) or None,
                       tool_calls=tool_calls or None)


# ---------------------------------------------------------------- Chat Completions -> Anthropic Messages

def messages_to_anthropic(messages: list[ChatMessage]) -> tuple[str | None, list[dict]]:
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
        role = m.role
        if role == "system":
            system = (system + "\n\n" + m.content) if system else m.content
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.tool_call_id,
                     "content": m.content or ""}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif role == "assistant" and m.tool_calls:
            content = []
            if m.content:
                content.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                content.append({"type": "tool_use", "id": tc.id, "name": tc.function.name,
                                "input": json.loads(tc.function.arguments or "{}")})
            out.append({"role": "assistant", "content": content})
        else:
            out.append({"role": role, "content": m.content or ""})
    return system, out


def tools_to_anthropic(tools: list[ChatTool] | None) -> list[dict]:
    """Flatten Chat Completions' nested tool schema into Anthropic's flat shape."""
    if not tools:
        return []
    return [{"name": t.function.name, "description": t.function.description,
             "input_schema": t.function.parameters} for t in tools]


def anthropic_response_to_message(content: list) -> ChatMessage:
    """Collapse an Anthropic response's content blocks into one Chat Completions message.

    Args:
        content: The `message.content` block list from an Anthropic Messages API call.

    Returns:
        The equivalent Chat Completions `ChatMessage`.
    """
    text_parts, tool_calls = [], []
    for block in content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(id=block.id,
                                       function=FunctionCall(name=block.name,
                                                             arguments=json.dumps(block.input))))
    return ChatMessage(role="assistant", content="\n".join(text_parts) or None,
                       tool_calls=tool_calls or None)


def dispatch(decision: Decision, messages: list[ChatMessage], tools: list[ChatTool] | None,
            openai_client: openai.OpenAI, anthropic_client: anthropic.Anthropic) -> ChatMessage:
    """Call the real provider for a routed decision and return a Chat Completions message.

    Args:
        decision: A `router_core.Decision` naming the chosen arm.
        messages: The incoming Chat Completions-style conversation.
        tools: The incoming Chat Completions-style tool definitions, if any.
        openai_client: Client used when the decision picked an OpenAI arm.
        anthropic_client: Client used when the decision picked an Anthropic arm.

    Returns:
        The provider's reply as a Chat Completions `ChatMessage`.
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


def dispatch_via_openrouter(decision: Decision, messages: list[ChatMessage],
                            tools: list[ChatTool] | None, client: openai.OpenAI) -> ChatMessage:
    """Call the routed model through OpenRouter instead of the provider directly.

    OpenRouter speaks the same Chat Completions dialect the incoming request already
    uses, so no format translation is needed -- only the model id gets a provider
    prefix and the request goes to a different base_url. Messages/tools are
    re-serialized back to plain dicts for the SDK call, keeping any fields our own
    models don't declare (via `extra="allow"` above).

    Args:
        decision: A `router_core.Decision` naming the chosen arm.
        messages: The incoming Chat Completions-style conversation, passed through as-is.
        tools: The incoming Chat Completions-style tool definitions, if any.
        client: An `openai.OpenAI` client pointed at OpenRouter's base_url.

    Returns:
        The provider's reply as a Chat Completions `ChatMessage`.
    """
    family = "anthropic" if decision.model.startswith("claude") else "openai"
    raw_messages = [m.model_dump(exclude_none=True) for m in messages]
    raw_tools = [t.model_dump(exclude_none=True) for t in tools] if tools else None
    r = client.chat.completions.create(
        model=f"{family}/{decision.model}", messages=raw_messages, tools=raw_tools)
    return ChatMessage.model_validate(r.choices[0].message.model_dump(exclude_none=True))


def message_preview(m: ChatMessage) -> str:
    """Render one message as `[role] json-snippet`, for the routing-embedding trajectory."""
    payload = m.content or ([tc.model_dump() for tc in m.tool_calls] if m.tool_calls else None)
    return f"[{m.role}] {json.dumps(payload)[:2000]}"


def make_app(router: Router, openai_client: openai.OpenAI, anthropic_client: anthropic.Anthropic,
            openrouter_client: openai.OpenAI | None = None) -> FastAPI:
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

    def embed_text(text: str) -> np.ndarray:
        """Embed `text` with the router's configured embedding model."""
        e = openai_client.embeddings.create(model=router.meta["embed_model"],
                                            input=text[:8000]).data[0].embedding
        return np.array(e, dtype=float)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """Route one Chat Completions request to a real model and return its reply.

        Embeds the whole conversation so far, routes it to an arm, dispatches to
        that arm's provider (or OpenRouter, if configured), and returns the reply
        in Chat Completions shape -- streamed as SSE chunks if `stream` was set.

        Args:
            request: The raw incoming HTTP request; its JSON body is Chat
                Completions shaped (`messages`, optional `tools`, optional `stream`).

        Returns:
            A `ChatCompletionResponse` JSON body for a normal request, or a
            `StreamingResponse` emitting one SSE chunk plus a `[DONE]` sentinel
            when `stream` is set.
        """
        body = await request.json()
        messages = [ChatMessage.model_validate(m) for m in body["messages"]]
        stream = bool(body.get("stream"))

        trajectory = "\n".join(message_preview(m) for m in messages)
        decision = router.route_embedding(embed_text(trajectory))
        tools = [parse_chat_tool(t) for t in (body.get("tools") or [])] or None
        message = (dispatch_via_openrouter(decision, messages, tools, openrouter_client)
                   if openrouter_client
                   else dispatch(decision, messages, tools, openai_client, anthropic_client))

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        print(f"routed -> {decision.model}@{decision.effort or 'default'}  "
              f"p_solve={decision.p_solve:.2f} off_dist={decision.off_distribution} "
              f"({len(messages)} messages in)")
        finish_reason = "tool_calls" if message.tool_calls else "stop"

        if not stream:
            resp = ChatCompletionResponse(
                id=completion_id, created=created, model=decision.model,
                choices=[Choice(index=0, message=message, finish_reason=finish_reason)])
            return JSONResponse(resp.model_dump(exclude_none=True))

        def sse():
            """Yield the reply as two SSE `chat.completion.chunk` lines plus `[DONE]`."""
            chunk = ChatCompletionChunk(
                id=completion_id, created=created, model=decision.model,
                choices=[ChunkChoice(index=0, delta=message, finish_reason=None)])
            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
            done = ChatCompletionChunk(
                id=completion_id, created=created, model=decision.model,
                choices=[ChunkChoice(index=0, delta=ChatMessage(), finish_reason=finish_reason)])
            yield f"data: {done.model_dump_json(exclude_none=True)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    @app.get("/v1/models")
    def models() -> ModelList:
        """List the one synthetic "auto" model this router exposes."""
        return ModelList(data=[ModelCard(id="auto", created=int(time.time()))])

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
