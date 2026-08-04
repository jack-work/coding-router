"""The wire: this server's public Chat Completions contract, and its translation
to each provider's native request shape.

Inbound and outbound bodies are modelled here so every consumer agrees on one
definition; `serve.py` owns the server, routing, and dispatch that use them.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from router.router_core import CachePolicy


# ---------------------------------------------------------------- Chat Completions shapes
# These mirror the OpenAI Chat Completions wire format, which is this server's public
# contract (what opencode et al. actually send/receive). `extra="allow"` lets an
# unrecognized field survive a parse -> re-serialize round trip (e.g. the OpenRouter
# passthrough path in `serve.py`), instead of silently dropping it.
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


class CacheControl(BaseModel):
    """A prompt-cache breakpoint marker on one content block."""

    model_config = ConfigDict(extra="allow")

    type: str = "ephemeral"
    ttl: str | None = None    # "5m" (default when absent) or "1h"


class ContentPart(BaseModel):
    """One structured content block of a Chat Completions message.

    Clients that mark cache breakpoints MUST send block lists rather than a bare
    string: a string has nowhere to hang a `cache_control` marker.
    """

    model_config = ConfigDict(extra="allow")

    type: str = "text"
    text: str | None = None
    cache_control: CacheControl | None = None


class ChatMessage(BaseModel):
    """One Chat Completions conversation message (system/user/assistant/tool)."""

    model_config = ConfigDict(extra="allow")

    role: str | None = None
    # Chat Completions has always permitted structured content, and every
    # cache-aware client sends it (a marker needs a block to sit on). Typing this
    # `str` alone made any such request fail validation before it reached routing.
    content: str | list[ContentPart] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    def text(self) -> str:
        """Flatten this message's content to plain text for embedding and logging.

        Returns:
            The concatenated text of every text block, or the bare string content;
            empty when the message carries no text (a pure tool call, say).
        """
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        return "\n".join(p.text for p in self.content if p.type == "text" and p.text)

    def blocks(self) -> list[ContentPart]:
        """This message's content as structured parts; a bare string becomes one text part."""
        if self.content is None:
            return []
        if isinstance(self.content, str):
            return [ContentPart(type="text", text=self.content)] if self.content else []
        return list(self.content)

    def markers(self) -> int:
        """Count the inbound `cache_control` breakpoints this message already carries."""
        if not isinstance(self.content, list):
            return 0
        return sum(1 for p in self.content if p.cache_control is not None)


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


class PromptTokensDetails(BaseModel):
    """Cache breakdown of the prompt tokens, as OpenAI and OpenRouter report it."""

    model_config = ConfigDict(extra="allow")

    cached_tokens: int = 0        # prompt tokens served from cache (a read)
    cache_write_tokens: int = 0   # prompt tokens written to cache


class TokenUsage(BaseModel):
    """Token counts for one dispatched request, Chat Completions `usage` shape.

    `prompt_tokens` stays inclusive of cached reads and writes, so it remains
    comparable across providers; the split lives in `prompt_tokens_details`.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: PromptTokensDetails | None = None

    @property
    def cache_read(self) -> int:
        """Prompt tokens read from cache, or 0 when the provider reported none."""
        return self.prompt_tokens_details.cached_tokens if self.prompt_tokens_details else 0

    @property
    def cache_write(self) -> int:
        """Prompt tokens written to cache, or 0 when the provider reported none."""
        return self.prompt_tokens_details.cache_write_tokens if self.prompt_tokens_details else 0

    @property
    def uncached_prompt(self) -> int:
        """Prompt tokens billed at the full input rate (never negative)."""
        return max(0, self.prompt_tokens - self.cache_read - self.cache_write)


class ChatCompletionResponse(BaseModel):
    """A non-streaming Chat Completions response body."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: TokenUsage | None = None


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
        text = m.text()
        if role == "system":
            system = (system + "\n\n" + text) if system else text
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": m.tool_call_id,
                          "output": text})
        elif role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                items.append({"type": "function_call", "call_id": tc.id,
                              "name": tc.function.name,
                              "arguments": tc.function.arguments})
            if text:
                items.append({"role": "assistant", "content": text})
        else:
            items.append({"role": role, "content": text})
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

    A client's `cache_control` markers are carried across for every role, so a
    breakpoint placed deliberately is not flattened away here. System markers are
    the exception: the system prompt is one string upstream, and `serve.py` stamps
    the last system block itself.

    Args:
        messages: Chat Completions-style message list.

    Returns:
        A (system_prompt, messages) pair for `client.messages.create()`.
    """
    system, out = None, []
    for m in messages:
        role = m.role
        text = m.text()
        mark = next((b.cache_control for b in m.blocks() if b.cache_control), None)
        if role == "system":
            system = (system + "\n\n" + text) if system else text
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.tool_call_id,
                     "content": text}
            if mark:
                block["cache_control"] = mark.model_dump(exclude_none=True)
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif role == "assistant" and m.tool_calls:
            content = []
            if text:
                block = {"type": "text", "text": text}
                if mark:
                    block["cache_control"] = mark.model_dump(exclude_none=True)
                content.append(block)
            for tc in m.tool_calls:
                content.append({"type": "tool_use", "id": tc.id, "name": tc.function.name,
                                "input": json.loads(tc.function.arguments or "{}")})
            out.append({"role": "assistant", "content": content})
        elif m.markers():
            # The client marked breakpoints on these blocks; forward them as blocks
            # so the markers survive, instead of flattening them away.
            out.append({"role": role, "content":
                        [b.model_dump(exclude_none=True) for b in m.blocks()]})
        else:
            out.append({"role": role, "content": text})
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


# ---------------------------------------------------------------- cache breakpoints

def count_cache_marks(*regions: list) -> int:
    """Count `cache_control` markers already present across native request regions.

    Args:
        *regions: Native block lists (system blocks, tool definitions, messages).
            Message content that is a plain string carries no marker by definition.

    Returns:
        The total number of marked blocks.
    """
    n = 0
    for region in regions:
        for item in region:
            if not isinstance(item, dict):
                continue
            if item.get("cache_control"):
                n += 1
            body = item.get("content")
            if isinstance(body, list):
                n += sum(1 for b in body if isinstance(b, dict) and b.get("cache_control"))
    return n


def _mark(block: dict, ttl: str | None) -> None:
    """Stamp one native block with a cache breakpoint.

    `type` is an enum of exactly ["ephemeral"]; a longer lifetime rides in the
    separate `ttl` field, so 1h is {"type": "ephemeral", "ttl": "1h"}.
    """
    cc: dict[str, str] = {"type": "ephemeral"}
    if ttl:
        cc["ttl"] = ttl
    block["cache_control"] = cc


def _tail_block(messages: list[dict]) -> dict | None:
    """The block that should carry the rolling breakpoint, promoting a string if needed.

    The mark goes on the SECOND-TO-LAST message: the final message is the turn
    being asked about and changes every request, so a cache written there would
    never be read back. Everything through the second-to-last is stable and is
    exactly what the next turn wants to read.
    """
    if len(messages) < 2:
        return None
    m = messages[-2]
    body = m.get("content")
    if isinstance(body, str):
        m["content"] = body = [{"type": "text", "text": body}]
    if not isinstance(body, list) or not body:
        return None
    return body[-1] if isinstance(body[-1], dict) else None


def mark_anthropic_cache(system: list[dict], tools: list[dict], messages: list[dict],
                         policy: CachePolicy) -> int:
    """Place prompt-cache breakpoints on a native Anthropic request, in place.

    Marks the rolling tail first, then the last system block, then the last tool
    definition -- the three-point pattern -- skipping any region already marked by
    the client and never exceeding the provider's breakpoint cap. The tail is
    marked first because it is the most valuable (a breakpoint caches everything
    before it) and because it must remain the LAST marker in wire order: Gemini
    honours only the final breakpoint, so ordering has to survive translation.

    Args:
        system: Native system blocks (mutated in place).
        tools: Native tool definitions (mutated in place).
        messages: Native messages (mutated in place; a string tail is promoted to
            a one-element block list so a marker has somewhere to sit).
        policy: The resolved arm's cache policy.

    Returns:
        The number of breakpoints this call added.
    """
    budget = policy.max_breakpoints - count_cache_marks(system, tools, messages)
    if budget <= 0:
        return 0
    added = 0
    for block in (_tail_block(messages),
                  system[-1] if system else None,
                  tools[-1] if tools else None):
        if budget <= 0:
            break
        if block is None or block.get("cache_control"):
            continue
        _mark(block, policy.ttl)
        added += 1
        budget -= 1
    return added
