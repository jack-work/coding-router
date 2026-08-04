"""The wire: this server's public Chat Completions contract, and its translation
to each provider's native request shape.

Inbound and outbound bodies are modelled here so every consumer agrees on one
definition; `serve.py` owns the server, routing, and dispatch that use them.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
from collections.abc import Iterator
from typing import Any
from urllib.parse import unquote_to_bytes

from pydantic import BaseModel, ConfigDict, Field

from router.router_core import CachePolicy

logger = logging.getLogger(__name__)


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
    # Required on STREAMED tool calls: Chat Completions identifies each call by
    # position, and clients key their accumulator on it. Absent on the buffered
    # response, where the array order carries the same information.
    index: int | None = None


class CacheControl(BaseModel):
    """A prompt-cache breakpoint marker on one content block."""

    model_config = ConfigDict(extra="allow")

    type: str = "ephemeral"
    ttl: str | None = None    # "5m" (default when absent) or "1h"


class ImageUrl(BaseModel):
    """The `image_url` payload of a Chat Completions image part.

    Chat Completions nests the location in an object; the Responses API takes the
    same string bare, with `detail` beside it rather than inside it.
    """

    model_config = ConfigDict(extra="allow")

    url: str = ""
    detail: str | None = None


class ContentPart(BaseModel):
    """One structured content block of a Chat Completions message.

    Clients that mark cache breakpoints MUST send block lists rather than a bare
    string: a string has nowhere to hang a `cache_control` marker.
    """

    model_config = ConfigDict(extra="allow")

    type: str = "text"
    text: str | None = None
    image_url: ImageUrl | None = None
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
    # Present only on the final chunk, and only when the client asked for it with
    # stream_options.include_usage. Every other chunk omits it.
    usage: TokenUsage | None = None


# OpenRouter caps the session key at 256 characters and reads it, in order, from
# the body's `session_id`, the `x-session-id` header, then `prompt_cache_key`.
SESSION_KEY_MAX = 256


def session_key(body: dict, headers) -> str | None:
    """Resolve the caller's session key, or derive a stable one from the conversation.

    Body beats header beats `prompt_cache_key`; when the client offers none, the
    opening of the conversation is hashed instead. Chat Completions is stateless,
    but a conversation is append-only, so its first system and first user message
    are the same bytes on every later turn -- the same trick the summary cache uses.

    The derived key is BEST-EFFORT CONTINUITY, not isolation: two conversations
    that open with identical messages derive the same key and share an arm
    preference (re-checked against the artifact every turn) and a cache-affinity
    key. They also genuinely share a cacheable prefix, so the affinity is wanted.
    Nothing stronger is possible from the request alone -- anything that told them
    apart would also change between turns of one conversation, which is the case
    stickiness exists for. A client that needs hard isolation sends `session_id`.

    Args:
        body: The parsed request body.
        headers: The request headers (any mapping with `.get`).

    Returns:
        A key of at most SESSION_KEY_MAX characters, or None for an empty request.

    NOTE: this value never reaches telemetry. It is derived from user content and
    is treated as user content.
    """
    for candidate in (body.get("session_id"),
                      headers.get("x-session-id") if headers is not None else None,
                      body.get("prompt_cache_key")):
        if isinstance(candidate, str) and candidate:
            return candidate[:SESSION_KEY_MAX]
    messages = body.get("messages") or []
    opening = [m for m in messages if isinstance(m, dict) and m.get("role") == "system"][:1]
    opening += [m for m in messages if isinstance(m, dict) and m.get("role") == "user"][:1]
    if not opening:
        return None
    return hashlib.sha256(json.dumps(opening, sort_keys=True).encode()).hexdigest()[:32]


class StreamOptions(BaseModel):
    """The `stream_options` object of a streaming Chat Completions request."""

    model_config = ConfigDict(extra="allow")

    include_usage: bool = False


def wants_usage(body: dict) -> bool:
    """Whether a streaming client asked for a final usage chunk.

    Args:
        body: The parsed Chat Completions request.

    Returns:
        True when `stream_options.include_usage` is set.
    """
    raw = body.get("stream_options")
    if not isinstance(raw, dict):
        return False
    return StreamOptions.model_validate(raw).include_usage


def usage_from_chat(raw) -> TokenUsage:
    """Read a Chat Completions `usage` object into ours, cache split included."""
    details = getattr(raw, "prompt_tokens_details", None)
    return TokenUsage(
        prompt_tokens=getattr(raw, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(raw, "completion_tokens", 0) or 0,
        total_tokens=getattr(raw, "total_tokens", 0) or 0,
        prompt_tokens_details=PromptTokensDetails(
            cached_tokens=getattr(details, "cached_tokens", 0) or 0,
            cache_write_tokens=getattr(details, "cache_write_tokens", 0) or 0))


def usage_from_responses(raw) -> TokenUsage:
    """Read a Responses API `usage` object into ours, cache split included."""
    details = getattr(raw, "input_tokens_details", None)
    return TokenUsage(
        prompt_tokens=getattr(raw, "input_tokens", 0) or 0,
        completion_tokens=getattr(raw, "output_tokens", 0) or 0,
        total_tokens=getattr(raw, "total_tokens", 0) or 0,
        prompt_tokens_details=PromptTokensDetails(
            cached_tokens=getattr(details, "cached_tokens", 0) or 0,
            cache_write_tokens=getattr(details, "cache_write_tokens", 0) or 0))


class StreamDelta(BaseModel):
    """One incremental piece of a streamed reply.

    Providers differ in what arrives when -- text lands token by token, tool calls
    usually land whole, usage lands last -- so a delta carries whichever of the
    three it has and the renderer decides how to shape it.
    """

    text: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: TokenUsage | None = None


def collect(deltas: Iterator[StreamDelta]) -> tuple[ChatMessage, TokenUsage]:
    """Drain a delta stream into the single message a non-streaming client expects.

    Dispatch is streaming-native so there is one code path to the provider; this
    is how the JSON response is assembled from it.

    Args:
        deltas: The provider's incremental output.

    Returns:
        The assembled assistant message and the reported token usage.
    """
    text: list[str] = []
    tool_calls: list[ToolCall] = []
    usage = TokenUsage()
    for delta in deltas:
        if delta.text:
            text.append(delta.text)
        if delta.tool_calls:
            tool_calls.extend(delta.tool_calls)
        if delta.usage is not None:
            usage = delta.usage
    return ChatMessage(role="assistant", content="".join(text) or None,
                       tool_calls=tool_calls or None), usage


def sse_passthrough(chunk_id: str, created: int, model: str,
                    deltas: Iterator[StreamDelta], include_usage: bool) -> Iterator[str]:
    """Render a provider's delta stream as SSE, forwarding each piece as it arrives.

    The first chunk carries `role`, as Chat Completions requires; later chunks
    carry only what changed. The finish chunk names `tool_calls` when any tool
    call was seen and `stop` otherwise, and the usage chunk (if the client asked)
    follows it with an empty `choices` list.

    Args:
        chunk_id: The completion id shared by every chunk.
        created: Unix timestamp shared by every chunk.
        model: The model producing the reply.
        deltas: The provider's incremental output.
        include_usage: Whether to emit a final usage chunk.

    Yields:
        Complete `data: ...` SSE lines, ending with the `[DONE]` sentinel.
    """
    def chunk(delta: ChatMessage, finish: str | None = None,
              usage: TokenUsage | None = None) -> str:
        """Serialize one chunk, with an empty choices list when it carries usage."""
        choices = [] if usage is not None else [ChunkChoice(index=0, delta=delta,
                                                            finish_reason=finish)]
        payload = ChatCompletionChunk(id=chunk_id, created=created, model=model,
                                      choices=choices, usage=usage)
        return f"data: {payload.model_dump_json(exclude_none=True)}\n\n"

    first, saw_tools, usage = True, False, TokenUsage()
    positions: dict[str, int] = {}
    for delta in deltas:
        if delta.usage is not None:
            usage = delta.usage
        if not delta.text and not delta.tool_calls:
            continue
        # The index is per-STREAM, not per-delta. A provider that reports one
        # completed call per delta -- the Responses API does -- would otherwise
        # have every call stamped 0, and a client keying by index collapses them
        # into a single call. Positions are held by id so a call keeps its own
        # across however many deltas it arrives in.
        calls = None
        if delta.tool_calls:
            calls = []
            for c in delta.tool_calls:
                key = c.id or f"#{len(positions)}"
                calls.append(c.model_copy(
                    update={"index": positions.setdefault(key, len(positions))}))
        out = ChatMessage(content=delta.text, tool_calls=calls)
        if first:
            out.role = "assistant"
            first = False
        saw_tools = saw_tools or bool(delta.tool_calls)
        yield chunk(out)
    if first:
        # An empty reply still needs a role-bearing chunk before the finish chunk,
        # or a client accumulating deltas never learns whose turn it was.
        yield chunk(ChatMessage(role="assistant", content=""))
    yield chunk(ChatMessage(), finish="tool_calls" if saw_tools else "stop")
    if include_usage:
        yield chunk(ChatMessage(), usage=usage)
    yield "data: [DONE]\n\n"


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

# Media types Anthropic accepts on an image block (its vision guide, read
# 2026-08-04): image/jpeg, image/png, image/gif, image/webp. A data: URI naming
# anything else is refused by the provider, so it is dropped here with a warning
# rather than sent and rejected.
ANTHROPIC_IMAGE_MEDIA = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})


def decode_data_uri(url: str) -> tuple[str, str] | None:
    """Split a `data:` URI into its media type and a base64 payload.

    RFC 2397 makes `;base64` optional: without it the payload is
    percent-encoded, not base64. Re-encoding that form is three lines and keeps
    a spec-valid image working; forwarding it under a base64 label instead would
    hand the provider bytes it cannot decode.

    Args:
        url: A `data:` URI.

    Returns:
        A (media type, base64 payload) pair, or None when there is no payload.
    """
    header, _, payload = url.partition(",")
    media = header[len("data:"):].split(";")[0]
    if not payload:
        return None
    if ";base64" in header:
        return media, payload
    return media, base64.b64encode(unquote_to_bytes(payload)).decode()


def _image_source(url: str) -> dict[str, str] | None:
    """Build Anthropic's image `source` from a Chat Completions image URL.

    Args:
        url: Either a `data:` URI carrying image bytes, or a remote URL.

    Returns:
        The `source` object, or None when the URL is empty, carries no payload,
        or names a media type Anthropic does not accept.
    """
    if not url:
        return None
    if not url.startswith("data:"):
        return {"type": "url", "url": url}
    decoded = decode_data_uri(url)
    if decoded is None:
        return None
    media, data = decoded
    if media not in ANTHROPIC_IMAGE_MEDIA:
        logger.warning(f"image: dropping data URI with unusable media type {media!r}")
        return None
    return {"type": "base64", "media_type": media, "data": data}


def to_anthropic_block(part: ContentPart) -> dict[str, Any] | None:
    """Translate one Chat Completions content block into its Anthropic shape.

    Returns None for anything this translation does not know, so an unrecognised
    block is dropped with a warning rather than sent in a shape the provider
    would reject.
    """
    cc: dict[str, Any] = ({"cache_control": part.cache_control.model_dump(exclude_none=True)}
                          if part.cache_control else {})
    if part.type == "text":
        return {"type": "text", "text": part.text or "", **cc}
    if part.type == "image_url":
        source = _image_source(part.image_url.url if part.image_url else "")
        return {"type": "image", "source": source, **cc} if source else None
    logger.warning(f"image: dropping unsupported content block {part.type!r} for anthropic")
    return None


def to_responses_block(part: ContentPart, marker: str | None = None) -> dict[str, Any] | None:
    """Translate one Chat Completions content block into its Responses API shape.

    The Responses API takes the image location as a bare string with `detail`
    beside it, where Chat Completions nests both inside an `image_url` object.
    """
    if part.type == "text":
        block: dict[str, Any] = {"type": "input_text", "text": part.text or ""}
        if marker and part.cache_control:
            block[marker] = {"mode": "explicit"}
        return block
    if part.type == "image_url":
        if not (part.image_url and part.image_url.url):
            return None
        url = part.image_url.url
        if url.startswith("data:"):
            # Normalise to the base64 form the vision guide documents; a
            # percent-encoded payload is spec-valid inbound but not what the
            # provider expects to receive.
            decoded = decode_data_uri(url)
            if decoded is None:
                return None
            url = f"data:{decoded[0]};base64,{decoded[1]}"
        block = {"type": "input_image", "image_url": url}
        if part.image_url.detail:
            block["detail"] = part.image_url.detail
        return block
    logger.warning(f"image: dropping unsupported content block {part.type!r} for openai")
    return None


def needs_blocks(m: ChatMessage, *, marks_apply: bool = True) -> bool:
    """Whether this message must be forwarded as blocks rather than flattened.

    A non-text block is content, and flattening drops it outright. A cache
    breakpoint only needs the block form where the provider can act on it;
    forcing it elsewhere would reshape the request to carry a hint that provider
    discards.
    """
    return any(b.type != "text" or (marks_apply and b.cache_control is not None)
               for b in m.blocks())


def messages_to_responses_input(messages: list[ChatMessage],
                                policy: CachePolicy | None = None,
                                ) -> tuple[str | None, list[dict]]:
    """Translate Chat Completions messages into a Responses API instructions/input pair.

    A client's `cache_control` markers are carried across rather than dropped when
    the chosen model accepts explicit breakpoints: OpenAI spells the same idea
    `prompt_cache_breakpoint`, and the two block markers are interchangeable. TTL
    is not translated -- OpenAI's explicit prefixes carry their own minimum
    lifetime -- so only the position of the breakpoint survives.

    Args:
        messages: Chat Completions-style message list.
        policy: The chosen arm's cache policy; when its `breakpoint_field` names
            OpenAI's spelling, marked messages are sent as blocks instead of text.

    Returns:
        A (system_instructions, input_items) pair for `client.responses.create()`.
    """
    marker = policy.breakpoint_field if policy else None
    keep_marks = marker == "prompt_cache_breakpoint"
    marked_positions: list[int] = []
    system, items = None, []
    for m in messages:
        role = m.role
        text = m.text()
        if (role not in ("system", "tool") and not m.tool_calls
                and needs_blocks(m, marks_apply=keep_marks)):
            if keep_marks and m.markers():
                marked_positions.append(len(items))
            native = [b for b in (to_responses_block(p, marker if keep_marks else None)
                                  for p in m.blocks()) if b]
            items.append({"role": role, "content": native or text})
            continue
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
    if keep_marks and not marked_positions:
        _mark_last_stable_user(items, marker)
    return system, items


def _mark_last_stable_user(items: list[dict], marker: str) -> None:
    """Put one breakpoint on the newest user turn that is not the current one.

    The final item is the question being asked and differs every turn, so a
    prefix cut there is written and never read. The user turn before it was sent
    byte-for-byte last time, which is the boundary worth cutting on. Mirrors the
    rolling tail the Anthropic path marks, and stays on `input_text`, the one
    block type this translation already emits.

    UNVERIFIED AGAINST A LIVE PROVIDER. `--via=openrouter` sends every arm down
    the Chat Completions path, so nothing in our measurements exercises this.

    Args:
        items: Responses API input items, mutated in place.
        marker: The breakpoint field name for this provider.
    """
    for i in range(len(items) - 2, -1, -1):
        item = items[i]
        if item.get("role") != "user" or not isinstance(item.get("content"), str):
            continue
        item["content"] = [{"type": "input_text", "text": item["content"],
                            marker: {"mode": "explicit"}}]
        return


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
        elif needs_blocks(m):
            # Images and cache breakpoints do not survive flattening, so this
            # message is translated block by block instead. Dumping the inbound
            # blocks raw would send Chat Completions shapes to a provider that
            # does not speak them.
            native = [b for b in (to_anthropic_block(p) for p in m.blocks()) if b]
            out.append({"role": role, "content": native or text})
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
