"""Regression tests for LLM JSON handling (user-reported bugs).

Covers:
1. LM Studio-style servers rejecting `response_format` (HTTP 400) — the
   client must retry without the field instead of failing the whole request.
2. "Extra data: line 1 column 308" — models emitting valid JSON followed by
   trailing prose/chatter; extract_json must recover the object.
"""
from __future__ import annotations

import json

import httpx
import pytest

from calliope.agent.llm import LLMClient, extract_json

# ---------- extract_json ----------

def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced_block():
    text = 'Here is your JSON:\n```json\n{"a": 1, "b": [2, 3]}\n```\nDone!'
    assert extract_json(text) == {"a": 1, "b": [2, 3]}


def test_extract_json_trailing_prose_extra_data():
    # The exact failure class the user hit: valid object + trailing text
    text = '{"title": "The Long Road", "beats": []} I hope this helps!'
    assert extract_json(text)["title"] == "The Long Road"


def test_extract_json_leading_prose():
    text = 'Sure! Here is the storyline you asked for: {"title": "X"}'
    assert extract_json(text)["title"] == "X"


def test_extract_json_two_objects_takes_first():
    text = '{"first": true} {"second": true}'
    assert extract_json(text) == {"first": True}


def test_extract_json_rejects_garbage():
    with pytest.raises(ValueError):
        extract_json("no json here at all")


def test_extract_json_empty():
    with pytest.raises(ValueError):
        extract_json("   ")


# ---------- LLMClient response_format fallback ----------

def _sse_body(message: dict, finish: str = "stop") -> bytes:
    """Encode a full assistant message as a minimal SSE stream."""
    chunks = []
    if message.get("reasoning_content"):
        chunks.append(
            {"choices": [{"delta": {"reasoning_content": message["reasoning_content"]}}]}
        )
    if message.get("content"):
        chunks.append({"choices": [{"delta": {"content": message["content"]}}]})
    chunks.append({"choices": [{"delta": {}, "finish_reason": finish}]})
    lines = [b"data: " + json.dumps(c).encode() for c in chunks]
    lines.append(b"data: [DONE]")
    return b"\n".join(lines) + b"\n"


class _FakeRouter:
    """httpx mock transport handler that 400s any request containing response_format.

    Serves SSE when the request asks for stream:true, plain JSON otherwise —
    chat()/chat_with_tools() stream internally, so both shapes occur.
    """

    def __init__(self, content: str, reject_response_format: bool) -> None:
        self.content = content
        self.reject = reject_response_format
        self.requests: list[dict] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        self.requests.append(body)
        if self.reject and "response_format" in body:
            return httpx.Response(
                400,
                json={"error": {"message": "response_format is not supported"}},
            )
        if body.get("stream"):
            return httpx.Response(
                200,
                content=_sse_body({"content": self.content}),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": self.content}}]},
        )


class _SequenceRouter(_FakeRouter):
    """Returns each reply in order instead of a fixed one."""

    def __init__(self, contents: list[str], reject_response_format: bool) -> None:
        super().__init__(contents[0], reject_response_format)
        self.contents = contents
        self.calls = 0

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        content = self.contents[min(self.calls, len(self.contents) - 1)]
        self.calls += 1
        self.content = content
        return await super().__call__(request)


async def test_chat_retries_without_response_format_on_400(monkeypatch):
    router = _FakeRouter(content='{"ok": true}', reject_response_format=True)
    transport = httpx.MockTransport(router)
    client = LLMClient()
    monkeypatch.setattr(client, "client", httpx.AsyncClient(transport=transport))

    text = await client.chat(
        [{"role": "user", "content": "hi"}],
        response_format={"type": "json_object"},
    )

    assert text == '{"ok": true}'
    assert len(router.requests) == 2
    assert "response_format" in router.requests[0]
    assert "response_format" not in router.requests[1]


async def test_generate_structured_no_response_format_by_default(monkeypatch):
    # Default path must NOT send response_format (LM Studio compatibility)
    router = _FakeRouter(content='{"a": 1}', reject_response_format=True)
    transport = httpx.MockTransport(router)
    client = LLMClient()
    monkeypatch.setattr(client, "client", httpx.AsyncClient(transport=transport))
    monkeypatch.setattr("calliope.agent.llm.LLMClient", lambda: client)

    result = await generate_structured_public([{"role": "user", "content": "hi"}])

    assert result == {"a": 1}
    assert len(router.requests) == 1
    assert "response_format" not in router.requests[0]


async def generate_structured_public(messages):
    from calliope.agent.llm import generate_structured

    return await generate_structured(messages)


async def test_generate_structured_recovers_via_json_mode_retry(monkeypatch):
    # First reply is garbage prose; the json_object retry must rescue it.
    router = _SequenceRouter(
        ["sorry, I cannot do that", '{"title": "Saved"}'],
        reject_response_format=False,
    )
    transport = httpx.MockTransport(router)
    client = LLMClient()
    monkeypatch.setattr(client, "client", httpx.AsyncClient(transport=transport))
    monkeypatch.setattr("calliope.agent.llm.LLMClient", lambda: client)

    result = await generate_structured_public([{"role": "user", "content": "hi"}])

    assert result == {"title": "Saved"}
    assert len(router.requests) == 2
    assert "response_format" not in router.requests[0]
    assert router.requests[1].get("response_format") == {"type": "json_object"}


async def test_generate_structured_raises_when_both_attempts_fail(monkeypatch):
    router = _SequenceRouter(
        ["all prose", "still prose"],
        reject_response_format=False,
    )
    transport = httpx.MockTransport(router)
    client = LLMClient()
    monkeypatch.setattr(client, "client", httpx.AsyncClient(transport=transport))
    monkeypatch.setattr("calliope.agent.llm.LLMClient", lambda: client)

    with pytest.raises(ValueError):
        await generate_structured_public([{"role": "user", "content": "hi"}])


# ---------- chat_stream: mid-stream error payloads ----------


def _sse_lines(*chunks: dict) -> list[bytes]:
    out = []
    for c in chunks:
        out.append(b"data: " + json.dumps(c).encode())
    out.append(b"data: [DONE]")
    return out


class _StreamRouter:
    """Serves a scripted SSE stream."""

    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        lines = _sse_lines(*self.chunks)
        body = b"\n".join(lines) + b"\n"
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream"},
        )


async def test_chat_stream_surfaces_error_payload(monkeypatch):
    """A mid-stream {error: ...} chunk must raise, not end as a blank reply."""
    router = _StreamRouter(
        [
            {"error": {"message": "model overloaded"}},
        ]
    )
    transport = httpx.MockTransport(router)
    client = LLMClient()
    monkeypatch.setattr(client, "client", httpx.AsyncClient(transport=transport))

    events = []
    with pytest.raises(RuntimeError, match="model overloaded"):
        async for ev in client.chat_stream([{"role": "user", "content": "hi"}]):
            events.append(ev)
    assert events == []  # nothing yielded before the failure surfaced


async def test_chat_stream_surfaces_string_error(monkeypatch):
    """Non-dict error payloads degrade to str(), still raising."""
    router = _StreamRouter([{"error": "bad gateway"}])
    transport = httpx.MockTransport(router)
    client = LLMClient()
    monkeypatch.setattr(client, "client", httpx.AsyncClient(transport=transport))

    with pytest.raises(RuntimeError, match="bad gateway"):
        async for _ in client.chat_stream([{"role": "user", "content": "hi"}]):
            pass


async def test_chat_stream_normal_tokens_unaffected(monkeypatch):
    """Happy path: deltas flow, done arrives, no error."""
    router = _StreamRouter(
        [
            {"choices": [{"delta": {"content": "Hi"}}]},
            {"choices": [{"delta": {"content": " there"}, "finish_reason": "stop"}]},
        ]
    )
    transport = httpx.MockTransport(router)
    client = LLMClient()
    monkeypatch.setattr(client, "client", httpx.AsyncClient(transport=transport))

    events = [ev async for ev in client.chat_stream([{"role": "user", "content": "hi"}])]
    types = [e["type"] for e in events]
    assert types == ["delta", "delta", "done"]
    assert events[0]["content"] == "Hi"


# ---------- top-level JSON arrays (models emitting items without the envelope) ----------
# Without special handling the balanced-object scan would confidently return just
# the array's FIRST element — e.g. one story beat instead of {"beats": [...]} —
# which surfaces downstream as "returned 0 beats" with no parse error anywhere.

def test_extract_json_rejects_bare_array():
    with pytest.raises(ValueError, match="top-level JSON array"):
        extract_json('[{"order_index": 1}, {"order_index": 2}]')


def test_extract_json_rejects_fenced_array():
    with pytest.raises(ValueError, match="top-level JSON array"):
        extract_json('```json\n[{"a": 1}, {"b": 2}]\n```')


def test_extract_json_rejects_array_in_prose():
    with pytest.raises(ValueError, match="top-level JSON array"):
        extract_json('Sure, here are the beats: [{"a": 1}, {"b": 2}] hope that helps')


def test_extract_json_array_inside_object_still_passes():
    # An "[" inside {"beats": [...]} must not shadow the enclosing object.
    text = 'Here you go: {"beats": [{"a": 1}], "characters": []} enjoy'
    assert list(extract_json(text)) == ["beats", "characters"]


async def test_generate_structured_recovers_from_bare_array_reply(monkeypatch):
    # First reply is the items without the envelope; the json_object retry rescues it.
    router = _SequenceRouter(
        ['[{"order_index": 1, "title": "Beat"}]', '{"beats": [{"order_index": 1}]}'],
        reject_response_format=False,
    )
    transport = httpx.MockTransport(router)
    client = LLMClient()
    monkeypatch.setattr(client, "client", httpx.AsyncClient(transport=transport))
    monkeypatch.setattr("calliope.agent.llm.LLMClient", lambda: client)

    result = await generate_structured_public([{"role": "user", "content": "hi"}])

    assert result == {"beats": [{"order_index": 1}]}
    assert router.requests[1].get("response_format") == {"type": "json_object"}


# ---------- reasoning-only replies (thinking models returning no `content`) ----------
# Reasoning models served via OpenAI-compatible endpoints can spend the whole
# completion in a reasoning channel (e.g. `reasoning_content`) and return a
# message with NO `content` key. That must be a retryable failure, not a
# KeyError bubbling up as an HTTP 500.

class _MessageRouter:
    """Returns full message dicts in sequence (to simulate reasoning-only replies).

    Streamed requests get the message re-encoded as SSE deltas — a
    reasoning-only message becomes a stream with reasoning chunks and no
    content chunks, which is exactly what oMLX-style servers emit."""

    def __init__(self, messages: list[dict]) -> None:
        self.messages = messages
        self.requests: list[dict] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        self.requests.append(body)
        message = self.messages[min(len(self.requests) - 1, len(self.messages) - 1)]
        if body.get("stream"):
            return httpx.Response(
                200,
                content=_sse_body(message),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": message, "finish_reason": "length"}]},
        )


async def test_chat_raises_value_error_on_missing_content(monkeypatch):
    router = _MessageRouter(
        [{"role": "assistant", "reasoning_content": "thinking forever..."}]
    )
    client = LLMClient()
    monkeypatch.setattr(
        client, "client", httpx.AsyncClient(transport=httpx.MockTransport(router))
    )

    with pytest.raises(ValueError, match="no content"):
        await client.chat([{"role": "user", "content": "hi"}])


async def test_generate_structured_recovers_from_reasoning_only_reply(monkeypatch):
    router = _MessageRouter(
        [
            {"role": "assistant", "reasoning_content": "hmm..."},
            {"role": "assistant", "content": '{"title": "Saved"}'},
        ]
    )
    client = LLMClient()
    monkeypatch.setattr(
        client, "client", httpx.AsyncClient(transport=httpx.MockTransport(router))
    )
    monkeypatch.setattr("calliope.agent.llm.LLMClient", lambda: client)

    result = await generate_structured_public([{"role": "user", "content": "hi"}])

    assert result == {"title": "Saved"}
    assert len(router.requests) == 2
    assert router.requests[1].get("response_format") == {"type": "json_object"}


# ---------- streaming-internal chat: fallbacks and accumulation ----------
# chat()/chat_with_tools() now consume chat_stream, so the 120 s client
# timeout bounds the gap BETWEEN chunks (liveness), not total generation time.
# These tests pin the two fallback ladders and the accumulation contract.


class _NoStreamRouter:
    """A server that rejects stream:true outright (400), accepts blocking."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.requests: list[dict] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        self.requests.append(body)
        if body.get("stream"):
            return httpx.Response(
                400, json={"error": {"message": "stream is not supported"}}
            )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": self.content}}]}
        )


async def test_chat_falls_back_to_blocking_when_stream_rejected(monkeypatch):
    router = _NoStreamRouter(content='{"ok": true}')
    client = LLMClient()
    monkeypatch.setattr(
        client, "client", httpx.AsyncClient(transport=httpx.MockTransport(router))
    )

    text = await client.chat([{"role": "user", "content": "hi"}])

    assert text == '{"ok": true}'
    # one rejected stream attempt, then one blocking call
    assert [b.get("stream", False) for b in router.requests] == [True, False]


async def test_chat_stream_drops_response_format_then_streams(monkeypatch):
    """The in-stream 400 ladder: response_format dropped, request retried."""
    router = _FakeRouter(content='{"ok": true}', reject_response_format=True)
    client = LLMClient()
    monkeypatch.setattr(
        client, "client", httpx.AsyncClient(transport=httpx.MockTransport(router))
    )

    text = await client.chat(
        [{"role": "user", "content": "hi"}],
        response_format={"type": "json_object"},
    )

    assert text == '{"ok": true}'
    assert len(router.requests) == 2
    assert "response_format" in router.requests[0]
    assert "response_format" not in router.requests[1]
    assert router.requests[1].get("stream") is True  # still streaming, not blocking


class _ToolStreamRouter:
    """Streams one content delta plus one chunked tool call."""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        self.requests.append(body)
        chunks = [
            {"choices": [{"delta": {"content": "Queuing now."}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "enqueue_asset", "arguments": '{"scene'},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '_id": 3}'}}
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        ]
        lines = [b"data: " + json.dumps(c).encode() for c in chunks]
        lines.append(b"data: [DONE]")
        return httpx.Response(
            200,
            content=b"\n".join(lines) + b"\n",
            headers={"content-type": "text/event-stream"},
        )


async def test_chat_with_tools_accumulates_streamed_tool_call(monkeypatch):
    router = _ToolStreamRouter()
    client = LLMClient()
    monkeypatch.setattr(
        client, "client", httpx.AsyncClient(transport=httpx.MockTransport(router))
    )

    msg = await client.chat_with_tools(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "enqueue_asset"}}],
    )

    assert msg["role"] == "assistant"
    assert msg["content"] == "Queuing now."
    assert len(msg["tool_calls"]) == 1
    call = msg["tool_calls"][0]
    assert call["function"]["name"] == "enqueue_asset"
    assert json.loads(call["function"]["arguments"]) == {"scene_id": 3}


async def test_chat_reasoning_only_stream_raises_value_error(monkeypatch):
    """A stream of reasoning deltas with zero content must be retryable."""
    router = _MessageRouter(
        [{"role": "assistant", "reasoning_content": "thinking forever..."}]
    )
    client = LLMClient()
    monkeypatch.setattr(
        client, "client", httpx.AsyncClient(transport=httpx.MockTransport(router))
    )

    with pytest.raises(ValueError, match="no content"):
        await client.chat([{"role": "user", "content": "hi"}])


# ---------- envelope guard + truncation (the extract_json first-object trap) ----------
# When the whole document is unparseable (truncated at a token cap, or malformed
# mid-document) the balanced-scan fallback "succeeds" on the first INNER object —
# a single beat instead of {"beats": [...]} — and the retry ladder never fires.
# expected_any guards the fallback only; finish_reason=length raises in chat().

TRUNCATED_BEATS = (
    'Here is your storyline:\n{"title": "X", "beats": ['
    '{"order_index": 1, "title": "Open", "description": "a"}, '
    '{"order_index": 2, "title": "Turn", "descri'  # cut mid-key by the token cap
)

MALFORMED_BEATS = (
    '{"title": "X", "beats": ['
    '{"order_index": 1, "title": "Open"} '
    '{"order_index": 2, "title": "Turn"}]}'  # missing comma mid-document
)


def test_extract_json_fallback_returns_first_object_without_guard():
    # Documents the trap this guard exists for: no expected_any -> first beat.
    assert extract_json(TRUNCATED_BEATS)["order_index"] == 1
    assert extract_json(MALFORMED_BEATS)["order_index"] == 1


def test_extract_json_fallback_rejects_wrong_envelope_truncated():
    with pytest.raises(ValueError, match="none of the expected keys"):
        extract_json(TRUNCATED_BEATS, expected_any=("beats",))


def test_extract_json_fallback_rejects_wrong_envelope_malformed():
    with pytest.raises(ValueError, match="none of the expected keys"):
        extract_json(MALFORMED_BEATS, expected_any=("beats",))


def test_extract_json_clean_parse_ignores_expected_any():
    # A complete, valid document is the model's actual answer, whatever its
    # keys — the guard applies to the fallback only.
    assert extract_json('{"title": "X"}', expected_any=("beats",)) == {"title": "X"}


def test_extract_json_fallback_passes_with_expected_key():
    text = 'Sure! {"beats": [{"order_index": 1}]} hope that helps'
    assert extract_json(text, expected_any=("beats",))["beats"]


async def test_chat_raises_on_length_finish(monkeypatch):
    class _Router:
        async def __call__(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=_sse_body({"content": '{"beats": ['}, finish="length"),
                headers={"content-type": "text/event-stream"},
            )

    client = LLMClient()
    monkeypatch.setattr(
        client, "client", httpx.AsyncClient(transport=httpx.MockTransport(_Router()))
    )
    with pytest.raises(ValueError, match="truncated"):
        await client.chat([{"role": "user", "content": "hi"}])


async def test_chat_stream_yields_finish_event_on_length(monkeypatch):
    class _Router:
        async def __call__(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=_sse_body({"content": "partial"}, finish="length"),
                headers={"content-type": "text/event-stream"},
            )

    client = LLMClient()
    monkeypatch.setattr(
        client, "client", httpx.AsyncClient(transport=httpx.MockTransport(_Router()))
    )
    events = [ev async for ev in client.chat_stream([{"role": "user", "content": "hi"}])]
    assert {"type": "finish", "reason": "length"} in events
    assert events[-1] == {"type": "done"}


async def test_generate_structured_recovers_from_truncated_reply(monkeypatch):
    """finish_reason=length on the first attempt -> retry rescues it."""

    class _Router:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        async def __call__(self, request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode())
            self.requests.append(body)
            if len(self.requests) == 1:
                return httpx.Response(
                    200,
                    content=_sse_body({"content": '{"beats": ['}, finish="length"),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(
                200,
                content=_sse_body({"content": '{"beats": [{"order_index": 1}]}'}),
                headers={"content-type": "text/event-stream"},
            )

    router = _Router()
    client = LLMClient()
    monkeypatch.setattr(
        client, "client", httpx.AsyncClient(transport=httpx.MockTransport(router))
    )
    monkeypatch.setattr("calliope.agent.llm.LLMClient", lambda: client)

    from calliope.agent.llm import generate_structured

    result = await generate_structured(
        [{"role": "user", "content": "hi"}], expected_any=("beats",)
    )

    assert result == {"beats": [{"order_index": 1}]}
    assert len(router.requests) == 2
    assert router.requests[1].get("response_format") == {"type": "json_object"}


# ---------- salvage: rescue well-formed items from an unparseable document ----------
# Observed live: a single corrupted token mid-array (`いorder_index":` where
# `{ "` belongs) malforms the WHOLE document while every other item is
# perfectly well-formed. With salvage markers, extract_json rebuilds the
# envelope from the surviving items instead of raising.

BEAT_SALVAGE = {"beats": ("order_index", "description")}

# Beat 2's opening `{ "` is corrupted to a stray token — the real specimen shape.
CORRUPTED_TOKEN_DOC = (
    '```json\n{"title": "X", "logline": "y", "beats": [\n'
    '    {"order_index": 1, "title": "Open", "description": "a"},\n'
    '    いorder_index": 2, "title": "Lost", "description": "b"},\n'
    '    {"order_index": 3, "title": "Turn", "description": "c"},\n'
    '    {"order_index": 4, "title": "End", "description": "d"}\n'
    '  ],\n'
    '  "characters": [{"name": "Mia", "role": "lead"}]\n'
    '}\n```'
)


def test_salvage_recovers_items_from_corrupted_token_doc():
    result = extract_json(
        CORRUPTED_TOKEN_DOC, expected_any=("beats",), salvage=BEAT_SALVAGE
    )
    beats = result["beats"]
    assert [b["order_index"] for b in beats] == [1, 3, 4]  # beat 2 stays lost
    assert all("description" in b for b in beats)


def test_salvage_excludes_fragments_without_markers():
    # The characters fragment parses but lacks order_index/description —
    # it must not be swept into the beats.
    result = extract_json(
        CORRUPTED_TOKEN_DOC, expected_any=("beats",), salvage=BEAT_SALVAGE
    )
    assert all("name" not in b for b in result["beats"])


def test_salvage_recovers_complete_items_from_truncated_doc():
    # Token-cap truncation mid-item: the complete beats before the cut survive.
    doc = (
        '{"title": "X", "beats": ['
        '{"order_index": 1, "title": "A", "description": "a"}, '
        '{"order_index": 2, "title": "B", "description": "b"}, '
        '{"order_index": 3, "title": "C", "description": "c"}, '
        '{"order_index": 4, "title": "D", "descri'
    )
    result = extract_json(doc, expected_any=("beats",), salvage=BEAT_SALVAGE)
    assert [b["order_index"] for b in result["beats"]] == [1, 2, 3]


def test_salvage_threshold_single_item_still_raises():
    # One surviving beat is indistinguishable from the first-object trap.
    doc = '{"beats": [{"order_index": 1, "description": "a"}, {"broken": '
    with pytest.raises(ValueError, match="none of the expected keys"):
        extract_json(doc, expected_any=("beats",), salvage=BEAT_SALVAGE)


def test_salvage_not_consulted_on_clean_parse():
    # A complete valid document wins outright; salvage never runs.
    doc = '{"title": "X"}'
    assert extract_json(doc, expected_any=("beats",), salvage=BEAT_SALVAGE) == {
        "title": "X"
    }


async def test_generate_structured_salvages_without_retry(monkeypatch):
    """A salvageable first reply returns immediately — no json_object retry."""

    class _Router:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        async def __call__(self, request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode())
            self.requests.append(body)
            return httpx.Response(
                200,
                content=_sse_body({"content": CORRUPTED_TOKEN_DOC}),
                headers={"content-type": "text/event-stream"},
            )

    router = _Router()
    client = LLMClient()
    monkeypatch.setattr(
        client, "client", httpx.AsyncClient(transport=httpx.MockTransport(router))
    )
    monkeypatch.setattr("calliope.agent.llm.LLMClient", lambda: client)

    from calliope.agent.llm import generate_structured

    result = await generate_structured(
        [{"role": "user", "content": "hi"}],
        expected_any=("beats",),
        salvage=BEAT_SALVAGE,
    )

    assert [b["order_index"] for b in result["beats"]] == [1, 3, 4]
    assert len(router.requests) == 1  # salvage succeeded on the first attempt


# ---------- salvage tiers: characters ride along with the beats anchor ----------

STORY_SALVAGE = {
    "beats": ("order_index", "description"),
    "characters": ("name", "role"),
}


def test_salvage_recovers_characters_alongside_beats():
    result = extract_json(
        CORRUPTED_TOKEN_DOC, expected_any=("beats",), salvage=STORY_SALVAGE
    )
    assert [b["order_index"] for b in result["beats"]] == [1, 3, 4]
    assert [c["name"] for c in result["characters"]] == ["Mia"]


def test_salvage_single_character_rides_on_beats_anchor():
    # One character is below the >=2 threshold on its own, but a tier with a
    # valid anchor (3 beats) carries it.
    result = extract_json(
        CORRUPTED_TOKEN_DOC, expected_any=("beats",), salvage=STORY_SALVAGE
    )
    assert len(result["characters"]) == 1


def test_salvage_no_anchor_still_raises():
    # One beat + one character: no tier reaches 2, so the wreck is not
    # distinguishable from the first-object trap — raise as before.
    doc = (
        '{"beats": [{"order_index": 1, "description": "a"}], '
        '"characters": [{"name": "Mia", "role": "lead"}], "broken": '
    )
    with pytest.raises(ValueError, match="none of the expected keys"):
        extract_json(doc, expected_any=("beats",), salvage=STORY_SALVAGE)


def test_salvage_fragment_claimed_by_first_matching_tier_only():
    # A fragment matching BOTH tiers' markers lands in beats (first key) and
    # is not duplicated into characters.
    doc = (
        'wreck: {"order_index": 1, "description": "a", "name": "Mia", "role": "lead"}, '
        '{"order_index": 2, "description": "b"}, broken {'
    )
    result = extract_json(doc, expected_any=("beats",), salvage=STORY_SALVAGE)
    assert len(result["beats"]) == 2
    assert "characters" not in result


# ---------- truncation + salvage: rescue the partial before burning a retry ----------
# (Observed live 2026-08-25: a 43-scene script regen truncated at the server
# cap with ~102K chars of content; the guard discarded text that salvage could
# have partially rescued — pre-guard, a truncated reply once yielded 42/42
# scenes. LLMTruncatedError now carries the partial.)

TRUNCATED_RICH = (
    '{"scenes": ['
    '{"order_index": 1, "heading": "EXT. CITY", "action": "a", "dialog": "L: hi"}, '
    '{"order_index": 2, "heading": "INT. SPIRE", "action": "b", "dialog": ""}, '
    '{"order_index": 3, "heading": "INT. HALL", "action": "c", "dialog": "K: no"}, '
    '{"order_index": 4, "heading": "INT. VAULT", "action": "d'  # cut by the cap
)

SCENE_SALVAGE = {"scenes": ("order_index", "action")}


async def test_chat_truncation_error_carries_partial(monkeypatch):
    from calliope.agent.llm import LLMTruncatedError

    class _Router:
        async def __call__(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=_sse_body({"content": TRUNCATED_RICH}, finish="length"),
                headers={"content-type": "text/event-stream"},
            )

    client = LLMClient()
    monkeypatch.setattr(
        client, "client", httpx.AsyncClient(transport=httpx.MockTransport(_Router()))
    )
    with pytest.raises(LLMTruncatedError) as ei:
        await client.chat([{"role": "user", "content": "hi"}])
    assert isinstance(ei.value, ValueError)  # retry-ladder compatibility
    assert ei.value.partial == TRUNCATED_RICH


async def test_generate_structured_rescues_truncated_partial(monkeypatch):
    """A truncated reply with >=2 complete items is salvaged with NO retry."""

    class _Router:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        async def __call__(self, request: httpx.Request) -> httpx.Response:
            self.requests.append(json.loads(request.content.decode()))
            return httpx.Response(
                200,
                content=_sse_body({"content": TRUNCATED_RICH}, finish="length"),
                headers={"content-type": "text/event-stream"},
            )

    router = _Router()
    client = LLMClient()
    monkeypatch.setattr(
        client, "client", httpx.AsyncClient(transport=httpx.MockTransport(router))
    )
    monkeypatch.setattr("calliope.agent.llm.LLMClient", lambda: client)

    from calliope.agent.llm import generate_structured

    result = await generate_structured(
        [{"role": "user", "content": "hi"}],
        expected_any=("scenes",),
        salvage=SCENE_SALVAGE,
    )

    assert [s["order_index"] for s in result["scenes"]] == [1, 2, 3]  # 4 lost at cut
    assert len(router.requests) == 1  # rescued without the json_object retry


async def test_generate_structured_truncated_below_threshold_still_retries(monkeypatch):
    """One complete item in the partial = the trap threshold — retry fires."""
    thin = '{"scenes": [{"order_index": 1, "action": "a"}, {"order_index": 2, "act'

    class _Router:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        async def __call__(self, request: httpx.Request) -> httpx.Response:
            self.requests.append(json.loads(request.content.decode()))
            if len(self.requests) == 1:
                return httpx.Response(
                    200,
                    content=_sse_body({"content": thin}, finish="length"),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(
                200,
                content=_sse_body({"content": '{"scenes": [{"order_index": 1, "action": "a"}]}'}),
                headers={"content-type": "text/event-stream"},
            )

    router = _Router()
    client = LLMClient()
    monkeypatch.setattr(
        client, "client", httpx.AsyncClient(transport=httpx.MockTransport(router))
    )
    monkeypatch.setattr("calliope.agent.llm.LLMClient", lambda: client)

    from calliope.agent.llm import generate_structured

    result = await generate_structured(
        [{"role": "user", "content": "hi"}],
        expected_any=("scenes",),
        salvage=SCENE_SALVAGE,
    )
    assert len(router.requests) == 2  # retry was needed
    assert result["scenes"]


async def test_generate_structured_rescues_truncated_retry(monkeypatch):
    """First reply garbage, retry truncated-but-rich -> rescued on the retry."""

    class _Router:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        async def __call__(self, request: httpx.Request) -> httpx.Response:
            self.requests.append(json.loads(request.content.decode()))
            if len(self.requests) == 1:
                return httpx.Response(
                    200,
                    content=_sse_body({"content": "sorry, prose only"}),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(
                200,
                content=_sse_body({"content": TRUNCATED_RICH}, finish="length"),
                headers={"content-type": "text/event-stream"},
            )

    router = _Router()
    client = LLMClient()
    monkeypatch.setattr(
        client, "client", httpx.AsyncClient(transport=httpx.MockTransport(router))
    )
    monkeypatch.setattr("calliope.agent.llm.LLMClient", lambda: client)

    from calliope.agent.llm import generate_structured

    result = await generate_structured(
        [{"role": "user", "content": "hi"}],
        expected_any=("scenes",),
        salvage=SCENE_SALVAGE,
    )
    assert len(router.requests) == 2
    assert [s["order_index"] for s in result["scenes"]] == [1, 2, 3]
