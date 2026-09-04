from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from calliope.config import settings

logger = logging.getLogger("calliope.llm")

class LLMTruncatedError(ValueError):
    """A completion cut off at the server's token cap (finish_reason=length).

    Carries the PARTIAL accumulated content: a truncated structured reply can
    still hold dozens of complete items (observed live 2026-08-24: 42/42
    scenes salvaged from a truncated reply, pre-guard), so generate_structured
    attempts salvage on `partial` before burning a full retry.
    Subclasses ValueError so every existing retry-ladder handler still fires.
    """

    def __init__(self, message: str, partial: str = "") -> None:
        super().__init__(message)
        self.partial = partial


# Status codes that mean "this server does not do SSE streaming at all" —
# chat()/chat_with_tools() then fall back to one plain blocking POST. Anything
# else (401, 429, 5xx) is a real error and re-raises.
_STREAM_UNSUPPORTED_STATUS = frozenset({400, 404, 405, 501})


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.api_key = api_key if api_key is not None else settings.llm_api_key
        # With every completion streamed (chat/chat_with_tools consume
        # chat_stream), the timeout bounds the gap BETWEEN chunks, not total
        # generation time: a thinking model streaming reasoning_content keeps
        # the connection fed for as long as it genuinely works, while a dead
        # server still fails fast. Shorter timeouts (e.g. the 30 s preview
        # path) trade headroom for a snappier deterministic fallback.
        self.client = httpx.AsyncClient(timeout=timeout)

    @classmethod
    def for_role(cls, role: str, *, timeout: float = 120.0) -> LLMClient:
        """Client for an agent role's assigned profile (active fallback)."""
        profile = settings.resolve_llm_for_role(role)
        return cls(
            base_url=profile.get("base_url"),
            model=profile.get("model"),
            api_key=profile.get("api_key") if isinstance(profile.get("api_key"), str) else None,
            timeout=timeout,
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        response_format: dict[str, str] | None = None,
    ) -> str:
        try:
            parts: list[str] = []
            reasoning_chars = 0
            truncated = False
            async for ev in self.chat_stream(
                messages, temperature=temperature, response_format=response_format
            ):
                if ev["type"] == "delta":
                    parts.append(ev["content"])
                elif ev["type"] == "reasoning":
                    reasoning_chars += len(ev["content"])
                elif ev["type"] == "finish" and ev["reason"] == "length":
                    truncated = True
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in _STREAM_UNSUPPORTED_STATUS:
                raise
            # Server rejected streaming itself (the in-stream field fallbacks
            # are exhausted) — one plain blocking call preserves old behavior.
            logger.warning(
                "Streaming unavailable (HTTP %s); falling back to blocking call",
                exc.response.status_code,
            )
            return await self._chat_blocking(messages, temperature, response_format)
        content = "".join(parts).strip()
        if truncated:
            # A completion cut off at the server's token cap is never a valid
            # structured reply — parsing it "successfully" returns the first
            # inner object (a single beat instead of the envelope). Raise so
            # the retry ladder fires — but carry the partial content: complete
            # items before the cut are salvageable.
            raise LLMTruncatedError(
                "LLM completion truncated (finish_reason=length, "
                f"content_chars={len(content)}, reasoning_chars={reasoning_chars})",
                partial=content,
            )
        if not content:
            # Thinking models can burn the whole completion in reasoning and
            # stream no content tokens at all. Raise ValueError so
            # generate_structured's retry ladder fires instead of returning a
            # silently blank reply.
            raise ValueError(
                f"LLM returned no content (reasoning_chars={reasoning_chars})"
            )
        return content

    async def _chat_blocking(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        response_format: dict[str, str] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        url = f"{self.base_url}/chat/completions"
        logger.info("LLM request to %s with model %s", url, self.model)
        resp = await self.client.post(url, headers=self._headers(), json=payload)
        if resp.status_code == 400 and "response_format" in payload:
            # Some OpenAI-compatible servers (e.g. LM Studio) reject the
            # response_format field outright — retry without it.
            logger.warning(
                "Server rejected response_format (HTTP 400); retrying without it"
            )
            payload.pop("response_format")
            resp = await self.client.post(url, headers=self._headers(), json=payload)
        resp.raise_for_status()
        data = resp.json()
        message = data["choices"][0]["message"]
        content = message.get("content")
        if not content:
            # Thinking models can burn the whole completion in reasoning (oMLX
            # surfaces it as reasoning_content) and return no content at all.
            # Raise ValueError so generate_structured's retry ladder fires
            # instead of a bare KeyError becoming an HTTP 500.
            reasoning = message.get("reasoning_content") or ""
            raise ValueError(
                "LLM returned no content "
                f"(finish_reason={data['choices'][0].get('finish_reason')!r}, "
                f"reasoning_chars={len(reasoning)})"
            )
        return content.strip()

    async def close(self) -> None:
        await self.client.aclose()

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """One tool-call round, streamed internally. Returns the full assistant
        message dict: {"role": "assistant", "content": str|None, "tool_calls": [...]}.

        Servers that reject the tools field get it dropped in-stream (the reply
        will have no tool_calls); servers that reject streaming itself get one
        plain blocking call.
        """
        try:
            parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            async for ev in self.chat_stream(
                messages, temperature=temperature, tools=tools, tool_choice=tool_choice
            ):
                if ev["type"] == "delta":
                    parts.append(ev["content"])
                elif ev["type"] == "tool_call":
                    tool_calls.append(ev["tool_call"])
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in _STREAM_UNSUPPORTED_STATUS:
                raise
            logger.warning(
                "Streaming unavailable (HTTP %s); falling back to blocking call",
                exc.response.status_code,
            )
            return await self._chat_with_tools_blocking(
                messages, temperature, tools, tool_choice
            )
        content = "".join(parts)
        return {
            "role": "assistant",
            "content": content if content else None,
            "tool_calls": tool_calls,
        }

    async def _chat_with_tools_blocking(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        url = f"{self.base_url}/chat/completions"
        logger.info("LLM tool-call request to %s with model %s", url, self.model)
        resp = await self.client.post(url, headers=self._headers(), json=payload)
        if resp.status_code == 400 and "tools" in payload:
            logger.warning("Server rejected tools (HTTP 400); retrying without them")
            payload.pop("tools")
            payload.pop("tool_choice", None)
            resp = await self.client.post(url, headers=self._headers(), json=payload)
        resp.raise_for_status()
        data = resp.json()
        message = data["choices"][0]["message"]
        if isinstance(message, dict):
            msg = dict(message)
            msg.setdefault("role", "assistant")
            msg.setdefault("content", None)
            msg.setdefault("tool_calls", [])
            return msg
        return {"role": "assistant", "content": str(message).strip(), "tool_calls": []}

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming completion. Yields event dicts:

        - {"type": "delta", "content": str}          — text token
        - {"type": "reasoning", "content": str}      — reasoning/thinking token
        - {"type": "tool_call", "tool_call": {...}}  — one complete tool call
          (argument fragments accumulated across chunks)
        - {"type": "finish", "reason": "length"}     — completion truncated at
          the server's token cap
        - {"type": "done"}                           — stream finished

        On HTTP 400 the optional fields are dropped one at a time
        (response_format first, then tools) and the request retried — the same
        LM-Studio-style fallbacks the blocking path has. A 400 that survives
        both drops surfaces as HTTPStatusError.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        if response_format:
            payload["response_format"] = response_format
        url = f"{self.base_url}/chat/completions"
        logger.info("LLM stream request to %s with model %s", url, self.model)
        tool_acc: dict[int, dict[str, Any]] = {}
        while True:
            retry_without: str | None = None
            async with self.client.stream("POST", url, headers=self._headers(), json=payload) as resp:
                if resp.status_code == 400:
                    # Read body for logging, then drop optional fields one at a
                    # time before giving up.
                    await resp.aread()
                    logger.warning("Stream request rejected (HTTP 400): %s", resp.text[:500])
                    if "response_format" in payload:
                        retry_without = "response_format"
                    elif "tools" in payload:
                        retry_without = "tools"
                if retry_without is None:
                    resp.raise_for_status()
                    async for ev in self._parse_sse(resp, tool_acc):
                        yield ev
            if retry_without is None:
                break
            logger.warning("Retrying stream without %s", retry_without)
            payload.pop(retry_without)
            if retry_without == "tools":
                payload.pop("tool_choice", None)
        # Some servers only send finish_reason=stop — flush anything accumulated.
        for idx in sorted(tool_acc):
            if tool_acc[idx]["function"]["name"]:
                yield {"type": "tool_call", "tool_call": tool_acc[idx]}
        yield {"type": "done"}

    async def _parse_sse(
        self, resp: httpx.Response, tool_acc: dict[int, dict[str, Any]]
    ) -> AsyncIterator[dict[str, Any]]:
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                # Mid-stream error payloads ({"error": {...}}) carry no
                # choices — surface them instead of ending the turn with
                # a silently blank assistant message.
                err = chunk.get("error")
                if err is not None:
                    message = (
                        err.get("message")
                        if isinstance(err, dict)
                        else str(err)
                    )
                    raise RuntimeError(f"LLM stream error: {message}")
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                yield {"type": "delta", "content": content}
            reasoning = delta.get("reasoning_content")
            if reasoning:
                yield {"type": "reasoning", "content": reasoning}
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                acc = tool_acc.get(idx)
                if acc is None:
                    acc = {
                        "id": tc.get("id") or f"call_{idx}",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                    tool_acc[idx] = acc
                if tc.get("id"):
                    acc["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    acc["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    acc["function"]["arguments"] += fn["arguments"]
            finish = choices[0].get("finish_reason")
            if finish == "tool_calls":
                for idx in sorted(tool_acc):
                    if tool_acc[idx]["function"]["name"]:
                        yield {"type": "tool_call", "tool_call": tool_acc[idx]}
                tool_acc.clear()
            elif finish == "length":
                # Completion hit the server's token cap — surface it so
                # structured consumers can treat the reply as unusable.
                # (Chat-loop consumers ignore unknown event types.)
                yield {"type": "finish", "reason": "length"}


def _salvage_items(
    text: str,
    salvage: dict[str, tuple[str, ...]],
    decoder: json.JSONDecoder,
) -> dict[str, Any] | None:
    """Rescue well-formed items from an unparseable document.

    A single corrupted token mid-array (observed live: `いorder_index":` where
    `{ "` belongs) malforms the WHOLE document while every other item stays
    perfectly well-formed. Collect the outermost balanced objects in document
    order, and classify them by marker keys into the envelope keys of
    `salvage` — each fragment is claimed by the FIRST key whose markers it
    fully matches, so tiers cannot double-collect. The salvage is valid only
    when at least one key recovers >= 2 items (a single fragment is
    indistinguishable from the first-object trap this path exists to avoid);
    once anchored, smaller tiers ride along with >= 1. Items are returned
    verbatim (a lost item stays lost; nothing is invented).
    """
    fragments: list[dict[str, Any]] = []
    pos = text.find("{")
    while pos != -1:
        try:
            obj, end = decoder.raw_decode(text[pos:])
            if isinstance(obj, dict):
                fragments.append(obj)
                pos = text.find("{", pos + end)
                continue
        except json.JSONDecodeError:
            pass
        pos = text.find("{", pos + 1)
    salvaged: dict[str, list[dict[str, Any]]] = {}
    for frag in fragments:
        for key, markers in salvage.items():
            if all(m in frag for m in markers):
                salvaged.setdefault(key, []).append(frag)
                break
    if salvaged and any(len(v) >= 2 for v in salvaged.values()):
        logger.warning(
            "Salvaged %s from an unparseable LLM reply (%d balanced fragments scanned)",
            ", ".join(f"{len(v)} '{k}'" for k, v in salvaged.items()),
            len(fragments),
        )
        return salvaged
    return None


def extract_json(
    text: str,
    expected_any: tuple[str, ...] | None = None,
    salvage: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Extract a JSON object from a model reply.

    Handles the messy shapes local models actually produce: raw JSON, fenced
    code blocks, JSON embedded in prose, and valid JSON followed by trailing
    chatter ("Extra data: line 1 column N" failures).

    `expected_any` guards the LAST-RESORT balanced-scan fallback only: when
    the whole document is unparseable (truncated at a token cap, or malformed
    mid-document) the scan would otherwise "succeed" on the first INNER object
    — e.g. a single beat instead of {"beats": [...]} — and the caller's retry
    never fires. If the fallback's result contains none of the expected keys,
    raise instead. Clean and fenced parses are never affected: a complete,
    valid document is the model's actual answer, whatever its keys.

    `salvage` (only consulted where the expected_any guard would raise) maps
    an envelope key to the marker keys its items must all carry, e.g.
    {"beats": ("order_index", "description")}. When >= 2 well-formed items can
    be recovered from the wreck, return {key: items} instead of raising —
    the caller's count validation then decides whether the partial is enough.
    """
    text = text.strip()
    if not text:
        raise ValueError("LLM returned empty content")

    # Fast path: clean, single JSON document
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            # A bare top-level array (some models emit the items without the
            # envelope). Do NOT fall through to the brace-scan — it would
            # confidently return just the FIRST element. Raising lets
            # generate_structured retry in json_object mode instead.
            raise ValueError(
                f"LLM returned a top-level JSON array ({len(parsed)} items), expected an object"
            )
    except json.JSONDecodeError:
        pass

    # Strip fenced code blocks (```json ... ``` or ``` ... ```)
    if "```" in text:
        lines = text.splitlines()
        chunks: list[str] = []
        inside = False
        for line in lines:
            if not inside and line.strip().startswith("```"):
                inside = True
                continue
            if inside and line.strip().startswith("```"):
                inside = False
                continue
            if inside:
                chunks.append(line)
        if chunks:
            candidate = "\n".join(chunks).strip()
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
                if isinstance(parsed, list):
                    raise ValueError(
                        f"LLM returned a top-level JSON array ({len(parsed)} items), expected an object"
                    )
            except json.JSONDecodeError:
                pass

    # A bare array embedded in prose/chatter: detect it BEFORE the object scan,
    # which would otherwise confidently return just the array's first element.
    # Only when the array opens before any object does — an "[" inside
    # {"beats": [...]} must not shadow the enclosing object.
    decoder = json.JSONDecoder()
    arr_start = text.find("[")
    obj_start = text.find("{")
    if arr_start != -1 and (obj_start == -1 or arr_start < obj_start):
        try:
            parsed, _ = decoder.raw_decode(text[arr_start:])
            if isinstance(parsed, list) and parsed and all(isinstance(x, dict) for x in parsed):
                raise ValueError(
                    f"LLM returned a top-level JSON array ({len(parsed)} items), expected an object"
                )
        except json.JSONDecodeError:
            pass

    # Last resort: scan for the first balanced {...} object and ignore
    # whatever prose or chatter follows it.
    start = text.find("{")
    while start != -1:
        try:
            parsed, _ = decoder.raw_decode(text[start:])
            if isinstance(parsed, dict):
                if expected_any and not any(k in parsed for k in expected_any):
                    if salvage:
                        salvaged = _salvage_items(text, salvage, decoder)
                        if salvaged is not None:
                            return salvaged
                    raise ValueError(
                        "LLM reply was unparseable as a whole and the recovered "
                        f"fragment has none of the expected keys {expected_any} "
                        f"(got {sorted(parsed)[:6]}) — likely a truncated or "
                        "malformed envelope"
                    )
                return parsed
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    raise ValueError(f"No JSON object found in LLM reply (len={len(text)})")


async def generate_structured(
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    expected_any: tuple[str, ...] | None = None,
    salvage: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    client = LLMClient()

    def _rescue_truncated(exc: ValueError) -> dict[str, Any] | None:
        """Salvage complete items out of a truncated reply's partial content."""
        if not (isinstance(exc, LLMTruncatedError) and exc.partial and salvage):
            return None
        try:
            rescued = extract_json(
                exc.partial, expected_any=expected_any, salvage=salvage
            )
        except ValueError:
            return None
        logger.warning(
            "Rescued a truncated reply via salvage (%s)",
            ", ".join(f"{len(v)} {k}" for k, v in rescued.items() if isinstance(v, list)),
        )
        return rescued

    try:
        # JSON mode is off by default: several OpenAI-compatible servers
        # (notably LM Studio) reject response_format, and the prompts already
        # instruct the model to answer with a single JSON object.
        try:
            # chat() itself can raise ValueError (reasoning-only reply with no
            # content, or a completion truncated at the token cap) — that must
            # reach the retry below, so it lives inside this try alongside the
            # parse.
            text = await client.chat(messages, temperature=temperature)
            return extract_json(text, expected_any=expected_any, salvage=salvage)
        except ValueError as exc:
            rescued = _rescue_truncated(exc)
            if rescued is not None:
                return rescued
            # One retry with JSON mode requested, for servers that support it
            logger.warning("LLM reply unusable (%s); retrying with json_object mode", exc)
            try:
                text = await client.chat(
                    messages, temperature=temperature, response_format={"type": "json_object"}
                )
                return extract_json(text, expected_any=expected_any, salvage=salvage)
            except ValueError as exc2:
                rescued = _rescue_truncated(exc2)
                if rescued is not None:
                    return rescued
                raise
    finally:
        await client.close()
