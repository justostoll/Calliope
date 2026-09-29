"""Render approval: a bare "yes" clicked on a non-render card is not a render
approval (found while reviewing session 167, 2026-09-29).

A card click is also a user/message, and user_allows_render read its text as
prose: "Yes" or "Yes, replace it" answering an info or destructive_replace
card also unlocked renders (and exposed the render tools). Now a bare
confirmation counts only when typed or clicked on a render card; a clicked
option whose own words ask for images/video still counts (as the operator's
"Generate reference images for locations/items first" did in session 167).
"""
from __future__ import annotations

from calliope.agent.harness import _render_approval_guard, build_harness
from calliope.agent.harness import log as session_log
from calliope.agent.harness.policy import user_allows_render
from calliope.agent.harness.registry import ToolContext
from calliope.config import settings
from calliope.db import get_db


def _ctx(client, message: str, *, card_scope: str | None = None) -> ToolContext:
    pid = client.post("/api/projects", json={"title": "Render Scope Film"}).json()["id"]
    conn = get_db(settings.db_path)
    try:
        sid = conn.execute(
            "INSERT INTO agent_sessions (title, project_id) VALUES (?, ?)", ("render-scope", pid)
        ).lastrowid
        conn.commit()
    finally:
        conn.close()
    if card_scope is not None:
        session_log.append_event(
            sid, session_log.QUESTION_ANSWERED, {"answer": message, "scope": card_scope}
        )
    session_log.append_event(sid, session_log.USER_MESSAGE, {"content": message})
    return ToolContext(session_id=sid, project_id=pid)


def _render_tool_offered(ctx: ToolContext) -> bool:
    registry, _ = build_harness()
    return "enqueue_asset_jobs" in {p["function"]["name"] for p in registry.openai_payload(ctx)}


def _guard(ctx: ToolContext) -> str:
    registry, _ = build_harness()
    return _render_approval_guard(ctx, registry.get("enqueue_asset_jobs"), {}).kind


def test_bare_yes_on_a_non_render_card_does_not_approve_renders(client):
    for message, scope in (
        ("Yes", "info"),
        ("Yes — generate the story", "info"),
        ("Yes, replace it", "destructive_replace"),
    ):
        ctx = _ctx(client, message, card_scope=scope)
        assert user_allows_render(ctx) is False, (message, scope)
        assert _guard(ctx) == "deny", (message, scope)
        assert not _render_tool_offered(ctx), (message, scope)


def test_render_worded_option_on_an_info_card_still_approves(client):
    ctx = _ctx(client, "Generate reference images for locations/items first", card_scope="info")
    assert user_allows_render(ctx) is True
    assert _guard(ctx) == "allow"
    assert _render_tool_offered(ctx)


def test_render_card_yes_and_typed_confirmations_still_approve(client):
    for message, scope in (
        ("Yes", "render"),
        ("yes, do it", None),
        ("yes", None),
    ):
        ctx = _ctx(client, message, card_scope=scope)
        assert user_allows_render(ctx) is True, (message, scope)
        assert _guard(ctx) == "allow", (message, scope)
