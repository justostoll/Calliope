"""Destructive guard: what a confirmation covers (observed live 2026-09-29).

Session 167: the user clicked "Yes — generate the story" on an `info` card.
The story-agent's plan (planner -> sub-agent, whose tool/calls carry no `turn`
field) then drafted a story, deleted 8 of its beats, wrote probe text over 3
more, and ran generate_story replace=true again, wiping the whole story. Four
gaps, one test group each:
  1. an `info` card answer confirmed a replace by its wording;
  2. a confirmation covered content created AFTER it (so it bought unlimited
     replacements);
  3. the delete_beat cap counted only turn-tagged calls, so orchestrated
     deletes were never counted;
  4. update_beat could overwrite a description with a stub, unguarded.
"""
from __future__ import annotations

import asyncio

from calliope.agent.harness import (
    BULK_ROW_DELETES_PER_TURN,
    GUARD_DESTRUCTIVE_REPLACE,
    _destructive_guard,
)
from calliope.agent.harness import log as session_log
from calliope.agent.harness.registry import ToolContext, ToolDefinition, ToolRegistry
from calliope.config import settings
from calliope.db import get_db

T_BEFORE = "2026-09-28 23:40:00"
T_MESSAGE = "2026-09-28 23:42:42"
T_AFTER = "2026-09-28 23:45:32"
DESCRIPTION = "Supergirl stands above the city at dawn, " * 6  # ~250 chars


async def _ok(ctx, args):
    return {"ok": True}


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    for name, props, destructive in (
        ("generate_story", {"replace": {"type": "boolean"}}, True),
        ("generate_script", {"replace": {"type": "boolean"}}, True),
        ("delete_beat", {"beat_id": {"type": "integer"}}, True),
        (
            "update_beat",
            {
                "beat_id": {"type": "integer"},
                "title": {"type": "string"},
                "description": {"type": "string"},
            },
            False,
        ),
    ):
        reg.register(
            ToolDefinition(
                name=name,
                description="stub",
                parameters={"type": "object", "properties": props},
                executor=_ok,
                destructive=destructive,
            )
        )
    reg.on_pre_execute(_destructive_guard)
    return reg


def _project(client, beats_created_at: str, n: int = 1) -> tuple[int, list[int]]:
    pid = client.post("/api/projects", json={"title": "Guard Scope Film"}).json()["id"]
    conn = get_db(settings.db_path)
    try:
        ids = [
            conn.execute(
                "INSERT INTO story_beats (project_id, order_index, title, description, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (pid, i + 1, f"Beat {i + 1}", DESCRIPTION, beats_created_at),
            ).lastrowid
            for i in range(n)
        ]
        conn.commit()
    finally:
        conn.close()
    return pid, ids


def _session(pid: int, message: str, *, card_scope: str | None = None) -> ToolContext:
    """A session whose latest user/message is `message` at T_MESSAGE; with
    card_scope, it is a card click (question/answered right before it). Like
    session 167, an earlier loop turn exists, and the work that follows the
    message is an orchestrated plan that logs no turn/start of its own."""
    conn = get_db(settings.db_path)
    try:
        sid = conn.execute(
            "INSERT INTO agent_sessions (title, project_id) VALUES (?, ?)", ("guard-scope", pid)
        ).lastrowid
        conn.commit()
    finally:
        conn.close()
    session_log.append_event(sid, session_log.USER_MESSAGE, {"content": "Create a new film"})
    session_log.append_event(sid, session_log.TURN_START, {"turn": 1})
    if card_scope is not None:
        session_log.append_event(
            sid, session_log.QUESTION_ANSWERED, {"answer": message, "scope": card_scope}
        )
    ev = session_log.append_event(sid, session_log.USER_MESSAGE, {"content": message})
    conn = get_db(settings.db_path)
    try:
        conn.execute(
            "UPDATE agent_events SET created_at = ? WHERE session_id = ? AND seq = ?",
            (T_MESSAGE, sid, ev.seq),
        )
        conn.commit()
    finally:
        conn.close()
    return ToolContext(session_id=sid, project_id=pid)


def _run(ctx: ToolContext, tool: str, args: dict) -> dict:
    return asyncio.run(_registry().execute(ctx, tool, args))


def _blocked(out: dict) -> bool:
    return out.get("ok") is False and (
        out.get("reason_code") == GUARD_DESTRUCTIVE_REPLACE or "blocked:" in str(out.get("error"))
    )


def _log_orchestrated_calls(ctx: ToolContext, tool: str, n: int) -> None:
    """Sub-agent tool/calls as the orchestrator logs them: no `turn` field."""
    for i in range(n):
        session_log.append_event(
            ctx.session_id,
            session_log.TOOL_CALL,
            {"call_id": f"s{i}", "tool_name": tool, "agent_name": "story-agent"},
        )


# 1. A card answer grants only what its scope says ───────────────────────────


def test_info_card_yes_does_not_confirm_a_replace(client):
    pid, _ = _project(client, T_BEFORE)
    ctx = _session(pid, "Yes — generate the story", card_scope="info")
    assert _blocked(_run(ctx, "generate_story", {"replace": True}))


def test_destructive_replace_card_and_typed_yes_still_confirm(client):
    pid, _ = _project(client, T_BEFORE)
    card = _session(pid, "Yes, replace it", card_scope="destructive_replace")
    assert _run(card, "generate_story", {"replace": True})["ok"] is True
    typed = _session(pid, "Yes — generate the story")
    assert _run(typed, "generate_story", {"replace": True})["ok"] is True


# 2. A confirmation covers only content that existed when it was given ──────


def test_confirmation_does_not_cover_content_created_after_it(client):
    pid, _ = _project(client, T_AFTER)  # the run's own first draft
    ctx = _session(pid, "yes, replace the story")
    out = _run(ctx, "generate_story", {"replace": True})
    assert _blocked(out)
    assert "created AFTER" in out["error"]


def test_confirmation_covers_content_the_user_had_seen(client):
    pid, _ = _project(client, T_BEFORE)
    ctx = _session(pid, "yes, replace the story")
    assert _run(ctx, "generate_story", {"replace": True})["ok"] is True


def test_same_second_content_counts_as_seen(client):
    pid, _ = _project(client, T_MESSAGE)
    ctx = _session(pid, "yes, replace the story")
    assert _run(ctx, "generate_story", {"replace": True})["ok"] is True


# 3. The delete cap counts orchestrated (turn-less) calls ───────────────────


def test_orchestrated_deletes_hit_the_cap(client):
    pid, ids = _project(client, T_BEFORE, n=BULK_ROW_DELETES_PER_TURN + 1)
    ctx = _session(pid, "make the story darker")
    _log_orchestrated_calls(ctx, "delete_beat", BULK_ROW_DELETES_PER_TURN - 1)
    assert _run(ctx, "delete_beat", {"beat_id": ids[0]})["ok"] is True
    _log_orchestrated_calls(ctx, "delete_beat", 1)
    out = _run(ctx, "delete_beat", {"beat_id": ids[1]})
    assert _blocked(out)
    assert "row by row" in out["error"]


def test_confirmed_bulk_delete_blocks_rows_newer_than_the_confirmation(client):
    pid, ids = _project(client, T_AFTER, n=2)
    ctx = _session(pid, "Regenerate the storyline from scratch, darker.")
    _log_orchestrated_calls(ctx, "delete_beat", BULK_ROW_DELETES_PER_TURN)
    assert _blocked(_run(ctx, "delete_beat", {"beat_id": ids[0]}))

    pid, ids = _project(client, T_BEFORE, n=2)
    ctx = _session(pid, "Regenerate the storyline from scratch, darker.")
    _log_orchestrated_calls(ctx, "delete_beat", BULK_ROW_DELETES_PER_TURN)
    assert _run(ctx, "delete_beat", {"beat_id": ids[0]})["ok"] is True


# 4. update_beat cannot wipe a description ──────────────────────────────────


def test_update_beat_wipe_is_blocked(client):
    pid, ids = _project(client, T_BEFORE)
    ctx = _session(pid, "make the story darker")
    for stub in ("TEST", "PROBE-KEPT", ""):
        out = _run(ctx, "update_beat", {"beat_id": ids[0], "description": stub})
        assert _blocked(out), stub
        assert "wipes it" in out["error"]


def test_update_beat_real_edits_stay_free(client):
    pid, ids = _project(client, T_BEFORE)
    ctx = _session(pid, "make the story darker")
    for args in (
        {"description": "At dawn Supergirl hovers over the city, alone and unreachable."},
        {"description": DESCRIPTION.replace("dawn", "dusk")},
        {"title": "X"},
    ):
        assert _run(ctx, "update_beat", {"beat_id": ids[0], **args})["ok"] is True, args


def test_update_beat_wipe_allowed_after_confirmation_of_seen_beat(client):
    pid, ids = _project(client, T_BEFORE)
    ctx = _session(pid, "yes, replace it")
    assert _run(ctx, "update_beat", {"beat_id": ids[0], "description": "Cut."})["ok"] is True

    pid, ids = _project(client, T_AFTER)
    ctx = _session(pid, "yes, replace it")
    assert _blocked(_run(ctx, "update_beat", {"beat_id": ids[0], "description": "Cut."}))


# 5. A replace that would delete nothing goes through ────────────────────────
# (12ca9c0 refused the first generate_script on a drafted story even after a
# yes: no scene predated the confirmation because there were no scenes.)


def _add_scene(pid: int, created_at: str) -> None:
    conn = get_db(settings.db_path)
    try:
        conn.execute(
            "INSERT INTO scenes (project_id, order_index, heading, created_at) VALUES (?, 1, ?, ?)",
            (pid, "INT. CHAPEL - NIGHT", created_at),
        )
        conn.commit()
    finally:
        conn.close()


def test_first_script_on_a_drafted_story_goes_through(client):
    pid, _ = _project(client, T_BEFORE, n=15)  # a story, no scenes: project 20 on 2026-09-29
    for message, scope in (
        ("Yes — generate the script", "info"),
        ("yes, generate the script", None),
        ("Yes", "destructive_replace"),
        ("make it darker", None),
    ):
        ctx = _session(pid, message, card_scope=scope)
        assert _run(ctx, "generate_script", {"replace": True})["ok"] is True, (message, scope)


def test_script_replace_is_still_guarded_when_scenes_exist(client):
    pid, _ = _project(client, T_BEFORE)
    _add_scene(pid, T_BEFORE)
    assert _blocked(_run(_session(pid, "Yes", card_scope="info"), "generate_script", {}))
    assert _run(_session(pid, "yes, replace the script"), "generate_script", {})["ok"] is True

    pid, _ = _project(client, T_BEFORE)
    _add_scene(pid, T_AFTER)
    out = _run(_session(pid, "yes, replace the script"), "generate_script", {"replace": True})
    assert _blocked(out) and "created AFTER" in out["error"]


def test_story_over_an_existing_script_needs_a_yes_not_a_seen_story(client):
    # Scenes only: a new story orphans the script (gated), but deletes no
    # story rows, so a yes is enough — the age rule has nothing to cover.
    pid = client.post("/api/projects", json={"title": "Script First"}).json()["id"]
    _add_scene(pid, T_BEFORE)
    assert _blocked(_run(_session(pid, "Yes", card_scope="info"), "generate_story", {}))
    assert _run(_session(pid, "yes, write a new story"), "generate_story", {})["ok"] is True


def test_second_story_replace_blocked_even_with_an_old_script(client):
    # The old script must not "cover" replacing the run's own fresh story.
    pid, _ = _project(client, T_AFTER)
    _add_scene(pid, T_BEFORE)
    out = _run(_session(pid, "yes, replace the story"), "generate_story", {"replace": True})
    assert _blocked(out) and "created AFTER" in out["error"]
