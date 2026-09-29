"""break_into_shots bulk check: counted per user message, and scenes the user
names count as asked (observed live 2026-09-29, session 169).

Asked only for the script, a script-agent broke 15 scenes into shots anyway:
three bulk calls were refused (> 3 scenes per call), then 3 + 3 passed because
the cap counted one call. And the refusal said "name the ones they want", but
a user naming "scenes 7–15" did not count.
"""
from __future__ import annotations

import asyncio

import pytest

from calliope.agent.harness import log as session_log
from calliope.agent.harness.plugins.script import t_break_into_shots
from calliope.agent.harness.registry import ToolContext
from calliope.config import settings
from calliope.db import get_db


@pytest.fixture
def expanded(monkeypatch):
    """Stub the LLM coverage pass; record which scene ids it was asked for."""
    calls: list[list[int]] = []

    async def fake(project_id, scene_ids, guidance=None):
        calls.append(list(scene_ids))
        return {"scenes": [{"scene_id": s, "clips": 2} for s in scene_ids]}

    monkeypatch.setattr("calliope.agent.coverage_agent.expand_scene_coverage", fake)
    return calls


def _film(client, message: str) -> tuple[ToolContext, dict[int, int]]:
    """A 15-scene project and a session whose latest user message is `message`;
    returns the context and {order: scene id}."""
    pid = client.post("/api/projects", json={"title": "Shots Film"}).json()["id"]
    conn = get_db(settings.db_path)
    try:
        ids = {
            order: conn.execute(
                "INSERT INTO scenes (project_id, order_index, heading) VALUES (?, ?, ?)",
                (pid, order, f"SCENE {order}"),
            ).lastrowid
            for order in range(1, 16)
        }
        sid = conn.execute(
            "INSERT INTO agent_sessions (title, project_id) VALUES (?, ?)", ("shots", pid)
        ).lastrowid
        conn.commit()
    finally:
        conn.close()
    session_log.append_event(sid, session_log.USER_MESSAGE, {"content": message})
    return ToolContext(session_id=sid, project_id=pid), ids


def _log_break(ctx: ToolContext, scene_ids: list[int], ok: bool = True) -> None:
    result = (
        {"ok": True, "scenes": [{"scene_id": s, "clips": 2} for s in scene_ids]}
        if ok
        else {"ok": False, "error": "bulk pass"}
    )
    session_log.append_event(
        ctx.session_id,
        session_log.TOOL_RESULT,
        {"call_id": "c", "tool_name": "break_into_shots", "result": result},
    )


def _break(ctx: ToolContext, **args) -> dict:
    return asyncio.run(t_break_into_shots(ctx, args))


def test_batches_count_against_the_same_message(client, expanded):
    ctx, ids = _film(client, "Regenerate the full script for this project")
    _log_break(ctx, [ids[1], ids[2], ids[3]])
    out = _break(ctx, scene_ids=[ids[4], ids[5], ids[6]])
    assert out["ok"] is False
    assert "3 already broken" in out["error"] and "smaller batches" in out["error"]
    assert expanded == []


def test_first_small_batch_is_still_free(client, expanded):
    ctx, ids = _film(client, "Regenerate the full script for this project")
    assert _break(ctx, scene_ids=[ids[1], ids[2], ids[3]])["ok"] is True
    assert expanded == [[ids[1], ids[2], ids[3]]]


def test_refused_attempts_do_not_count(client, expanded):
    ctx, ids = _film(client, "Regenerate the full script for this project")
    _log_break(ctx, list(ids.values()), ok=False)
    assert _break(ctx, scene_ids=[ids[1], ids[2], ids[3]])["ok"] is True


def test_a_new_user_message_starts_a_new_count(client, expanded):
    ctx, ids = _film(client, "break scenes 1 to 3 into shots")
    _log_break(ctx, [ids[1], ids[2], ids[3]])
    session_log.append_event(ctx.session_id, session_log.USER_MESSAGE, {"content": "now 4-6"})
    assert _break(ctx, scene_ids=[ids[4], ids[5], ids[6]])["ok"] is True


def test_scenes_the_user_named_are_asked_for(client, expanded):
    ctx, ids = _film(client, "Break scenes 7–15 into shots")
    wanted = [ids[o] for o in range(7, 16)]
    assert _break(ctx, scene_ids=wanted)["ok"] is True
    assert _break(ctx, orders=list(range(7, 16)))["ok"] is True
    out = _break(ctx, scene_ids=[ids[6], *wanted])  # scene 6 was not named
    assert out["ok"] is False


def test_bulk_words_still_allow_the_whole_pass(client, expanded):
    ctx, ids = _film(client, "Break all remaining scenes into shots")
    _log_break(ctx, [ids[o] for o in range(1, 7)])
    assert _break(ctx, orders=list(range(7, 16)))["ok"] is True


def test_named_scene_orders():
    from calliope.agent.harness.policy import named_scene_orders

    assert named_scene_orders("Break scenes 7–15 into shots") == set(range(7, 16))
    assert named_scene_orders("redo scene 4") == {4}
    assert named_scene_orders("scenes 2, 5 and 9 please") == {2, 5, 9}
    assert named_scene_orders("scenes 1 to 3, but not scenes 7-9") == {1, 2, 3}
    assert named_scene_orders("regenerate the full script") == set()
