"""Tool-result cap + get_workspace paging.

At a 4,000-char cap a 3-minute film's beats section (15 beats, 6.1k chars) was
cut at beat 11, and TRUNCATE_NOTE's only advice (sections=["beats"]) could not
get under it: the agent deleted beats to see the rest (observed live
2026-09-29). These pin the fix: that story fits whole, longer lists page with
offset/limit, and the note teaches paging instead of editing.
"""
from __future__ import annotations

import asyncio
import json

from calliope.agent.harness import log as session_log
from calliope.agent.harness.registry import ToolContext
from calliope.agent.harness.tools import execute_tool
from calliope.config import settings
from calliope.db import get_db

# Description lengths of the 15 beats in the 2026-09-29 incident (chunk 1 = 12
# short beats, the continuation chunk = 3 long ones).
INCIDENT_BEAT_CHARS = [209, 230, 227, 224, 229, 218, 217, 233, 222, 210, 209, 218, 427, 479, 490]


def _project_with_beats(client, lengths: list[int]) -> ToolContext:
    pid = client.post("/api/projects", json={"title": "Paging Film"}).json()["id"]
    conn = get_db(settings.db_path)
    try:
        for i, n in enumerate(lengths, start=1):
            conn.execute(
                "INSERT INTO story_beats (project_id, order_index, title, description) "
                "VALUES (?, ?, ?, ?)",
                (pid, i, f"Beat {i}", ("w" * (n - 1)) + "."),
            )
        sid = conn.execute(
            "INSERT INTO agent_sessions (title, project_id) VALUES (?, ?)", ("paging-test", pid)
        ).lastrowid
        conn.commit()
    finally:
        conn.close()
    return ToolContext(session_id=sid, project_id=pid)


def _ws(ctx: ToolContext, args: dict) -> dict:
    return asyncio.run(execute_tool(ctx, "get_workspace", args))


def test_incident_story_beats_fit_one_result(client):
    ctx = _project_with_beats(client, INCIDENT_BEAT_CHARS)
    out = _ws(ctx, {"sections": ["beats"]})
    text = session_log._truncate_result(out)
    assert session_log.TRUNCATE_NOTE not in text
    assert [b["order_index"] for b in json.loads(text)["beats"]] == list(range(1, 16))


def test_offset_limit_pages_beats_in_story_order(client):
    ctx = _project_with_beats(client, INCIDENT_BEAT_CHARS)

    first = _ws(ctx, {"sections": ["beats"], "offset": 0, "limit": 6})
    assert [b["order_index"] for b in first["beats"]] == [1, 2, 3, 4, 5, 6]
    assert first["page"] == {"offset": 0, "limit": 6, "totals": {"beats": 15}, "next_offset": 6}

    last = _ws(ctx, {"sections": ["beats"], "offset": 12, "limit": 6})
    assert [b["order_index"] for b in last["beats"]] == [13, 14, 15]
    assert "next_offset" not in last["page"]

    # offset alone = the rest of each list
    rest = _ws(ctx, {"sections": ["beats"], "offset": 10})
    assert [b["order_index"] for b in rest["beats"]] == [11, 12, 13, 14, 15]
    assert rest["page"]["totals"] == {"beats": 15}


def test_no_page_args_is_unchanged(client):
    ctx = _project_with_beats(client, INCIDENT_BEAT_CHARS)
    out = _ws(ctx, {"sections": ["beats"]})
    assert len(out["beats"]) == 15
    assert "page" not in out


def test_bad_page_args_are_errors_not_full_fetches(client):
    ctx = _project_with_beats(client, [100, 100])
    for args in ({"offset": -1}, {"limit": 0}, {"limit": "many"}):
        out = _ws(ctx, {"sections": ["beats"], **args})
        assert out["ok"] is False, args
        assert "beats" not in out


def test_truncation_note_teaches_paging_not_editing():
    note = session_log.TRUNCATE_NOTE
    assert "offset" in note and "limit" in note and "next_offset" in note
    assert "Never delete or edit content" in note
