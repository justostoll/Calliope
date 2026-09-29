"""Bulk scope and render consent read from prose (found by the Laya shadow
eval on the 2026-09-29 snapshots, constellation-build measurements/
n1-laya-shadow-eval-2026-09-29.md).

1. allows_bulk_enqueue missed "all 8 clips", "all the remaining clips" and
   "all of them" (real card answers "Yes — images + all 7 clips" and "Yes,
   generate all 4" read as NOT asking for all), and passed "not all scenes,
   just 4 and 5" (no negation check). A number now caps the ask, and a
   negated ask counts for nothing, clause-scoped like every other cue.
2. user_allows_render fell back to is_confirmation, whose command verbs made
   a typed "Regenerate the full script …" (chat 169) a render approval. Only
   a plain yes / ok / go ahead counts now; is_confirmation itself is unchanged
   (the destructive guard still reads "regenerate from scratch" as consent).
"""
from __future__ import annotations

import pytest

from calliope.agent.harness import _render_approval_guard, build_harness
from calliope.agent.harness import log as session_log
from calliope.agent.harness.policy import (
    allows_bulk_enqueue,
    allows_bulk_video_enqueue,
    is_affirmation,
    is_confirmation,
    user_allows_render,
)
from calliope.agent.harness.registry import ToolContext
from calliope.config import settings
from calliope.db import get_db

# ── 1. bulk scope ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "count"),
    [
        ("Yes — images + all 7 clips", 7),
        ("then render all 8 H3 clips", 8),
        ("Yes, generate all 4", 4),
        ("render all the remaining clips", 9),
        ("all of them", 8),
        ("yes, all of these", 5),
        ("render all 20 scenes", 20),
    ],
)
def test_bulk_asks_the_regex_missed(text: str, count: int):
    assert allows_bulk_enqueue(text, count) is True, text
    assert allows_bulk_video_enqueue(text, count) is True, text


@pytest.mark.parametrize(
    ("text", "count"),
    [
        ("render all 8 clips", 24),  # the number caps the ask
        ("Yes, generate all 4", 12),
        ("not all scenes, just 4 and 5", 15),  # negated ask
        ("don't render all scenes", 15),
        ("never do every clip", 10),
    ],
)
def test_bulk_capped_or_negated_asks_do_not_count(text: str, count: int):
    assert allows_bulk_enqueue(text, count) is False, text


@pytest.mark.parametrize(
    ("text", "count", "want"),
    [
        ("yes", 2, True),  # small batch
        ("yes", 24, False),
        ("generate all scenes", 24, True),
        ("do every clip", 10, True),
        ("Break all remaining scenes into shots", 9, True),  # session 169, 2026-09-29
        ("don't overthink, render all scenes", 15, True),  # the veto is in another clause
        ("regenerate the full script", 15, False),
    ],
)
def test_bulk_controls_unchanged(text: str, count: int, want: bool):
    assert allows_bulk_enqueue(text, count) is want, text


# ── 2. render consent ─────────────────────────────────────────────────────


def _ctx(client, message: str) -> ToolContext:
    pid = client.post("/api/projects", json={"title": "Consent Film"}).json()["id"]
    conn = get_db(settings.db_path)
    try:
        sid = conn.execute(
            "INSERT INTO agent_sessions (title, project_id) VALUES (?, ?)", ("consent", pid)
        ).lastrowid
        conn.commit()
    finally:
        conn.close()
    session_log.append_event(sid, session_log.USER_MESSAGE, {"content": message})
    return ToolContext(session_id=sid, project_id=pid)


def _guard(ctx: ToolContext) -> str:
    registry, _ = build_harness()
    return _render_approval_guard(ctx, registry.get("enqueue_asset_jobs"), {}).kind


def test_typed_command_verb_is_not_a_render_approval(client):
    for message in (
        "Regenerate the full script for this project — turn the storyline into "
        "ordered scenes with action and dialogue.",  # chat 169, 2026-09-29
        "rewrite the logline",
        "delete beat 4",
    ):
        ctx = _ctx(client, message)
        assert user_allows_render(ctx) is False, message
        assert _guard(ctx) == "deny", message


def test_typed_yes_and_render_words_still_approve(client):
    for message in (
        "yes",
        "yes, do it",
        "go ahead",
        "regenerate the images",
        "redo the video for scene 5",
    ):
        ctx = _ctx(client, message)
        assert user_allows_render(ctx) is True, message
        assert _guard(ctx) == "allow", message


def test_affirmation_is_the_plain_yes_subset_of_confirmation():
    assert is_affirmation("Regenerate the full script") is False
    assert is_confirmation("Regenerate the full script") is True  # destructive guard unchanged
    assert is_confirmation("regenerate the whole script from scratch") is True
    assert is_affirmation("sure, go ahead") is True
    assert is_affirmation("no, don't") is False
    assert is_affirmation("don't overthink it, yes") is True
