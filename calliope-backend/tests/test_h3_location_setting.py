"""H3 prompts keep the scene's location, whether or not it gets an image slot.

The location used to reach the H3 rewrite only as a <Subject N> with an image
slot, so a 1-slot workflow with one character dropped the environment, text
included, and a location subject carried its image prompt ("no people").
"""
from __future__ import annotations

import asyncio
import json

from calliope.agent.prompts import (
    MINIMAX_H3_REF_SYSTEM,
    build_minimax_h3_ref_messages,
    minimax_h3_ref_fallback,
    video_setting,
)
from calliope.config import settings
from calliope.db import get_db

HARBOR = {
    "name": "Harbor Warehouse",
    "description": "rusted containers under sodium light",
    "consistency_prompt": (
        "ENVIRONMENT REFERENCE — Harbor Warehouse, wide establishing, no people, no text."
    ),
}
SETTING = {"name": "Harbor Warehouse", "description": "rusted containers under sodium light"}


def test_video_setting_uses_description_not_consistency_prompt():
    assert video_setting(HARBOR) == SETTING
    assert video_setting(None) is None
    assert video_setting({"name": "", "description": ""}) is None
    assert video_setting({"name": "", "description": "fog"}) == {
        "name": "the location",
        "description": "fog",
    }


def test_system_prompt_has_setting_rule():
    assert "9. Setting:" in MINIMAX_H3_REF_SYSTEM


def test_messages_always_carry_the_setting():
    scene = {"heading": "INT. WAREHOUSE - NIGHT", "action": "She waits.", "duration_sec": 4}
    text = build_minimax_h3_ref_messages(scene, [], setting=SETTING)[1]["content"]
    assert "Setting (the environment of every shot" in text
    assert "Harbor Warehouse — rusted containers under sodium light" in text
    none = build_minimax_h3_ref_messages(scene, [])[1]["content"]
    assert "(none given — infer the environment from the heading)" in none


def test_fallback_with_setting():
    scene = {"heading": "INT. WAREHOUSE - NIGHT", "action": "She waits."}
    subjects = [{"index": 1, "kind": "character", "name": "Maya", "appearance": "olive jacket"}]
    text = minimax_h3_ref_fallback(scene, subjects, setting=SETTING)
    assert (
        "[reference generation] INT. WAREHOUSE - NIGHT, set in Harbor Warehouse, "
        "featuring <Subject 1>." in text
    )
    assert (
        "[Shot 1] INT. WAREHOUSE - NIGHT The scene takes place in Harbor Warehouse: "
        "rusted containers under sodium light. She waits." in text
    )
    assert "overall_soundscape:\nNatural ambience of Harbor Warehouse" in text
    bare = minimax_h3_ref_fallback(scene, [], setting=SETTING)
    assert "[reference generation] INT. WAREHOUSE - NIGHT, set in Harbor Warehouse." in bare


def test_fallback_without_setting_is_unchanged():
    scene = {"heading": "INT. WAREHOUSE - NIGHT", "action": "She waits."}
    subjects = [{"index": 1, "kind": "character", "name": "Maya", "appearance": "olive jacket"}]
    text = minimax_h3_ref_fallback(scene, subjects)
    assert "[reference generation] INT. WAREHOUSE - NIGHT featuring <Subject 1>." in text
    assert "[Shot 1] INT. WAREHOUSE - NIGHT She waits." in text
    assert "overall_soundscape:\nN/A" in text
    assert "set in" not in text


def test_location_subject_uses_description():
    from calliope.agent.video_agent import resolve_h3_references
    from calliope.comfyui.parser import parse_dynamic_inputs

    inputs = parse_dynamic_inputs(_workflow(slots=2))
    subjects, paths, _ = resolve_h3_references(
        inputs,
        {},
        [{"name": "Maya", "appearance": "olive jacket", "sheet_path": "maya.png"}],
        HARBOR,
        "harbor.png",
    )
    loc = [s for s in subjects if s["kind"] == "location"]
    assert len(loc) == 1
    assert loc[0]["name"] == "Harbor Warehouse"
    assert loc[0]["appearance"] == "rusted containers under sodium light"
    assert "no people" not in loc[0]["appearance"]
    assert paths == ["maya.png", "harbor.png"]


# ── call sites: a 1-slot workflow whose slot goes to the character ─────


def _workflow(slots: int) -> dict:
    wf = {
        "10": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": ""},
            "_meta": {"title": "Main Prompt (Input:prompt)"},
        }
    }
    for i in range(slots):
        wf[str(20 + i)] = {
            "class_type": "LoadImage",
            "inputs": {"image": ""},
            "_meta": {"title": f"Ref {i + 1} (Input:image)"},
        }
    return wf


def _one_slot_clip(client) -> tuple[int, int]:
    """Project + clip: one character with a sheet, a location with an image, wf of 1 slot."""
    pid = client.post("/api/projects", json={"title": "harbor"}).json()["id"]
    scene = client.post(
        f"/api/projects/{pid}/scenes",
        json={"order_index": 1, "heading": "INT. WAREHOUSE - NIGHT", "action": "She waits."},
    ).json()
    assets = settings.assets_dir
    assets.mkdir(parents=True, exist_ok=True)
    sheet = assets / "maya_sheet.png"
    plate = assets / "harbor.png"
    sheet.write_bytes(b"\x89PNG\r\n\x1a\n")
    plate.write_bytes(b"\x89PNG\r\n\x1a\n")
    conn = get_db(settings.db_path)
    try:
        char_id = conn.execute(
            "INSERT INTO characters (project_id, name, appearance, sheet_path, consistency_prompt) "
            "VALUES (?, 'Maya', 'olive jacket', ?, 'clean neutral backdrop, studio lighting')",
            (pid, str(sheet)),
        ).lastrowid
        loc_id = conn.execute(
            "INSERT INTO locations (project_id, name, description, consistency_prompt, "
            "reference_image_path) VALUES (?, ?, ?, ?, ?)",
            (pid, HARBOR["name"], HARBOR["description"], HARBOR["consistency_prompt"], str(plate)),
        ).lastrowid
        conn.execute(
            "INSERT INTO scene_characters (scene_id, character_id) VALUES (?, ?)",
            (scene["id"], char_id),
        )
        conn.execute("UPDATE scenes SET location_id = ? WHERE id = ?", (loc_id, scene["id"]))
        wf_id = conn.execute(
            "INSERT INTO workflows (name, kind, workflow_json, input_schema, output_schema, "
            "prompt_profile, is_enabled) VALUES ('H3 1-slot', 'video', ?, '[]', '[]', "
            "'minimax_h3_ref', 1)",
            (json.dumps(_workflow(slots=1)),),
        ).lastrowid
        clip_id = conn.execute(
            "SELECT id FROM clips WHERE scene_id = ? ORDER BY order_index, id LIMIT 1",
            (scene["id"],),
        ).fetchone()["id"]
        conn.execute("UPDATE clips SET workflow_id = ? WHERE id = ?", (wf_id, clip_id))
        conn.commit()
    finally:
        conn.close()
    return pid, int(clip_id)


def _fake_plan(clip_id: int):
    async def fake_plan(project_id, **kwargs):
        return {
            "based_on": "p",
            "overview": {},
            "requirements": {},
            "shots": [{"clip_id": clip_id}],
        }

    return fake_plan


def test_preview_passes_the_setting_when_the_location_gets_no_slot(client, monkeypatch):
    from calliope.agent.video_agent import preview_clip_prompt

    pid, clip_id = _one_slot_clip(client)
    seen: dict = {}

    async def fake_rewrite(scene_, subjects, **kwargs):
        seen["subjects"] = subjects
        seen["setting"] = kwargs.get("setting")
        return "COMPILED"

    async def fake_critic(prompt, plan, clip_id_):
        return {"ok": True, "notes": []}

    monkeypatch.setattr("calliope.agent.video_agent.ensure_continuity_plan", _fake_plan(clip_id))
    monkeypatch.setattr("calliope.agent.video_agent._h3_rewrite", fake_rewrite)
    monkeypatch.setattr("calliope.agent.video_agent.critique_prompt", fake_critic)

    result = asyncio.run(preview_clip_prompt(pid, clip_id))
    assert result["prompt"] == "COMPILED"
    assert [s["kind"] for s in seen["subjects"]] == ["character"]
    assert seen["setting"] == SETTING


def test_enqueue_passes_the_setting_when_the_location_gets_no_slot(client, monkeypatch):
    from calliope.agent.video_agent import enqueue_video_jobs

    pid, clip_id = _one_slot_clip(client)
    seen: dict = {}

    async def fake_rewrite(scene_, subjects, **kwargs):
        seen["subjects"] = subjects
        seen["setting"] = kwargs.get("setting")
        return "COMPILED"

    monkeypatch.setattr("calliope.agent.video_agent.ensure_continuity_plan", _fake_plan(clip_id))
    monkeypatch.setattr("calliope.agent.video_agent._h3_rewrite", fake_rewrite)

    jobs = asyncio.run(enqueue_video_jobs(pid, clip_ids=[clip_id]))
    assert len(jobs) == 1
    assert [s["kind"] for s in seen["subjects"]] == ["character"]
    assert seen["setting"] == SETTING
