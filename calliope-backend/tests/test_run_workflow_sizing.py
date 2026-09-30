"""run_workflow's sizing guidance names sizes the render lanes deliver as asked.

The agent takes its sizes from this tool's description. It used to say 16:9 → 1280×720 / 1920×1080 and
1:1 → 1080×1080. The AI1 lanes round each side to a multiple of 32 (half-to-even), so 1280×720 rendered as
1280×704 while the agent reported 1280×720 (Calliope job 642, 2026-09-30). Every size the description
recommends is a multiple of 64, which every lane keeps as is. Each one is also exactly what AI1 already
delivered for the old size, so an AI1 render does not change; only the size the agent states does.
"""
from __future__ import annotations

import asyncio
import re

import pytest

from calliope.agent.harness import build_harness

_PAIR_RE = re.compile(r"(\d{3,4})×(\d{3,4})")


@pytest.fixture(autouse=True)
def _scratch_db(monkeypatch, tmp_path):
    import calliope.config as config_module

    monkeypatch.setattr(config_module.settings, "data_dir", tmp_path)
    from calliope.db import migrate_db

    asyncio.run(migrate_db(tmp_path / "calliope.db"))
    yield


def _tool():
    registry, _ = build_harness()
    tool = registry.get("run_workflow")
    assert tool is not None
    return tool


def _ai1_round(v: int, step: int = 32) -> int:
    """The AI1 appliance's rounding for an in-range side (comfyui_mcp_server.py `_snap_one_dim`)."""
    return step * round(v / step)


def test_every_recommended_size_is_a_multiple_of_64():
    pairs = [(int(w), int(h)) for w, h in _PAIR_RE.findall(_tool().description)]
    assert pairs, "the description recommends no W×H size"
    off = [(w, h) for w, h in pairs if w % 64 or h % 64]
    assert not off, f"sizes a lane would round: {off}"


def test_the_old_off_grid_sizes_are_gone():
    desc = _tool().description
    for old in ("1280×720", "1920×1080", "1080×1080", "720×1280", "1080×1920"):
        assert old not in desc


def test_the_table_still_covers_16_9_9_16_and_1_1():
    desc = _tool().description
    assert "16:9" in desc and "9:16 swaps those" in desc and "1:1 →" in desc
    wide = [(int(w), int(h)) for w, h in re.findall(r"(\d{3,4})×(\d{3,4}) \((?:1080|720)p", desc)]
    assert sorted(wide) == [(1280, 704), (1920, 1088)]
    for w, h in wide:
        assert abs(w / h - 16 / 9) / (16 / 9) < 0.025, (w, h)
    square = re.search(r"1:1 → (\d{3,4})×(\d{3,4})", desc)
    assert square and square.group(1) == square.group(2)


def test_each_new_size_is_what_ai1_already_delivered_for_the_old_one():
    # the old recommendation -> what AI1's /32 lanes rendered for it == the new recommendation
    old_to_new = {(1280, 720): (1280, 704), (1920, 1080): (1920, 1088), (1080, 1080): (1088, 1088)}
    desc = _tool().description
    for (ow, oh), (nw, nh) in old_to_new.items():
        assert (_ai1_round(ow), _ai1_round(oh)) == (nw, nh)
        assert f"{nw}×{nh}" in desc


def test_width_and_height_parameters_say_multiple_of_64():
    props = _tool().parameters["properties"]
    for key in ("width", "height"):
        assert "multiple of 64" in props[key]["description"], key
