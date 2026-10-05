"""Agent harness package.

Composition root: builds the shared ToolRegistry + SystemPromptService once,
loads all plugins, and installs the built-in policy hooks (destructive-action
guard). Modeled on the deepseek-harness plugin composition: the loop, the
orchestrator, and tests all program against these services, never against
individual plugins.
"""
from __future__ import annotations

import logging

from calliope.agent.harness.plugins import load_all as _load_plugins
from calliope.agent.harness.policy import (
    is_confirmation,
    is_render_request,
    user_allows_render,
)
from calliope.agent.harness.prompts import (
    SystemPromptService,
    register_builtin_sections,
)
from calliope.agent.harness.registry import (
    PreExecuteDecision,
    ToolContext,
    ToolDefinition,
    ToolRegistry,
    allow,
    deny,
)

logger = logging.getLogger("calliope.harness")

# Stable machine-readable refusal codes (consumed by the UI and tests — the
# human guidance text stays in the deny message for the model).
GUARD_RENDER_APPROVAL = "guard_render_approval"
GUARD_DESTRUCTIVE_REPLACE = "guard_destructive_replace"
# Agent shell (run_command) denials — defined here so the contracts exporter
# (which scans this module + registry.py) picks them up.
GUARD_SHELL_DISABLED = "guard_shell_disabled"
GUARD_SHELL_APPROVAL = "guard_shell_approval"
GUARD_SHELL_OUTSIDE_WORKSPACE = "guard_shell_outside_workspace"
GUARD_SHELL_DENIED_PATH = "guard_shell_denied_path"
GUARD_SHELL_DENIED_COMMAND = "guard_shell_denied_command"
# Scene tool scope guard lives in registry.py (circular-import free); the
# contracts exporter reads guard codes from THIS module, so re-export here.
from calliope.agent.harness.registry import (  # noqa: E402
    GUARD_SCENE_TOOL_SCOPE,
    GUARD_NO_EXPORT_VIDEO_TOOL,
    GUARD_WORKSPACE_PATH,
    EXPORT_VIDEO_DENIED_NAMES,
    _scene_scoped,
)

# Re-exports for tests / older imports.
_is_confirmation = is_confirmation
_is_render_request = is_render_request


def _user_confirmed_replacement(ctx: ToolContext) -> bool:
    """True when the user's latest input explicitly asks for / confirms a
    destructive replace — a scoped question-card answer or prose (derived
    from the event log — no separate state).

    A card click is ALSO a user/message, and it grants only what its scope
    says: an `info` or `render` card answered "Yes — generate the story" is
    not a replace confirmation, whatever its wording (observed live
    2026-09-29: that answer let a second generate_story replace=true wipe
    the story the first call had just drafted)."""
    from calliope.agent.harness import log as session_log
    from calliope.agent.harness.policy import has_structured_approval, latest_answer

    if has_structured_approval(ctx, "destructive_replace"):
        return True
    if latest_answer(ctx.session_id) is not None:
        return False
    return is_confirmation(session_log.latest_user_message(ctx.session_id) or "")


def _latest_user_message_mark(session_id: int) -> tuple[int, str | None] | None:
    """(seq, created_at) of the session's latest user/message, or None."""
    from calliope.agent.harness import log as session_log
    from calliope.agent.harness.registry import _db

    conn = _db()
    try:
        row = conn.execute(
            "SELECT seq, created_at FROM agent_events WHERE session_id = ? AND type = ? "
            "ORDER BY seq DESC LIMIT 1",
            (session_id, session_log.USER_MESSAGE),
        ).fetchone()
    finally:
        conn.close()
    return (int(row["seq"]), row["created_at"]) if row else None


def _rows_confirmed(
    ctx: ToolContext,
    tables: tuple[str, ...],
    row_id: int | None = None,
    *,
    any_age: bool = False,
) -> int:
    """Rows of `tables` in the project that already existed when the user sent
    their latest message — the only content a confirmation in it can cover.
    One confirmation therefore buys one replacement: afterwards everything is
    newer than the message. Same-second rows count as existing (an agent's
    content is always an LLM step newer); NULL created_at counts as existing.
    any_age=True counts every row, whatever its age."""
    from calliope.agent.harness.registry import _db

    mark = None if any_age else _latest_user_message_mark(ctx.session_id)
    since = mark[1] if mark else None
    conn = _db()
    try:
        n = 0
        for table in tables:
            sql = f"SELECT COUNT(*) AS n FROM {table} WHERE project_id = ?"
            params: list = [ctx.project_id]
            if row_id is not None:
                sql += " AND id = ?"
                params.append(row_id)
            if since is not None:
                sql += " AND (created_at IS NULL OR datetime(created_at) <= datetime(?))"
                params.append(since)
            n += conn.execute(sql, params).fetchone()["n"]
        return n
    finally:
        conn.close()


def _render_approval_guard(ctx: ToolContext, t: ToolDefinition, args: dict) -> PreExecuteDecision:
    """Human-in-the-loop guard for image/video generation.

    Rendering (enqueue_asset_jobs / enqueue_video_jobs / run_workflow)
    queues real, expensive, side-effecting work. It must never run unless
    the user's latest message explicitly asks for generation, or tersely confirms
    the agent's offer. Text edits (add_item, add_character, add_location,
    update_scene, generate_story/script, …) do NOT grant render permission —
    this is what previously let the assets-agent auto-generate images after a
    "create a Misc. Item" request.
    """
    if not t.requires_approval:
        return allow()
    if user_allows_render(ctx):
        return allow()
    return deny(
        "Image/video generation is human-in-the-loop and needs explicit user approval. "
        "The user's latest message does not ask to render. Complete the requested text "
        "edits, then ASK whether they want images/videos generated — do not enqueue "
        "until they confirm (e.g. 'yes, generate the images').",
        code=GUARD_RENDER_APPROVAL,
    )


# Row-by-row tools that can reassemble a full replace without ever touching a
# `replace` parameter. Observed live 2026-09-07 (session 157): asked to redo the
# storyline, the agent called delete_beat 20 times and add_beat 8 times and
# never met _destructive_guard. The Nth such call in ONE unconfirmed turn is
# refused; the first N-1 stay free for genuine targeted edits.
# tool -> (table, id argument).
BULK_ROW_DELETE_TOOLS = {"delete_beat": ("story_beats", "beat_id")}
BULK_ROW_DELETES_PER_TURN = 3

# An update that cuts a long text field to a stub is a wipe, not an edit
# (observed live 2026-09-29: a story-agent wrote "PROBE-KEPT" over a 222-char
# beat to test its own tools). tool -> (table, id argument, field). A wipe
# needs the same confirmation as a delete; a real rewrite, even a one-line
# one, stays free.
ROW_WIPE_TOOLS = {"update_beat": ("story_beats", "beat_id", "description")}
WIPE_MAX_CHARS = 40
WIPE_KEEP_RATIO = 0.25

# What each replace tool deletes (generate_story: routers/story.py;
# generate_script: script_agent.py) — the rows a confirmation must cover.
# Other replace tools: all content tables.
REPLACE_TOOL_TABLES = {
    "generate_story": ("story_beats", "characters", "locations", "items"),
    "generate_script": ("scenes",),
}
# What gates a replace tool at all. A new story orphans an existing script,
# so generate_story is gated by any content; generate_script only by scenes —
# the story is its input, not what it replaces.
GUARDED_TABLES = {"generate_script": ("scenes",)}
_CONTENT_TABLES = ("scenes", "story_beats", "characters", "locations", "items")


def _row_deletes_this_turn(session_id: int, tool_name: str) -> int:
    """tool/call events for `tool_name` since the user's latest message.

    That is the whole "turn": the loop's steps AND an orchestrated plan,
    whose sub-agent tool/calls carry no `turn` field (matching on the turn
    number counted them as 0, so a story-agent's 8 delete_beat calls in one
    plan never met the cap — observed live 2026-09-29). The loop and the
    orchestrator log tool/call BEFORE pre-execute hooks run, so the count
    includes the call being guarded (and any earlier denied ones, which also
    stay counted — a refused delete is still an attempt at bulk deletion)."""
    from calliope.agent.harness import log as session_log

    mark = _latest_user_message_mark(session_id)
    after = mark[0] if mark else -1
    n = 0
    for ev in session_log.read_events(session_id):
        if ev.seq <= after or ev.type != session_log.TOOL_CALL:
            continue
        if ev.data.get("tool_name") == tool_name:
            n += 1
    return n


def _row_id(args: dict, key: str) -> int | None:
    try:
        return int(args[key])
    except (KeyError, TypeError, ValueError):
        return None


def _row_exists(ctx: ToolContext, table: str, row_id: int) -> bool:
    from calliope.agent.harness.registry import _db

    conn = _db()
    try:
        return conn.execute(
            f"SELECT 1 FROM {table} WHERE id = ? AND project_id = ?", (row_id, ctx.project_id)
        ).fetchone() is not None
    finally:
        conn.close()


def _bulk_row_delete_guard(ctx: ToolContext, t: ToolDefinition, args: dict) -> PreExecuteDecision:
    if ctx.project_id is None:
        return allow()
    n = _row_deletes_this_turn(ctx.session_id, t.name)
    if n < BULK_ROW_DELETES_PER_TURN:
        return allow()
    if _user_confirmed_replacement(ctx):
        table, id_arg = BULK_ROW_DELETE_TOOLS[t.name]
        row_id = _row_id(args, id_arg)
        if row_id is None or not _row_exists(ctx, table, row_id):
            return allow()  # nothing to destroy; the executor reports it
        if _rows_confirmed(ctx, (table,), row_id):
            return allow()
        return deny(
            f"This is {t.name} call #{n} since the user's last message, and row #{row_id} was "
            "created after that message — the user has never seen it, so their confirmation "
            "does not cover deleting it row by row. Stop deleting; show them the current "
            "content and ask with ask_user(scope=\"destructive_replace\").",
            code=GUARD_DESTRUCTIVE_REPLACE,
        )
    return deny(
        f"This is {t.name} call #{n} since the user's last message — deleting the board row "
        "by row is a replace, and the user has not asked for one. Stop deleting. Either ask "
        "the user with ask_user(scope=\"destructive_replace\") and wait for the answer, or — "
        "once they have confirmed (e.g. 'yes, replace it') — call generate_story / "
        "generate_script with replace=true, which rewrites the board in one step.",
        code=GUARD_DESTRUCTIVE_REPLACE,
    )


def _row_wipe_guard(ctx: ToolContext, t: ToolDefinition, args: dict) -> PreExecuteDecision:
    table, id_arg, field = ROW_WIPE_TOOLS[t.name]
    new = args.get(field)
    row_id = _row_id(args, id_arg)
    if ctx.project_id is None or new is None or row_id is None:
        return allow()
    from calliope.agent.harness.registry import _db

    conn = _db()
    try:
        row = conn.execute(
            f"SELECT {field} AS v FROM {table} WHERE id = ? AND project_id = ?",
            (row_id, ctx.project_id),
        ).fetchone()
    finally:
        conn.close()
    old = str(row["v"] or "") if row else ""
    kept = len(str(new).strip())
    if kept >= WIPE_MAX_CHARS or kept >= WIPE_KEEP_RATIO * len(old):
        return allow()
    if _user_confirmed_replacement(ctx) and _rows_confirmed(ctx, (table,), row_id):
        return allow()
    return deny(
        f"{t.name} would cut row #{row_id}'s {field} from {len(old)} to {kept} characters — "
        "that wipes it, and the user has not asked for it. Never overwrite content with "
        "placeholder or test text. If the user wants it removed or shortened this far, ask "
        "with ask_user(scope=\"destructive_replace\") first.",
        code=GUARD_DESTRUCTIVE_REPLACE,
    )


def _destructive_guard(ctx: ToolContext, t: ToolDefinition, args: dict) -> PreExecuteDecision:
    """Block silent destructive *regeneration* unless the user asked for it.

    Scope: tools with a `replace` parameter (generate_story / generate_script)
    — "replace=true (default) deletes everything first". A destructive tool
    WITHOUT a replace param (delete_scene, comfy lifecycle tools) acts on one
    explicit target the model named, so it is not blocked here.

    A non-empty project is protected unless the user's latest message is an
    explicit confirmation / regeneration request (e.g. "yes, replace it").
    That turns "ask the user first" from a dead-end into a real confirm step.
    The confirmation covers only content that existed when the user gave it.
    """
    if t.name in ROW_WIPE_TOOLS:
        return _row_wipe_guard(ctx, t, args)
    if not t.destructive:
        return allow()
    if t.name in BULK_ROW_DELETE_TOOLS:
        return _bulk_row_delete_guard(ctx, t, args)
    has_replace_param = "replace" in t.parameters.get("properties", {})
    if not has_replace_param:
        return allow()
    replace = bool(args.get("replace", True))
    # BOTH modes are gated on a non-empty project: replace=true deletes the
    # existing content, and replace=false APPENDS a second full set alongside
    # it (observed live 2026-08-25: an unconfirmed replace=false generate_story
    # silently doubled a project's beats, 25 -> 50). Only allow when the
    # project is effectively empty, or the user explicitly confirmed.
    from calliope.agent.harness.registry import _db

    if ctx.project_id is None:
        return allow()
    conn = _db()
    try:
        scenes = conn.execute(
            "SELECT COUNT(*) AS n FROM scenes WHERE project_id = ?", (ctx.project_id,)
        ).fetchone()["n"]
        beats = conn.execute(
            "SELECT COUNT(*) AS n FROM story_beats WHERE project_id = ?", (ctx.project_id,)
        ).fetchone()["n"]
        chars = conn.execute(
            "SELECT COUNT(*) AS n FROM characters WHERE project_id = ?", (ctx.project_id,)
        ).fetchone()["n"]
        locs = conn.execute(
            "SELECT COUNT(*) AS n FROM locations WHERE project_id = ?", (ctx.project_id,)
        ).fetchone()["n"]
        items = conn.execute(
            "SELECT COUNT(*) AS n FROM items WHERE project_id = ?", (ctx.project_id,)
        ).fetchone()["n"]
    finally:
        conn.close()
    has_content = (scenes + beats + chars + locs + items) > 0
    if not has_content:
        return allow()
    # The first generate_script on a drafted story replaces nothing, so it is
    # not gated; 12ca9c0 refused it even after a yes (observed live
    # 2026-09-29) because it gated on beats and then found no covered scene.
    if not _rows_confirmed(ctx, GUARDED_TABLES.get(t.name, _CONTENT_TABLES), any_age=True):
        return allow()
    confirmed = _user_confirmed_replacement(ctx)
    counts = (
        f"Project #{ctx.project_id} already has content ({scenes} scenes, {beats} beats, "
        f"{chars} characters, {locs} locations, {items} items)."
    )
    if replace:
        deleted = REPLACE_TOOL_TABLES.get(t.name, _CONTENT_TABLES)
        if confirmed and (
            not _rows_confirmed(ctx, deleted, any_age=True) or _rows_confirmed(ctx, deleted)
        ):
            return allow()
        if confirmed:
            return deny(
                counts + f" Everything {t.name} would replace was created AFTER the "
                "user's last message, so the user has not seen it and their earlier "
                "yes does not cover it (one confirmation covers one replacement). "
                "Show them what is there now and ask again with "
                "ask_user(scope=\"destructive_replace\"); for targeted changes use "
                "the granular tools (add/update/delete) instead.",
                code=GUARD_DESTRUCTIVE_REPLACE,
            )
        return deny(
            counts + " replace=true DELETES all of it first, and the user's "
            "latest message did not ask for a replacement. Ask them once "
            "(ask_user(scope=\"destructive_replace\") or plain text: 'replace "
            "the existing content?'); after an affirmative reply to that "
            "question, retry replace=true — it will be allowed. Do not retry "
            "replace=true again before they answer; do not call get_workspace "
            "repeatedly instead. For targeted changes use the granular tools "
            "(add/update/delete) instead.",
            code=GUARD_DESTRUCTIVE_REPLACE,
        )
    if confirmed:
        return allow()
    return deny(
        counts + " replace=false would APPEND a second full set alongside "
        "the existing content (e.g. doubling the beat list). Ask the user "
        "whether to replace or append (ask_user(scope=\"destructive_replace\") "
        "or plain text); after an affirmative reply retry — it will be "
        "allowed. Do not retry either mode again before they answer. For "
        "targeted changes use the granular tools (add/update/delete) instead.",
        code=GUARD_DESTRUCTIVE_REPLACE,
    )


_BUILT = False
_registry: ToolRegistry | None = None
_prompts: SystemPromptService | None = None


def build_harness() -> tuple[ToolRegistry, SystemPromptService]:
    """Compose the harness (idempotent). Returns (registry, prompts)."""
    global _BUILT, _registry, _prompts
    if _BUILT:
        assert _registry is not None and _prompts is not None
        return _registry, _prompts
    registry = ToolRegistry()
    prompts = SystemPromptService()
    _load_plugins()
    from calliope.agent.harness.plugins import (  # noqa: E402
        canvas,
        files,
        interaction,
        memory,
        render,
        script,
        shot_builder,
        skills,
        story,
        system,
        workspace,
    )

    workspace.register(registry)
    story.register(registry)
    script.register(registry)
    render.register(registry)
    canvas.register(registry)
    interaction.register(registry)
    memory.register(registry)
    skills.register(registry)
    shot_builder.register(registry)
    system.register(registry)
    files.register(registry)
    registry.on_pre_execute(_destructive_guard)
    registry.on_pre_execute(_render_approval_guard)
    register_builtin_sections(prompts)
    _registry, _prompts = registry, prompts
    _BUILT = True
    logger.info("Harness composed: %d tools", len(registry.tools))
    return registry, prompts


def get_registry() -> ToolRegistry:
    registry, _ = build_harness()
    return registry


def get_prompts() -> SystemPromptService:
    _, prompts = build_harness()
    return prompts
