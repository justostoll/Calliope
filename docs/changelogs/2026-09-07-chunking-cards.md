# Calliope 1.4.1 — chunked generation + working question cards

Two fixes to the agent pipeline, both aimed at long-running generations
stalling without feedback. **Updated 2026-09-11:** chunked writes are now
*durable* (every chunk commits to the DB as it lands), the Script stage
shows the real Scripts→Clips count, and the per-scene "Break Into Shots"
button breaks only that scene.

## Fixed

- **Draft Storyline / Regenerate Script no longer hang on long targets.**
  Both asked the LLM for the whole board in one response — a 50-beat
  storyline or a 20-scene script is a multi-thousand-token JSON answer that
  small/local models take minutes to stream, often under-deliver, and then a
  single short chunk retried the *entire* generation. Story beats are now
  drafted in chunks of 12 (a brief call with title/logline/cast, then
  continuations that see the established cast and recent beats); script
  scenes in chunks of 4, each fed the full context plus the last two written
  scenes for continuity. Both renumber their boards gaplessly, retry only a
  short chunk, and publish progress per chunk ("Writing beats 25–36
  (chunk 3/5)…"). Small projects keep the exact single-call behavior.
- **The agent's question cards actually appear.** When the agent asked you
  to confirm something (regenerate a script, replace content, render), the
  chat froze at "working…" with no question on screen — you had to press
  Stop or refresh to see anything. The turn now records the question before
  ending, and the card renders it with clickable options. Clicking an option
  or typing "yes" both count as the approval; "No, …" refuses.

## Updated 2026-09-11

- **Chunked writes are durable — story → script → clips all insert chunk by
  chunk.** The agent writes each generated chunk into SQLite the moment the
  LLM call returns, and commits before starting the next call:
  - *Story:* the brief (title/logline/cast) + its first 12 beats commit
    first; every continuation chunk of 12 beats commits on arrival. A crash
    or Stop mid-draft keeps every beat already written instead of losing the
    whole story in one failed transaction. The old board is only cleared
    once the first replacement chunk is in hand.
  - *Script:* each chunk of 4 scenes is persisted + committed immediately
    (with its default clip), so "Saved 8/20 scenes…" reflects real DB rows.
  - *Shot clips:* after the script finishes, the coverage pass breaks each
    scene into shots **one scene per commit** ("Breaking scene 6 into shots
    (2/14)…") — already-expanded scenes survive an interruption.
  Progress events now mirror what is actually on disk.
- **Scripts→Clips count fixed.** The Script stage header showed only scene
  counts ("4 scenes"), hiding the shot list the pipeline had produced. It
  now reads "4 scenes / 11 shot clips", summed from the scenes' clip rows.
- **"Break Into Shots" on a scene card breaks only that scene.** The
  per-scene button POSTed to the project-level endpoint, which ignored the
  scene id and re-ran the coverage pass for *every* scene — identical to
  "Break All Into Shots". The endpoint now honors `scene_ids`, so clicking
  it on scene #3 re-breaks scene #3 only (one LLM call, one scene's shots
  replaced). The header button remains the explicit whole-board action, and
  a regression test pins the contract.

## Notes

- The agent loop is unchanged otherwise — chunked calls are internal to the
  story/script generators, so the AI Canvas agent tools (`generate_story`,
  `generate_script`) benefit automatically.
- If a long generation still feels slow, the progress events now show which
  chunk the model is on — a stuck chunk means the LLM endpoint is the
  bottleneck, not Calliope.

## Upgrade

```bash
git pull
```

No new dependencies, no schema changes (existing boards are untouched —
regeneration just gets faster and can no longer lose committed chunks).
Restart the backend and the frontend dev server.
