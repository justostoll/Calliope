"""Regression tests for render-intent detection (policy.is_render_request).

The failure that motivated this: "all 20 scenes use this text2video workflow.
Do not overthinking, no image need." read as NON-render (compound word
"text2video" unmatched + a message-global negation veto from "Do not"/"no"),
so the registry hid enqueue_video_jobs/run_workflow, and the agent — unable
to say why — fabricated job ids instead of admitting the tools were hidden.
"""
from __future__ import annotations

import pytest

from calliope.agent.harness.policy import _COMPOUND_SPLIT_RE, is_render_request


@pytest.mark.parametrize(
    ("text", "want"),
    [
        # The exact messages from the canvas/62 incident
        ("all 20 scenes use this text2video workflow. Do not overthinking, no image need.", True),
        ("you are not doing the job go away", False),
        # Compound / model-name cues
        ("make a text2video clip of the establishing shot", True),
        ("txt2vid for scene 3 please", True),
        ("use fastvideoH3_t2v-API for all 20 scenes", True),
        ("image-to-video on scene 2", True),
        # Negation must stay authoritative when it governs the render cue
        ("do not generate videos", False),
        ("no images please", False),
        ("don't render anything", False),
        ("no render, just fix the script", False),
        ("i said do not render", False),
        # …but a negated clause must not veto an unrelated render clause
        ("fix the script but render afterwards", True),
        ("no image need, render all scenes", True),
        # Plain positive cues
        ("render all scenes now", True),
        ("yes, generate the images", True),
        # Clip/shot are render-unit nouns (session 908: "shot clips" left the
        # video sub-agent's enqueue tools hidden → "generation tool is not
        # exposed")
        ("Break the beats into the 8-scene script + shot clips", True),
        ("break scene 3 into clips", True),
        ("render the shots for scene 2", True),
        # Non-render text must not flip positive
        ("generate the story", False),
        ("provide the lyrics", False),
        # 'clip' as a noun of speech ("clip the beard") is not a cue — but the
        # verb is rare in film asks; keep the broad match and rely on
        # clause-scoped negation for safety.
    ],
)
def test_render_intent(text: str, want: bool):
    assert is_render_request(text) is want, text


# The compound splitter split "to" between ANY two letters: "storyboard" became
# "s ryboard" (the cue could never match) and "stop" became "s p" (the negation
# vanished, so "stop rendering" read as a render request).
@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("turn scene 3 into a storyboard", True),
        ("storyboard the opening", True),
        ("make a storyboard", True),
        ("generate storyboards for all scenes", True),
        ("stop rendering", False),
        ("please stop generating the videos", False),
        ("stop generating images", False),
        # the storyboard cue obeys negation like every other cue
        ("don't make a storyboard", False),
        ("no storyboard, just the script", False),
        # "stop motion" is an animation style, not a stop
        ("make a stop-motion clip of scene 2", True),
        ("stop motion video for the opening", True),
        # "to" / "2" still join a compound before a media noun
        ("textToImage for the poster", True),
        ("ref2image for Ana", True),
    ],
)
def test_compound_splitter_keeps_plain_words(text: str, want: bool):
    assert is_render_request(text) is want, text


def test_compound_splitter_splits_only_compounds():
    for word in ("storyboard", "stop", "custom", "history", "photos"):
        assert _COMPOUND_SPLIT_RE.sub(" ", word) == word
    assert _COMPOUND_SPLIT_RE.sub(" ", "text2video") == "text video"
    assert _COMPOUND_SPLIT_RE.sub(" ", "textToImage") == "text Image"
    assert _COMPOUND_SPLIT_RE.sub(" ", "image-to-video") == "image to video"
