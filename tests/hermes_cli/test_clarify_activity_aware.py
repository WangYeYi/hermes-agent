"""Activity-aware clarify countdown — every "Other" entry point.

Local patch 2026-09-16 added the activity awareness; the 2026-09-19 fix closed a gap in it. The
rule the user asked for: the CLI keeps a SHORT window (clarify.timeout_cli, 60s) so an empty
keyboard frees the turn quickly, but the moment the user is TYPING the clock must stop racing
them — deadline = now + max(window, CLARIFY_PAUSE_GRACE_SECONDS) = 600s.

Entries that put the panel into freetext: the number key for the "Other" row, typing without
picking anything, and ↑/↓ + Enter on the "Other" row. The third one was never wired, and
_tui_clarify_any_key() returns early once _clarify_freetext is set, so the keystrokes that
followed picking "Other" could not pause it either. These tests pin all three.

Re-applied 2026-10-07 on the post-#18450 CLI shape: the panel is batch-only, so "↑/↓ + Enter on
Other" now funnels through _clarify_batch_enter(), which pauses on both of its freetext paths
(selected == len(choices), and multi-select with the Other row checked). The state these tests
drive is therefore the batch state the panel really uses.
"""

import queue
import time
import types

from cli import HermesCLI
from hermes_cli.cli_modal_mixin import CLARIFY_PAUSE_GRACE_SECONDS
from unittest.mock import MagicMock

SHORT_WINDOW = 60  # what clarify.timeout_cli is set to on this host


def _stub(window=SHORT_WINDOW):
    cli = HermesCLI.__new__(HermesCLI)
    cli._clarify_timeout_window = window
    cli._clarify_paused = False
    cli._clarify_freetext = False
    cli._clarify_prefill = ""
    cli._clarify_multi_base = None
    cli._clarify_deadline = time.monotonic() + window
    cli._clarify_state = None
    cli._persist_prompt_summary = MagicMock()
    cli._clarify_flash = MagicMock()
    cli._ring_bell = MagicMock()
    cli._paint_now = MagicMock()
    return cli


def _event(text=""):
    buf = types.SimpleNamespace(text=text, cursor_position=0, reset=MagicMock())  # real prompt_toolkit Buffer has reset()
    return types.SimpleNamespace(app=types.SimpleNamespace(invalidate=MagicMock(), current_buffer=buf))


def _remaining(cli):
    return round(cli._clarify_deadline - time.monotonic())


def _state(**kw):
    """The batch state the panel installs: one active question + its flat mirrors."""
    entry = {"qid": "q0", "id": None, "question": "问题一", "choices": ["a", "b"],
             "choices_offered": ["a", "b"], "multi_select": False}
    base = {
        "questions": [entry],
        "answers": {},
        "answer_meta": {},
        "active": 0,
        "response_queue": queue.Queue(),
        "flash_stopped": False,
        "selected": 0,
        "choices": ["a", "b"],
        "multi_select": False,
        "selected_indices": None,
    }
    base.update(kw)
    return base


def test_other_row_via_arrow_enter_pauses_the_countdown():
    """The entry that was missing: ↑/↓ to "Other" (index == len(choices)) then Enter."""
    cli = _stub()
    cli._clarify_state = _state(selected=2)

    cli._tui_enter_clarify_choice(_event())

    assert cli._clarify_freetext is True
    assert cli._clarify_paused is True
    assert _remaining(cli) == CLARIFY_PAUSE_GRACE_SECONDS == 600


def test_multi_select_checking_other_pauses_the_countdown():
    cli = _stub()
    cli._clarify_state = _state(multi_select=True, selected_indices={2})  # only "Other" checked

    cli._tui_enter_clarify_choice(_event())

    assert cli._clarify_freetext is True and cli._clarify_paused is True
    assert _remaining(cli) == 600


def test_real_choice_neither_pauses_nor_stops_the_clock():
    """Control: picking a real option submits and must NOT pause (clock keeps its short window)."""
    cli = _stub()
    q = queue.Queue()
    cli._clarify_state = _state(selected=0, response_queue=q)

    cli._tui_enter_clarify_choice(_event())

    assert cli._clarify_freetext is False and cli._clarify_paused is False
    assert _remaining(cli) <= SHORT_WINDOW
    assert q.get_nowait() == {"q0": "a"}  # the panel resolves the batch by qid


def test_number_key_other_entry_still_pauses():
    """Regression guard for the entry the 2026-09-16 patch did wire up."""
    cli = _stub()
    cli._clarify_state = _state()

    cli._tui_make_clarify_number_handler(2)(_event())  # index 2 == len(choices) → the "Other" row

    assert cli._clarify_freetext is True and cli._clarify_paused is True
    assert _remaining(cli) == 600


def test_typing_without_picking_other_still_pauses():
    """Regression guard for the any-key entry (and it must keep working after the fix)."""
    cli = _stub()
    cli._clarify_state = _state()

    cli._tui_clarify_any_key(types.SimpleNamespace(data="做", app=types.SimpleNamespace(invalidate=MagicMock())))

    assert cli._clarify_freetext is True and cli._clarify_paused is True
    assert _remaining(cli) == 600


def test_submitting_a_typed_answer_resumes_the_short_window():
    """Control: after the answer is submitted the clock must go back to the short window."""
    cli = _stub()
    cli._clarify_pause_deadline()
    cli._clarify_state = _state()
    ev = _event(text="自定义答案")

    cli._tui_enter_clarify_freetext(ev)

    assert cli._clarify_paused is False
    assert _remaining(cli) <= SHORT_WINDOW
