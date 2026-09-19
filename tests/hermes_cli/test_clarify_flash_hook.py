"""Clarify → approval-observer bridge (local patch 2026-09-19).

A clarify prompt *is* the agent waiting on a human, but the clarify path emitted no hook at all,
so surfaces that key off ``pre_approval_request`` / ``post_approval_response`` (a taskbar flash
signal) never covered it. These tests pin the wiring:

* the payload the observer receives (pattern_key/keys, surface, session_key, choice on post)
* single question: exactly one pre, one post on every exit (answer / timeout / exception)
* batch: one pre for the panel, and the **first locked answer** stops it — not the last one
* the stop is idempotent: later locks and the backstop never re-fire it
"""

import queue

import pytest
from unittest.mock import MagicMock, patch

from cli import HermesCLI


def _stub():
    cli = HermesCLI.__new__(HermesCLI)
    cli._clarify_state = None
    cli._clarify_freetext = False
    cli._clarify_multi_base = None
    cli._clarify_prefill = ""
    cli._clarify_deadline = None
    cli._clarify_timeout_window = None
    cli._clarify_paused = False
    cli._paint_now = MagicMock()
    cli._persist_prompt_summary = MagicMock()
    cli._ring_bell = MagicMock()
    cli._clarify_flash = MagicMock()
    return cli


def _flashes(cli):
    """[(phase, choice)] in call order; choice is None for the pre call."""
    out = []
    for call in cli._clarify_flash.call_args_list:
        phase = call.args[0] if call.args else call.kwargs.get("phase")
        choice = call.args[2] if len(call.args) > 2 else call.kwargs.get("choice")
        out.append((phase, choice))
    return out


def _q(index, question, choices=None, multi_select=False):
    return {
        "qid": f"q{index}",
        "id": None,
        "question": question,
        "choices": list(choices) if choices else None,
        "choices_offered": list(choices) if choices else None,
        "multi_select": bool(multi_select) and bool(choices),
    }


def _batch_state(cli, questions):
    state = {
        "questions": list(questions),
        "answers": {},
        "answer_meta": {},
        "active": 0,
        "response_queue": queue.Queue(),
        "flash_stopped": False,
        "question": "",
        "choices": [],
        "selected": 0,
        "multi_select": False,
        "selected_indices": None,
    }
    cli._clarify_batch_set_active = MagicMock()
    return state


# --- the payload handed to the approval observers -------------------------------

def test_payload_reaches_the_approval_observer():
    from tools.clarify_tool import fire_clarify_hook

    seen = []
    with patch("tools.approval_context._fire_approval_hook",
               side_effect=lambda name, **kw: seen.append((name, kw))):
        fire_clarify_hook("pre", "要不要做这件事？")
        fire_clarify_hook("post", "要不要做这件事？", "timeout")

    assert seen[0][0] == "pre_approval_request"
    kw = seen[0][1]
    assert kw["pattern_key"] == "clarify"
    assert kw["pattern_keys"] == ["clarify"]
    assert kw["surface"] == "cli"
    assert isinstance(kw["session_key"], str)
    assert kw["command"].startswith("clarify: 要不要做这件事")
    assert "choice" not in kw

    assert seen[1][0] == "post_approval_response"
    assert seen[1][1]["choice"] == "timeout"


def test_long_question_is_truncated_in_the_command_field():
    from tools.clarify_tool import fire_clarify_hook

    seen = []
    with patch("tools.approval_context._fire_approval_hook",
               side_effect=lambda name, **kw: seen.append((name, kw))):
        fire_clarify_hook("pre", "长" * 500)
    assert len(seen[0][1]["command"]) == len("clarify: ") + 80


def test_observer_failure_never_breaks_the_question():
    from tools.clarify_tool import fire_clarify_hook

    with patch("tools.approval_context._fire_approval_hook", side_effect=RuntimeError("boom")):
        fire_clarify_hook("pre", "坏了也别影响提问")  # must not raise


# --- single question: one pre, one post on every exit ---------------------------

def test_single_question_stops_after_the_answer():
    cli = _stub()
    cli._poll_modal_queue = MagicMock(return_value="做")
    out = cli._clarify_callback("要不要做？", ["做", "不做"])
    assert out == "做"  # this layer returns the raw answer; clarify_tool wraps it as JSON
    assert _flashes(cli) == [("pre", None), ("post", "answered")]


def test_single_question_stops_on_timeout():
    from hermes_cli.cli_modal_mixin import _TIMED_OUT, _CLARIFY_TIMEOUT_REPLY

    cli = _stub()
    cli._clarify_teardown = MagicMock()
    cli._poll_modal_queue = MagicMock(return_value=_TIMED_OUT)
    out = cli._clarify_callback("要不要做？", None)
    assert out == _CLARIFY_TIMEOUT_REPLY
    assert _flashes(cli) == [("pre", None), ("post", "timeout")]


def test_single_question_stops_when_the_ui_raises():
    cli = _stub()
    cli._poll_modal_queue = MagicMock(side_effect=RuntimeError("ui died"))
    with pytest.raises(RuntimeError):
        cli._clarify_callback("要不要做？", None)
    assert _flashes(cli) == [("pre", None), ("post", "timeout")]


# --- batch: one pre, first locked answer stops it -------------------------------

def test_batch_panel_fires_one_pre_and_a_backstop_post():
    cli = _stub()
    cli._poll_modal_queue = MagicMock(return_value={"q1": "a"})
    out = cli._clarify_callback("", None, questions=[_q(1, "问题一", ["a", "b"])])
    assert out == {"answers": {"q1": "a"}}
    assert _flashes(cli) == [("pre", None), ("post", "answered")]


def test_batch_first_locked_answer_stops_the_flash_and_later_locks_do_not_refire():
    cli = _stub()
    state = _batch_state(cli, [_q(1, "问题一", ["a", "b"]), _q(2, "问题二", ["c", "d"]), _q(3, "问题三")])

    cli._clarify_batch_lock(state, "a", meta={"kind": "choice"})
    assert state["flash_stopped"] is True
    assert _flashes(cli) == [("post", "answered")], "first locked answer must stop the signal"

    # Remaining questions are answered (including a freetext one) → still exactly one stop.
    state["active"] = 1
    cli._clarify_batch_lock(state, "c", meta={"kind": "choice"})
    state["active"] = 2
    cli._clarify_batch_lock(state, "自由文本", meta={"kind": "other"})
    assert _flashes(cli) == [("post", "answered")]


def test_batch_backstop_does_not_refire_after_the_lock_stopped_it():
    cli = _stub()
    questions = [_q(1, "问题一", ["a", "b"]), _q(2, "问题二", ["c", "d"])]
    state = _batch_state(cli, questions)

    def _answer_first_then_finish(_queue, _deadline_attr):
        # The panel blocks here in real life; simulate the user locking 问题一 和 问题二.
        # Read the state the callback installed (in real code that IS cli._clarify_state).
        live = cli._clarify_state
        cli._clarify_batch_lock(live, "a", meta={"kind": "choice"})
        live["active"] = 1
        cli._clarify_batch_lock(live, "c", meta={"kind": "choice"})
        return dict(live["answers"])

    cli._poll_modal_queue = MagicMock(side_effect=_answer_first_then_finish)
    out = cli._clarify_callback("", None, questions=questions)
    assert out == {"answers": {"q1": "a", "q2": "c"}}
    assert _flashes(cli) == [("pre", None), ("post", "answered")], "no duplicate stop from the backstop"
