"""Regression tests for the execute_code approval guard (#65592).

Pins the helper contracts added by PR #65592 and extended after the
2026-08-25 andrexibiza review:

1. Sensitive file writes (#49578): builtin ``open()`` write modes and
   pathlib ``Path.write_text`` / ``write_bytes`` / ``Path.open`` must be
   flagged; read-only forms must pass.
2. Process-kill hard block: direct, import-aliased, assignment-aliased
   (incl. chains), star-imported, ``getattr`` and ``__dict__`` dynamic
   forms must be caught; ordinary os usage must pass.
3. Conversation-loop user-denial halt: plain-text and JSON-wrapped
   BLOCKED tool results must be recognized.

The denial tests fail on the pre-review implementation (which skipped
star imports, assignment aliases, and pathlib writes).
"""

import pytest

from tools.approval import (
    _execute_code_has_dangerous_ops,
    _execute_code_has_self_destructive_ops,
    check_execute_code_guard,
)


# ─────────────────────────────────────────────────────────────────────
# Blocker 1: #49578 sensitive file writes (open + pathlib)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("code", [
    # raw open() write modes (the #49578 reproducer shape)
    'open("/path/.hermes/config.yaml", "a").write("injected")',
    'open("/path/.hermes/config.yaml", "w").write("injected")',
    'open("/path/x", "x").write("injected")',
    'open("/path/x", "r+").write("injected")',
    'with open("/path/.hermes/config.yaml", "a") as f:\n    f.write("injected")',
    # open() with non-literal mode must fail closed (conservative)
    'mode = "a"\nopen("/path/x", mode).write("x")',
    # pathlib write surfaces (review: not recognized before)
    'from pathlib import Path\nPath("/path/.hermes/config.yaml").write_text("injected")',
    'from pathlib import Path\nPath("/path/.hermes/config.yaml").write_bytes(b"injected")',
    'from pathlib import Path\nwith Path("/path/.hermes/config.yaml").open("a") as f:\n    f.write("x")',
    'import pathlib\npathlib.Path("/path/x").write_text("x")',
    # aliased pathlib constructor
    'from pathlib import Path as P\nP("/path/x").write_text("x")',
])
def test_sensitive_file_write_is_flagged(code):
    assert _execute_code_has_dangerous_ops(code) == "open-write"


@pytest.mark.parametrize("code", [
    'open("/path/x", "r").read()',
    'open("/path/x").read()',  # default mode is read
    'with open("/path/x", "rb") as f:\n    data = f.read()',
    'from pathlib import Path\nPath("/path/x").read_text()',
    'from pathlib import Path\nwith Path("/path/x").open() as f:\n    print(f.read())',
    'from pathlib import Path\nwith Path("/path/x").open("r") as f:\n    print(f.read())',
])
def test_read_only_file_access_passes(code):
    assert _execute_code_has_dangerous_ops(code) is None


# ─────────────────────────────────────────────────────────────────────
# Blocker 2: process-kill hard block — alias / dynamic forms
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("code", [
    # direct (baseline)
    'import os\nos.kill(os.getpid(), 15)',
    'import os\nos.killpg(0, 9)',
    # import aliases (baseline from July review)
    'import os as o\no.kill(os.getpid(), 15)',
    'from os import kill\nkill(os.getpid(), 15)',
    # assignment alias (review: not caught before)
    'import os\nkiller = os.kill\nkiller(os.getpid(), 15)',
    'import os\nk = os.killpg\nk(0, 9)',
    # chained assignment alias
    'import os\na = os\nb = a.kill\nb(os.getpid(), 15)',
    # star import (review: skipped before)
    'from os import *\nkill(os.getpid(), 15)',
    # getattr dynamic (review-adjacent)
    'import os\ngetattr(os, "kill")(os.getpid(), 15)',
    'import os\ngetattr(os, "killpg")(0, 9)',
    # __dict__ access
    'import os\nos.__dict__["kill"](os.getpid(), 15)',
])
def test_process_kill_forms_are_hard_blocked(code):
    assert _execute_code_has_self_destructive_ops(code) is not None


@pytest.mark.parametrize("code", [
    'import os\nprint(os.getpid())',
    'import os\nprint(os.environ.get("HOME", ""))',
    'import os\np = os.path.join("/tmp", "x")\nprint(p)',
    'import os\nprint(getattr(os, "environ").get("HOME", ""))',
    'import sys\nprint(sys.version)',
    'import math\nprint(math.sqrt(16))',
    'import json\nprint(json.dumps({"a": 1}))',
    'import subprocess\nprint("no call")',  # import alone is not a call
])
def test_benign_os_sys_usage_passes(code):
    assert _execute_code_has_self_destructive_ops(code) is None
    assert _execute_code_has_dangerous_ops(code) is None


# ─────────────────────────────────────────────────────────────────────
# Dangerous ops: alias / star / exec* forms
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("code,reason", [
    ('import subprocess as sp\nsp.run(["rm", "-rf", "/"], capture_output=True)', "command-exec"),
    ('import subprocess as sp\nrun = sp.run\nrun(["ls"])', "command-exec"),
    ('from shutil import *\nrmtree("/important")', "file-delete"),
    ('import os\nos.execv("/bin/sh", ["sh"])', "command-exec"),
    ('import os\nos.system("rm -rf /")', "command-exec"),
    ('from os import remove\nremove("/path/x")', "file-delete"),
])
def test_dangerous_call_aliases_are_flagged(code, reason):
    assert _execute_code_has_dangerous_ops(code) == reason


# ─────────────────────────────────────────────────────────────────────
# Guard end-to-end: hard block returns outcome=hard_blocked
# ─────────────────────────────────────────────────────────────────────

def test_guard_returns_hard_blocked_outcome():
    result = check_execute_code_guard(
        'import os\nos.kill(os.getpid(), 15)', env_type="local"
    )
    assert result["approved"] is False
    assert result["outcome"] == "hard_blocked"
    assert "HARD BLOCKED" in result["message"]


def test_guard_hard_blocks_alias_after_yolo_check():
    # Even with yolo, hard block must fire (it runs before the mode gate).
    result = check_execute_code_guard(
        'import os\nkiller = os.kill\nkiller(os.getpid(), 15)', env_type="local"
    )
    assert result["approved"] is False
    assert result["outcome"] == "hard_blocked"


# ─────────────────────────────────────────────────────────────────────
# Conversation-loop user-denial halt (plain-text + JSON-wrapped BLOCKED)
# ─────────────────────────────────────────────────────────────────────

from agent.conversation_loop import _tool_results_contain_user_blocked


def _tool_msgs(*contents):
    """Build trailing tool messages (all role=tool) around a user ask."""
    messages = [{"role": "user", "content": "do it"}]
    for c in contents:
        messages.append({"role": "tool", "content": c})
    return messages


def test_plain_text_blocked_detected():
    assert _tool_results_contain_user_blocked(
        _tool_msgs("BLOCKED: User denied dangerous command")
    ) is True


def test_bracket_blocked_list_detected():
    assert _tool_results_contain_user_blocked(
        _tool_msgs('["BLOCKED: User denied", "more context"]')
    ) is True


def test_json_wrapped_blocked_detected():
    # execute_code denial format: {"status":"error","error":"BLOCKED: ..."}
    assert _tool_results_contain_user_blocked(
        _tool_msgs('{"status": "error", "error": "BLOCKED: User denied"}')
    ) is True


def test_non_blocked_tool_result_passes():
    assert _tool_results_contain_user_blocked(
        _tool_msgs("command output ok", '{"status": "error", "error": "boom"}')
    ) is False


def test_only_trailing_tool_messages_scanned():
    # A BLOCKED in an OLD tool message (followed by a newer non-BLOCKED
    # tool message) must not halt the turn — only the trailing batch is
    # the current tool execution's output.
    messages = [
        {"role": "user", "content": "go"},
        {"role": "tool", "content": "BLOCKED: old denial"},
        {"role": "assistant", "content": "trying again"},
        {"role": "tool", "content": "all good"},
    ]
    assert _tool_results_contain_user_blocked(messages) is False
