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
from tools.exec_code_policy import (
    _execute_code_has_sensitive_write,
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
# Re-review (2026-08-25, andrexibiza) Blocker 1: sensitive-write invariant
# ─────────────────────────────────────────────────────────────────────

SENSITIVE_WRITE_CASES = [
    # literal protected targets
    'open("/root/.hermes/config.yaml", "a").write("injected")',
    'open("/root/.ssh/authorized_keys", "a").write("key")',
    'with open("/root/.hermes/config.yaml", "a") as f:\n    f.write("injected")',
    # expanduser forms (the #49578 reproducer shape)
    'import os\ntarget = os.path.expanduser("~/.hermes/config.yaml")\nwith open(target, "a") as f:\n    f.write("injected")',
    'import os\nopen(os.path.expanduser("~/.hermes/config.yaml"), "a").write("x")',
    # pathlib to protected targets
    'from pathlib import Path\nPath("/root/.ssh/authorized_keys").write_text("key")',
    'from pathlib import Path\nwith Path("/etc/passwd").open("a") as f:\n    f.write("x")',
    # simple variable alias
    'target = "/root/.hermes/config.yaml"\nopen(target, "a").write("x")',
]


@pytest.mark.parametrize("code", SENSITIVE_WRITE_CASES)
def test_sensitive_write_target_detected(code):
    assert _execute_code_has_sensitive_write(code) is not None


@pytest.mark.parametrize("code", [
    'open("/tmp/x.txt", "w").write("x")',
    'open("/home/user/project/a.py", "w").write("x")',
    'open("/root/.hermes/config.yaml", "r").read()',
    'from pathlib import Path\nPath("/root/.hermes/config.yaml").read_text()',
])
def test_non_sensitive_write_passes(code):
    assert _execute_code_has_sensitive_write(code) is None


@pytest.mark.parametrize("mode_gate", ["normal", "yolo", "off"])
def test_sensitive_write_hard_blocked_in_all_modes(monkeypatch, mode_gate):
    """The #49578 destination invariant must hold even when approval is
    turned off — the sensitive-write check runs before the yolo/mode-off
    bypass gates (re-review Blocker 1)."""
    code = ('import os\ntarget = os.path.expanduser("~/.hermes/config.yaml")\n'
            'with open(target, "a") as f:\n    f.write("injected")')

    import tools.approval as approval_module
    if mode_gate == "yolo":
        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", True)
    elif mode_gate == "off":
        monkeypatch.setattr(
            approval_module, "_get_approval_mode", lambda: "off"
        )
    # normal: nothing to patch — env_type="local" goes through the guard

    result = check_execute_code_guard(code, env_type="local")
    assert result["approved"] is False
    assert result["outcome"] == "hard_blocked"
    assert "protected path" in result["message"]


# ─────────────────────────────────────────────────────────────────────
# Re-review (2026-08-25) Blocker 2: eval/exec dynamic-exec detection
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("code", [
    'import os\neval("os.kill")(os.getpid(), 15)',
    'exec("os.kill(os.getpid(), 15)")',
    'import os\ncompile("os.kill(1,9)", "<s>", "exec")',
])
def test_eval_exec_dynamic_code_flagged(code):
    """eval("os.kill")(...) escapes the hard-block resolver (outer callee
    is eval) but must not fall through to auto-approve — it is flagged as
    dynamic-exec and requires approval."""
    assert _execute_code_has_dangerous_ops(code) == "dynamic-exec"


def test_eval_exec_guard_not_auto_approved(monkeypatch):
    """dynamic-exec must fall through to the approval prompt rather than
    the danger_reason-is-None auto-approve path."""
    import tools.approval as approval_module
    # Force CLI-like path: not gateway, not ask, not yolo, not off.
    monkeypatch.setattr(approval_module, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_is_single_query_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_is_cron_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")
    result = check_execute_code_guard(
        'import os\neval("os.kill")(os.getpid(), 15)', env_type="local"
    )
    # It must NOT be auto-approved; it should be blocked/ask for approval.
    assert result["approved"] is False or result.get("outcome") in (
        "blocked", "denied", "pending",
    )


# ─────────────────────────────────────────────────────────────────────
# Conversation-loop user-denial halt (plain-text + JSON-wrapped BLOCKED)
# ─────────────────────────────────────────────────────────────────────

from agent.conversation_loop import (
    _tool_results_contain_user_blocked,
    _user_blocked_halt_response,
)


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


# ─────────────────────────────────────────────────────────────────────
# Loop-boundary control flow: exit reason + no-retry semantics
# (2026-08-25 re-review: parser recognition alone does not prove
# termination semantics)
# ─────────────────────────────────────────────────────────────────────

class _FakeAgent:
    """Minimal agent stub with the halt side-effect surface."""

    def __init__(self):
        self.emitted = []
        self.printed = []
        self.streamed = []
        self.stream_delta_callback = self._stream

    def _emit_status(self, text):
        self.emitted.append(text)

    def _safe_print(self, text):
        self.printed.append(text)

    def _stream(self, text):
        self.streamed.append(text)


def test_user_blocked_halt_sets_exit_reason_and_appends_response():
    agent = _FakeAgent()
    messages = _tool_msgs('{"status": "error", "error": "BLOCKED: User denied"}')
    result = _user_blocked_halt_response(agent, messages)

    assert result == ("user_blocked", "操作被拒绝。请指示下一步。")
    # Side effects: status emitted, response appended, stream flushed.
    assert agent.emitted and "用户拒绝了危险操作" in agent.emitted[0]
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == "操作被拒绝。请指示下一步。"
    assert agent.printed and "操作被拒绝" in agent.printed[0]


def test_user_blocked_halt_returns_none_without_denial():
    agent = _FakeAgent()
    messages = _tool_msgs("command output ok")
    assert _user_blocked_halt_response(agent, messages) is None
    # No side effects when there is no denial.
    assert agent.emitted == []
    assert agent.printed == []
    assert messages[-1]["role"] == "tool"


# ─────────────────────────────────────────────────────────────────────
# Builtin 别名绕过回归（2026-08-26 复现发现）
# ``op = open`` / ``e = eval`` / ``from builtins import open as op`` 曾绕过
# open/eval/exec 检测分支（只匹配裸名字 func.id），在 CLI 交互模式下
# auto-approve；``op = open`` + expanduser 函数别名组合还击穿了 #49578
# 敏感写不变量。修复：_resolve_alias_value 将 builtin 名解析为
# ("builtins", name)，open/eval/exec 检测统一走 _resolve_call_target。
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("code,reason", [
    # open 赋值别名 → 写模式必须触发 open-write
    ('op = open\nwith op("/tmp/t.txt", "w") as f:\n    f.write("x")', "open-write"),
    # 链式赋值别名 g = f; f = open
    ('f = open\ng = f\nwith g("/tmp/t.txt", "w") as fh:\n    fh.write("x")', "open-write"),
    # builtins import 别名
    ('from builtins import open as op\nop("/tmp/t.txt", "w").write("x")', "open-write"),
    # eval/exec 赋值别名 → dynamic-exec
    ('e = eval\ne("os.kill")(1, 15)', "dynamic-exec"),
    ('x = exec\nx("import os; os.kill(1, 15)")', "dynamic-exec"),
])
def test_builtin_alias_forms_flagged(code, reason):
    """builtin 别名（open/eval/exec）必须被危险扫描识别（曾绕过）。"""
    assert _execute_code_has_dangerous_ops(code) == reason


@pytest.mark.parametrize("code", [
    # 只读别名不误报
    'op = open\nwith op("data.txt", "r") as f:\n    print(f.read())',
    # 非危险 builtin 别名不误报
    'p = print\np("hi")',
    'l = len\nprint(l([1, 2, 3]))',
])
def test_builtin_alias_benign_passes(code):
    """无害 builtin 别名不得触发危险扫描。"""
    assert _execute_code_has_dangerous_ops(code) is None


def test_builtin_open_alias_not_auto_approved(monkeypatch):
    """op = open 写文件必须落到审批弹窗，不能 auto-approve（2026-08-26 漏洞）。"""
    import tools.approval as approval_module
    monkeypatch.setattr(approval_module, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_is_single_query_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_is_cron_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")
    result = check_execute_code_guard(
        "op = open\nwith op('/tmp/t.txt', 'w') as f:\n    f.write('x')",
        env_type="local",
    )
    assert result["approved"] is False or result.get("outcome") in (
        "blocked", "denied", "pending",
    )


@pytest.mark.parametrize("code", [
    # op = open + expanduser 变量目标（#49578 原始形状的别名版）
    ('import os\nop = open\ntarget = os.path.expanduser("~/.hermes/config.yaml")\n'
     'with op(target, "a") as f:\n    f.write("x")'),
    # op = open + expanduser 函数别名组合（曾完整击穿不变量）
    ('import os\nop = open\nh = os.path.expanduser\n'
     'with op(h("~/.hermes/config.yaml"), "a") as f:\n    f.write("x")'),
    # builtins import 别名 + 敏感目标
    ('from builtins import open as op\n'
     'op("~/.ssh/authorized_keys", "a").write("x")'),
])
def test_builtin_open_alias_sensitive_write_hard_blocked(code):
    """builtin 别名写敏感目标必须命中 #49578 不变量（曾 auto-approved）。"""
    assert _execute_code_has_sensitive_write(code) is not None


def test_expanduser_function_alias_sensitive_write():
    """h = os.path.expanduser 函数引用别名写敏感目标必须硬阻断（曾退化为审批）。"""
    code = ('import os\nh = os.path.expanduser\n'
            'with open(h("~/.hermes/config.yaml"), "a") as f:\n    f.write("x")')
    assert _execute_code_has_sensitive_write(code) is not None
