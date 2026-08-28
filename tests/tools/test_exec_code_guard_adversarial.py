"""Adversarial + completeness tests for the execute_code approval guard (#65592).

Designed from the FULL review thread of PR #65592 (teknium1 sweeper review,
three andrexibiza review rounds, and the author's own bypass re-tests), then
extended by 举一反三 — inferring new attack shapes from the classes the
reviewers already named:

  Review claim                          →  New shapes derived
  -------------------------------------   ------------------------------------
  "assignment aliases incl. chains"      →  walrus (:=), tuple-unpack, for-target,
                                            dict/list subscript, vars(),
                                            __dict__.get(), globals(), sys.modules
  "getattr / __dict__ dynamic access"    →  getattr keyword form, __import__ chain,
                                            string-concat fold ("ki"+"ll")
  "os.kill/os.killpg never allowed"      →  signal.kill / signal.pthread_kill /
                                            psutil.kill / psutil.Process().kill() /
                                            functools.partial — same capability
  "#49578 destination invariant"         →  expandvars / f-string / join /
                                            double-slash targets (static, not runtime)
  "BLOCKED plain + JSON-wrapped"         →  leading-whitespace JSON, list/dict
                                            content blocks, every deny-arm contract

Suite layout:
  Section A — functional completeness: gate ordering, env/mode matrix, call
              families, mode edges, target normalization, library writers,
              approval gates, deny-message contracts, classifier, logging.
  Section B — 2026-08-25 复测发现的缺陷面（已修复，普通断言）：绑定图、
              kill 等价物、敏感写静态目标、BLOCKED 格式鲁棒性。
  Section C — 无法静态修复的残余面（XFAIL 标注，属于 sandbox/运行时边界）：
              函数/λ 间接调用、非字面量 for 可迭代、动态 f-string 插值。
  Section D — benign no-false-positive controls for every new capability.

All Section B tests were empirically verified to FAIL on head 5902589454
(auto-approve in local CLI) and PASS on the fix head — they pin the fixes.

Run:  PYTHONPATH=<repo> python3 -m pytest tests/tools/test_exec_code_guard_adversarial.py -v
"""

import pytest

from tools.approval import (
    _execute_code_has_dangerous_ops,
    _execute_code_has_self_destructive_ops,
    check_execute_code_guard,
)
from tools.exec_code_policy import (
    _execute_code_has_sensitive_write,
    _execute_code_touches_sensitive_path,
    _classify_exec_code_imports,
    _log_blocked_exec_code,
)
from agent.conversation_loop import _tool_results_contain_user_blocked


# ═════════════════════════════════════════════════════════════════════════
# Section A1 — Gate ordering (orchestration seam, tools/approval.py:5222)
# ═════════════════════════════════════════════════════════════════════════

def test_hard_block_priority_over_sensitive_write():
    """kill + sensitive write in one script → kill reason wins (checked first)."""
    code = ('import os\nos.kill(os.getpid(), 15)\n'
            'open("/root/.hermes/config.yaml", "a").write("x")')
    result = check_execute_code_guard(code, env_type="local")
    assert result["outcome"] == "hard_blocked"
    assert "process-killing" in result["message"]


def test_sensitive_write_priority_over_open_write():
    """literal sensitive write → hard_blocked, NOT the recoverable open-write prompt."""
    result = check_execute_code_guard(
        'open("/root/.hermes/config.yaml", "w").write("x")', env_type="local"
    )
    assert result["approved"] is False
    assert result["outcome"] == "hard_blocked"
    assert "protected path" in result["message"]


def test_eval_kill_is_dynamic_exec_not_hard_blocked(monkeypatch):
    """eval(\"os.kill\") is dynamic-exec (prompt), NOT hard_blocked and NOT
    auto-approved — per scope honesty."""
    import tools.approval as approval_module
    monkeypatch.setattr(approval_module, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_is_single_query_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_is_cron_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")
    result = check_execute_code_guard(
        'import os\neval("os.kill")(os.getpid(), 15)', env_type="local"
    )
    assert result.get("outcome") != "hard_blocked"
    assert result["approved"] is False  # reaches the approval surface, not auto-approve


def test_hard_block_fires_before_container_skip():
    """Hard block precedes the container-skip gate — os.kill is blocked even in
    sandboxed env_types where benign scripts auto-skip."""
    for env_type in ("docker", "vercel_sandbox", "modal"):
        result = check_execute_code_guard(
            "import os\nos.kill(os.getpid(), 15)", env_type=env_type
        )
        assert result["outcome"] == "hard_blocked", env_type


# ═════════════════════════════════════════════════════════════════════════
# Section A2 — env_type / mode matrix
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("env_type", ["docker", "vercel_sandbox", "local", "ssh"])
def test_container_skip_matrix_benign(env_type):
    """Benign scripts in container envs skip the guard; local runs the scan
    (benign → danger_reason None → approved anyway)."""
    result = check_execute_code_guard("print('hello')", env_type=env_type)
    assert result["approved"] is True


def test_docker_with_host_access_not_skipped(monkeypatch):
    """docker + has_host_access=True must NOT skip — host paths are reachable.
    NOTE: use monkeypatch here — a bare module write would leak the frozen
    yolo flag into every later test in the file."""
    import tools.approval as approval_module
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", True)
    result = check_execute_code_guard(
        "import os\nos.kill(os.getpid(), 15)", env_type="docker", has_host_access=True
    )
    assert result["outcome"] == "hard_blocked"


@pytest.mark.parametrize("mode_gate", ["yolo", "off"])
def test_ordinary_dangerous_op_approved_under_trust_gate(monkeypatch, mode_gate):
    """Yolo / approvals=off deliberately approve ORDINARY dangerous ops
    (subprocess, os.remove) — that is the documented trust boundary. Only
    hard-block classes (kill / sensitive-write) are non-negotiable."""
    import tools.approval as approval_module
    if mode_gate == "yolo":
        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", True)
    else:
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "off")
    result = check_execute_code_guard(
        'import subprocess\nsubprocess.run(["ls"])', env_type="local"
    )
    assert result["approved"] is True


@pytest.mark.parametrize("mode_gate", ["yolo", "off"])
def test_kill_never_overridden_by_trust_gate(monkeypatch, mode_gate):
    import tools.approval as approval_module
    if mode_gate == "yolo":
        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", True)
    else:
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "off")
    result = check_execute_code_guard(
        "import os\nos.kill(os.getpid(), 15)", env_type="local"
    )
    assert result["outcome"] == "hard_blocked"


def test_session_approval_respected(monkeypatch):
    """A session-approved execute_code key skips the per-script prompt (after the
    hard blocks) — the #39275 gate is consulted."""
    import tools.approval as approval_module
    monkeypatch.setattr(approval_module, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_is_single_query_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_is_cron_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval_module, "is_approved", lambda *a, **k: True)
    result = check_execute_code_guard(
        'import os\nos.remove("/tmp/x")', env_type="local"
    )
    assert result["approved"] is True


def test_smart_approve_verdict(monkeypatch):
    """smart mode APPROVE → approved with smart_approved flag; no prompt."""
    import tools.approval as approval_module
    monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(approval_module, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_is_single_query_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_is_cron_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_smart_approve", lambda c, d: "approve")
    monkeypatch.setattr(approval_module, "_prepare_smart_approval_observer", lambda **k: {})
    monkeypatch.setattr(approval_module, "_observe_smart_approval_verdict", lambda *a, **k: None)
    result = check_execute_code_guard(
        'import os\nos.remove("/tmp/x")', env_type="local"
    )
    assert result["approved"] is True
    assert result.get("smart_approved") is True


def test_smart_deny_verdict_blocked(monkeypatch):
    """smart mode DENY in a non-gateway context → hard denial, no retry."""
    import tools.approval as approval_module
    monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(approval_module, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_is_single_query_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_is_cron_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_smart_approve", lambda c, d: "deny")
    monkeypatch.setattr(approval_module, "_prepare_smart_approval_observer", lambda **k: {})
    monkeypatch.setattr(approval_module, "_observe_smart_approval_verdict", lambda *a, **k: None)
    result = check_execute_code_guard(
        'import os\nos.remove("/tmp/x")', env_type="local"
    )
    assert result["approved"] is False
    assert "BLOCKED by smart approval" in result["message"]
    assert "Do NOT retry" in result["message"]


def test_single_query_deny_mode(monkeypatch):
    """-q mode with deny policy → blocked (escape hatch no longer auto-approves)."""
    import tools.approval as approval_module
    monkeypatch.setattr(approval_module, "_is_single_query_approval_context", lambda: True)
    monkeypatch.setattr(approval_module, "_get_single_query_approval_mode", lambda: "deny")
    result = check_execute_code_guard("print('hi')", env_type="local")
    assert result["approved"] is False
    assert result["outcome"] == "blocked"


def test_gateway_deny_via_transport(monkeypatch):
    """Gateway ask-mode: transport deny → BLOCKED denied outcome."""
    import tools.approval as approval_module
    monkeypatch.setattr(approval_module, "_is_gateway_approval_context", lambda: True)
    monkeypatch.setattr(approval_module, "_present_with_selected_transport",
                       lambda **k: {"selected": True, "choice": "deny"})
    result = check_execute_code_guard(
        'import os\nos.remove("/tmp/x")', env_type="local"
    )
    assert result["approved"] is False
    assert result["outcome"] == "denied"
    assert result["message"].startswith("BLOCKED")


def test_gateway_deny_via_decision(monkeypatch):
    """Gateway ask-mode: _await_gateway_decision deny → BLOCKED denied outcome."""
    import tools.approval as approval_module
    monkeypatch.setattr(approval_module, "_is_gateway_approval_context", lambda: True)
    monkeypatch.setattr(approval_module, "_present_with_selected_transport",
                       lambda **k: {"selected": False})
    session_key = approval_module.get_current_session_key()
    monkeypatch.setattr(approval_module, "_gateway_notify_cbs", {session_key: object()})
    monkeypatch.setattr(approval_module, "_await_gateway_decision",
                       lambda *a, **k: {"resolved": True, "choice": "deny", "reason": "nope"})
    result = check_execute_code_guard(
        'import os\nos.remove("/tmp/x")', env_type="local"
    )
    assert result["approved"] is False
    assert result["outcome"] == "denied"
    assert result["message"].startswith("BLOCKED")


# ═════════════════════════════════════════════════════════════════════════
# Section A3 — dangerous-call families not yet pinned by the 134-case file
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("code,reason", [
    # file-delete family
    ('import os\nos.unlink("/path/x")', "file-delete"),
    # file-mutate family (config-write bypass family, #49578)
    ('import shutil\nshutil.copy("/a", "/b")', "file-mutate"),
    ('import shutil\nshutil.copy2("/a", "/b")', "file-mutate"),
    ('import shutil\nshutil.move("/a", "/b")', "file-mutate"),
    ('import shutil\nshutil.copytree("/a", "/b")', "file-mutate"),
    ('import os\nos.rename("/a", "/b")', "file-mutate"),
    ('import os\nos.replace("/a", "/b")', "file-mutate"),
    # command-exec family
    ('import subprocess\nsubprocess.Popen(["ls"])', "command-exec"),
    ('import subprocess\nsubprocess.check_output(["ls"])', "command-exec"),
    ('import subprocess\nsubprocess.check_call(["ls"])', "command-exec"),
    ('import subprocess\nsubprocess.call(["ls"])', "command-exec"),
    ('import os\nos.popen("ls")', "command-exec"),
    # exec* process-replace family (review-adjacent, added by analogy)
    ('import os\nos.execl("/bin/sh", "sh")', "command-exec"),
    ('import os\nos.execle("/bin/sh", "sh")', "command-exec"),
    ('import os\nos.execlp("sh", "sh")', "command-exec"),
    ('import os\nos.execlpe("sh", "sh")', "command-exec"),
    ('import os\nos.execve("/bin/sh", ["sh"], {})', "command-exec"),
    ('import os\nos.execvp("sh", ["sh"])', "command-exec"),
    ('import os\nos.execvpe("sh", ["sh"], {})', "command-exec"),
    ('import os\nos.posix_spawn("/bin/sh", ["sh"], {})', "command-exec"),
    ('import os\nos.posix_spawnp("sh", ["sh"], {})', "command-exec"),
    # aliased exec* (the PR fixed the alias resolver — pin it here)
    ('import os as o\no.execv("/bin/sh", ["sh"])', "command-exec"),
    ('from os import execl\nexecl("/bin/sh", "sh")', "command-exec"),
])
def test_dangerous_call_family_flagged(code, reason):
    assert _execute_code_has_dangerous_ops(code) == reason


@pytest.mark.parametrize("code", [
    "import ctypes",
    "import ctypes as c",
    "import ctypes\nctypes.CDLL(None).unlink('/tmp/x')",
    "from ctypes import CDLL",
])
def test_ctypes_whole_module_gate(code):
    """ctypes import alone triggers the gate (CDLL(None).unlink needs no os)."""
    assert _execute_code_has_dangerous_ops(code) == "ctypes-import"


# ═════════════════════════════════════════════════════════════════════════
# Section A4 — open()/Path.open() mode edges
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("code", [
    # keyword mode
    'open("/tmp/t", mode="w").write("x")',
    # positional mode with extra kwargs
    'open("/tmp/t", "w", encoding="utf-8").write("x")',
    # combined binary write
    'open("/tmp/t", "wb").write(b"x")',
    'open("/tmp/t", "w+b").write(b"x")',
    'open("/tmp/t", "r+").write("x")',
    # non-literal mode → fail closed (conservative)
    'm = "a"\nopen("/tmp/t", m).write("x")',
    'open("/tmp/t", "w" + "").write("x")',
    # pathlib keyword mode (Path.open(mode=...) — arg 0 differs from builtin)
    'from pathlib import Path\nPath("/tmp/t").open(mode="w").write("x")',
    'from pathlib import Path\nPath("/tmp/t").open("wb").write(b"x")',
])
def test_open_write_mode_edges_flagged(code):
    assert _execute_code_has_dangerous_ops(code) == "open-write"


@pytest.mark.parametrize("code", [
    'open("/tmp/t", "r").read()',
    'open("/tmp/t", "rb").read()',
    'from pathlib import Path\nPath("/tmp/t").open("r").read()',
    'from pathlib import Path\nPath("/tmp/t").open(mode="r").read()',
])
def test_open_read_mode_edges_pass(code):
    assert _execute_code_has_dangerous_ops(code) is None


# ═════════════════════════════════════════════════════════════════════════
# Section A5 — sensitive-target normalization (_write_target_is_sensitive)
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("code", [
    # tilde without expanduser (expanded at comparison time)
    'open("~/.ssh/authorized_keys", "w").write("x")',
    # $HOME literal (expandvars at comparison time)
    'open("$HOME/.ssh/authorized_keys", "w").write("x")',
    # dot-segments normalized by normpath
    'open("/root/.ssh/../.hermes/config.yaml", "w").write("x")',
    'open("/root/./.hermes/config.yaml", "w").write("x")',
    # mode as variable (conservative write) + literal sensitive target
    'm = "w"\nopen("/root/.hermes/config.yaml", m).write("x")',
    # r+ is still a write
    'open("/root/.ssh/authorized_keys", "r+").write("x")',
    # pathlib var object → sensitive target
    'from pathlib import Path\np = Path("~/.ssh/authorized_keys")\np.write_text("x")',
])
def test_sensitive_target_normalization_detected(code):
    assert _execute_code_has_sensitive_write(code) is not None


@pytest.mark.parametrize("code", [
    'open("/tmp/x", "w").write("x")',
    'open("/root/.ssh/authorized_keys", "r").read()',
    'from pathlib import Path\nPath("/root/.hermes/config.yaml").exists()',
])
def test_sensitive_normalization_controls_pass(code):
    assert _execute_code_has_sensitive_write(code) is None


# ═════════════════════════════════════════════════════════════════════════
# Section A6 — library writers (#49578 residual surface) — extra shapes
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("code", [
    # DataFrame stored in a variable (method name + path arg are what matter)
    'import pandas as pd\ndf = pd.DataFrame({"a": [1]})\ndf.to_csv("/root/.ssh/authorized_keys")',
    # keyword path on a writer with extra kwargs
    'import pandas as pd\npd.DataFrame({"a": [1]}).to_csv("/root/.ssh/x", index=False)',
    'import pandas as pd\npd.DataFrame({"a": [1]}).to_csv(path="/root/.ssh/x")',
    # non-dataframe writers with sensitive path args
    'import matplotlib.pyplot as plt\nplt.savefig("/root/.ssh/plot.png")',
    'import numpy as np\nnp.savez("/root/.hermes/x.npz", a=1)',
    # destructive file ops on sensitive paths → hard-blocked via touches_path
    'import shutil\nshutil.copy("/root/.ssh/id_rsa", "/tmp")',
    'import os\nos.remove("/root/.ssh/authorized_keys")',
    'import os\nos.rename("/root/.ssh/a", "/root/.ssh/b")',
])
def test_library_writer_extra_shapes_detected(code):
    assert _execute_code_touches_sensitive_path(code) is not None


@pytest.mark.parametrize("code", [
    'import pandas as pd\npd.read_csv("/root/.ssh/config.csv")',
    'import numpy as np\nnp.load("/root/.hermes/data.npy")',
    'import os\nos.path.exists("/root/.ssh")',
    'import os\nos.listdir("/root/.ssh")',
    'import pandas as pd\npd.DataFrame({"a": [1]}).to_csv("/tmp/ok.csv")',
])
def test_library_readonly_sensitive_passes(code):
    assert _execute_code_touches_sensitive_path(code) is None


# ═════════════════════════════════════════════════════════════════════════
# Section A7 — builtins.open family (resolved via ("builtins", name))
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("code", [
    'import builtins\nbuiltins.open("/tmp/t", "w").write("x")',
    'import builtins as b\nb.open("/tmp/t", "w").write("x")',
    'from builtins import open\nopen("/tmp/t", "w").write("x")',
    'from builtins import open as op\nop("/tmp/t", "w").write("x")',
    # star-import of builtins
    'from builtins import *\nopen("/tmp/t", "w").write("x")',
    # builtins alias → sensitive target (must hit the invariant)
    'import builtins\nbuiltins.open("/root/.hermes/config.yaml", "w").write("x")',
])
def test_builtins_open_family_flagged(code):
    assert _execute_code_has_dangerous_ops(code) == "open-write"


def test_builtins_open_sensitive_target_hard_blocked():
    result = check_execute_code_guard(
        'import builtins\nbuiltins.open("/root/.hermes/config.yaml", "w").write("x")',
        env_type="local",
    )
    assert result["outcome"] == "hard_blocked"


# ═════════════════════════════════════════════════════════════════════════
# Section A8 — deny-arm message contracts (every BLOCKED arm must halt the loop)
# Each of these is the REAL message a deny arm produces; the dispatch-layer
# halt must recognize every one, plain or JSON-wrapped.
# ═════════════════════════════════════════════════════════════════════════

BLOCKED_MESSAGE_CONTRACTS = [
    # execute_code CLI deny (tools/approval.py:5608)
    "BLOCKED: User denied execute_code script execution (matched 'execute_code "
    "script execution. The script can spawn subprocesses or mutate files'). "
    "Do NOT retry — the user has explicitly rejected it.",
    # execute_code CLI timeout (:5588)
    "BLOCKED: Action timed out without user response. The user has NOT "
    "consented to this action. Do NOT retry it, do NOT rephrase it, and do "
    "NOT attempt the same outcome via a different path. Silence is not consent.",
    # execute_code transport deny (:5516)
    "BLOCKED: User denied execute_code through the selected approval transport. "
    "The user has NOT consented.",
    # execute_code notify failure (:5686)
    "BLOCKED: Failed to send execute_code approval request to user. Do NOT retry.",
    # execute_code gateway deny/timeout (:5708)
    "BLOCKED: execute_code script denied by user. Reason given by the user: "
    '"no". The user has NOT consented to running this code. Do NOT retry, do '
    "NOT rephrase the script, and do NOT attempt the same outcome via a "
    "different tool.",
    # smart approval deny (:5456)
    "BLOCKED by smart approval: execute_code script execution was assessed as "
    "genuinely dangerous. Do NOT retry.",
    # terminal deny arms (check_all_command_guards)
    "BLOCKED: User denied this command. The user has NOT consented to running it.",
    "BLOCKED (hardline): this command is in the hardline blocklist.",
    "BLOCKED: Failed to send approval request to user. Do NOT retry.",
    "BLOCKED: Command flagged as dangerous (rm -rf). The user has NOT consented.",
    "BLOCKED: Command timed out without user response. The user has NOT consented.",
]

@pytest.mark.parametrize("message", BLOCKED_MESSAGE_CONTRACTS)
def test_every_deny_arm_message_halts(message):
    msgs = [{"role": "user", "content": "go"}, {"role": "tool", "content": message}]
    assert _tool_results_contain_user_blocked(msgs) is True


@pytest.mark.parametrize("message", BLOCKED_MESSAGE_CONTRACTS)
def test_every_deny_arm_message_halts_when_json_wrapped(message):
    """Same contracts as the execute_code tool result shape."""
    import json as _json
    wrapped = _json.dumps({"status": "error", "error": message})
    msgs = [{"role": "user", "content": "go"}, {"role": "tool", "content": wrapped}]
    assert _tool_results_contain_user_blocked(msgs) is True


def test_deny_message_with_breaker_addendum_still_halts():
    """_denial_breaker_addendum appends after the BLOCKED sentence — the
    leading prefix must still match."""
    msg = ("BLOCKED: User denied execute_code script execution (matched 'x'). "
           "Do NOT retry — the user has explicitly rejected it. "
           "Consecutive denials: 2/3 — the next denial will hard-stop this session.")
    msgs = [{"role": "user", "content": "go"}, {"role": "tool", "content": msg}]
    assert _tool_results_contain_user_blocked(msgs) is True


# ═════════════════════════════════════════════════════════════════════════
# Section A9 — import classifier (Layer 1 diagnostics)
# ═════════════════════════════════════════════════════════════════════════

def test_classifier_safe_imports():
    s, d, u = _classify_exec_code_imports("import json, math, re\nimport datetime")
    assert d == [] and u == []
    assert set(s) == {"json", "math", "re", "datetime"}


def test_classifier_dangerous_imports():
    s, d, u = _classify_exec_code_imports("import os, subprocess, ctypes, socket")
    assert s == [] and u == []
    assert set(d) == {"os", "subprocess", "ctypes", "socket"}


def test_classifier_unknown_third_party():
    s, d, u = _classify_exec_code_imports("import pandas as pd")
    assert s == [] and d == []
    assert u == ["pandas"]


@pytest.mark.parametrize("code,expect_top", [
    ("import os.path", "os"),
    ("import numpy as np, sys", "sys"),
    ("from pathlib import Path", "pathlib"),
    ("import pathlib as pl", "pathlib"),
    ("from os import remove", "os"),
    ("import a.b.c", "a"),
])
def test_classifier_alias_dotted_forms(code, expect_top):
    s, d, u = _classify_exec_code_imports(code)
    assert expect_top in (d or u or s)


def test_classifier_syntax_error():
    s, d, u = _classify_exec_code_imports("syntax error here !!!")
    assert s == [] and d == []
    assert "(syntax error)" in u


# ═════════════════════════════════════════════════════════════════════════
# Section A10 — logging: truncation + redaction safety
# ═════════════════════════════════════════════════════════════════════════

def test_block_log_truncates_long_scripts(caplog):
    long_code = 'x = "a" * 100\n' + "print('data')\n" * 1000  # ~13k chars
    with caplog.at_level("WARNING"):
        _log_blocked_exec_code(long_code, "open-write")
    assert len(caplog.records) == 1
    text = caplog.records[0].getMessage()
    assert "open-write" in text
    assert "... [truncated, " in text  # truncation marker present


def test_block_log_does_not_crash_on_short_script(caplog):
    with caplog.at_level("WARNING"):
        _log_blocked_exec_code("print(1)", "command-exec")
    assert len(caplog.records) == 1


# ═════════════════════════════════════════════════════════════════════════
# Section B — 2026-08-25 复测缺陷面（已修复，普通断言固定）
# 修复前（head 5902589454）全部实测 ESCAPE → CLI 本地 auto-approve；
# 修复后全部命中硬阻断/不变量。测试即修复的回归锁。
# ═════════════════════════════════════════════════════════════════════════

# ── B1. 硬阻断绑定图：普通 Python 绑定形式（修复：_collect_exec_code_bindings
#       扩展 walrus/元组解包/for 目标 + _resolve_binding_expr 表达式级解析）──

HARD_BLOCK_BINDING_FIXED = [
    ("walrus direct", "import os\n(k := os.kill)(os.getpid(), 15)"),
    ("walrus chained", "import os\n(k := os.kill)(1, 15)"),
    ("tuple unpacking", "import os\nk1, k2 = os.kill, os.killpg\nk1(os.getpid(), 15)"),
    ("dict subscript", "import os\nk = {'kill': os.kill}['kill']\nk(os.getpid(), 15)"),
    ("list subscript", "import os\nk = [os.kill][0]\nk(os.getpid(), 15)"),
    ("vars()", "import os\nvars(os)['kill'](os.getpid(), 15)"),
    ("__dict__.get", "import os\nos.__dict__.get('kill')(os.getpid(), 15)"),
    ("dict .get literal", "import os\n{'kill': os.kill}.get('kill')(os.getpid(), 15)"),
    ("__import__ chain", "__import__('os').kill(1, 15)"),
    ("__import__ RHS", "import os\nk = __import__('os')\nk.kill(os.getpid(), 15)"),
    ("globals()", "import os\nglobals()['os'].kill(os.getpid(), 15)"),
    ("sys.modules", "import sys\nsys.modules['os'].kill(os.getpid(), 15)"),
    ("getattr keyword", "import os\ngetattr(os, name='kill')(os.getpid(), 15)"),
    ("getattr object kw", "import os\ngetattr(object=os, name='kill')(os.getpid(), 15)"),
    ("for-loop target", "import os\nfor f in [os.kill]:\n    f(os.getpid(), 15)"),
    ("for-loop multi", "import os\nfor f in [os.kill, os.killpg]:\n    f(os.getpid(), 15)"),
    # 字符串常量折叠（"ki"+"ll" / name = "kill" 别名）
    ("str-concat __dict__", 'import os\nos.__dict__["ki" + "ll"](os.getpid(), 15)'),
    ("str-alias __dict__", 'import os\nname = "kill"\nos.__dict__[name](os.getpid(), 15)'),
    # partial 包装
    ("partial direct", "import os\nfrom functools import partial\npartial(os.kill, os.getpid(), 15)()"),
    ("partial alias", "import os\nfrom functools import partial as pt\nk = pt(os.kill, 1, 15)\nk()"),
]

@pytest.mark.parametrize("label,code", HARD_BLOCK_BINDING_FIXED,
                         ids=[l for l, _ in HARD_BLOCK_BINDING_FIXED])
def test_hard_block_binding_fixed(label, code):
    """2026-08-25 复测缺陷（修复前 auto-approve）：绑定形状必须命中硬阻断。"""
    assert _execute_code_has_self_destructive_ops(code) is not None


@pytest.mark.parametrize("code", [
    'import os\na = b = os.kill\na(os.getpid(), 15)',
    'import os\no = os\no.__dict__["kill"](os.getpid(), 15)',
    'import os\nk = {"kill": os.kill}["kill"]\nk(os.getpid(), 15)',
])
def test_hard_block_binding_fixed_guard_end_to_end(code):
    """guard 端到端：修复形状返回 hard_blocked（review 要求的回归通道）。"""
    result = check_execute_code_guard(code, env_type="local")
    assert result["approved"] is False
    assert result["outcome"] == "hard_blocked"


# ── B1b. 进程 kill 等价物（修复：_HARD_BLOCKED_CALLS 增补 signal/psutil +
#        psutil.Process() 实例方法链解析）──

@pytest.mark.parametrize("code", [
    "import signal\nsignal.kill(os.getpid(), 15)",
    "import signal\nsignal.pthread_kill(1, 15)",
    "import psutil\npsutil.kill(1, 15)",
    "import psutil\npsutil.Process(1).kill()",
], ids=["signal.kill", "signal.pthread_kill", "psutil.kill", "psutil.Process.kill"])
def test_process_kill_equivalent_fixed(code):
    """os.kill 等价物（signal/psutil）必须同样不可审批（修复前 auto-approve）。"""
    assert _execute_code_has_self_destructive_ops(code) is not None


@pytest.mark.parametrize("code", [
    "import signal\nsignal.kill(os.getpid(), 15)",
    "import psutil\npsutil.Process(1).kill()",
], ids=["signal.kill-guard", "psutil.Process.kill-guard"])
def test_process_kill_equivalent_guard_end_to_end(code):
    result = check_execute_code_guard(code, env_type="local")
    assert result["outcome"] == "hard_blocked"


# ── B2. #49578 不变量：静态可解析目标（修复：_resolve_expr_path 扩展
#       expandvars/join/JoinedStr + 双斜杠折叠）──

SENSITIVE_WRITE_STATIC_FIXED = [
    ("expandvars", 'import os\nopen(os.path.expandvars("$HOME/.ssh/authorized_keys"), "w").write("x")'),
    ("f-string literal", "open(f'/root/.hermes/config.yaml', 'w').write('x')"),
    ("f-string pathlib", "from pathlib import Path\nPath(f'/root/.hermes/config.yaml').write_text('x')"),
    ("f-string interpolated", 'import os\nhome = os.path.expanduser("~")\nopen(f"{home}/.ssh/authorized_keys", "w").write("x")'),
    ("join literals", 'import os\nopen(os.path.join("/root", ".hermes", "config.yaml"), "w").write("x")'),
    ("join expanduser", 'import os\nopen(os.path.join(os.path.expanduser("~"), ".ssh", "authorized_keys"), "w").write("x")'),
    ("double-slash", "open('//root/.hermes/config.yaml', 'w').write('x')"),
]

@pytest.mark.parametrize("label,code", SENSITIVE_WRITE_STATIC_FIXED,
                         ids=[l for l, _ in SENSITIVE_WRITE_STATIC_FIXED])
def test_sensitive_write_static_fixed_detected(label, code):
    """静态可解析的敏感目标必须命中不变量（修复前仅降级为可恢复审批）。"""
    assert _execute_code_has_sensitive_write(code) is not None


@pytest.mark.parametrize("label,code", SENSITIVE_WRITE_STATIC_FIXED,
                         ids=[l for l, _ in SENSITIVE_WRITE_STATIC_FIXED])
def test_sensitive_write_static_fixed_hard_blocked_in_yolo(monkeypatch, label, code):
    """yolo/approvals=off 下同样不可覆盖——不变量在信任门之前强制执行。"""
    import tools.approval as approval_module
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", True)
    result = check_execute_code_guard(code, env_type="local")
    assert result["outcome"] == "hard_blocked"
    assert "protected path" in result["message"]


# ── B3. BLOCKED 检测格式鲁棒性（修复：_extract_tool_content_text +
#        lstrip 前导空白容忍）──

@pytest.mark.parametrize("content", [
    '  {"status": "error", "error": "BLOCKED: denied"}',
    '\n{"status": "error", "error": "BLOCKED: denied"}',
    '{"status": "error", "error": ["BLOCKED: denied", "more"]}',
    [{"type": "text", "text": "BLOCKED: denied"}],
    {"error": "BLOCKED: denied"},
    {"status": "error", "error": {"message": "BLOCKED: nested"}},
], ids=["ws-prefix", "nl-prefix", "error-list", "content-list", "content-dict", "error-dict"])
def test_blocked_detection_robustness_fixed(content):
    """非精确 content 形状（前导空白/列表/字典）必须仍能触发 halt（修复前漏检）。"""
    msgs = [{"role": "user", "content": "go"}, {"role": "tool", "content": content}]
    assert _tool_results_contain_user_blocked(msgs) is True


def test_normal_output_containing_blocked_word_not_halted():
    """正常命令输出中出现 BLOCKED 字样（echo BLOCKED 等）不得误停——只扫前缀。"""
    msgs = [{"role": "user", "content": "go"},
            {"role": "tool", "content": "some output\nBLOCKED: is just text here"}]
    assert _tool_results_contain_user_blocked(msgs) is False


# ── B4. andrexibiza re-review（head ead8c83cc2，2026-08-25T10:47）——
#      Blocker 1 作用域/控制流盲 + Blocker 2 psutil 终止能力 +
#      Blocker 3 receiver-bound 变异 / 容器误报。修复前全部 auto-approve。──

# B4a. 嵌套作用域不得污染模块级绑定
NESTED_SCOPE_BINDING_FIXED = [
    ("nested import shadow",
     "import os\n\ndef shadow():\n    import math as os\n\nos.kill(os.getpid(), 15)\n"),
    ("nested import shadow guard-visible",
     "import os\n\ndef shadow():\n    import math as os\n\nos.kill(os.getpid(), 15)\n"),
]

@ pytest.mark.parametrize("label,code", NESTED_SCOPE_BINDING_FIXED,
                          ids=[l for l, _ in NESTED_SCOPE_BINDING_FIXED])
def test_nested_scope_import_does_not_pollute_module_binding(label, code):
    """函数体内 ``import math as os`` 不得覆盖模块级 ``import os``——
    顶层 os.kill 必须仍解析为 ('os','kill') 并硬阻断（修复前放行）。"""
    assert _execute_code_has_self_destructive_ops(code) is not None


def test_nested_scope_import_fixed_guard_end_to_end():
    result = check_execute_code_guard(
        "import os\n\ndef shadow():\n    import math as os\n\nos.kill(os.getpid(), 15)\n",
        env_type="local",
    )
    assert result["outcome"] == "hard_blocked"


# B4b. 死分支赋值不得覆盖可达的危险绑定
DEAD_BRANCH_BINDING_FIXED = [
    ("if False overwrite",
     "import os\nkiller = os.kill\nif False:\n    killer = print\nkiller(os.getpid(), 15)\n"),
    ("if False overwrite star",
     "import os\nfrom os import kill\nif False:\n    kill = print\nkill(os.getpid(), 15)\n"),
]

@ pytest.mark.parametrize("label,code", DEAD_BRANCH_BINDING_FIXED,
                          ids=[l for l, _ in DEAD_BRANCH_BINDING_FIXED])
def test_dead_branch_assignment_does_not_shadow_reachable_binding(label, code):
    """不可达分支的赋值（if False: killer = print）不得覆盖可达的
    killer = os.kill——调用点必须保留危险身份（修复前放行）。"""
    assert _execute_code_has_self_destructive_ops(code) is not None


def test_dead_branch_binding_fixed_guard_end_to_end():
    result = check_execute_code_guard(
        "import os\nkiller = os.kill\nif False:\n    killer = print\nkiller(os.getpid(), 15)\n",
        env_type="local",
    )
    assert result["outcome"] == "hard_blocked"


# B4c. psutil.Process 进程终止能力（terminate/send_signal/suspend）
PSUTIL_PROCESS_TERMINATE_FIXED = [
    ("terminate", "import os\nimport psutil\npsutil.Process(os.getppid()).terminate()\n"),
    ("send_signal", "import os\nimport signal\nimport psutil\npsutil.Process(os.getppid()).send_signal(signal.SIGKILL)\n"),
    ("suspend", "import os\nimport psutil\npsutil.Process(os.getppid()).suspend()\n"),
]

@ pytest.mark.parametrize("label,code", PSUTIL_PROCESS_TERMINATE_FIXED,
                          ids=[l for l, _ in PSUTIL_PROCESS_TERMINATE_FIXED])
def test_psutil_process_terminate_capability_hard_blocked(label, code):
    """按进程终止能力定义硬阻断，而非单个方法名 kill——terminate()/
    send_signal()/suspend() 可终止/冻结 Hermes 父进程（修复前放行）。"""
    assert _execute_code_has_self_destructive_ops(code) is not None


@ pytest.mark.parametrize("label,code", PSUTIL_PROCESS_TERMINATE_FIXED,
                          ids=[l + "-guard" for l, _ in PSUTIL_PROCESS_TERMINATE_FIXED])
def test_psutil_process_terminate_guard_end_to_end(label, code):
    result = check_execute_code_guard(code, env_type="local")
    assert result["outcome"] == "hard_blocked"


# B4d. #49578 receiver-bound 变异（目标在 receiver，无参数）
PATHLIB_RECEIVER_MUTATION_FIXED = [
    ("unlink", "from pathlib import Path\nPath('/root/.ssh/authorized_keys').unlink()\n"),
    ("touch", "from pathlib import Path\nPath('/root/.hermes/config.yaml').touch()\n"),
    ("chmod", "from pathlib import Path\nPath('/root/.hermes/config.yaml').chmod(0o644)\n"),
    ("chown", "from pathlib import Path\nPath('/root/.hermes/config.yaml').chown(0, 0)\n"),
    ("mkdir", "from pathlib import Path\nPath('/root/.ssh').mkdir()\n"),
    ("symlink_to", "from pathlib import Path\nPath('/root/.ssh/newkey').symlink_to('/root/.ssh/authorized_keys')\n"),
    ("rename receiver", "from pathlib import Path\nPath('/root/.ssh/authorized_keys').rename('/tmp/x')\n"),
]

@ pytest.mark.parametrize("label,code", PATHLIB_RECEIVER_MUTATION_FIXED,
                          ids=[l for l, _ in PATHLIB_RECEIVER_MUTATION_FIXED])
def test_pathlib_receiver_mutation_sensitive_hard_blocked(label, code):
    """receiver-bound 变异（无路径参数）：目标在 Path 构造参数里，参数型
    检测看不到——必须硬阻断（修复前 auto-approve）。"""
    assert _execute_code_has_sensitive_write(code) is not None


@ pytest.mark.parametrize("label,code", PATHLIB_RECEIVER_MUTATION_FIXED,
                          ids=[l + "-guard" for l, _ in PATHLIB_RECEIVER_MUTATION_FIXED])
def test_pathlib_receiver_mutation_guard_end_to_end(label, code):
    result = check_execute_code_guard(code, env_type="local")
    assert result["outcome"] == "hard_blocked"


@ pytest.mark.parametrize("label,code", PATHLIB_RECEIVER_MUTATION_FIXED,
                          ids=[l + "-yolo" for l, _ in PATHLIB_RECEIVER_MUTATION_FIXED])
def test_pathlib_receiver_mutation_hard_blocked_in_yolo(monkeypatch, label, code):
    """yolo/approvals=off 下同样不可覆盖——receiver 变异是 #49578 目标
    不变量的变异侧，必须在信任门之前强制执行。"""
    import tools.approval as approval_module
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", True)
    result = check_execute_code_guard(code, env_type="local")
    assert result["outcome"] == "hard_blocked"
    assert "protected path" in result["message"]


# B4e. 容器方法携带敏感字符串 ≠ 文件 I/O（误报修复）
def test_container_append_sensitive_string_not_hard_blocked():
    """paths.append('/etc/passwd') 只是往内存列表加字符串，无文件 I/O——
    不得触发 protected-path 硬阻断（修复前误报）。"""
    assert _execute_code_touches_sensitive_path(
        "paths = []\npaths.append('/etc/passwd')\n") is None
    assert _execute_code_touches_sensitive_path(
        "data = {}\ndata.setdefault('/root/.hermes/config.yaml', [])\n") is None
    assert _execute_code_touches_sensitive_path(
        "s = '/etc/passwd'\ns.replace('passwd', 'hosts')\n") is None


def test_pathlib_replace_dest_sensitive_still_hard_blocked():
    """Path.replace 与 str.replace 同名——但前者是文件覆盖，目标参数
    敏感必须硬阻断，不能因 NO_IO 白名单漏掉（同名的不同语义）。"""
    code = "from pathlib import Path\nPath('/tmp/x').replace('/root/.ssh/authorized_keys')\n"
    assert _execute_code_touches_sensitive_path(code) is not None
    result = check_execute_code_guard(code, env_type="local")
    assert result["outcome"] == "hard_blocked"


def test_pathlib_mutation_non_sensitive_target_triggers_approval_not_silent():
    """非敏感目标的 Path.unlink 不是静默放行——落入 file-delete 审批
    （与 os.remove 同级），保证删除操作永远走显式决策。"""
    assert _execute_code_has_dangerous_ops(
        "from pathlib import Path\nPath('/tmp/x').unlink()\n") == "file-delete"
    assert _execute_code_has_sensitive_write(
        "from pathlib import Path\nPath('/tmp/x').unlink()\n") is None


# ═════════════════════════════════════════════════════════════════════════
# Section C — 无法静态修复的残余面（XFAIL 标注，文档化）
# 属于 sandbox/运行时边界的职责（模块 docstring 已诚实声明）：
# 跨函数数据流（def/lambda 返回）、非字面量可迭代、动态插值。
# 若未来引入运行时沙箱，这些测试应翻转成普通断言。
# ═════════════════════════════════════════════════════════════════════════

STATIC_BOUNDARY_RESIDUALS = [
    # 跨函数数据流：需要过程间分析
    ("fn indirection", "import os\ndef f():\n    return os.kill\nf()(os.getpid(), 15)"),
    ("lambda indirection", "import os\nk = lambda: os.kill\nk()(os.getpid(), 15)"),
    ("fn wrapper", "import os\ndef call(fn, *a):\n    return fn(*a)\ncall(os.kill, os.getpid(), 15)"),
    # 非字面量 for 可迭代（运行时才知道 iterable 内容）
    ("for non-literal iter", "import os\nfuncs = [os.kill]\nfor f in funcs:\n    f(os.getpid(), 15)"),
    # 动态 f-string 插值（插值值静态不可解析）
    ("f-string unresolvable", "import os\nopen(f'{name}/.ssh/authorized_keys', 'w').write('x')"),
]

@pytest.mark.parametrize("label,code", STATIC_BOUNDARY_RESIDUALS,
                         ids=[l for l, _ in STATIC_BOUNDARY_RESIDUALS])
@pytest.mark.xfail(reason="静态分析边界（文档化残余）：跨函数数据流/运行时值需过程间分析或运行时沙箱，"
                          "属 sandbox 边界职责，当前模块 docstring 已诚实声明。",
                   strict=False)
def test_static_boundary_residual(label, code):
    """无法静态修复的形状——XFAIL 标注为文档化残余，不得误报为已修复。"""
    assert _execute_code_has_self_destructive_ops(code) is not None


# ═════════════════════════════════════════════════════════════════════════
# Section D — benign controls for every new capability (zero false positives)
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("code", [
    # walrus used benignly
    "(n := len([1, 2, 3]))\nprint(n)",
    # tuple unpacking of values, not functions
    "a, b = 1, 2\nprint(a + b)",
    "x, y = [1, 2]\nprint(x + y)",
    # dict/list subscripts of harmless content
    "k = {'x': 1}['x']\nprint(k)",
    "k = [1, 2][0]\nprint(k)",
    # getattr keyword with harmless module
    "import json\nprint(getattr(json, name='dumps')({'a': 1}))",
    # for-loop over benign callables / non-callable iterables
    "for f in [str, len]:\n    print(f('abc'))",
    "for i in [1, 2, 3]:\n    print(i)",
    # __import__ / sys.modules / globals / vars benign uses
    "__import__('json').dumps({'a': 1})",
    "import sys\nprint(sys.modules['json'].__name__)",
    "import json\nprint(globals()['json'].__name__)",
    "import json\nprint(vars(json).get('dumps'))",
    # signal used benignly
    "import signal\nprint(signal.SIGTERM)",
    # dict .get benign
    "print({'a': 1}.get('a'))",
    # f-string / join / expandvars to NON-sensitive paths
    "open(f'/tmp/x.txt', 'w').write('x')",
    'import os\nprint(os.path.join("/tmp", "x"))',
    'import os\nprint(os.path.expandvars("$HOME/tmp/x"))',
    # string concat of harmless names
    'import os\nname = "getpid"\nprint(os.__dict__[name]())',
])
def test_benign_controls_pass_all_layers(code):
    assert _execute_code_has_self_destructive_ops(code) is None
    assert _execute_code_has_sensitive_write(code) is None
    assert _execute_code_touches_sensitive_path(code) is None


def test_benign_script_auto_approves_cli(monkeypatch):
    """Pure-data script with safe imports → approved without any prompt."""
    import tools.approval as approval_module
    monkeypatch.setattr(approval_module, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_is_single_query_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_is_cron_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")
    result = check_execute_code_guard(
        'import json, math\nprint(json.dumps({"x": math.sqrt(4)}))',
        env_type="local",
    )
    assert result["approved"] is True


def test_syntax_error_script_not_crash(monkeypatch):
    """Unparseable script → analyzer yields None (approved in CLI); the script
    fails at runtime anyway — must not raise inside the guard."""
    result = check_execute_code_guard("def broken(:\n    pass", env_type="local")
    assert result["approved"] is True


def test_empty_script_auto_approves():
    result = check_execute_code_guard("", env_type="local")
    assert result["approved"] is True
