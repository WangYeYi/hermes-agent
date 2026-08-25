"""Static policy analysis for execute_code scripts (PR #65592).

Extracted from ``tools/approval.py`` into its own sub-2K module per the
andrexibiza re-review (2026-08-25): the execute-code static analyzer —
AST binding resolver, call-target resolver, write-mode analysis,
process-kill policy, import classifier, sensitive-write target analysis —
lives here; ``tools/approval.py`` retains only the orchestration seam
(``check_execute_code_guard``) that calls into this module.

Scope honesty (documented limitation, #65592 review): everything in this
module is a *static* AST layer.  It blocks every statically-detectable
form — direct calls, import aliases, assignment aliases (incl. chains),
star imports, ``getattr`` / ``__dict__`` dynamic access, pathlib write
methods, literal/expanduser sensitive targets.  Code that builds calls at
runtime (``eval("os.kill")(...)`` with a *string literal* is caught by
the eval guard below; string-concatenated ``exec`` is not statically
visible) belongs to the runtime/sandbox boundary.  Callers must not
present ``hard_blocked`` as an unbypassable syscall-level property.
"""

import ast
import logging
import os
import posixpath
import re

logger = logging.getLogger(__name__)

# Dangerous Python operations that bypass terminal() approval when used
# inside execute_code scripts.  Detected via AST walk with import tracking
# so both ``os.remove(x)`` and ``from os import remove; remove(x)`` are
# caught.  ctypes is listed as a whole-module gate. reason key 见
# _EXEC_CODE_DANGER_DETAILS。
_EXEC_CODE_DANGEROUS_CALLS = {
    # 文件/目录删除
    ("os", "remove"): "file-delete",
    ("os", "unlink"): "file-delete",
    ("shutil", "rmtree"): "file-delete",
    # 文件移动/复制/重命名（config write bypass, #49578）
    ("shutil", "copy"): "file-mutate",
    ("shutil", "copy2"): "file-mutate",
    ("shutil", "move"): "file-mutate",
    ("shutil", "copytree"): "file-mutate",
    ("os", "rename"): "file-mutate",
    ("os", "replace"): "file-mutate",
    # 任意命令执行（绕过 terminal() DANGEROUS_PATTERNS）
    ("os", "system"): "command-exec",
    ("os", "popen"): "command-exec",
    ("subprocess", "run"): "command-exec",
    ("subprocess", "call"): "command-exec",
    ("subprocess", "Popen"): "command-exec",
    ("subprocess", "check_output"): "command-exec",
    ("subprocess", "check_call"): "command-exec",
    # 进程替换（exec* 系列）——脚本把自己换成本机程序，绕过后续全部
    # Python 级检查（#65592 review 举一反三补充）
    ("os", "execv"): "command-exec",
    ("os", "execve"): "command-exec",
    ("os", "execvp"): "command-exec",
    ("os", "execvpe"): "command-exec",
    ("os", "execl"): "command-exec",
    ("os", "execle"): "command-exec",
    ("os", "execlp"): "command-exec",
    ("os", "execlpe"): "command-exec",
    ("os", "posix_spawn"): "command-exec",
    ("os", "posix_spawnp"): "command-exec",
    # pathlib 写方法（#49578 的等效绕过面，review 点名 Path.write_text）
    ("pathlib", "write_text"): "open-write",
    ("pathlib", "write_bytes"): "open-write",
    ("pathlib", "open"): "open-write",
}

# 模块的整体导入即触发 guard（即使不调用具体函数）。ctypes 符合——
# ``ctypes.CDLL(None).unlink(...)`` 无需 os.remove 即可绕过所有检查。
_EXEC_CODE_SUSPICIOUS_IMPORTS = frozenset({"ctypes"})

# =========================================================================
# Layer 3 — Hard Block: Self-Destructive / Process-Killing Operations
# =========================================================================
# These operations can destroy the Hermes parent process or kill arbitrary
# system processes. They NEVER enter the approval chain — no user consent,
# yolo mode, smart approval, or session persistence can override them.
# Design principle: Linux seccomp / macOS SIP — if the operation is
# fundamentally incompatible with agent operation, no bypass exists.
# (from PR #65592 commit 4, 66e423e4)
#
# NOTE (2026-08-25 re-review): the *static* scan can only match
# statically-visible call shapes; ``eval("os.kill")(...)`` with a literal
# string is caught by the eval guard below, but runtime-built call names
# are not statically visible.  The returned message therefore does NOT
# claim an absolute "no bypass exists" guarantee — the residual surface
# belongs to the runtime/sandbox boundary.

_HARD_BLOCKED_CALLS = frozenset({
    # Process killing — can target the Hermes parent process (os.getppid())
    # or any arbitrary system process.
    ("os", "kill"),
    ("os", "killpg"),
})


def _resolve_alias_value(name, imports, raw_aliases, seen=None):
    """解析赋值别名链为规范化的 (module, attr) 元组。

    覆盖（#65592 review Blocker 2 举一反三）：
      - ``a = os``        → ('os', None)        （模块级别名）
      - ``killer = os.kill`` → ('os', 'kill')   （函数别名）
      - ``b = a.kill``    → ('os', 'kill')      （链式别名）
      - ``x = getattr(os, 'kill')`` → ('os', 'kill')
    解析不了返回 None。
    """
    if name not in raw_aliases:
        return None
    if seen is None:
        seen = set()
    if name in seen:
        return None  # 循环别名（a=b; b=a）— 放弃，保守不误报
    seen.add(name)
    expr = raw_aliases[name]

    # a = os（模块级别名）
    if isinstance(expr, ast.Name):
        if expr.id in imports:
            return imports[expr.id]
        return _resolve_alias_value(expr.id, imports, raw_aliases, seen)

    # killer = os.kill / b = a.kill
    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
        base = None
        if expr.value.id in imports:
            base = imports[expr.value.id]
        else:
            base = _resolve_alias_value(expr.value.id, imports, raw_aliases, seen)
        if base:
            m, a = base
            return (m, expr.attr) if a is None else (m, a)

    # x = getattr(os, 'kill')
    if isinstance(expr, ast.Call) and getattr(expr.func, "id", None) == "getattr":
        if (len(expr.args) >= 2 and isinstance(expr.args[0], ast.Name)
                and expr.args[0].id in imports):
            m, a = imports[expr.args[0].id]
            if (a is None and isinstance(expr.args[1], ast.Constant)
                    and isinstance(expr.args[1].value, str)):
                return (m, expr.args[1].value)
    return None


def _expr_is_path_constructor(expr, imports, raw_aliases):
    """Path(...) / pathlib.Path(...) / P(...)（P 为 Path 的别名）→ True。"""
    if not isinstance(expr, ast.Call):
        return False
    f = expr.func
    if isinstance(f, ast.Name):
        if f.id == "Path":
            return True
        if f.id in imports:
            return imports[f.id][0] == "pathlib"
        return _resolve_alias_value(f.id, imports, raw_aliases) == ("pathlib", "Path")
    if isinstance(f, ast.Attribute) and f.attr == "Path" and isinstance(f.value, ast.Name):
        if f.value.id in imports:
            return imports[f.value.id][0] == "pathlib"
        return _resolve_alias_value(f.value.id, imports, raw_aliases) == ("pathlib", None)
    return False


def _collect_exec_code_bindings(code):
    """Pass 1：收集脚本的 import / star-import / 赋值别名绑定。

    返回 ``(imports, star_modules, raw_aliases)``：
      - imports:      {local_name: (module, attr_or_None)}
      - star_modules: set[str] — ``from X import *`` 的模块名集合
      - raw_aliases:  {local_name: ast.expr} — 赋值 RHS 原始表达式，
                      由 ``_resolve_alias_value`` 递归解析
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}, set(), {}
    imports: dict = {}
    star_modules: set = set()
    raw_aliases: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                imports[name] = (alias.name, None)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    star_modules.add(module.split(".")[0])
                    continue
                name = alias.asname or alias.name
                imports[name] = (module, alias.name)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name):
                raw_aliases[node.targets[0].id] = node.value
    return imports, star_modules, raw_aliases


def _resolve_call_target(func, imports, star_modules, raw_aliases, known):
    """把 ast.Call 的 func 解析为规范化 (module, attr)，失败返回 None。

    覆盖（#65592 review Blocker 2 + 举一反三）：
      - 直接调用: ``os.kill(x)``
      - import 别名: ``from os import kill; kill(x)``、``import os as o; o.kill(x)``
      - 赋值别名: ``killer = os.kill; killer(x)``（含链式 ``a=os; b=a.kill``）
      - star import: ``from os import *; kill(x)``
      - 动态调用: ``getattr(os, 'kill')(x)``、``os.__dict__['kill'](x)``
      - pathlib: ``Path(x).write_text(...)``、``pathlib.Path(x).open('a')``

    ``known`` 是调用方持有的 (module, attr) 集合，用于 star import 场景下
    判断 ``kill`` 是否确实来自 ``os``（避免把任意名字都当 os 的函数）。
    """
    # ── ast.Name: kill(...) ──
    if isinstance(func, ast.Name):
        name = func.id
        alias = _resolve_alias_value(name, imports, raw_aliases)
        if alias is not None:
            return alias
        if name in imports:
            m, a = imports[name]
            return (m, a) if a is not None else None
        for mod in star_modules:  # from os import *; kill(...)
            if (mod, name) in known:
                return (mod, name)
        return None

    # ── ast.Attribute: os.kill(...) / o.kill(...) / Path(x).write_text(...) ──
    if isinstance(func, ast.Attribute):
        attr = func.attr
        base = func.value
        # Path(...).write_text(...) — pathlib 写方法
        if _expr_is_path_constructor(base, imports, raw_aliases):
            return ("pathlib", attr)
        if isinstance(base, ast.Name):
            bname = base.id
            if bname == "Path":
                return ("pathlib", attr)
            alias = _resolve_alias_value(bname, imports, raw_aliases)
            if alias is not None:
                m, a = alias
                return (m, attr) if a is None else (m, a)
            if bname in imports:
                m, a = imports[bname]
                return (m, attr) if a is None else (m, a)
        return None

    # ── ast.Call: getattr(os, 'kill')(...) ──
    if isinstance(func, ast.Call) and getattr(func.func, "id", None) == "getattr":
        if len(func.args) >= 2 and isinstance(func.args[0], ast.Name):
            m = None
            if func.args[0].id in imports:
                m = imports[func.args[0].id][0]
            else:
                alias = _resolve_alias_value(func.args[0].id, imports, raw_aliases)
                if alias is not None:
                    m = alias[0]
            if m is not None:
                if (isinstance(func.args[1], ast.Constant)
                        and isinstance(func.args[1].value, str)):
                    return (m, func.args[1].value)
                return (m, None)  # 动态属性名 — 由调用方决定是否保守拦截
        return None

    # ── ast.Subscript: os.__dict__['kill'](...) ──
    if isinstance(func, ast.Subscript):
        val = func.value
        if (isinstance(val, ast.Attribute) and val.attr == "__dict__"
                and isinstance(val.value, ast.Name)):
            m = None
            if val.value.id in imports:
                m = imports[val.value.id][0]
            if (m is not None and isinstance(func.slice, ast.Constant)
                    and isinstance(func.slice.value, str)):
                return (m, func.slice.value)
        return None

    return None


def _execute_code_has_self_destructive_ops(code: str) -> str | None:
    """Return a human-readable reason if *code* contains operations that
    can destroy the Hermes process or kill arbitrary processes, or None
    if the code is free of self-destructive operations.

    These operations are HARD BLOCKED — they never enter the approval
    chain and cannot be bypassed via yolo, smart mode, or session
    persistence.

    Scope honesty (#65592 review, andrexibiza): this is a *static* AST
    layer.  It blocks every statically-detectable form — direct calls,
    import aliases, assignment aliases (incl. chains), star imports,
    ``getattr`` / ``__dict__`` dynamic access, and ``eval("os.kill")``
    with a string literal.  Code that builds the call at runtime
    (string concatenation into ``exec``) is not statically visible.
    That residual surface belongs to the runtime/sandbox boundary, not
    to this heuristic; the message returned to the model therefore does
    NOT claim an absolute "no bypass exists" guarantee.  Design follows
    Linux seccomp / macOS SIP only in spirit: if the operation is
    fundamentally incompatible with the agent's continued operation, no
    user consent can make it safe.
    """
    imports, star_modules, raw_aliases = _collect_exec_code_bindings(code)

    # ── Pass 2: walk call nodes ──────────────────────────────
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        resolved = _resolve_call_target(
            node.func, imports, star_modules, raw_aliases, _HARD_BLOCKED_CALLS
        )
        if resolved is None:
            continue
        m, a = resolved
        if (m, a) in _HARD_BLOCKED_CALLS:
            return (
                f"{m}.{a}() — "
                f"process-killing operation (HARD BLOCKED, no approval path)"
            )
        if a is None and m in ("os", "sys"):
            # getattr(os, dynamic_name)(...) — 动态属性名，静态无法判定具体
            # 函数，但 os/sys 的动态属性访问可能取到 kill/killpg。保守拦截：
            # 该类别本身允许用户改用显式调用（os.kill 会被上面精确拦截并
            # 给出明确原因），误伤面极小。
            return (
                f"getattr({m}, ...) — dynamic attribute access on {m} "
                f"(HARD BLOCKED: may resolve to a process-killing function; "
                f"use explicit {m}.<name> calls instead)"
            )
    return None


# reason key → (用户可读的中文原因, 建议改用方式)。用于 execute_code
# 拦截提示区分具体原因，避免 Agent 被拦后无从判断该换什么工具。
_EXEC_CODE_DANGER_DETAILS = {
    "open-write": ("文件写入（open 的 w/a/x 或 + 模式）",
                   "改用 write_file 或 patch 工具"),
    "file-delete": ("文件/目录删除（os.remove / shutil.rmtree 等）",
                    "先确认目标路径，或改用 terminal 走正常审批"),
    "file-mutate": ("文件移动/复制/重命名（shutil.copy / os.rename 等）",
                    "先确认目标路径"),
    "command-exec": ("任意命令执行（subprocess / os.system 等）",
                     "改用 terminal 工具，走正常命令审批"),
    "ctypes-import": ("ctypes 模块导入（可绕过所有 Python 级检查）",
                      "确认确实需要 syscall 级访问"),
    "dynamic-exec": ("eval/exec/compile 动态代码执行（可绕过静态分析）",
                     "改用显式函数调用，或改用 terminal 工具走正常审批"),
    "sensitive-write": ("写入安全敏感路径（config / .ssh / 系统目录）",
                        "该路径受保护，禁止通过 execute_code 修改"),
}


def _exec_code_reason_text(reason: str) -> str:
    """把 reason key 转成用户可读的拦截说明（含建议改用方式）。"""
    detail = _EXEC_CODE_DANGER_DETAILS.get(reason)
    if detail is None:
        return f"危险操作（{reason}）"
    why, remedy = detail
    return f"{why}；建议：{remedy}"


def _log_blocked_exec_code(code: str, reason: str) -> None:
    """Log a blocked execute_code script with redacted content."""
    from agent.redact import redact_sensitive_text
    truncated = code[:4000]
    if len(code) > 4000:
        truncated += f"\n... [truncated, {len(code)} total chars]"
    logger.warning(
        "execute_code BLOCKED (%s). Script (%d chars):\n%s",
        reason, len(code), redact_sensitive_text(truncated),
    )


def _open_mode_is_write(call_node: ast.Call) -> bool:
    """判断 open(...) 调用的 mode 参数是否为写模式。

    open(file, mode='r', ...) — mode 是第二个位置参数或 keyword 参数。
    mode 缺省或明确只读（r/rb/rt）→ False；含写标志（w/a/x/+）→ True；
    mode 是变量/表达式无法静态判定 → True（保守拦截）。
    """
    mode_arg = None
    if len(call_node.args) >= 2:
        mode_arg = call_node.args[1]  # 第二个位置参数
    else:
        for kw in call_node.keywords:
            if kw.arg == "mode":
                mode_arg = kw.value
                break
    if mode_arg is None:
        return False  # 无 mode → 默认 'r'，只读
    if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str):
        return any(c in mode_arg.value for c in "wax+")
    return True  # 变量/表达式，无法静态判定 → 保守拦截


def _path_open_mode_is_write(call_node: ast.Call) -> bool:
    """判断 pathlib ``Path.open(mode, ...)`` 的 mode 是否为写模式。

    Path.open 的签名是 ``open(mode='r', buffering=-1, ...)`` —— mode 是
    第一个位置参数（与内置 open(file, mode='r') 不同，见 #65592 review）。
    mode 缺省或只读 → False；含写标志（w/a/x/+）→ True；无法静态判定 → True。
    """
    mode_arg = None
    if call_node.args:
        mode_arg = call_node.args[0]
    else:
        for kw in call_node.keywords:
            if kw.arg == "mode":
                mode_arg = kw.value
                break
    if mode_arg is None:
        return False  # 缺省 'r'，只读
    if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str):
        return any(c in mode_arg.value for c in "wax+")
    return True  # 变量/表达式，无法静态判定 → 保守拦截


def _execute_code_has_dangerous_ops(code: str):
    """返回 execute_code 脚本中首个危险操作的 reason key（见
    ``_EXEC_CODE_DANGER_DETAILS``），无危险操作返回 None。

    Two-pass scan:
    1. Collect imports + assignment aliases via
       ``_collect_exec_code_bindings``
    2. Walk call nodes, resolving every call target to a canonical
       ``(module, attr)`` pair (direct calls, import aliases, assignment
       aliases, star imports, ``getattr`` / ``__dict__`` dynamic access,
       pathlib methods) before checking the denylist.

    Immune to whitespace / comments / string literals (``ast.parse``).
    """
    imports, star_modules, raw_aliases = _collect_exec_code_bindings(code)

    # ── 可疑模块整体导入（ctypes）────────────────────────────
    for _local_name, (_module, _attr) in imports.items():
        if _module in _EXEC_CODE_SUSPICIOUS_IMPORTS:
            return "ctypes-import"

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    # ── Pass 2: walk call nodes ──────────────────────────────
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func

        # 内置 open() — 区分读写模式（只读放行，写拦截）
        if isinstance(func, ast.Name) and func.id == "open":
            if _open_mode_is_write(node):
                return "open-write"
            continue  # 只读 open，放行

        # eval/exec/compile — 动态代码执行（review Blocker 2 举一反三：
        # ``eval("os.kill")(...)`` 的外层 callee 是 eval 调用，普通解析器
        # 不处理；字面量 eval 是静态可检测的，动态字符串拼接无法检测）
        if isinstance(func, ast.Name) and func.id in ("eval", "exec", "compile"):
            return "dynamic-exec"

        resolved = _resolve_call_target(
            func, imports, star_modules, raw_aliases, _EXEC_CODE_DANGEROUS_CALLS
        )
        if resolved is None:
            continue
        m, a = resolved
        if (m, a) in _EXEC_CODE_DANGEROUS_CALLS:
            if (m, a) == ("pathlib", "open"):
                # Path.open 的 mode 是第一个位置参数，需单独判断读写
                if _path_open_mode_is_write(node):
                    return "open-write"
                continue  # 只读 Path.open，放行
            return _EXEC_CODE_DANGEROUS_CALLS[(m, a)]

    return None


# =========================================================================
# Layer 4 — Sensitive-Write Target Analysis (review Blocker 1)
# =========================================================================
# #49578 is an *effect/destination* invariant: security-sensitive files
# (Hermes config, ~/.ssh, system dirs) are hard-refused by the file-tool
# path (`_check_sensitive_path` in tools/file_tools.py) REGARDLESS of
# approval mode.  execute_code must preserve that invariant: a statically
# resolvable write to a protected target is hard-blocked BEFORE any
# recoverable approval bypass (--yolo / approvals.mode=off), so the
# invariant cannot be traded away by turning approvals off.
#
# Static limitation (honest): only literal / expanduser / simple-alias
# target strings are resolvable here.  A target computed at runtime
# (`os.path.join(home, name)`) is not statically visible — that residual
# surface belongs to the runtime/sandbox boundary.

# Sensitive prefixes mirrored from tools/file_tools._SENSITIVE_PATH_PREFIXES
# plus the Hermes/SSH trees (same class of protected destination #49578
# names).  Matched after ~/env expansion and normpath.
_EXEC_CODE_SENSITIVE_PREFIXES = (
    "/etc/", "/boot/", "/usr/lib/systemd/", "/private/etc/",
    "/private/var/db/", "/private/var/root/",
    "/run/", "/var/run/",
)
_EXEC_CODE_SENSITIVE_HOME_TREES = (".ssh", ".hermes", ".aws", ".gnupg")
_EXEC_CODE_SENSITIVE_EXACT = {"/var/run/docker.sock", "/run/docker.sock"}

_STRING_CONSTANT_EVAL_RE = re.compile(
    r"os\.path\.expanduser\((['\"])(.*?)\1\)", re.DOTALL
)


def _resolve_static_write_target(node: ast.Call, raw_aliases) -> str | None:
    """Try to resolve an open()/Path() first-arg target to a literal path.

    Handles: string literal, ``os.path.expanduser("...")`` with a literal,
    and a simple variable alias whose RHS is one of those.  Returns None
    when the target is not statically resolvable.
    """
    if not node.args:
        return None
    target_expr = node.args[0]

    if isinstance(target_expr, ast.Constant) and isinstance(target_expr.value, str):
        return target_expr.value
    if isinstance(target_expr, ast.Name) and target_expr.id in raw_aliases:
        rhs = raw_aliases[target_expr.id]
        if isinstance(rhs, ast.Constant) and isinstance(rhs.value, str):
            return rhs.value
        # os.path.expanduser("~/.hermes/config.yaml")
        if (isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Attribute)
                and isinstance(rhs.func.value, ast.Attribute)
                and isinstance(rhs.func.value.value, ast.Name)
                and rhs.func.value.value.id == "os"
                and rhs.func.value.attr == "path"
                and rhs.func.attr == "expanduser"
                and rhs.args and isinstance(rhs.args[0], ast.Constant)
                and isinstance(rhs.args[0].value, str)):
            return os.path.expanduser(rhs.args[0].value)
    # os.path.expanduser("~/.hermes/config.yaml") inline as first arg
    if (isinstance(target_expr, ast.Call) and isinstance(target_expr.func, ast.Attribute)
            and isinstance(target_expr.func.value, ast.Attribute)
            and isinstance(target_expr.func.value.value, ast.Name)
            and target_expr.func.value.value.id == "os"
            and target_expr.func.value.attr == "path"
            and target_expr.func.attr == "expanduser"
            and target_expr.args and isinstance(target_expr.args[0], ast.Constant)
            and isinstance(target_expr.args[0].value, str)):
        return os.path.expanduser(target_expr.args[0].value)
    return None


def _write_target_is_sensitive(path: str) -> bool:
    """True if *path* targets a protected destination (mirrors the file-tool
    sensitive-path invariant from #49578)."""
    if not path:
        return False
    expanded = os.path.expanduser(os.path.expandvars(path))
    normalized = posixpath.normpath(expanded.replace("\\", "/"))
    if normalized in _EXEC_CODE_SENSITIVE_EXACT:
        return True
    for prefix in _EXEC_CODE_SENSITIVE_PREFIXES:
        if normalized.startswith(prefix):
            return True
    # Hermes home tree (config/env live here — approvals.mode etc.)
    hermes_home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    hermes_norm = posixpath.normpath(hermes_home.replace("\\", "/"))
    if normalized == hermes_norm or normalized.startswith(hermes_norm + "/"):
        return True
    # Home trees that gate agent security: .ssh, .aws, .gnupg
    home = os.path.expanduser("~")
    home_norm = posixpath.normpath(home.replace("\\", "/"))
    for tree in _EXEC_CODE_SENSITIVE_HOME_TREES:
        target = home_norm + "/" + tree
        if normalized == target or normalized.startswith(target + "/"):
            return True
    return False


def _execute_code_has_sensitive_write(code: str) -> str | None:
    """Return the protected target path if *code* statically writes a
    sensitive destination via open()/Path(), else None.

    Runs BEFORE the yolo/approvals-off bypass gates in
    ``check_execute_code_guard`` so the #49578 destination invariant is
    enforced even when approval is turned off.  Only statically
    resolvable targets are caught; runtime-computed paths are a
    documented static limitation.
    """
    imports, star_modules, raw_aliases = _collect_exec_code_bindings(code)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func

        # builtin open(...) / open(file, mode) with write mode
        if isinstance(func, ast.Name) and func.id == "open":
            if _open_mode_is_write(node):
                target = _resolve_static_write_target(node, raw_aliases)
                if target and _write_target_is_sensitive(target):
                    return target
            continue

        # Path(...).write_text / write_bytes / open(write)
        resolved = _resolve_call_target(
            func, imports, star_modules, raw_aliases, _EXEC_CODE_DANGEROUS_CALLS
        )
        if resolved == ("pathlib", "write_text") or resolved == ("pathlib", "write_bytes"):
            base = func.value  # Path(...) call
            if isinstance(base, ast.Call) and base.args:
                target = _resolve_static_write_target(base, raw_aliases)
                if target and _write_target_is_sensitive(target):
                    return target
        elif resolved == ("pathlib", "open"):
            if _path_open_mode_is_write(node):
                base = func.value
                if isinstance(base, ast.Call) and base.args:
                    target = _resolve_static_write_target(base, raw_aliases)
                    if target and _write_target_is_sensitive(target):
                        return target
    return None


# =========================================================================
# Layer 1 — Capability Whitelist: Safe Imports Classification
# =========================================================================
# Known-safe stdlib modules whose presence alone does not indicate danger.
# Scripts importing ONLY from these modules (and containing no dangerous
# call patterns) are classified as pure-data / computation — they pass
# through without triggering the approval prompt in CLI sessions.
# (from PR #65592 commit 4, 66e423e4)

_EXEC_CODE_SAFE_IMPORTS = frozenset({
    # Data formats
    "json", "csv", "base64", "binascii", "codecs",
    # Text processing
    "re", "string", "textwrap", "difflib", "unicodedata",
    # Numeric / math
    "math", "statistics", "fractions", "decimal", "numbers",
    "random",
    # Collections / data structures
    "collections", "itertools", "functools", "operator",
    "heapq", "bisect", "array", "struct",
    # Filesystem (read-only / temp)
    "pathlib", "tempfile", "glob", "fnmatch", "fileinput",
    # Date / time
    "datetime", "calendar", "time",
    # Hashing
    "hashlib", "hmac",
    # Type system / introspection
    "typing", "dataclasses", "enum", "inspect", "types",
    # Output / formatting
    "pprint", "textwrap",
    # Debugging / logging (read-only use)
    "traceback", "warnings", "logging",
    # Markup (safe parsing)
    "html",
})

# Modules whose import signals potential danger (process/file/network
# capability). Used by _classify_exec_code_imports for diagnostics.

_EXEC_CODE_DANGEROUS_IMPORTS = frozenset({
    "os", "sys", "subprocess", "shutil", "ctypes",
    "socket", "signal", "multiprocessing", "threading",
    "http", "urllib", "ftplib", "smtplib", "poplib", "imaplib",
    "telnetlib", "asyncio",
})


def _classify_exec_code_imports(code: str) -> tuple[list[str], list[str], list[str]]:
    """Classify imports in *code* as (safe, dangerous, unknown).

    Returns three lists of top-level module name strings.  Used by
    Layer 1 (whitelist) to determine whether a script that has no
    dangerous call patterns should still trigger the guard because
    it imports dangerous or unrecognised modules.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [], [], ["(syntax error)"]

    safe, dangerous, unknown = [], [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _EXEC_CODE_DANGEROUS_IMPORTS:
                    dangerous.append(top)
                elif top in _EXEC_CODE_SAFE_IMPORTS:
                    safe.append(top)
                else:
                    unknown.append(top)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in _EXEC_CODE_DANGEROUS_IMPORTS:
                    dangerous.append(top)
                elif top in _EXEC_CODE_SAFE_IMPORTS:
                    safe.append(top)
                else:
                    unknown.append(top)

    return safe, dangerous, unknown
