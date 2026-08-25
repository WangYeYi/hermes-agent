"""Static policy analysis for execute_code scripts (PR #65592).

Extracted from ``tools/approval.py`` into its own sub-2K module per the
andrexibiza re-review (2026-08-25): the execute-code static analyzer —
AST binding resolver, call-target resolver, write-mode analysis,
process-kill policy, import classifier, sensitive-write target analysis —
lives here; ``tools/approval.py`` retains only the orchestration seam
(``check_execute_code_guard``) that calls into this module.

Scope honesty (documented limitation, #65592 review): everything in this
module is a *static* AST layer.  It blocks every statically-detectable
form — direct calls, import aliases, assignment aliases (incl. chains,
walrus, tuple-unpack, for-targets), container subscripts (dict/list
literals), star imports, ``getattr`` / ``__dict__`` dynamic access,
``sys.modules`` / ``globals()`` / ``vars()`` / ``__import__`` chains,
``functools.partial``, process-kill equivalents (``signal.kill`` /
``psutil.kill``), pathlib write methods, literal/expanduser/expandvars/
f-string/join sensitive targets.  Code that builds calls at runtime
(string-concatenated ``exec``, dynamic f-string interpolation, function/
lambda/partial indirection through user-defined callables, non-literal
for-iterables) is not statically visible and belongs to the
runtime/sandbox boundary.  Callers must not present ``hard_blocked`` as
an unbypassable syscall-level property.
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

# Builtin 危险名称（2026-08-25 复现发现）：``op = open``、``e = eval``、
# ``from builtins import open as op`` 等别名形式曾绕过基于 ``func.id`` 的
# 直接检测（open/eval/exec 检测分支只匹配裸名字）。解析为
# ``("builtins", name)`` 后与模块属性检测统一走 ``_resolve_call_target``。
_EXEC_CODE_DANGEROUS_BUILTINS = frozenset({
    "open", "eval", "exec", "compile", "__import__",
})

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
    # os.kill 的等价物（2026-08-25 复测发现：signal.kill 是 POSIX os.kill，
    # psutil.kill 是跨平台 os.kill；psutil.Process(...).kill() 经
    # _resolve_call_target 的 psutil 方法链解析到 ("psutil", "kill")）。
    # 同类能力必须同样不可审批——否则 signal.kill(getppid(), 9) 绕过硬阻断。
    ("signal", "kill"),
    ("signal", "pthread_kill"),
    ("psutil", "kill"),
})


def _resolve_binding_expr(expr, imports, raw_aliases, seen=None):
    """解析赋值 RHS 表达式为规范化的 (module, attr) 元组。

    2026-08-25 复测重构（#65592）：原 _resolve_alias_value 只收 name 字符串，
    容器下标 / vars / globals / sys.modules / __import__ / partial 等
    绑定形状无法表达。现统一由本函数解析任意 RHS 表达式，覆盖：
      - ``a = os``        → ('os', None)        （模块级别名）
      - ``killer = os.kill`` → ('os', 'kill')   （函数别名）
      - ``b = a.kill``    → ('os', 'kill')      （链式别名）
      - ``x = getattr(os, 'kill')`` → ('os', 'kill')
      - ``h = os.path.expanduser`` → ('os.path', 'expanduser')
      - ``op = open`` / ``e = eval`` → ('builtins', name)
      - ``(k := os.kill)`` → ('os', 'kill')     （walrus，NamedExpr）
      - ``k = {'kill': os.kill}['kill']`` → ('os', 'kill')（dict 字面量下标）
      - ``k = [os.kill][0]`` → ('os', 'kill')   （list 字面量下标）
      - ``x = vars(os)['kill']`` / ``x = globals()['os']`` / ``sys.modules['os']``
      - ``k = partial(os.kill, 1, 15)`` → ('os', 'kill')
      - ``k = __import__('os')`` → ('os', None)
    解析不了返回 None（调用方决定是否保守拦截）。
    """
    if seen is None:
        seen = set()

    # (k := os.kill) — walrus 表达式本身（2026-08-25 复测：NamedExpr 不是
    # Assign，整个逃出绑定图 → (k := os.kill)(1, 15) 曾直接放行）
    if isinstance(expr, ast.NamedExpr):
        return _resolve_binding_expr(expr.value, imports, raw_aliases, seen)

    # a = b（别名链）/ a = os（模块级别名）
    if isinstance(expr, ast.Name):
        if expr.id in _EXEC_CODE_DANGEROUS_BUILTINS:
            return ("builtins", expr.id)
        if expr.id in imports:
            return imports[expr.id]
        if expr.id in seen:
            return None  # 循环别名（a=b; b=a）— 放弃，保守不误报
        seen.add(expr.id)
        if expr.id in raw_aliases:
            return _resolve_binding_expr(raw_aliases[expr.id], imports, raw_aliases, seen)
        return None

    # killer = os.kill / b = a.kill / h = p.expanduser（p = os.path）
    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
        base = None
        if expr.value.id in imports:
            base = imports[expr.value.id]
        else:
            base = _resolve_binding_expr(expr.value, imports, raw_aliases, seen)
        if base:
            m, a = base
            # 组合属性链（2026-08-25 re-review）：base 已带属性时（如
            # p = os.path → ('os', 'path')），下一层属性必须组合成
            # ('os.path', attr)，否则 p.expanduser 会被错误解析成
            # ('os', 'path')，敏感写目标解析彻底丢失。
            if a is None:
                return (m, expr.attr)
            return (f"{m}.{a}", expr.attr)

    # h = os.path.expanduser — os.path 子模块属性链（2026-08-25：敏感写
    # 目标解析曾漏掉此函数引用别名，退化为审批而非硬阻断）
    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Attribute):
        inner = expr.value
        if (inner.attr == "path" and isinstance(inner.value, ast.Name)
                and inner.value.id in imports
                and imports[inner.value.id][0] == "os"):
            return ("os.path", expr.attr)
        if (inner.attr == "path" and isinstance(inner.value, ast.Name)
                and _resolve_binding_expr(inner.value, imports, raw_aliases, seen)
                == ("os", None)):
            return ("os.path", expr.attr)

    # x = getattr(os, 'kill') / getattr(o, 'kill')（别名 base）/ 关键字形式
    # （2026-08-25 复测：getattr(os, name='kill') 关键字 attr 曾完全逃逸）
    if isinstance(expr, ast.Call) and getattr(expr.func, "id", None) == "getattr":
        obj_expr, attr_expr = _getattr_args(expr)
        if obj_expr is not None and isinstance(obj_expr, ast.Name):
            m = None
            if obj_expr.id in imports:
                m = imports[obj_expr.id][0]
            else:
                base = _resolve_binding_expr(obj_expr, imports, raw_aliases, seen)
                if base:
                    m = base[0]
            if m is not None:
                if (isinstance(attr_expr, ast.Constant)
                        and isinstance(attr_expr.value, str)):
                    return (m, attr_expr.value)
                return (m, None)  # 动态属性名 — 调用方决定是否保守拦截
        return None

    # k = partial(os.kill, 1, 15) — functools.partial 首参即被调用目标
    # （2026-08-25 复测举一反三：partial 包装曾完全逃过解析）
    if (isinstance(expr, ast.Call) and expr.args
            and _resolve_binding_expr(expr.func, imports, raw_aliases, seen)
            in (("functools", "partial"), ("builtins", "partial"))):
        return _resolve_binding_expr(expr.args[0], imports, raw_aliases, seen)

    # k = __import__('os') — 返回模块对象
    if (isinstance(expr, ast.Call) and getattr(expr.func, "id", None) == "__import__"
            and expr.args and isinstance(expr.args[0], ast.Constant)
            and isinstance(expr.args[0].value, str)):
        return (expr.args[0].value, None)

    # x = o.__dict__["killpg"] / x = {'kill': os.kill}['kill'] /
    #     x = [os.kill][0] / x = sys.modules['os'] / x = globals()['os'] /
    #     x = vars(os)['kill'] — 下标取值形状
    if isinstance(expr, ast.Subscript):
        return _resolve_subscript_expr(expr, imports, raw_aliases, seen)
    return None


def _fold_str_expr(e, raw_aliases=None):
    """静态折叠字符串表达式（字面量 + 常量拼接 + 简单别名）。

    2026-08-25 复测举一反三：``os.__dict__["ki" + "ll"](...)`` /
    ``os.__dict__[name](...)``（name = "kill"）曾因 slice 不是 Constant
    而逃过所有下标分支。折叠失败返回 None。
    """
    if isinstance(e, ast.Constant) and isinstance(e.value, str):
        return e.value
    if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Add):
        left = _fold_str_expr(e.left, raw_aliases)
        right = _fold_str_expr(e.right, raw_aliases)
        if left is not None and right is not None:
            return left + right
        return None
    if raw_aliases is not None and isinstance(e, ast.Name) and e.id in raw_aliases:
        return _fold_str_expr(raw_aliases[e.id], raw_aliases)
    return None


def _getattr_args(call_node):
    """提取 getattr(object, name[, default]) 的位置/关键字参数对。

    2026-08-25 复测：``getattr(os, name='kill')`` 关键字形式曾绕过——
    _resolve_call_target 的 getattr 分支只读位置参数。
    """
    obj_expr, attr_expr = None, None
    if call_node.args:
        obj_expr = call_node.args[0]
        if len(call_node.args) >= 2:
            attr_expr = call_node.args[1]
    if attr_expr is None:
        for kw in call_node.keywords:
            if kw.arg == "name":
                attr_expr = kw.value
    if obj_expr is None:
        for kw in call_node.keywords:
            if kw.arg == "object":
                obj_expr = kw.value
    return obj_expr, attr_expr


def _resolve_subscript_expr(expr, imports, raw_aliases, seen=None):
    """解析下标表达式 x[slice] 为规范化的 (module, attr)（调用位与赋值 RHS 共用）。

    覆盖（2026-08-25 复测举一反三）：
      - ``o.__dict__['kill']`` / ``os.__dict__['kill']``（含别名 base）
      - ``{'kill': os.kill}['kill']`` — dict 字面量下标
      - ``[os.kill][0]`` — list/tuple 字面量下标
      - ``sys.modules['os']`` — 运行时模块表（'os' 为字面量）
      - ``globals()['os']`` — 仅当名字是脚本显式 import 的
      - ``vars(os)['kill']`` — vars(x) 等价 x.__dict__
    """
    val = expr.value
    # slice 常量折叠：字面量 / "ki"+"ll" 拼接 / name = "kill" 别名
    # （2026-08-25 复测举一反三：非常量 slice 曾逃过全部下标分支）
    slice_str = _fold_str_expr(expr.slice, raw_aliases)
    # __dict__ 动态访问（原逻辑 + 别名 base）
    if (isinstance(val, ast.Attribute) and val.attr == "__dict__"
            and isinstance(val.value, ast.Name)):
        m = None
        if val.value.id in imports:
            m = imports[val.value.id][0]
        else:
            base = _resolve_binding_expr(val.value, imports, raw_aliases, seen)
            if base:
                m = base[0]
        if m is not None and slice_str is not None:
            return (m, slice_str)
    # {'kill': os.kill}['kill'] — dict 字面量
    if (isinstance(val, ast.Dict) and slice_str is not None):
        for key_n, value_n in zip(val.keys, val.values):
            key_s = _fold_str_expr(key_n, raw_aliases)
            if key_s is not None and key_s == slice_str:
                return _resolve_binding_expr(value_n, imports, raw_aliases, seen)
    # [os.kill][0] / (os.kill,)[0] — list/tuple 字面量（slice 为整型常量）
    if (isinstance(val, (ast.List, ast.Tuple))
            and isinstance(expr.slice, ast.Constant)
            and isinstance(expr.slice.value, int)):
        idx = expr.slice.value
        if 0 <= idx < len(val.elts):
            return _resolve_binding_expr(val.elts[idx], imports, raw_aliases, seen)
    # sys.modules['os'] — 模块表查询（折叠后字符串 → 该模块）
    if (isinstance(val, ast.Attribute) and val.attr == "modules"
            and isinstance(val.value, ast.Name)
            and val.value.id in imports
            and imports[val.value.id][0] == "sys"
            and slice_str is not None):
        return (slice_str, None)
    # globals()['os'] — 仅当名字是脚本显式 import（否则无法静态判定内容）
    if (isinstance(val, ast.Call) and getattr(val.func, "id", None) == "globals"
            and not val.args and slice_str is not None
            and slice_str in imports):
        return imports[slice_str]
    # vars(os)['kill'] — vars(x) 等价 x.__dict__
    if (isinstance(val, ast.Call) and getattr(val.func, "id", None) == "vars"
            and val.args and isinstance(val.args[0], ast.Name)):
        m = None
        if val.args[0].id in imports:
            m = imports[val.args[0].id][0]
        else:
            base = _resolve_binding_expr(val.args[0], imports, raw_aliases, seen)
            if base:
                m = base[0]
        if m is not None and slice_str is not None:
            return (m, slice_str)
    return None


def _resolve_alias_value(name, imports, raw_aliases, seen=None):
    """解析赋值别名链为规范化的 (module, attr) 元组（name 字符串入口）。

    直接调用（open(...)）与别名 RHS（op = open）都先检查 builtin 名；
    其余交给 _resolve_binding_expr 做表达式级解析。解析不了返回 None。
    """
    if name in _EXEC_CODE_DANGEROUS_BUILTINS:
        return ("builtins", name)
    if name not in raw_aliases:
        return None
    if seen is None:
        seen = set()
    if name in seen:
        return None  # 循环别名（a=b; b=a）— 放弃，保守不误报
    seen.add(name)
    return _resolve_binding_expr(raw_aliases[name], imports, raw_aliases, seen)


def _resolve_attribute_chain(expr, imports, raw_aliases, seen=None):
    """把嵌套属性表达式解析为规范化 (module, attr)。

    2026-08-25 re-review 补充：``os.path.expanduser`` 的 func 是三层
    Attribute（expanduser → os.path → os），``_resolve_call_target`` 的
    Attribute 分支只认 ``base`` 为 Name 的形状，嵌套属性 base 曾直接
    return None——导致组合属性调用（``p = os.path; p.expanduser(...)``
    之外还有 ``os.path.expanduser(...)`` 内联形式）解析失败。
    """
    if isinstance(expr, ast.Name):
        if expr.id in imports:
            return imports[expr.id]
        return _resolve_alias_value(expr.id, imports, raw_aliases, seen)
    if isinstance(expr, ast.Attribute):
        base = _resolve_attribute_chain(expr.value, imports, raw_aliases, seen)
        if base:
            m, a = base
            if a is None:
                return (m, expr.attr)
            return (f"{m}.{a}", expr.attr)
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


def _resolve_path_constructor(expr, raw_aliases, imports):
    """把 Path 构造调用解析出来：直接 ``Path("...")`` / ``pathlib.Path("...")``
    或存进变量的对象 ``p = Path("...")``（含链式 ``q = p``）。

    2026-08-25 re-review（andrexibiza Blocker 2b）：call-valued RHS 不在
    抽象绑定图里，``p = Path("~/.hermes/config.yaml"); p.write_text("x")``
    曾完全逃过敏感写形状检测。返回 ast.Call 节点（可继续取构造参数），
    解析不了返回 None。
    """
    if isinstance(expr, ast.Name):
        rhs = raw_aliases.get(expr.id)
        if rhs is not None:
            return _resolve_path_constructor(rhs, raw_aliases, imports)
        return None
    if _expr_is_path_constructor(expr, imports, raw_aliases):
        return expr
    return None


def _collect_exec_code_bindings(code):
    """Pass 1：收集脚本的 import / star-import / 赋值别名绑定。

    返回 ``(imports, star_modules, raw_aliases)``：
      - imports:      {local_name: (module, attr_or_None)}
      - star_modules: set[str] — ``from X import *`` 的模块名集合
      - raw_aliases:  {local_name: ast.expr} — 赋值 RHS 原始表达式，
                      由 ``_resolve_binding_expr`` 递归解析

    2026-08-25 复测扩展（#65592）——普通 Python 绑定形式补全：
      - walrus ``(k := os.kill)``（ast.NamedExpr，曾完全逃出绑定图）
      - 元组解包 ``k1, k2 = os.kill, os.killpg``（Tuple 目标按位置配对）
      - for 循环目标 ``for f in [os.kill]``（字面量可迭代时取首元素——
        第一次迭代即执行，取第一个可解析元素是正确且保守的）
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
        elif isinstance(node, ast.Assign):
            # 多重赋值 a = b = os.kill → targets=[a, b]，每个名字都绑定同一 RHS
            # （2026-08-25 re-review：曾只记录 len(targets)==1，多重赋值完全
            # 逃出绑定图 → a = b = os.kill 绕过 hard block）
            for t in node.targets:
                if isinstance(t, ast.Name):
                    raw_aliases[t.id] = node.value
                elif isinstance(t, ast.Tuple) and isinstance(
                        node.value, (ast.Tuple, ast.List)):
                    # 元组解包 k1, k2 = os.kill, os.killpg → 按位置配对
                    # （2026-08-25 复测：Tuple 目标曾整段跳过 → k1(...) 放行）
                    for target_elt, value_elt in zip(t.elts, node.value.elts):
                        if isinstance(target_elt, ast.Name):
                            raw_aliases[target_elt.id] = value_elt
        elif isinstance(node, ast.NamedExpr):
            # (k := os.kill) — walrus 绑定（2026-08-25 复测发现）
            if isinstance(node.target, ast.Name):
                raw_aliases[node.target.id] = node.value
        elif isinstance(node, ast.For):
            # for f in [os.kill]: f(...) — 字面量可迭代的目标绑定
            # （2026-08-25 复测：for 目标曾完全逃逸）
            if (isinstance(node.target, ast.Name)
                    and isinstance(node.iter, (ast.List, ast.Tuple))
                    and node.iter.elts):
                first = node.iter.elts[0]
                if isinstance(first, (ast.Name, ast.Attribute,
                                      ast.Subscript, ast.Call, ast.NamedExpr)):
                    raw_aliases[node.target.id] = first
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
    # ── ast.NamedExpr: (k := os.kill)(...) — walrus 作调用目标
    #    （2026-08-25 复测：NamedExpr func 曾直接 return None → 放行）
    if isinstance(func, ast.NamedExpr):
        return _resolve_call_target(func.value, imports, star_modules, raw_aliases, known)

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

    # ── ast.Attribute: os.kill(...) / o.kill(...) / os.path.expanduser(...)
    #    / Path(x).write_text(...) / p.write_text(...) ──
    if isinstance(func, ast.Attribute):
        attr = func.attr
        base = func.value
        # Path 构造（直接 / pathlib.Path / P 别名 / 变量存对象 p = Path(...)）
        if _resolve_path_constructor(base, raw_aliases, imports) is not None:
            return ("pathlib", attr)
        resolved_base = None
        if isinstance(base, ast.Name):
            if base.id == "Path":
                return ("pathlib", attr)
            if base.id in imports:
                resolved_base = imports[base.id]
            else:
                # 赋值别名：killer = os.kill 的 base 是 os；o.kill 的 base 是 o；
                # p.expanduser 的 base 是 p（p = os.path，组合属性）
                resolved_base = _resolve_alias_value(base.id, imports, raw_aliases)
        elif isinstance(base, ast.Attribute):
            # 嵌套属性 base：os.path.expanduser 的 base 是 os.path
            # （2026-08-25 re-review：此前只认 Name base，直接 return None）
            resolved_base = _resolve_attribute_chain(base, imports, raw_aliases)
        elif isinstance(base, ast.Call):
            # psutil.Process(1).kill() — 实例方法链（2026-08-25 复测：
            # psutil.Process(...).kill() 曾完全逃逸）。base 是可调用构造，
            # 先解析构造器本身；psutil 家族的方法 kill 与 os.kill 同能力。
            inner = _resolve_call_target(base.func, imports, star_modules,
                                         raw_aliases, known)
            if inner is not None and inner[0] == "psutil":
                return ("psutil", attr)
            # __import__('os').kill(...) — base 是模块导入调用
            resolved_base = _resolve_binding_expr(base, imports, raw_aliases)
        elif isinstance(base, ast.Subscript):
            # sys.modules['os'].kill(...) / vars(os)['kill'](...) /
            # {'kill': os.kill}['kill'](...) — base 是下标取值
            resolved_base = _resolve_binding_expr(base, imports, raw_aliases)
        elif isinstance(base, ast.NamedExpr):
            # (k := os).kill(...) — walrus base
            resolved_base = _resolve_binding_expr(base, imports, raw_aliases)
        if resolved_base is not None:
            m, a = resolved_base
            # 组合属性链：o = os.path 后 o.expanduser(...) 必须解析成
            # ('os.path', 'expanduser') 而非 ('os', 'path')。
            if a is None:
                return (m, attr)
            return (f"{m}.{a}", attr)
        return None

    # ── ast.Call: getattr(os, 'kill')(...) / os.__dict__.get('kill')(...) ──
    if isinstance(func, ast.Call):
        # getattr（2026-08-25 复测：关键字形式 getattr(os, name='kill')
        # 曾绕过——分支只读位置参数；别名 base o=os 一并支持）
        if getattr(func.func, "id", None) == "getattr":
            obj_expr, attr_expr = _getattr_args(func)
            if obj_expr is not None and isinstance(obj_expr, ast.Name):
                m = None
                if obj_expr.id in imports:
                    m = imports[obj_expr.id][0]
                else:
                    alias = _resolve_alias_value(obj_expr.id, imports, raw_aliases)
                    if alias is not None:
                        m = alias[0]
                if m is not None:
                    if (isinstance(attr_expr, ast.Constant)
                            and isinstance(attr_expr.value, str)):
                        return (m, attr_expr.value)
                    return (m, None)  # 动态属性名 — 由调用方决定是否保守拦截
            return None
        # X.__dict__.get('kill')(...) — .get 方法链（2026-08-25 复测：
        # os.__dict__.get('kill')(1, 15) 曾完全逃逸）
        if isinstance(func.func, ast.Attribute) and func.func.attr == "get":
            key = None
            if func.args:
                key = _fold_str_expr(func.args[0], raw_aliases)
            if key is not None:
                base = func.func.value
                if (isinstance(base, ast.Attribute) and base.attr == "__dict__"
                        and isinstance(base.value, ast.Name)):
                    m = None
                    if base.value.id in imports:
                        m = imports[base.value.id][0]
                    else:
                        alias = _resolve_alias_value(base.value.id, imports, raw_aliases)
                        if alias is not None:
                            m = alias[0]
                    if m is not None:
                        return (m, key)
                if isinstance(base, ast.Dict):
                    # {'kill': os.kill}.get('kill')(...) — dict 字面量 .get
                    for key_n, value_n in zip(base.keys, base.values):
                        if (isinstance(key_n, ast.Constant)
                                and key_n.value == key):
                            return _resolve_binding_expr(value_n, imports, raw_aliases)
            return None
        # partial(os.kill, 1, 15)(...) — functools.partial 首参即被调用目标
        if (func.args and _resolve_binding_expr(func.func, imports, raw_aliases)
                in (("functools", "partial"), ("builtins", "partial"))):
            return _resolve_binding_expr(func.args[0], imports, raw_aliases)
        return None

    # ── ast.Subscript: os.__dict__['kill'](...) / o.__dict__['kill'](...) /
    #    sys.modules['os'].kill(...) / vars(os)['kill'](...) /
    #    {'kill': os.kill}['kill'](...) ──
    # （2026-08-25 复测：统一委托 _resolve_subscript_expr——下标取值形状
    #   全家族覆盖：__dict__、dict/list 字面量、sys.modules、globals()、
    #   vars()；调用位与赋值 RHS 共用同一解析）
    if isinstance(func, ast.Subscript):
        return _resolve_subscript_expr(func, imports, raw_aliases)

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
    import aliases, assignment aliases (incl. chains, walrus, tuple-unpack,
    for-targets), container subscripts, star imports, ``getattr`` /
    ``__dict__`` dynamic access, ``sys.modules`` / ``globals()`` /
    ``vars()`` / ``__import__`` chains, ``functools.partial``, the
    os.kill-equivalents ``signal.kill`` / ``signal.pthread_kill`` /
    ``psutil.kill`` / ``psutil.Process(...).kill()``, and
    ``eval("os.kill")`` with a string literal.  Code that builds the call
    at runtime (string concatenation into ``exec``, calls routed through
    user-defined functions/lambdas, non-literal for-iterables) is not
    statically visible.  That residual surface belongs to the
    runtime/sandbox boundary, not to this heuristic; the message returned
    to the model therefore does NOT claim an absolute "no bypass exists"
    guarantee.  Design follows Linux seccomp / macOS SIP only in spirit:
    if the operation is fundamentally incompatible with the agent's
    continued operation, no user consent can make it safe.
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

        resolved = _resolve_call_target(
            func, imports, star_modules, raw_aliases, _EXEC_CODE_DANGEROUS_CALLS
        )

        # 内置 open() — 区分读写模式（只读放行，写拦截）。resolved 覆盖
        # 直接调用、``op = open`` 赋值别名、``from builtins import open as op``
        # （2026-08-25 复现：赋值别名曾绕过此检测）。
        if resolved == ("builtins", "open"):
            if _open_mode_is_write(node):
                return "open-write"
            continue  # 只读 open，放行

        # eval/exec/compile — 动态代码执行（review Blocker 2 举一反三：
        # ``eval("os.kill")(...)`` 的外层 callee 是 eval 调用，普通解析器
        # 不处理；字面量 eval 是静态可检测的，动态字符串拼接无法检测。
        # 别名形式（``e = eval``）同样解析为 ("builtins", "eval")）
        if resolved in (
            ("builtins", "eval"), ("builtins", "exec"), ("builtins", "compile"),
        ):
            return "dynamic-exec"

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

# 只读/查询方法白名单（2026-08-25 补：#49578 残余面——pandas/numpy 等库
# 写方法的路径参数绕过 open()/Path() 形状检测）。带敏感路径参数的方法调用
# 若不在本集合中，一律 hard-block。集合覆盖：纯路径操作（无 IO）、存在性/
# 元数据查询、目录列举、文件内容读取（read/load/read_* 系列）。
_EXEC_CODE_READONLY_QUERY_NAMES = frozenset({
    # --- 纯路径操作（os.path.* / pathlib 属性，无文件 IO）---
    "join", "basename", "dirname", "split", "splitext", "abspath", "realpath",
    "normpath", "normcase", "expanduser", "expandvars", "commonpath",
    "commonprefix", "relpath", "samefile", "sameopenfile", "samestat",
    "name", "suffix", "suffixes", "stem", "anchor", "parent", "parents",
    "parts", "as_posix", "as_uri", "cwd", "home",
    # --- 存在性 / 元数据查询（不读取内容）---
    "exists", "isfile", "isdir", "islink", "ismount", "lexists",
    "is_file", "is_dir", "is_symlink", "is_socket", "is_fifo",
    "is_block_device", "is_char_device", "is_absolute", "is_relative_to",
    "stat", "lstat", "fstat", "getsize", "getmtime", "getctime", "getatime",
    "access", "walk", "scandir", "listdir", "glob", "iglob", "rglob",
    "iterdir", "absolute", "resolve",
    # --- 文件内容读取（读敏感目标 = #46900 的 secret 读取面，单独管控；
    #   与 open() 只读放行行为保持一致）---
    "read", "read_text", "read_bytes", "readlines", "readline",
    "load", "loads", "loadtxt", "loadmat", "load_npy", "fromfile",
    "fromstring", "frombuffer", "memmap", "imread", "imdecode", "mmap",
    "read_csv", "read_json", "read_excel", "read_parquet", "read_pickle",
    "read_hdf", "read_sql", "read_html", "read_xml", "read_fwf",
    "read_table", "read_sas", "read_spss", "read_clipboard", "read_feather",
    "read_orc", "read_stata", "read_gbq", "read_sql_table", "read_sql_query",
})

_STRING_CONSTANT_EVAL_RE = re.compile(
    r"os\.path\.expanduser\((['\"])(.*?)\1\)", re.DOTALL
)


def _resolve_expr_path(expr, raw_aliases, imports) -> str | None:
    """Try to resolve *expr* (any AST expression) to a literal path string.

    Handles: string literal, ``os.path.expanduser("...")`` with a literal,
    a simple variable alias whose RHS is one of those, and (2026-08-25) a
    function-reference alias ``h = os.path.expanduser; h("...")``.  Returns
    None when the target is not statically resolvable.
    """
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    # f-string（2026-08-25 复测：#49578 不变量曾漏掉——f'/root/.hermes/...'
    # 是 JoinedStr，不是 Constant；全字面量/可解析插值拼接，否则视为运行时）
    if isinstance(expr, ast.JoinedStr):
        parts = []
        for v in expr.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            elif isinstance(v, ast.FormattedValue):
                r = _resolve_expr_path(v.value, raw_aliases, imports)
                if isinstance(r, str):
                    parts.append(r)
                else:
                    return None
            else:
                return None
        return "".join(parts)
    if isinstance(expr, ast.Name) and expr.id in raw_aliases:
        rhs = raw_aliases[expr.id]
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
    # 任意可解析的 os.path 函数调用 — 统一走 _resolve_call_target：
    #   h = os.path.expanduser; h("...")（函数引用别名）
    #   p = os.path; p.expanduser("...")（组合属性别名，2026-08-25 re-review）
    #   os.path.expanduser("...")（直接调用）
    # 曾各自写死 AST 形状，组合属性链漏掉导致敏感写目标解析丢失。
    # 2026-08-25 复测扩展：expandvars（$HOME/...）与 join（全字面量参数）
    # 同为静态可解析形状，曾降级为可恢复审批 → yolo 下击穿 #49578 不变量。
    if (isinstance(expr, ast.Call) and expr.args
            and isinstance(expr.args[0], ast.Constant)
            and isinstance(expr.args[0].value, str)
            and _resolve_call_target(
                expr.func, imports, set(), raw_aliases, frozenset()
            ) == ("os.path", "expanduser")):
        return os.path.expanduser(expr.args[0].value)
    if (isinstance(expr, ast.Call) and expr.args
            and isinstance(expr.args[0], ast.Constant)
            and isinstance(expr.args[0].value, str)
            and _resolve_call_target(
                expr.func, imports, set(), raw_aliases, frozenset()
            ) == ("os.path", "expandvars")):
        return os.path.expandvars(expr.args[0].value)
    if (isinstance(expr, ast.Call) and expr.args
            and _resolve_call_target(
                expr.func, imports, set(), raw_aliases, frozenset()
            ) == ("os.path", "join")):
        # os.path.join("lit", "lit", ...) — 每个参数递归解析（expanduser/
        # expandvars/字面量/简单别名），任一不可解析 → 视为运行时拼接
        parts = []
        for a in expr.args:
            r = _resolve_expr_path(a, raw_aliases, imports)
            if isinstance(r, str):
                parts.append(r)
            else:
                return None
        if parts:
            return posixpath.join(*parts)
    return None


def _resolve_static_write_target(node: ast.Call, raw_aliases, imports) -> str | None:
    """Try to resolve an open()/Path() first-arg target to a literal path.

    Thin wrapper over ``_resolve_expr_path`` for the legacy call shape
    (first positional argument).
    """
    if not node.args:
        return None
    return _resolve_expr_path(node.args[0], raw_aliases, imports)


def _write_target_is_sensitive(path: str) -> bool:
    """True if *path* targets a protected destination (mirrors the file-tool
    sensitive-path invariant from #49578)."""
    if not path:
        return False
    expanded = os.path.expanduser(os.path.expandvars(path))
    normalized = posixpath.normpath(expanded.replace("\\", "/"))
    # POSIX 规范：路径以 // 开头是实现定义，Linux 下 // == /；但
    # posixpath.normpath 会保留开头的双斜杠前缀，导致后续 startswith
    # 比较失败——``open("//root/.hermes/config.yaml", "w")`` 曾逃过不变量
    # （2026-08-25 复测）。折叠开头多斜杠为单斜杠再比较。
    if normalized.startswith("//"):
        normalized = "/" + normalized.lstrip("/")
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

        resolved = _resolve_call_target(
            func, imports, star_modules, raw_aliases, _EXEC_CODE_DANGEROUS_CALLS
        )

        # builtin open(...) / open(file, mode) with write mode
        # （resolved 覆盖 op = open 别名 / from builtins import open as op）
        if resolved == ("builtins", "open"):
            if _open_mode_is_write(node):
                target = _resolve_static_write_target(node, raw_aliases, imports)
                if target and _write_target_is_sensitive(target):
                    return target
            continue

        # Path(...).write_text / write_bytes / open(write) — 含变量存对象
        # p = Path("..."); p.write_text("x")（2026-08-25 re-review Blocker 2b）
        if resolved in (("pathlib", "write_text"), ("pathlib", "write_bytes")):
            if not isinstance(func, ast.Attribute):
                continue
            ctor = _resolve_path_constructor(func.value, raw_aliases, imports)
            if ctor is not None and ctor.args:
                target = _resolve_expr_path(ctor.args[0], raw_aliases, imports)
                if target and _write_target_is_sensitive(target):
                    return target
        elif resolved == ("pathlib", "open"):
            if _path_open_mode_is_write(node):
                if not isinstance(func, ast.Attribute):
                    continue
                ctor = _resolve_path_constructor(func.value, raw_aliases, imports)
                if ctor is not None and ctor.args:
                    target = _resolve_expr_path(ctor.args[0], raw_aliases, imports)
                    if target and _write_target_is_sensitive(target):
                        return target
    return None


def _execute_code_touches_sensitive_path(code: str) -> str | None:
    """Return the protected target if any *library* call's path argument
    statically references a sensitive destination.

    Closes the #49578 residual surface found 2026-08-25: pandas/numpy and
    other third-party writers (``to_csv``/``save``/``dump``/...) accept
    arbitrary path strings that never match the open()/Path() AST shapes
    checked by ``_execute_code_has_sensitive_write`` — so
    ``pd.DataFrame(...).to_csv('/root/.ssh/authorized_keys')`` sailed
    straight through.  Any call whose method is NOT in the read-only
    query whitelist and whose positional/keyword arguments statically
    resolve to a sensitive path is hard-blocked.  Read-only access
    (os.path queries, existence checks, directory listing, content
    reads) stays allowed, matching the open() read-mode behaviour.
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

        # 只检查属性方法调用（obj.method(...)）；裸函数名调用（Path(...)、
        # open(...) 等构造函数/内置）由各自的形状检测负责——否则
        # ``Path('/root/.hermes').exists()`` 的构造参数会被误伤为敏感引用。
        if isinstance(func, ast.Attribute):
            method = func.attr
        else:
            continue

        if method in _EXEC_CODE_READONLY_QUERY_NAMES:
            continue
        # open()/Path() 写形状已在 _execute_code_has_sensitive_write 单独
        # 处理；这里跳过避免重复判定（其只读形态合法，不升级）。
        resolved = _resolve_call_target(
            func, imports, star_modules, raw_aliases, _EXEC_CODE_DANGEROUS_CALLS
        )
        if resolved in (("builtins", "open"), ("pathlib", "open")):
            continue

        # 检查所有位置参数 + 关键字参数是否静态解析为敏感路径
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            target = _resolve_expr_path(arg, raw_aliases, imports)
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
