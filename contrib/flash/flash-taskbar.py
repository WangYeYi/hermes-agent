"""Windows Terminal 任务栏闪黄，用于危险命令审批提醒。

停止机制（Windows Temp 信号文件，无跨边界依赖）：
  信号目录: %TEMP%\hermes-flash\ （WSL 和 Windows 均可直接访问）
  flash 每 ~0.35s 检查 os.path.exists(signal_file)
  检测到 → FlashWindow(FALSE) → sys.exit(0)
  atexit 完整执行。120s 超时安全网。

参数 (命令行): max_timeout interval session_key signal_dir_win [pid_file]
  注意: WSL subprocess 无法向 Windows 进程传递环境变量，所有参数通过 argv 传递。
"""
import atexit
import ctypes
import hashlib
import json
import os
import sys
import time

# ── 完整性校验 ───────────────────────────────────────────────────
HASH_FILE = os.path.join(os.path.dirname(__file__), "flash-taskbar.py.sha256")
def verify_integrity():
    if not os.path.exists(HASH_FILE):
        print("WARNING: hash file missing", file=sys.stderr)
        return
    with open(HASH_FILE) as hf:
        expected = hf.read().strip()
    with open(__file__, "rb") as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    if actual != expected:
        print(f"INTEGRITY FAILED", file=sys.stderr)
        sys.exit(2)
verify_integrity()

# ── 参数 (命令行，非环境变量 —— WSL→Windows 环境变量传递失败) ──
max_timeout = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
interval = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35
SESSION_KEY = sys.argv[3] if len(sys.argv) > 3 else ""
SIGNAL_DIR_WIN = sys.argv[4] if len(sys.argv) > 4 else ""

SIGNAL_FILE = os.path.join(SIGNAL_DIR_WIN, f"{SESSION_KEY}.stop") if (SESSION_KEY and SIGNAL_DIR_WIN) else None
ALL_STOP_FILE = os.path.join(SIGNAL_DIR_WIN, "_all.stop") if SIGNAL_DIR_WIN else None

# ── Windows API ──────────────────────────────────────────────────
user32 = ctypes.windll.user32
WT_CLASS = "CASCADIA_HOSTING_WINDOW_CLASS"
WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)

wt_hwnd = None
def find_wt(h, _):
    global wt_hwnd
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(h, buf, 256)
    if buf.value == WT_CLASS:
        wt_hwnd = h
        return False
    return True
user32.EnumWindows(WNDPROC(find_wt), 0)

if not wt_hwnd:
    print("ERROR: WT window not found", file=sys.stderr)
    sys.exit(1)

# ── atexit ───────────────────────────────────────────────────────
# 仅负责 FlashWindow(FALSE)，不删除任何文件。
# 文件清理由 hook-approval-flash-stop.py Phase 3 统一负责。
STOPPED = False
def _cleanup():
    global STOPPED
    if not STOPPED and wt_hwnd:
        user32.FlashWindow(wt_hwnd, False)
        STOPPED = True
atexit.register(_cleanup)

_start_marker = {
    "event": "flash_start",
    "hwnd": wt_hwnd,
    "key": SESSION_KEY,
    "timeout": max_timeout,
    "signal": SIGNAL_FILE,
}
print(f"FLASH_MARKER::{json.dumps(_start_marker)}", flush=True)

# ── 主循环 ──────────────────────────────────────────────────────
elapsed = 0.0
cycle = 0
flash_api_ok = 0
flash_api_fail = 0

while elapsed < max_timeout:
    ret = user32.FlashWindow(wt_hwnd, True)
    if ret:
        flash_api_ok += 1
    else:
        flash_api_fail += 1
    time.sleep(interval)
    elapsed += interval
    cycle += 1

    if (SIGNAL_FILE and os.path.exists(SIGNAL_FILE)) or (ALL_STOP_FILE and os.path.exists(ALL_STOP_FILE)):
        user32.FlashWindow(wt_hwnd, False)
        STOPPED = True
        _stop_marker = {
            "event": "flash_stop",
            "reason": "signal",
            "elapsed": round(elapsed, 1),
            "flash_api_ok": flash_api_ok,
            "flash_api_fail": flash_api_fail,
        }
        print(f"FLASH_MARKER::{json.dumps(_stop_marker)}", flush=True)
        sys.exit(0)

user32.FlashWindow(wt_hwnd, False)
STOPPED = True
_timeout_marker = {
    "event": "flash_stop",
    "reason": "timeout",
    "elapsed": round(elapsed, 1),
    "flash_api_ok": flash_api_ok,
    "flash_api_fail": flash_api_fail,
}
print(f"FLASH_MARKER::{json.dumps(_timeout_marker)}", flush=True)
