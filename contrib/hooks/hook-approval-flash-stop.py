#!/usr/bin/env python3
"""post_approval_response 钩子：审批结束后停止任务栏闪黄。

v5.2 (2026-07-27):
  - 写信号后立即返回（不等、不验证、不 PowerShell 杀）
  - flash 进程每 0.35s 检查信号，/init 写入零 9p 延迟 → 最多 0.35s 自退
  - 原则：审批结束 = 结果已获得，不需要等待。超时安全网是给「审批未响应」的，
    不是给「审批已完成」的。
  - /init 不可用时 fallback 到 WSL fsync（v4 兼容）
  - 备份: hook-approval-flash-stop.py.bak-v4

stdin JSON: {"hook_event_name":"post_approval_response","extra":{"session_key":"...","choice":"..."}}
"""
import json
import os
import subprocess
import sys
import time

PID_DIR = "/tmp/hermes-approval-flash"
SIGNAL_SUBDIR = "temp/flash"
TIMING_LOG = "/tmp/hermes-flash-timing.log"
JSONL_LOG = os.path.expanduser("~/.hermes/logs/flash-hooks.jsonl")

_INIT_BIN = "/init"
_CMD_EXE_WSL = "/mnt/c/Windows/System32/cmd.exe"

def _jsonl(entry: dict) -> None:
    try:
        from datetime import datetime, timezone
        entry["ts"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:23]
        entry["hook"] = "post_approval"
        os.makedirs(os.path.dirname(JSONL_LOG), exist_ok=True)
        with open(JSONL_LOG, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    try:
        with open(TIMING_LOG, "a") as f:
            f.write(f"[{ts}] {msg}\n")
    except OSError:
        pass

def _get_win_temp():
    candidates = ["/mnt/e/0AIHermes"]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return candidates[0]

def _wsl_to_win_path(wsl_path: str) -> str:
    return wsl_path.replace("/mnt/e/", "E:\\").replace("/", "\\").rstrip("\\")

def _write_signal_via_init(win_path: str) -> bool:
    try:
        result = subprocess.run(
            [_INIT_BIN, _CMD_EXE_WSL, "/c", f"echo stop> {win_path}"],
            capture_output=True, timeout=10
        )
        return result.returncode == 0
    except Exception as e:
        _log(f"/init write failed: {e}")
        return False

def _write_signal_wsl_fsync(wsl_path: str) -> bool:
    try:
        with open(wsl_path, "w") as f:
            f.write("stop")
            f.flush()
            os.fsync(f.fileno())
        return True
    except Exception:
        return False

def stop_flash(session_key, signal_dir_wsl):
    t_start = time.time()
    log_entry = {"session": session_key, "event": "stop"}
    pid_file = os.path.join(PID_DIR, f"{session_key}.pid")

    # 写入 _all.stop（全局）— flash 进程检查此文件停止所有闪黄
    all_stop_wsl = os.path.join(signal_dir_wsl, "_all.stop")

    if os.path.exists(_INIT_BIN):
        all_stop_win = _wsl_to_win_path(all_stop_wsl)
        ok = _write_signal_via_init(all_stop_win)
        _log(f"[{session_key}] _all.stop (/init): {'OK' if ok else 'FAIL'}")
        log_entry["signal_ok"] = ok
        log_entry["via"] = "/init"
    else:
        ok = _write_signal_wsl_fsync(all_stop_wsl)
        _log(f"[{session_key}] _all.stop (fsync): {'OK' if ok else 'FAIL'}")
        log_entry["signal_ok"] = ok
        log_entry["via"] = "fsync"

    # 清理 WSL 侧文件（信号文件留到下次 start hook 清理）
    for f in [pid_file]:
        try:
            os.remove(f)
        except (FileNotFoundError, OSError):
            pass
    for fname in [os.path.join(signal_dir_wsl, "flash-taskbar.py"),
                  os.path.join(signal_dir_wsl, "flash-taskbar.py.sha256"),
                  os.path.join(signal_dir_wsl, f"{session_key}.flash.log")]:
        try:
            os.remove(fname)
        except (FileNotFoundError, OSError):
            pass

    elapsed = time.time() - t_start
    _log(f"[{session_key}] STOP done: {elapsed:.2f}s")
    log_entry["total_ms"] = round(elapsed * 1000)
    _jsonl(log_entry)

def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    session_key = payload.get("extra", {}).get("session_key", "default")
    choice = payload.get("extra", {}).get("choice", "?")
    _log(f"[{session_key}] STOP choice={choice}")

    win_temp_wsl = _get_win_temp()
    signal_dir_wsl = os.path.join(win_temp_wsl, SIGNAL_SUBDIR)
    stop_flash(session_key, signal_dir_wsl)

if __name__ == "__main__":
    main()
