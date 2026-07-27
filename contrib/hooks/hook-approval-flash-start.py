#!/usr/bin/env python3
"""pre_approval_request 钩子：危险命令审批前启动任务栏闪黄。

v9 (2026-07-27):
  - PowerShell 残留进程清理：启动前杀所有 flash-taskbar.py 进程
    解决连续审批时前一个 flash 进程未退出（9p 信号延迟）导致进程堆积
  - v8: /init 降级路径：WSLInterop 丢失时用 /init + powershell.exe 启动闪黄脚本

v7 (2026-07-17): 
  - 移除 os.walk，改用硬编码 python.exe 路径列表（避免 9p 文件系统挂死）
  - 添加详细时间戳日志 /tmp/hermes-flash-timing.log
  - 将 flash 脚本复制到 Windows Temp 执行（避免 \\wsl$\ 网络路径延迟）
  - 修复竞技条件：stop 信号文件不再被 stop hook 删除

stdin JSON: {"hook_event_name":"pre_approval_request","extra":{"session_key":"...","surface":"cli"}}
"""
import json
import os
import shutil
import subprocess
import sys
import time

FLASH_SCRIPT = "/mnt/e/0AIHermes/scripts/flash/flash-taskbar.py"
PID_DIR = "/tmp/hermes-approval-flash"
STALE_AGE = 600
SIGNAL_SUBDIR = "temp/flash"  # 相对于 E:\0AIHermes\
TIMING_LOG = "/tmp/hermes-flash-timing.log"
JSONL_LOG = os.path.expanduser("~/.hermes/logs/flash-hooks.jsonl")

# /init 降级路径：当 binfmt_misc WSLInterop 丢失时，
# /init 可直接调用 Windows 可执行文件（绕过 binfmt_misc）
_INIT_BIN = "/init"
_POWERSHELL_WSL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"

def _jsonl(entry: dict) -> None:
    try:
        from datetime import datetime, timezone
        entry["ts"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:23]
        entry["hook"] = "pre_approval"
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
    """获取 Windows 侧工作目录的 WSL 路径"""
    candidates = [
        "/mnt/e/0AIHermes",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return candidates[0]

def _get_approval_timeout():
    try:
        import yaml
        cfg_path = os.path.expanduser("~/.hermes/config.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            return int(cfg.get("approvals", {}).get("timeout", 60))
    except Exception:
        pass
    return 60

# 硬编码 python.exe 路径列表
# WSL binfmt_misc 路由在 cmd.exe 超时后会降级挂死，因此全路径优先
_PYTHON_EXE_CANDIDATES = [
    "/mnt/c/Users/041701/AppData/Local/Programs/Python/Python311/python.exe",
    "/mnt/c/Users/041701/AppData/Local/Programs/Python/Python312/python.exe",
    "/mnt/c/Users/041701/AppData/Local/Programs/Python/Python310/python.exe",
    "/mnt/c/Program Files/Python311/python.exe",
    "/mnt/c/Program Files/Python312/python.exe",
    "python.exe",  # binfmt_misc 兜底（可能不可靠）
]

_python_exe_cache = None

def _find_windows_python():
    global _python_exe_cache
    if _python_exe_cache:
        return _python_exe_cache
    for py in _PYTHON_EXE_CANDIDATES:
        try:
            if subprocess.run([py, "--version"], capture_output=True, timeout=5).returncode == 0:
                _python_exe_cache = py
                return py
        except Exception:
            continue
    return None

def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def _cleanup_stale():
    try:
        now = time.time()
        for fname in os.listdir(PID_DIR):
            fpath = os.path.join(PID_DIR, fname)
            try:
                if now - os.path.getmtime(fpath) > STALE_AGE:
                    os.remove(fpath)
            except OSError:
                pass
    except FileNotFoundError:
        pass

def _kill_flash_processes_via_powershell() -> int:
    """通过 /init + PowerShell 查找并强制终止所有 flash-taskbar.py 进程。
    v9: 启动前清理，解决连续审批时的进程堆积。
    返回杀掉的进程数，-1 表示 /init 不可用。
    """
    if not os.path.exists(_INIT_BIN):
        return -1
    try:
        ps_cmd = (
            "Get-Process python -ErrorAction SilentlyContinue | "
            "ForEach-Object { "
            "$cim = Get-CimInstance Win32_Process -Filter \"ProcessId=$($_.Id)\"; "
            "if($cim.CommandLine -like '*flash-taskbar*') { "
            "Write-Host \"flash_kill:$($_.Id)\"; "
            "Stop-Process -Id $_.Id -Force "
            "} "
            "}"
        )
        result = subprocess.run(
            [_INIT_BIN, _POWERSHELL_WSL, "-Command", ps_cmd],
            capture_output=True, timeout=15
        )
        killed = result.stdout.decode("utf-8", "replace").count("flash_kill:")
        if killed > 0:
            _log(f"pre-cleanup: killed {killed} residual flash process(es)")
        return killed
    except Exception as e:
        _log(f"pre-cleanup exception: {e}")
        return -1

def start_flash(session_key, win_temp_wsl, win_temp_win):
    t_start = time.time()
    log_entry = {"session": session_key, "event": "start", "error": None}
    try:
        os.makedirs(PID_DIR, exist_ok=True)
        _cleanup_stale()

        # v9: 启动前清理残留 flash 进程（解决连续审批进程堆积）
        _kill_flash_processes_via_powershell()

        pid_file = os.path.join(PID_DIR, f"{session_key}.pid")
        # 清理残留
        for f in [pid_file]:
            try:
                with open(f) as fh:
                    old_pid = int(fh.read().strip())
                if not _pid_alive(old_pid):
                    os.remove(f)
            except (FileNotFoundError, ValueError, OSError):
                try:
                    os.remove(f)
                except FileNotFoundError:
                    pass

        t0 = time.time()
        python_exe = _find_windows_python()
        _log(f"[{session_key}] find_python: {time.time()-t0:.2f}s → {python_exe}")
        log_entry["python_exe"] = python_exe
        log_entry["find_python_ms"] = round((time.time() - t0) * 1000)
        
        use_init_fallback = False
        if not python_exe:
            # WSLInterop binfmt_misc 注册丢失（如 OpenClaw 触发 WSL 重启后）
            # /init 可直接调用 Windows 可执行文件，绕过 binfmt_misc
            if os.path.exists(_INIT_BIN) and os.path.exists(_POWERSHELL_WSL):
                use_init_fallback = True
                _log(f"[{session_key}] WSLInterop lost — using /init + powershell fallback")
                log_entry["fallback"] = "/init"
            else:
                _log(f"[{session_key}] FAIL: no python.exe found, /init not available")
                log_entry["error"] = "no python.exe found"
                _jsonl(log_entry)
                return

        # 确保信号目录存在
        signal_dir_wsl = os.path.join(win_temp_wsl, SIGNAL_SUBDIR)
        os.makedirs(signal_dir_wsl, exist_ok=True)
        # 清理该 session 旧信号文件 + 全局残留 _all.stop
        old_stop = os.path.join(signal_dir_wsl, f"{session_key}.stop")
        try:
            os.remove(old_stop)
        except FileNotFoundError:
            pass
        old_all_stop = os.path.join(signal_dir_wsl, "_all.stop")
        try:
            os.remove(old_all_stop)
        except FileNotFoundError:
            pass

        flash_timeout = _get_approval_timeout() + 10
        signal_dir_win = win_temp_win + "\\" + SIGNAL_SUBDIR

        # v8: 复制 flash 脚本 + sha256 校验文件到 Windows Temp，避免 \\wsl$\\ 网络路径
        t0 = time.time()
        temp_script_win = os.path.join(signal_dir_win, "flash-taskbar.py")
        temp_script_wsl = os.path.join(signal_dir_wsl, "flash-taskbar.py")
        shutil.copy2(FLASH_SCRIPT, temp_script_wsl)
        # 同时复制 sha256 校验文件，否则 flash-taskbar.py 的完整性校验会失败
        hash_src = FLASH_SCRIPT + ".sha256"
        hash_dst_wsl = temp_script_wsl + ".sha256"
        if os.path.exists(hash_src):
            shutil.copy2(hash_src, hash_dst_wsl)
        # 9p 文件系统缓存: 写入后短暂等待确保 Windows 侧可见
        time.sleep(0.5)
        _log(f"[{session_key}] copy_script: {time.time()-t0:.2f}s")

        env = os.environ.copy()
        
        t0 = time.time()
        # 记录闪黄脚本 stderr 到日志文件，用于诊断「第一次不闪」问题
        flash_log = os.path.join(signal_dir_wsl, f"{session_key}.flash.log")
        with open(flash_log, "w") as flog:
            if use_init_fallback:
                # /init + powershell 降级路径（WSLInterop 丢失时可用）
                ps_cmd = 'python "{}" {} 0.35 {} "{}" "{}"'.format(
                    temp_script_win, flash_timeout, session_key,
                    signal_dir_win, temp_script_win)
                proc = subprocess.Popen(
                    [_INIT_BIN, _POWERSHELL_WSL, "-Command", ps_cmd],
                    stdout=flog, stderr=subprocess.STDOUT,
                    env=env, start_new_session=True,
                )
            else:
                proc = subprocess.Popen(
                    [python_exe, temp_script_win,
                     str(flash_timeout), "0.35",
                     session_key, signal_dir_win,
                     temp_script_win],
                    stdout=flog, stderr=subprocess.STDOUT,
                    env=env, start_new_session=True,
                )
        _log(f"[{session_key}] Popen: {time.time()-t0:.2f}s pid={proc.pid}")
        log_entry["flash_pid"] = proc.pid
        log_entry["flash_timeout"] = flash_timeout
        log_entry["popen_ms"] = round((time.time() - t0) * 1000)

        with open(pid_file, "w") as f:
            f.write(str(proc.pid))

        _log(f"[{session_key}] TOTAL start_flash: {time.time()-t_start:.2f}s")
        log_entry["total_ms"] = round((time.time() - t_start) * 1000)
        _jsonl(log_entry)
    except Exception as e:
        _log(f"[{session_key}] ERROR: {e}")
        log_entry["error"] = str(e)
        log_entry["total_ms"] = round((time.time() - t_start) * 1000)
        _jsonl(log_entry)

def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    extra = payload.get("extra", {})
    if extra.get("surface", "") not in ("cli", ""):
        _log(f"SKIP surface={extra.get('surface')}")
        return

    session_key = extra.get("session_key", "default")
    _log(f"[{session_key}] START pre_approval_request hook")

    win_temp_wsl = _get_win_temp()
    win_temp_win = win_temp_wsl.replace("/mnt/e/", "E:\\").replace("/", "\\")
    start_flash(session_key, win_temp_wsl, win_temp_win)

if __name__ == "__main__":
    main()
