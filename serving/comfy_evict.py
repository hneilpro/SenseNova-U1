#!/usr/bin/env python3
"""ComfyUI VRAM eviction hook for the Nova launcher (custom/nova-serving only).

Why: ComfyUI on this box (v0.40.0, 127.0.0.1:8188) holds VRAM even when
idle. Nova in fast mode pins ~21.4GB while generating, and a 2026-11-09
test failed with only ~7GB free because ComfyUI (plus other apps) was
holding the card. Nova must therefore claim the GPU *before* it starts.

Why this lives in the LAUNCHER, not in nova_api.py's per-generation
hook (unlike ollama_evict.py): nova_api.py loads the model at startup
in main(), before it ever serves a request. By the time a per-generation
hook could run, the model load has already competed with ComfyUI for
VRAM. The launcher (nova_launcher.py) is the only place that runs
before the Nova process exists, so it evicts there, in start_nova().
nova_api.py main() also calls this module as a backstop for the direct
`serving/start_nova_api.sh` path, and refuses to start if ComfyUI is busy.

Rules (owner-authorized 2026-11-09):
1. Check ComfyUI /queue. Evict ONLY if queue_running AND queue_pending
   are both empty. If a job is running or queued, do NOT stop anything
   and do NOT start Nova: the caller gets a clear "comfyui_busy" error
   and retries later. No waiting, no partial eviction.
2. If ComfyUI is unreachable, treat it as nothing to evict (fail-open).
3. Stop by exact PID only: scan /proc for the python process whose
   cmdline is `main.py --listen 127.0.0.1 --port 8188` in the ComfyUI
   dir. Never pattern-kill (pgrep -af matches the invoking shell).
   SIGTERM first, SIGKILL only after a 30s grace period, and verify
   :8188 is actually down before reporting success.
4. Restart after Nova stops: the launcher calls restart_comfy() from
   stop_nova(), and also on any start_nova() failure after an eviction,
   so ComfyUI is never left down by a failed Nova start. Restart uses
   the exact command ComfyUI was launched with and waits for
   /system_stats to answer. A state file (~/.nova-comfy-evict.json)
   records an eviction so a launcher restart can recover ComfyUI even
   if the launcher itself died while it was down.

Controls: disable globally with env NOVA_EVICT_COMFYUI=0, or per
request with {"evict_comfyui": false} in the generation/edit body
(launcher reads the body before starting Nova).

Stdlib only. Fail-safe, not fail-open, on the busy/error paths: when
in doubt the generation is refused rather than risking a job or an OOM.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path

COMFYUI_DIR = Path("/mnt/c/Users/Harish/Documents/GitHub/hneilpro/ComfyUI")
COMFYUI_URL = "http://127.0.0.1:8188"
COMFYUI_QUEUE_URL = COMFYUI_URL + "/queue"
COMFYUI_STATS_URL = COMFYUI_URL + "/system_stats"
# Exact launch command ComfyUI was started with on this box (parent bash
# 4001, 2026-11-09): same dir, same log file, so restarts continue it.
COMFYUI_CMD = (
    "cd /mnt/c/Users/Harish/Documents/GitHub/hneilpro/ComfyUI && "
    "nohup .venv/bin/python main.py --listen 127.0.0.1 --port 8188 "
    "> /home/hneil/comfy-hstore-restart.log 2>&1"
)
COMFYUI_LOG_FILE = Path("/home/hneil/comfy-hstore-restart.log")
STATE_FILE = Path.home() / ".nova-comfy-evict.json"
NVIDIA_SMI_EXE = "/mnt/c/Windows/System32/nvidia-smi.exe"

STOP_WAIT_SECONDS = 30.0   # SIGTERM grace period before SIGKILL
KILL_WAIT_SECONDS = 10.0   # wait after SIGKILL before giving up
RESTART_WAIT_SECONDS = 90.0  # ComfyUI loads custom nodes; allow time

EVICT_ENABLED = os.environ.get("NOVA_EVICT_COMFYUI", "1") != "0"

# evict_if_idle() outcomes. "busy" and "error" must abort the Nova start;
# "evicted" / "not_running" / "disabled" let it proceed.
BUSY = "busy"
ERROR = "error"
EVICTED = "evicted"
NOT_RUNNING = "not_running"
DISABLED = "disabled"


def _log(msg: str) -> None:
    print("[comfy-evict] %s" % msg, flush=True)


def _http_get_json(url, timeout=10):
    """GET url and parse JSON; raises on any failure (caller decides)."""
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def queue_status(http_get=None):
    """ComfyUI /queue as a dict, or None if ComfyUI is unreachable.

    Returns {"queue_running": [...], "queue_pending": [...]} on success.
    Any transport/parse failure returns None: nothing to evict.
    """
    http_get = http_get or _http_get_json
    try:
        data = http_get(COMFYUI_QUEUE_URL, timeout=10)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def queue_is_empty(status):
    """True only when both running and pending queues are empty."""
    if status is None:
        return True  # unreachable == nothing running here
    return not status.get("queue_running") and not status.get("queue_pending")


def comfy_is_up(http_get=None):
    """True if /system_stats answers (process up, regardless of queue)."""
    http_get = http_get or _http_get_json
    try:
        http_get(COMFYUI_STATS_URL, timeout=10)
        return True
    except Exception:
        return False


def find_comfy_pid(proc_root="/proc"):
    """Exact PID of the ComfyUI python process, or None.

    Scans /proc directly (no pgrep pattern, which matches the invoking
    shell): cmdline must contain main.py + --port 8188 and the process
    cwd must be the ComfyUI dir. Returns None if zero or more than one
    match (ambiguous is fail-safe: do not kill).
    """
    matches = []
    try:
        pids = os.listdir(proc_root)
    except OSError:
        return None
    for name in pids:
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == os.getpid():
            continue
        try:
            cmdline = Path(proc_root, name, "cmdline").read_bytes()
            parts = [p.decode("utf-8", "replace") for p in cmdline.split(b"\x00") if p]
            if "main.py" not in parts:
                continue
            if "8188" not in parts:
                continue
            cwd = os.readlink(os.path.join(proc_root, name, "cwd"))
            if Path(cwd) != COMFYUI_DIR:
                continue
            matches.append(pid)
        except (OSError, ValueError):
            continue
    if len(matches) != 1:
        return None
    return matches[0]


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _wait_for(predicate, timeout, sleep, now):
    deadline = now() + timeout
    while now() < deadline:
        if predicate():
            return True
        sleep(0.5)
    return predicate()


def vram_free_mib():
    """Free VRAM in MiB via the Windows nvidia-smi, or None if unknown."""
    try:
        out = subprocess.run(
            [NVIDIA_SMI_EXE, "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip().splitlines()[0].strip())
    except Exception:
        return None


def _record_eviction(pid):
    try:
        STATE_FILE.write_text(json.dumps({
            "pid": pid, "evicted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "cmd": COMFYUI_CMD,
        }) + "\n")
    except OSError:
        pass


def _clear_eviction():
    try:
        STATE_FILE.unlink()
    except OSError:
        pass


def load_eviction_state():
    """Recorded eviction (dict) if the launcher died mid-eviction, else None."""
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, ValueError):
        return None


def evict_if_idle(queue_fn=None, pid_fn=None, kill_fn=None,
                  sleep=None, now=None, log=None):
    """Stop idle ComfyUI so Nova gets the GPU. See module docstring.

    Returns one of BUSY / ERROR / EVICTED / NOT_RUNNING / DISABLED.
    The *_fn / sleep / now seams exist for unit tests; production
    callers pass nothing.
    """
    queue_fn = queue_fn or queue_status
    pid_fn = pid_fn or find_comfy_pid
    kill_fn = kill_fn or os.kill
    sleep = sleep or time.sleep
    now = now or time.monotonic
    log = log or _log

    if not EVICT_ENABLED:
        return DISABLED
    status = queue_fn()
    if status is None:
        return NOT_RUNNING
    if not queue_is_empty(status):
        log("comfyui busy: %d running, %d pending; refusing to evict"
            % (len(status.get("queue_running") or []),
               len(status.get("queue_pending") or [])))
        return BUSY
    pid = pid_fn()
    if pid is None:
        log("comfyui answers on :8188 but no unique main.py pid found; "
            "refusing to evict")
        return ERROR
    free_before = vram_free_mib()
    _record_eviction(pid)
    log("comfyui idle, stopping pid=%d (VRAM free before: %s MiB)"
        % (pid, free_before))
    try:
        kill_fn(pid, signal.SIGTERM)
    except OSError as exc:
        log("SIGTERM to comfyui pid=%d failed (%r)" % (pid, exc))
        return ERROR
    if not _wait_for(lambda: not _pid_alive(pid), STOP_WAIT_SECONDS,
                    sleep, now):
        log("comfyui pid=%d still alive %.0fs after SIGTERM; SIGKILL"
            % (pid, STOP_WAIT_SECONDS))
        try:
            kill_fn(pid, signal.SIGKILL)
        except OSError:
            pass
        if not _wait_for(lambda: not _pid_alive(pid), KILL_WAIT_SECONDS,
                        sleep, now):
            log("comfyui pid=%d would not die; leaving state file for "
                "recovery" % pid)
            return ERROR
    if not _wait_for(lambda: queue_fn() is None, 15.0, sleep, now):
        log("comfyui :8188 still answering after pid=%d stopped" % pid)
        return ERROR
    free_after = vram_free_mib()
    log("comfyui stopped pid=%d (VRAM free: %s -> %s MiB)"
        % (pid, free_before, free_after))
    return EVICTED


def restart_comfy(http_get=None, popen_fn=None, sleep=None, now=None,
                  log=None):
    """Restart ComfyUI with its original launch command. Returns bool.

    No-op True if it is already up. Clears the eviction state file on
    success so a later launcher start does not restart it twice.
    """
    http_get = http_get or _http_get_json
    popen_fn = popen_fn or subprocess.Popen
    sleep = sleep or time.sleep
    now = now or time.monotonic
    log = log or _log

    if comfy_is_up(http_get):
        _clear_eviction()
        return True
    log("restarting comfyui: %s" % COMFYUI_CMD)
    try:
        COMFYUI_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(COMFYUI_LOG_FILE, "ab")
    except OSError as exc:
        log("cannot open comfyui log %s: %r" % (COMFYUI_LOG_FILE, exc))
        return False
    try:
        proc = popen_fn(["bash", "-c", COMFYUI_CMD],
                        cwd=str(COMFYUI_DIR),
                        stdin=subprocess.DEVNULL,
                        stdout=log_file, stderr=subprocess.STDOUT,
                        start_new_session=True, close_fds=True)
    except OSError as exc:
        log("failed to restart comfyui: %r" % (exc,))
        return False
    log("comfyui restart started pid=%s" % getattr(proc, "pid", "?"))
    if _wait_for(lambda: comfy_is_up(http_get), RESTART_WAIT_SECONDS,
                sleep, now):
        _clear_eviction()
        log("comfyui healthy again on :8188")
        return True
    log("comfyui did NOT come back within %.0fs; state file kept for "
        "recovery" % RESTART_WAIT_SECONDS)
    return False


def recover_if_needed(log=None):
    """Launcher-startup recovery: restart ComfyUI if a previous launcher
    died while it was evicted. Returns True if ComfyUI is up afterwards
    (or was never evicted), False if a restart was needed and failed."""
    log = log or _log
    state = load_eviction_state()
    if state is None:
        return True
    if comfy_is_up():
        _clear_eviction()
        return True
    log("previous launcher died while comfyui was evicted "
        "(pid=%s); restarting it" % state.get("pid"))
    return restart_comfy(log=log)
