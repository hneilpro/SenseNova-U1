#!/usr/bin/env python3
"""Nova auto-start launcher proxy (4090 box, WSL).

Listens on 127.0.0.1:8001, the port the Cloudflare tunnel
(nova.hneil.net) and nova_bridge.py (127.0.0.1:8002 -> 127.0.0.1:8001)
already point at. Nova itself is moved to an internal port (8003).

Rule: Nova only runs when there is an actual generation request.
  * Trigger ONLY: POST /v1/images/generations, POST /v1/images/edits.
  * Everything else (GET /health, GET /v1/models, /robots.txt, ...)
    returns a clear "nova_stopped" JSON WITHOUT starting Nova.
    Health probes are frequent and must never wake Nova.
  * On a generation request with Nova down: start Nova
    (--vram_mode fast, port 8003), wait up to 180s for TCP readiness
    (plus the "model loaded" log line when visible), proxy the
    original request through with a 1800s timeout, return it unchanged.
  * Idle shutdown: if no generation request has completed for
    IDLE_TIMEOUT_S (default 1800s = 30 min) and none is in flight,
    stop Nova to free VRAM.
  * ComfyUI eviction (serving/comfy_evict.py, owner-authorized
    2026-11-09): before Nova starts, if ComfyUI is idle (empty
    /queue) it is stopped by exact PID so Nova gets the full card;
    it is restarted when Nova stops, and also if Nova fails to
    start. If ComfyUI is running or has a queued job, Nova is NOT
    started at all (503 comfyui_busy) and nothing is stopped.
    Disable globally with NOVA_EVICT_COMFYUI=0, or per request
    with {"evict_comfyui": false}. A state file
    ~/.nova-comfy-evict.json lets a fresh launcher recover ComfyUI
    if a previous launcher died while it was down.

Security / privacy:
  * Stdlib only. No request bodies, no headers, and never any
    Authorization/token value are logged. Only method, path, status,
    duration, and start/stop events are logged.
  * This launcher never reads ~/.nova-api/token.txt. Clients
    (tunnel callers, nova_bridge) keep supplying their own token;
    it is forwarded to Nova untouched on generation requests.
  * Nova is started with the same pattern as serving/start_nova_api.sh
    (setsid, nohup-equivalent via start_new_session, append log),
    but with its own pid/log so the original files are not confused:
      Nova pid:  ~/nova-api-ondemand.pid
      Nova log:  ~/nova-api-ondemand.log
      Launcher pid: ~/nova-launcher.pid
      Launcher log: ~/nova-launcher.log

Start (on the box, in WSL):
  setsid nohup python3 ~/nova_launcher.py >> ~/nova-launcher.log 2>&1 < /dev/null &
  echo $! > ~/nova-launcher.pid
Canonical source: serving/nova_launcher.py in hneilpro/SenseNova-U1
(custom/nova-serving); edit the repo copy and redeploy to
~/nova_launcher.py, keeping a dated .bak of the deployed copy.
Stop:
  kill "$(cat ~/nova-launcher.pid)"
Rollback (Nova back on 8001 directly):
  kill "$(cat ~/nova-launcher.pid)"   # stop launcher, frees 8001
  # then, if Nova on 8003 is still running, stop it:
  kill "$(cat ~/nova-api-ondemand.pid)" 2>/dev/null || true
  cd /mnt/c/Users/Harish/Documents/GitHub/hneilpro/SenseNova-U1
  ./serving/start_nova_api.sh   # original script, port 8001, fast mode
"""

import http.client
import http.server
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import threading
import time

LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 8001
NOVA_HOST = "127.0.0.1"
NOVA_PORT = 8003  # internal only; tunnel/bridge keep using 8001
NOVA_DIR = pathlib.Path("/mnt/c/Users/Harish/Documents/GitHub/hneilpro/SenseNova-U1")
NOVA_PYTHON = NOVA_DIR / ".venv" / "bin" / "python"
NOVA_SCRIPT = NOVA_DIR / "serving" / "nova_api.py"
HOME = pathlib.Path.home()
NOVA_PID_FILE = HOME / "nova-api-ondemand.pid"
NOVA_LOG_FILE = HOME / "nova-api-ondemand.log"

GENERATION_PATHS = {"/v1/images/generations", "/v1/images/edits"}
START_TIMEOUT_S = 180
PROXY_TIMEOUT_S = 1800
IDLE_TIMEOUT_S = 1800  # 30 minutes; change here only with user approval
IDLE_CHECK_INTERVAL_S = 30

_lock = threading.Lock()
_nova_proc = None  # subprocess.Popen or None
_in_flight = 0
_last_generation_done = 0.0  # monotonic time of last completed generation
_comfy_evicted = False  # True while we stopped ComfyUI for this Nova run
_last_start_error = None  # "comfyui_busy" | "comfyui_evict_failed" | None

# ComfyUI eviction hook (see serving/comfy_evict.py). Import is optional
# so a missing/broken hook module can never take the launcher down.
# NOVA_EVICT_COMFYUI=0 disables it globally.
sys.path.insert(0, str(NOVA_DIR / "serving"))
try:
    import comfy_evict
except Exception as _comfy_import_exc:  # pragma: no cover - box-only path
    print("comfyui eviction hook unavailable: %r" % (_comfy_import_exc,),
          flush=True)
    comfy_evict = None
EVICT_COMFYUI_ENABLED = os.environ.get("NOVA_EVICT_COMFYUI", "1") != "0"


def _restore_comfy_if_evicted():
    """Restart ComfyUI if we stopped it for Nova. Safe to call repeatedly."""
    global _comfy_evicted
    if not _comfy_evicted:
        return
    _comfy_evicted = False
    if comfy_evict is None:
        log("ERROR: comfyui was evicted but the hook module is gone; "
            "restart it manually (see ~/.nova-comfy-evict.json)")
        return
    if comfy_evict.restart_comfy():
        log("comfyui restarted after Nova")
    else:
        log("ERROR: comfyui restart FAILED after Nova; state kept in "
            "~/.nova-comfy-evict.json - restart it manually")


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _nova_tcp_ready():
    try:
        with socket.create_connection((NOVA_HOST, NOVA_PORT), timeout=2):
            return True
    except OSError:
        return False


def _nova_log_says_loaded():
    try:
        text = NOVA_LOG_FILE.read_text(errors="replace")
    except OSError:
        return False
    return "model loaded" in text


def _nova_pid_alive():
    global _nova_proc
    if _nova_proc is not None:
        return _nova_proc.poll() is None
    try:
        pid = int(NOVA_PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _write_nova_pid(pid):
    try:
        NOVA_PID_FILE.write_text(str(pid) + "\n")
    except OSError as exc:
        log(f"warn: could not write nova pid file: {exc!r}")


def start_nova(allow_comfy_evict=True):
    """Start Nova on the internal port if it is not already up. Returns True if ready.

    allow_comfy_evict=False (per-request {"evict_comfyui": false}) still
    refuses to start while ComfyUI is busy, but never stops it.
    """
    global _nova_proc, _last_start_error, _comfy_evicted
    _last_start_error = None
    with _lock:
        if _nova_proc is not None and _nova_proc.poll() is None:
            # Another generation request is already waiting on this start;
            # do NOT spawn a duplicate Nova. Caller waits for TCP below.
            log("nova start already in progress, waiting for it")
            return True
        if _nova_tcp_ready() and _nova_pid_alive():
            log("nova already running on internal port, reusing it")
            return True
        if _nova_tcp_ready():
            # Something else answers on 8003; do not start a duplicate.
            log("warn: something already answers on internal port 8003, reusing it")
            return True
        # Claim the GPU from ComfyUI BEFORE Nova loads its model: the
        # model load happens at nova_api startup, so evicting later
        # would be too late. Busy ComfyUI => no Nova at all.
        if EVICT_COMFYUI_ENABLED and comfy_evict is not None:
            if allow_comfy_evict:
                status = comfy_evict.evict_if_idle()
                if status == comfy_evict.BUSY:
                    log("comfyui is running a job; refusing to start Nova")
                    _last_start_error = "comfyui_busy"
                    return False
                if status == comfy_evict.ERROR:
                    log("comfyui eviction failed; refusing to start Nova")
                    _last_start_error = "comfyui_evict_failed"
                    return False
                if status == comfy_evict.EVICTED:
                    _comfy_evicted = True
            else:
                # Opt-out: never stop ComfyUI, but still refuse to start
                # Nova into a busy ComfyUI job.
                queue = comfy_evict.queue_status()
                if queue is not None and not comfy_evict.queue_is_empty(queue):
                    log("comfyui is running a job and eviction was opted "
                        "out; refusing to start Nova")
                    _last_start_error = "comfyui_busy"
                    return False
        log("starting nova (vram_mode=fast, internal port 8003)")
        try:
            log_file = open(NOVA_LOG_FILE, "ab")
        except OSError as exc:
            log(f"error: cannot open nova log {NOVA_LOG_FILE}: {exc!r}")
            _restore_comfy_if_evicted()
            return False
        try:
            # Tell the child whether it may evict ComfyUI itself (its
            # main() backstop). Opt-out must survive into the child.
            child_env = dict(os.environ)
            if not allow_comfy_evict:
                child_env["NOVA_EVICT_COMFYUI"] = "0"
            proc = subprocess.Popen(
                [str(NOVA_PYTHON), str(NOVA_SCRIPT),
                 "--vram_mode", "fast", "--port", str(NOVA_PORT)],
                cwd=str(NOVA_DIR),
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
                env=child_env,
            )
        except OSError as exc:
            log(f"error: failed to start nova: {exc!r}")
            _restore_comfy_if_evicted()
            return False
        _nova_proc = proc
        _write_nova_pid(proc.pid)
        log(f"nova started pid={proc.pid}")
    # Wait for readiness outside the lock (polling only).
    deadline = time.monotonic() + START_TIMEOUT_S
    saw_log = False
    while time.monotonic() < deadline:
        if _nova_proc is not None and _nova_proc.poll() is not None:
            log(f"error: nova exited early with code {_nova_proc.poll()}")
            _restore_comfy_if_evicted()
            return False
        if _nova_log_says_loaded():
            saw_log = True
        if _nova_tcp_ready():
            log(f"nova ready on internal port (log 'model loaded' seen: {saw_log})")
            return True
        time.sleep(2)
    log("error: nova did not become ready within 180s")
    _restore_comfy_if_evicted()
    return False


def stop_nova(reason):
    """Stop the Nova process we started, if any. Safe to call repeatedly."""
    global _nova_proc
    with _lock:
        proc = _nova_proc
        pid = None
        if proc is not None:
            pid = proc.pid
        else:
            try:
                pid = int(NOVA_PID_FILE.read_text().strip())
            except (OSError, ValueError):
                pid = None
        if pid is None:
            _restore_comfy_if_evicted()
            return
        # Only kill if it still looks like a nova_api process.
        try:
            cmdline = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes()
            if b"nova_api.py" not in cmdline:
                log(f"warn: pid {pid} is not nova_api.py, not killing it")
                _restore_comfy_if_evicted()
                return
        except OSError:
            # Process already gone (or /proc unavailable); clear state.
            _nova_proc = None
            _restore_comfy_if_evicted()
            return
        log(f"stopping nova pid={pid} ({reason})")
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    # Wait outside the lock, then SIGKILL if needed.
    for _ in range(10):
        try:
            os.kill(pid, 0)
        except OSError:
            break
        time.sleep(1)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    with _lock:
        if _nova_proc is not None and _nova_proc.pid == pid:
            _nova_proc = None
    log(f"nova pid={pid} stopped")
    _restore_comfy_if_evicted()


def idle_watchdog():
    while True:
        time.sleep(IDLE_CHECK_INTERVAL_S)
        with _lock:
            in_flight = _in_flight
            last_done = _last_generation_done
        if in_flight > 0 or last_done == 0.0:
            continue
        idle_for = time.monotonic() - last_done
        if idle_for >= IDLE_TIMEOUT_S and _nova_pid_alive():
            stop_nova(f"idle for {idle_for:.0f}s (timeout {IDLE_TIMEOUT_S}s)")


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "NovaLauncher/1.0"

    def log_message(self, *args):  # keep our own log format; suppress default
        pass

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stopped_response(self):
        self._send_json(503, {
            "status": "nova_stopped",
            "detail": "Nova is stopped and will start on a generation request "
                      "(POST /v1/images/generations or POST /v1/images/edits).",
            "nova": "stopped",
        })

    def _handle(self, method):
        started = time.monotonic()
        # Split off query string for path matching only.
        path_only = self.path.split("?", 1)[0]
        is_generation = (method == "POST" and path_only in GENERATION_PATHS)
        if not is_generation:
            # Never wake Nova for probes, health, models, or anything else.
            self._stopped_response()
            log(f"{method} {path_only} -> 503 nova_stopped (no start)")
            return
        # Generation request: wake Nova if needed, then proxy unchanged.
        global _in_flight, _last_generation_done
        with _lock:
            _in_flight += 1
        try:
            # Read the body BEFORE starting Nova: the per-request
            # {"evict_comfyui": false} opt-out must be known up front.
            # Bodies and headers are never logged.
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length > 0 else None
            allow_comfy_evict = True
            if body:
                try:
                    allow_comfy_evict = json.loads(
                        body.decode("utf-8")).get("evict_comfyui", True) is not False
                except Exception:
                    allow_comfy_evict = True
            if not start_nova(allow_comfy_evict=allow_comfy_evict):
                if _last_start_error == "comfyui_busy":
                    self._send_json(503, {
                        "status": "comfyui_busy",
                        "detail": "ComfyUI is running a job; Nova was not "
                                  "started and nothing was stopped. Retry "
                                  "when the ComfyUI queue is empty.",
                    })
                elif _last_start_error == "comfyui_evict_failed":
                    self._send_json(503, {
                        "status": "comfyui_evict_failed",
                        "detail": "ComfyUI could not be safely stopped; "
                                  "Nova was not started.",
                    })
                else:
                    self._send_json(503, {
                        "status": "nova_start_failed",
                        "detail": "Nova did not become ready within 180s; see nova-api-ondemand.log.",
                    })
                log(f"POST {path_only} -> 503 {_last_start_error or 'nova_start_failed'}")
                return
            # Forward headers unchanged except hop-by-hop / host handling.
            # Never log header values (tokens live here).
            forward_headers = {}
            for key, value in self.headers.items():
                lk = key.lower()
                if lk in ("host", "connection", "content-length",
                          "transfer-encoding", "keep-alive", "upgrade"):
                    continue
                forward_headers[key] = value
            conn = http.client.HTTPConnection(NOVA_HOST, NOVA_PORT,
                                              timeout=PROXY_TIMEOUT_S)
            try:
                conn.request(method, self.path, body=body, headers=forward_headers)
                resp = conn.getresponse()
                resp_body = resp.read()
            finally:
                conn.close()
            self.send_response(resp.status, resp.reason)
            for key, value in resp.getheaders():
                lk = key.lower()
                if lk in ("connection", "transfer-encoding", "content-length",
                          "keep-alive"):
                    continue
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
            log(f"POST {path_only} -> {resp.status} in {time.monotonic() - started:.1f}s")
        except Exception as exc:  # proxy / client errors; never leak internals
            log(f"POST {path_only} -> proxy error: {type(exc).__name__}")
            try:
                self._send_json(502, {
                    "status": "nova_proxy_error",
                    "detail": "Failed proxying the generation request to Nova.",
                })
            except Exception:
                pass
        finally:
            with _lock:
                _in_flight -= 1
                _last_generation_done = time.monotonic()

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def do_PUT(self):
        self._handle("PUT")

    def do_DELETE(self):
        self._handle("DELETE")

    def do_HEAD(self):
        self._handle("HEAD")


def main():
    server = http.server.ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    server.daemon_threads = True
    threading.Thread(target=idle_watchdog, daemon=True).start()
    log(f"nova launcher listening on {LISTEN_HOST}:{LISTEN_PORT} "
        f"(nova internal port {NOVA_PORT}, idle timeout {IDLE_TIMEOUT_S}s)")
    log(f"nova already running: {_nova_pid_alive()} (not started by launcher)")
    if comfy_evict is not None:
        comfy_evict.recover_if_needed()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
