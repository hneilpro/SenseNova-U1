#!/usr/bin/env python3
"""Ollama VRAM eviction hook for the Nova API (custom/nova-serving only).

Why: SillyTavern on this box pins its Ollama model (Cydonia-24B, ~16.9GB
VRAM) forever via config.yaml `ollama.keepAlive: -1`. When a Nova generation
arrives, that residency leaves the 24GB RTX 4090 almost no headroom and
Windows starts paging Ollama's VRAM to system RAM (slow thrash).

What this does, once per Nova generation request (called from
nova_api.Handler._run, inside the generation lock):

1. Ask Ollama which models are loaded. None -> done, no-op.
2. Busy-check: never evict mid-generation. ST replies are the user's, so
   sample llama-server.exe's CPU over SAMPLE_SECONDS; a generating runner
   burns CPU continuously, an idle one reads ~0.03s per 4s window
   (measured on this box, 2026-10-21). If busy, wait and re-sample for up
   to BUSY_WAIT_SECONDS. Still busy at the deadline -> proceed WITHOUT
   evicting: a shared-GPU Nova generation beats killing an RP reply.
3. Idle -> POST keep_alive:0 for each loaded model, then poll /api/ps
   until Ollama reports nothing loaded (UNLOAD_WAIT_SECONDS cap) so the
   generation really starts with the VRAM free.
4. Nothing to restore: Ollama reloads the model automatically on the
   next SillyTavern message (one slow first reply per switch).

Transport quirk: Nova runs inside WSL, Ollama is a Windows-native app,
and WSL localhost does NOT reach Windows services on this box. So all
Ollama calls go through the Windows curl.exe / powershell.exe binaries
via WSL interop, where 127.0.0.1 is the Windows loopback. Stdlib only.

Fail-open everywhere: any error (Ollama down, interop missing, parse
failure) logs and lets the Nova generation proceed.
"""

from __future__ import annotations

import json
import subprocess
import time

OLLAMA_PS_URL = "http://127.0.0.1:11434/api/ps"
OLLAMA_GENERATE_URL = "http://127.0.0.1:11434/api/generate"
CURL_EXE = "/mnt/c/Windows/System32/curl.exe"
POWERSHELL_EXE = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"

SAMPLE_SECONDS = 4.0        # llama-server CPU sampling window
BUSY_CPU_THRESHOLD = 0.5    # CPU-seconds inside the window that mean "busy"
BUSY_WAIT_SECONDS = 60.0    # max time to wait for an in-flight ST reply
UNLOAD_WAIT_SECONDS = 30.0  # max time to wait for VRAM to actually free


def _log(msg: str) -> None:
    print("[evict] %s" % msg, flush=True)


def _run(cmd, timeout):
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout)


def loaded_models():
    """Names of models Ollama currently holds in VRAM (GET /api/ps)."""
    out = _run([CURL_EXE, "-s", "-m", "10", OLLAMA_PS_URL], timeout=15)
    data = json.loads(out.stdout)
    return [m["name"] for m in data.get("models", [])]


def llama_cpu_seconds():
    """Total CPU seconds of llama-server.exe, or None if it isn't running."""
    script = ("(Get-Process llama-server -ErrorAction SilentlyContinue | "
              "Measure-Object -Property CPU -Sum).Sum")
    out = _run([POWERSHELL_EXE, "-NoProfile", "-Command", script],
               timeout=20)
    text = (out.stdout or "").strip()
    if not text:
        return None
    return float(text.splitlines()[-1].strip())


def unload_model(name):
    """POST /api/generate with keep_alive 0: Ollama unloads immediately."""
    body = json.dumps({"model": name, "keep_alive": 0})
    out = _run([CURL_EXE, "-s", "-m", "30", "-X", "POST",
                "-H", "Content-Type: application/json",
                "-d", body, OLLAMA_GENERATE_URL], timeout=40)
    return out.stdout or ""


def _is_busy(cpu_fn, sleep):
    """Sample the runner CPU across one window. (busy, cpu_delta)."""
    t0 = cpu_fn()
    if t0 is None:
        return False, 0.0
    sleep(SAMPLE_SECONDS)
    t1 = cpu_fn()
    if t1 is None:
        return False, 0.0
    delta = max(0.0, t1 - t0)
    return delta > BUSY_CPU_THRESHOLD, delta


def evict_if_needed(ps_fn=None, cpu_fn=None, unload_fn=None,
                    sleep=None, now=None, log=None):
    """Evict Ollama's loaded model(s) so Nova gets the GPU. Fail-open.

    Returns True if a model was evicted and confirmed gone, else False.
    The *_fn / sleep / now seams exist for unit tests; production callers
    pass nothing.
    """
    ps_fn = ps_fn or loaded_models
    cpu_fn = cpu_fn or llama_cpu_seconds
    unload_fn = unload_fn or unload_model
    sleep = sleep or time.sleep
    now = now or time.monotonic
    log = log or _log

    try:
        models = ps_fn()
    except Exception as exc:
        log("ollama unreachable (%r); generating without eviction" % (exc,))
        return False
    if not models:
        return False

    # Busy-wait: an actively generating ST reply always finishes first.
    deadline = now() + BUSY_WAIT_SECONDS
    while True:
        try:
            busy, delta = _is_busy(cpu_fn, sleep)
        except Exception as exc:
            log("busy check failed (%r); proceeding without eviction"
                % (exc,))
            return False
        if not busy:
            break
        if now() >= deadline:
            log("ollama still generating after %.0fs; proceeding WITHOUT "
                "evicting %s" % (BUSY_WAIT_SECONDS, models))
            return False
        log("ollama busy (llama-server CPU +%.2fs in %.0fs window); "
            "waiting for the ST reply to finish" % (delta, SAMPLE_SECONDS))

    for name in models:
        try:
            unload_fn(name)
            log("unload requested for %s" % name)
        except Exception as exc:
            log("unload of %s failed (%r); proceeding anyway" % (name, exc))
            return False

    deadline = now() + UNLOAD_WAIT_SECONDS
    while now() < deadline:
        try:
            if not ps_fn():
                log("evicted %s; VRAM free for Nova" % (models,))
                return True
        except Exception:
            pass  # transient; keep polling until the deadline
        sleep(1.0)
    log("models still listed %.0fs after unload request; proceeding "
        "anyway" % UNLOAD_WAIT_SECONDS)
    return False
