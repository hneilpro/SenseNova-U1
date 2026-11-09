#!/usr/bin/env python3
"""Nova API - thin OpenAI-ish HTTP wrapper around SenseNova-U1.5 local inference.

Custom branch (custom/nova-serving) addition. NOT part of upstream SenseNova-U1.

Design:
- Stdlib only (http.server). No new dependencies in the locked venv.
- One resident model, loaded once at startup via the same utils the CLI
  examples use (load_model_and_tokenizer + make_offload_ctx), calling
  model.t2i_generate / model.it2i_generate directly - the exact calls
  examples/t2i/inference.py and examples/editing/inference.py make.
- vram_mode is HARDCODED to "fast" in this file (HARDCODED_VRAM_MODE):
  upstream's "full" default pins ~22.5GB and hangs on the 24GB RTX 4090.
  See the constant's comment block before changing it.
- Binds 127.0.0.1 only. Remote access is via the cloudflared named tunnel
  (nova.hneil.net). Every endpoint requires the bearer token.
- One generation at a time (GPU is serial); concurrent callers get 429.
- Before each generation, the Ollama eviction hook (ollama_evict.py)
  claims the GPU: if SillyTavern's Ollama model is resident (~16.9GB
  VRAM, pinned by keepAlive -1) and NOT actively generating, it is
  unloaded so Nova generates with the full card; Ollama reloads it
  automatically on the next ST message. An in-flight ST reply is waited
  out (60s cap), never evicted mid-generation. Disable globally with
  NOVA_EVICT_OLLAMA=0, or per request with {"evict_ollama": false}.
- ComfyUI eviction (comfy_evict.py) is owned by the launcher
  (nova_launcher.py), which runs it BEFORE starting this process:
  the model loads here at startup, so a per-generation hook would
  be too late. As a backstop for the direct ./start_nova_api.sh
  path, main() below also calls it before loading the model and
  refuses to start (exit 3) if ComfyUI is busy or the eviction
  fails. The launcher restarts ComfyUI when Nova stops; the
  direct path restarts it in main()'s finally block (SIGTERM
  bypasses it - the ~/.nova-comfy-evict.json state file lets a
  fresh launcher recover it). Disable with NOVA_EVICT_COMFYUI=0.

Endpoints:
  GET  /health                  -> {status, model, vram_mode, busy}
  GET  /v1/models               -> OpenAI-style model list
  POST /v1/images/generations   {prompt, width, height, seed, num_steps,
                                 cfg_scale, think_mode}
                                -> {created, data: [{b64_json, url}], think?}
  POST /v1/images/edits         {prompt, images: [b64, ...], width, height,
                                 seed, num_steps, cfg_scale, img_cfg_scale,
                                 input_max_pixels, think_mode}
                                -> same shape as generations
  GET  /outputs/<file.png>      -> a previously generated PNG

Token: read from --token_file (created with 0600 on first run if missing),
or the NOVA_API_TOKEN env var when set.
"""

from __future__ import annotations

import argparse
import base64
import hmac
import io
import json
import os
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

MODEL_ALIAS = "sensenova-u1"
MAX_BODY_BYTES = 64 * 1024 * 1024  # base64 inputs for edits can be large

# ---------------------------------------------------------------------------
# HARDCODED FOR THE RTX 4090 (24GB) - DO NOT CHANGE WITHOUT A GPU TEST.
# vram_mode MUST default to "fast" (async prefetch, generation layers
# kept resident). Changed from "balanced" 2026-10-21 after a same-seed
# benchmark on this box (2048x2048, 50 steps): fast 76.5s warm vs balanced
# 86.0s, byte-identical output. Fast pins ~21.4GB VRAM while generating
# and its first run in a session is slow (~147s) while layers are pinned.
# Upstream's default is "full" (whole 8B model resident): on this card
# that pins ~22.5GB VRAM and HUNG two generation runs - it effectively
# runs forever or crashes. "balanced" remains the proven fallback. This
# constant is deliberately independent of the patched sensenova_u1
# DEFAULT_VRAM_MODE so an upstream merge can never silently drop the
# 4090 default. Override per-experiment with the --vram_mode CLI arg only.
HARDCODED_VRAM_MODE = "fast"
# Edit inputs are downscaled to this pixel budget by default:
# 2048 * 2048 = 4194304. This is upstream's editing default
# (DEFAULT_INPUT_MAX_PIXELS = 2048 * 2048 in
# examples/editing/inference.py) and matches upstream's best-quality
# guidance to pre-resize inputs to ~2048x2048, original aspect ratio
# preserved, before inference. Changed 2026-10-22 at the owner's
# instruction; the previous local default was 1048576 (1024x1024),
# taken from local CLI run scripts, and is superseded.
HARDCODED_INPUT_MAX_PIXELS = 4194304

STATE = {
    "model_path": None,
    "vram_mode": None,
    "outputs_dir": None,
    "token": None,
    "engine": None,  # NovaEngine, set after load
    "busy": False,
}
GEN_LOCK = threading.Lock()

# Ollama eviction hook (see ollama_evict.py). Import is optional so a
# missing/broken hook module can never take the API down; the hook itself
# is fail-open as well. NOVA_EVICT_OLLAMA=0 disables it globally.
try:
    from ollama_evict import evict_if_needed
except Exception as _evict_import_exc:  # pragma: no cover - box-only path
    print("ollama eviction hook unavailable: %r" % (_evict_import_exc,),
          flush=True)
    evict_if_needed = None
EVICT_OLLAMA_ENABLED = os.environ.get("NOVA_EVICT_OLLAMA", "1") != "0"

# ComfyUI eviction hook (see comfy_evict.py). Same optional-import
# pattern: a missing/broken hook can never take the API down. The
# launcher is the primary caller; main() below uses it as a backstop
# for the direct start path. NOVA_EVICT_COMFYUI=0 disables it globally.
try:
    import comfy_evict
except Exception as _comfy_import_exc:  # pragma: no cover - box-only path
    print("comfyui eviction hook unavailable: %r" % (_comfy_import_exc,),
          flush=True)
    comfy_evict = None
EVICT_COMFYUI_ENABLED = os.environ.get("NOVA_EVICT_COMFYUI", "1") != "0"


# --------------------------------------------------------------------------
# Engine: one resident model, t2i + edit through the upstream utils.
# --------------------------------------------------------------------------
class NovaEngine:
    def __init__(self, model_path: str, vram_mode: str, device: str = "cuda"):
        import torch

        from sensenova_u1.utils import (
            load_model_and_tokenizer,
            make_offload_ctx,
            vram_mode_keeps_generation_resident,
            vram_mode_to_prefetch_count,
        )

        self._torch = torch
        self._make_offload_ctx = make_offload_ctx
        self._keeps_resident = vram_mode_keeps_generation_resident
        self.device = device
        self.vram_mode = vram_mode
        self.prefetch_count = vram_mode_to_prefetch_count(vram_mode)
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_path,
            dtype=torch.bfloat16,
            device=device,
            for_offload=self.prefetch_count > 0,
        )

    def _ctx(self):
        return self._make_offload_ctx(
            self.model,
            self.prefetch_count,
            self.device,
            keep_generation_resident=self._keeps_resident(self.vram_mode),
        )

    def generate(self, prompt, width, height, seed, num_steps, cfg_scale,
                 think_mode):
        torch = self._torch
        with torch.inference_mode():
            with self._ctx() as offloaded:
                out = offloaded.t2i_generate(
                    self.tokenizer,
                    prompt,
                    image_size=(width, height),
                    cfg_scale=cfg_scale,
                    cfg_norm="none",
                    timestep_shift=3.0,
                    cfg_interval=(0.0, 1.0),
                    num_steps=num_steps,
                    batch_size=1,
                    seed=seed,
                    think_mode=think_mode,
                )
        if think_mode:
            tensor, think_text = out
        else:
            tensor, think_text = out, ""
        return tensor_to_pil(tensor), think_text

    def edit(self, prompt, images, width, height, seed, num_steps, cfg_scale,
             img_cfg_scale, think_mode):
        torch = self._torch
        with torch.inference_mode():
            with self._ctx() as offloaded:
                out = offloaded.it2i_generate(
                    self.tokenizer,
                    prompt,
                    images,
                    image_size=(width, height),
                    cfg_scale=cfg_scale,
                    img_cfg_scale=img_cfg_scale,
                    cfg_norm="none",
                    timestep_shift=3.0,
                    cfg_interval=(0.0, 1.0),
                    num_steps=num_steps,
                    batch_size=1,
                    think_mode=think_mode,
                    seed=seed,
                )
        if think_mode:
            return tensor_to_pil(out[0]), out[1]
        return tensor_to_pil(out), ""


def tensor_to_pil(batch):
    """Same denorm the CLI examples use: (x * 0.5 + 0.5) clamped to [0, 1]."""
    import numpy as np
    import torch
    from PIL import Image

    mean = torch.tensor((0.5, 0.5, 0.5), device=batch.device, dtype=batch.dtype)
    std = torch.tensor((0.5, 0.5, 0.5), device=batch.device, dtype=batch.dtype)
    x = (batch.float() * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)).clamp(0, 1)
    arr = (x.permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    return [Image.fromarray(a) for a in arr]


def resize_to_budget(img, max_pixels: int):
    """Downscale (never upscale) so w*h <= max_pixels, dims multiple of 16.

    Approximation of the editing CLI's smart_resize budget step.
    The serving default budget is --input_max_pixels 4194304
    (2048x2048); see HARDCODED_INPUT_MAX_PIXELS above.
    """
    import math

    from PIL import Image

    w, h = img.size
    if w * h <= max_pixels:
        return img
    scale = math.sqrt(max_pixels / float(w * h))
    nw = max(16, int(w * scale) // 16 * 16)
    nh = max(16, int(h * scale) // 16 * 16)
    return img.resize((nw, nh), Image.LANCZOS)


# --------------------------------------------------------------------------
# HTTP layer (stdlib only)
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "NovaAPI/1.0"
    protocol_version = "HTTP/1.1"

    # -- helpers -----------------------------------------------------------
    def log_message(self, fmt, *args):  # keep stdout log concise
        sys.stdout.write("%s %s\n" % (self.address_string(), fmt % args))
        sys.stdout.flush()

    def _authed(self) -> bool:
        hdr = self.headers.get("Authorization", "")
        token = hdr[7:].strip() if hdr.lower().startswith("bearer ") else ""
        if not token:
            token = self.headers.get("X-API-Key", "")
        return bool(token) and hmac.compare_digest(token, STATE["token"])

    def _send_json(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, message: str):
        self._send_json(code, {"error": {"message": message, "code": code}})

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ValueError("bad or oversized body")
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    # -- routing -----------------------------------------------------------
    def do_GET(self):
        path = urlparse(self.path).path
        if not self._authed():
            return self._error(401, "missing or invalid token")
        if path == "/health":
            return self._send_json(200, {
                "status": "ok",
                "model": STATE["model_path"],
                "alias": MODEL_ALIAS,
                "vram_mode": STATE["vram_mode"],
                "busy": STATE["busy"],
                "ollama_evict": bool(EVICT_OLLAMA_ENABLED
                                     and evict_if_needed is not None),
                "comfy_evict": bool(EVICT_COMFYUI_ENABLED
                                    and comfy_evict is not None),
            })
        if path == "/v1/models":
            return self._send_json(200, {
                "object": "list",
                "data": [{"id": MODEL_ALIAS, "object": "model",
                          "owned_by": "sensenova"}],
            })
        if path.startswith("/outputs/"):
            return self._serve_output(path[len("/outputs/"):])
        return self._error(404, "not found")

    def _serve_output(self, name: str):
        name = os.path.basename(name)  # no path traversal
        fpath = Path(STATE["outputs_dir"]) / name
        if not fpath.is_file() or fpath.suffix.lower() != ".png":
            return self._error(404, "no such output")
        data = fpath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/v1/images/generations", "/v1/images/edits"):
            return self._error(404, "not found")
        if not self._authed():
            return self._error(401, "missing or invalid token")
        try:
            req = self._read_json()
        except Exception:
            return self._error(400, "invalid JSON body")
        prompt = (req.get("prompt") or "").strip()
        if not prompt:
            return self._error(400, "prompt is required")
        if not GEN_LOCK.acquire(blocking=False):
            return self._error(429, "busy: a generation is already running")
        STATE["busy"] = True
        try:
            result = self._run(path, req, prompt)
            return self._send_json(200, result)
        except Exception as exc:  # surface model errors as JSON, keep serving
            sys.stdout.write("generation failed: %r\n" % (exc,))
            sys.stdout.flush()
            return self._error(500, "generation failed: %s" % (exc,))
        finally:
            STATE["busy"] = False
            GEN_LOCK.release()

    def _run(self, path, req, prompt):
        # Claim the GPU from Ollama first (busy-waits an in-flight ST
        # reply, fail-open: any hook problem still lets us generate).
        if (EVICT_OLLAMA_ENABLED and evict_if_needed is not None
                and req.get("evict_ollama", True)):
            try:
                evict_if_needed()
            except Exception as exc:
                sys.stdout.write("[evict] hook error: %r\n" % (exc,))
                sys.stdout.flush()
        eng = STATE["engine"]
        width = int(req.get("width", 2048))
        height = int(req.get("height", 2048))
        seed = int(req.get("seed", 0))
        num_steps = int(req.get("num_steps", 50))
        cfg_scale = float(req.get("cfg_scale", 4.0))
        think_mode = bool(req.get("think_mode", False))
        if path == "/v1/images/generations":
            images, think = eng.generate(prompt, width, height, seed,
                                         num_steps, cfg_scale, think_mode)
        else:
            from PIL import Image

            raw_images = req.get("images") or []
            if not raw_images:
                raise ValueError("images (base64 list) is required for edits")
            budget = int(req.get("input_max_pixels", HARDCODED_INPUT_MAX_PIXELS))
            inputs = []
            for b64 in raw_images:
                img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                inputs.append(resize_to_budget(img, budget))
            images, think = eng.edit(
                prompt, inputs, width, height, seed, num_steps, cfg_scale,
                float(req.get("img_cfg_scale", 1.0)), think_mode)
        out_dir = Path(STATE["outputs_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        data = []
        for i, img in enumerate(images):
            name = "nova-api-%s-seed%d%s.png" % (
                stamp, seed, "" if len(images) == 1 else "-%d" % i)
            fpath = out_dir / name
            img.save(fpath)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            data.append({
                "b64_json": base64.b64encode(buf.getvalue()).decode(),
                "url": "/outputs/" + name,
            })
            sys.stdout.write("[saved] %s\n" % (fpath,))
            sys.stdout.flush()
        payload = {"created": int(time.time()), "data": data}
        if think:
            payload["think"] = think
        return payload


# --------------------------------------------------------------------------
def load_or_create_token(token_file: str) -> str:
    env = os.environ.get("NOVA_API_TOKEN")
    if env:
        return env.strip()
    p = Path(token_file).expanduser()
    if p.is_file():
        return p.read_text().strip()
    token = secrets.token_urlsafe(32)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(token + "\n")
    os.chmod(p, 0o600)
    return token


def main():
    ap = argparse.ArgumentParser(description="Nova API - SenseNova-U1.5 wrapper")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--model_path", default="sensenova/SenseNova-U1.5-8B-MoT")
    ap.add_argument("--vram_mode", default=HARDCODED_VRAM_MODE,
                    help="default: 'fast' - HARDCODED for the 24GB "
                         "RTX 4090 (see HARDCODED_VRAM_MODE). Upstream's "
                         "'full' hangs/crashes on this card.")
    ap.add_argument("--token_file", default="~/.nova-api/token.txt")
    ap.add_argument("--outputs_dir",
                    default="/mnt/c/Users/Harish/Documents/GitHub/outputs")
    args = ap.parse_args()

    if args.vram_mode == "full":
        print("WARNING: --vram_mode full on the 24GB RTX 4090 pins ~22.5GB "
              "and has hung generation runs before. 'fast' is the "
              "hardcoded default for a reason.", flush=True)

    STATE["model_path"] = args.model_path
    STATE["vram_mode"] = args.vram_mode
    STATE["outputs_dir"] = args.outputs_dir
    STATE["token"] = load_or_create_token(args.token_file)

    # ComfyUI backstop (launcher is the primary caller and normally
    # already evicted; here queue_status() then returns None and this
    # is a no-op). Refuse to start into a busy ComfyUI job, or if the
    # eviction itself fails: the model load would fight it for VRAM.
    comfy_status = None
    if EVICT_COMFYUI_ENABLED and comfy_evict is not None:
        comfy_status = comfy_evict.evict_if_idle()
        if comfy_status == comfy_evict.BUSY:
            print("comfyui is running a job; refusing to start Nova "
                  "(comfyui_busy).", flush=True)
            sys.exit(3)
        if comfy_status == comfy_evict.ERROR:
            print("comfyui eviction failed; refusing to start Nova.",
                  flush=True)
            sys.exit(3)

    print("loading %s (vram_mode=%s) ..." % (args.model_path, STATE["vram_mode"]),
          flush=True)
    STATE["engine"] = NovaEngine(args.model_path, STATE["vram_mode"])
    print("model loaded; listening on http://%s:%d" % (args.host, args.port),
          flush=True)
    try:
        ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # Direct-start path only: the launcher restarts ComfyUI itself
        # in stop_nova(). SIGTERM bypasses this; the state file lets a
        # fresh launcher recover ComfyUI in that case.
        if comfy_evict is not None and comfy_status == comfy_evict.EVICTED:
            comfy_evict.restart_comfy()


if __name__ == "__main__":
    main()
