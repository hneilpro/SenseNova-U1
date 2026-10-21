# Nova API — serving SenseNova-U1.5 on this box

Custom-branch (`custom/nova-serving`) addition; nothing here is upstream.

## What it is

`nova_api.py` is a thin, stdlib-only HTTP wrapper around the same local
inference the CLI examples run. It loads
`sensenova/SenseNova-U1.5-8B-MoT` once, resident, with the locally patched
default VRAM mode (`balanced` layer offload — see
`src/sensenova_u1/utils/offload.py`), and calls `t2i_generate` /
`it2i_generate` through `load_model_and_tokenizer` + `make_offload_ctx`,
exactly like `examples/t2i/inference.py` and `examples/editing/inference.py`.

- Binds `127.0.0.1:8001` only. Public access is via the cloudflared named
  tunnel `nova` → `https://nova.hneil.net` (config
  `~/.cloudflared/nova-config.yml`, run manually like the other tunnels).
- Every endpoint requires `Authorization: Bearer <token>`. The token lives
  in `~/.nova-api/token.txt` (0600, created on first start), or set
  `NOVA_API_TOKEN`. It is never committed.
- One generation at a time; concurrent requests get `429`.
- Outputs are saved to `/mnt/c/Users/Harish/Documents/GitHub/outputs/`
  (same folder the CLI runs use) as `nova-api-<timestamp>-seed<N>.png`.

Note: `examples/serving/client.py` targets the LightLLM deployment and is
**not** compatible with this wrapper; use the endpoints below.

## The 4090 flag — hardcoded, do not remove

Generation on this box requires **`--vram_mode balanced`**
(async-prefetch layer offload). Upstream's default, `full`, keeps the whole
8B model resident: on the 24GB RTX 4090 that pins ~22.5GB VRAM and hung two
runs — it takes forever or crashes. Every proven CLI run here passed
`--vram_mode balanced` explicitly.

So it is set in three places, on purpose:

1. `nova_api.py`: `HARDCODED_VRAM_MODE = "balanced"` is the argparse
   default — independent of the upstream
   `sensenova_u1.utils.DEFAULT_VRAM_MODE` patch, so a merge can't lose it.
2. `start_nova_api.sh` passes `--vram_mode balanced` explicitly
   (later args can still override it deliberately; the server also warns
   loudly if started with `full`).
3. `src/sensenova_u1/utils/offload.py` carries the same default for the
   CLI examples (local patch commit on this branch).

Second proven default, also hardcoded in `nova_api.py`:
edit inputs downscale to `input_max_pixels = 1048576`, matching the CLI
run scripts.

## Run

```bash
cd /mnt/c/Users/Harish/Documents/GitHub/hneilpro/SenseNova-U1
serving/start_nova_api.sh          # detached, log ~/nova-api.log, pid ~/nova-api.pid
tail -f ~/nova-api.log             # wait for "model loaded; listening"
kill "$(cat ~/nova-api.pid)"       # stop
```

## Endpoints

```bash
TOKEN=$(cat ~/.nova-api/token.txt)

curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8001/health
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8001/v1/models

# text-to-image (defaults: 2048x2048, seed 0, 50 steps, cfg 4.0)
curl -X POST -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  http://127.0.0.1:8001/v1/images/generations \
  -d '{"prompt": "A cozy coffee shop storefront", "seed": 42}'

# image edit: images are base64 PNG/JPEG strings, downscaled to
# input_max_pixels (default 1048576, same as the CLI run scripts)
curl -X POST -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  http://127.0.0.1:8001/v1/images/edits \
  -d '{"prompt": "...", "images": ["<base64>", "..."], "width": 1536, "height": 2720, "seed": 401}'
```

Responses are OpenAI-shaped: `{created, data: [{b64_json, url}]}` where
`url` is `/outputs/<file>` on the same server (auth required).

## Tunnel

```bash
serving/start_nova_tunnel.sh       # detached, log ~/nova-tunnel.log, pid ~/nova-tunnel.pid
kill "$(cat ~/nova-tunnel.pid)"    # stop
# config: ~/.cloudflared/nova-config.yml
# ingress: nova.hneil.net -> http://127.0.0.1:8001
```

After a Windows sleep/reboot or WSL restart, both pieces need relaunching
(they are plain detached processes, like every tunnel on this box):
`serving/start_nova_api.sh` and `serving/start_nova_tunnel.sh`, in any order.

The main tunnels are root-owned and cannot be edited from this account,
so Nova has its own tunnel — same isolation pattern as the old monitor's
quick tunnel, but a named tunnel with a stable hostname.

## Update workflow (upstream → main → custom)

`main` is kept pristine upstream. All local work lives on this branch.

```bash
git fetch upstream
git checkout main && git merge --ff-only upstream/main
git push origin main                # keep the fork's main in sync
# smoke-test pristine upstream first, e.g. one CLI generation:
.venv/bin/python examples/t2i/inference.py \
  --model_path sensenova/SenseNova-U1.5-8B-MoT --prompt "test" \
  --output /tmp/main-smoke.png --vram_mode balanced
git checkout custom/nova-serving && git merge main
# watch for conflicts in src/sensenova_u1/utils/offload.py
# (the DEFAULT_VRAM_MODE = "balanced" patch must survive)
serving/start_nova_api.sh           # restart, then /health + one generation
git push origin custom/nova-serving
```
