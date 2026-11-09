#!/bin/bash
# Start the Nova API server detached. Extra args pass through to nova_api.py
# (e.g. --port 8002). Stop with: kill "$(cat ~/nova-api.pid)"
set -e
cd "$(dirname "$0")/.."
LOG="$HOME/nova-api.log"
PIDFILE="$HOME/nova-api.pid"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running: pid $(cat "$PIDFILE")"
  exit 1
fi
# --vram_mode fast is passed explicitly on purpose: benchmarked fastest
# warm on the 24GB RTX 4090 (2026-10-21; see nova_api.py). Upstream
# default "full" pins ~22.5GB and hangs. "$@" comes after, so a
# deliberate override still wins.
setsid nohup .venv/bin/python serving/nova_api.py --vram_mode fast "$@" >> "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "started pid $(cat "$PIDFILE") - log: $LOG (model load takes a minute or two)"
