#!/bin/bash
# Start the `nova` cloudflared named tunnel detached (nova.hneil.net -> :8001).
# The tunnels on this box are manual processes: after a Windows sleep/reboot
# or WSL restart, run this AND start_nova_api.sh to bring Nova back.
# Stop with: kill "$(cat ~/nova-tunnel.pid)"
set -e
PIDFILE="$HOME/nova-tunnel.pid"
LOG="$HOME/nova-tunnel.log"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running: pid $(cat "$PIDFILE")"
  exit 1
fi
setsid nohup cloudflared tunnel --config "$HOME/.cloudflared/nova-config.yml" \
  run nova >> "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "tunnel started pid $(cat "$PIDFILE") - log: $LOG"
