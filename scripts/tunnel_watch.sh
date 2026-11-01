#!/usr/bin/env bash
# Keep the SSH tunnel to vLLM alive (for runs on the Mac): restart it when /v1/models fails twice in a row.
# usage (repo root): scripts/tunnel_watch.sh [log=result/tunnel_watch.log]
cd "$(dirname "$0")/.."
LOG=${1:-result/tunnel_watch.log}; mkdir -p "$(dirname "$LOG")"
TPID=""
start() {
  [ -n "$TPID" ] && { pkill -P "$TPID" 2>/dev/null; kill "$TPID" 2>/dev/null; }
  python3 scripts/vast.py tunnel >/dev/null 2>&1 &
  TPID=$!
  sleep 8
  echo "[$(date +%T)] tunnel (re)started pid $TPID" >> "$LOG"
}
start
fails=0
while true; do
  if curl -s -m 6 -o /dev/null localhost:8000/v1/models; then fails=0; else fails=$((fails+1)); fi
  if [ "$fails" -ge 2 ]; then echo "[$(date +%T)] tunnel hung" >> "$LOG"; start; fails=0; fi
  sleep 4
done
