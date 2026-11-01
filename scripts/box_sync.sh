#!/usr/bin/env bash
# Copy run results from the GPU box to this machine every N seconds (default 120). Ctrl-C to stop.
# usage (on the Mac, repo root): scripts/box_sync.sh [interval_s]
cd "$(dirname "$0")/.."
read -r HOST PORT < <(python3 scripts/vast.py target)
mkdir -p result/runs result/box_logs
while true; do
  rsync -az -e "ssh -p $PORT -o ConnectTimeout=15" "$HOST":/root/repo/result/runs/ result/runs/ 2>/dev/null
  rsync -az -e "ssh -p $PORT -o ConnectTimeout=15" "$HOST":/root/repo/result/box_logs/ result/box_logs/ 2>/dev/null
  echo "synced $(date +%H:%M:%S)"
  sleep "${1:-120}"
done
