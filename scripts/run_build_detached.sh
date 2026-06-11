#!/usr/bin/env bash
# Run index build detached from any IDE terminal (survives Cursor/SSH disconnect).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
LOG="$ROOT/logs/build_paragraph.log"
PIDFILE="$ROOT/logs/build_paragraph.pid"

if pgrep -f "$ROOT/scripts/build_index.py" >/dev/null 2>&1 || pgrep -f "python -u scripts/build_index.py" >/dev/null 2>&1; then
  echo "Build already running. Check: tail -f $LOG"
  exit 1
fi

nohup python -u scripts/build_index.py >>"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "Started paragraph index build (PID $(cat "$PIDFILE"))"
echo "Log: $LOG"
echo "Resume after crash: $ROOT/scripts/run_build_detached.sh (if not already running)"
