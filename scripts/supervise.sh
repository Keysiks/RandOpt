#!/bin/bash
# Keeps a long job alive: runs a command, restarts it when it crashes or hangs.
#   PROGRESS_DIR=<dir> PROGRESS_GLOB='<find -path pattern>' scripts/supervise.sh <command> [args...]
# - progress = number of files matching PROGRESS_GLOB under PROGRESS_DIR (finished seeds / grid points);
# - a crash is retried until MAX_FAILS attempts in a row produced no new file;
# - an attempt with no new file for STALL_MIN minutes is killed (hang watchdog) and restarted;
# - the whole process group is killed between attempts so no worker keeps the GPUs busy.
# Exit code 0 only when the command itself finished normally.
PROGRESS_DIR=${PROGRESS_DIR:?set PROGRESS_DIR}
PROGRESS_GLOB=${PROGRESS_GLOB:?set PROGRESS_GLOB}
MAX_FAILS=${MAX_FAILS:-5}
STALL_MIN=${STALL_MIN:-60}
POLL_S=${POLL_S:-60}          # how often the watchdog looks at the attempt
RETRY_S=${RETRY_S:-60}        # pause before a restart
KILL_WAIT_S=${KILL_WAIT_S:-15}
PID=""

count_done() { find "$PROGRESS_DIR" -path "$PROGRESS_GLOB" 2>/dev/null | wc -l | tr -d ' '; }
newest_time() { find "$PROGRESS_DIR" -path "$PROGRESS_GLOB" -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1; }
kill_group() {
  [[ -z "$PID" ]] && return
  local pgid
  pgid=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
  [[ -z "$pgid" ]] && pgid=$PID
  kill -TERM -- "-$pgid" 2>/dev/null
  sleep "$KILL_WAIT_S"
  kill -KILL -- "-$pgid" 2>/dev/null
}
trap 'echo "[supervise] got a termination signal"; kill_group; exit 143' TERM INT

fails=0
attempt=0
while true; do
  attempt=$((attempt + 1))
  before=$(count_done)
  echo "[supervise] $(date '+%F %T') attempt $attempt, $before files so far"
  setsid "$@" &
  PID=$!
  start=$(date +%s)
  while kill -0 "$PID" 2>/dev/null; do
    sleep "$POLL_S"
    last=$(newest_time); last=${last:-0}
    ref=$(( last > start ? last : start ))
    if (( $(date +%s) - ref > STALL_MIN * 60 )); then
      echo "[supervise] no new file for ${STALL_MIN} min, killing the attempt"
      kill_group
      break
    fi
  done
  wait "$PID" 2>/dev/null
  code=$?
  kill_group   # leftovers (vLLM workers) of this attempt
  PID=""
  if (( code == 0 )); then
    echo "[supervise] finished normally"
    exit 0
  fi
  after=$(count_done)
  if (( after > before )); then fails=0; else fails=$((fails + 1)); fi
  echo "[supervise] attempt failed (exit $code), $((after - before)) new files, $fails/$MAX_FAILS failures in a row"
  if (( fails >= MAX_FAILS )); then
    echo "[supervise] giving up"
    exit 1
  fi
  sleep "$RETRY_S"
done
