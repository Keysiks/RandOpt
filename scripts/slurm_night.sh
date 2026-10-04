#!/bin/bash
# Overnight RandOpt: Qwen2.5-32B-Instruct on MATH-500 and GSM8K (500 test problems), seeds shared by both
# datasets. 2 GPUs (tensor parallel, the base-weights copy lives in host RAM). Submit from the repo root:
#   sbatch scripts/slurm_night.sh                  # extra arguments are passed to evaluate.py
#   sbatch scripts/slurm_night.sh --population_size 2 --out_dir logs/night_smoke     # quick check first
#
# Does not die halfway:
#   - every finished seed is stored atomically, a restart continues with the next seed;
#   - if evaluate.py crashes it is restarted (up to MAX_FAILS attempts in a row without a new seed);
#   - if no new seed appears for STALL_MIN minutes the attempt is killed and restarted (hang watchdog);
#   - the whole process group is killed between attempts so no worker keeps the GPUs busy.
# After the time limit (or scancel) rerun the same sbatch command to continue, and
#   python evaluate.py --aggregate_only <same arguments>      to rebuild summary.json from the logs.
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o slurm_night_%j.log

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
source .venv/bin/activate
export VLLM_NO_USAGE_STATS=1
export PYTHONUNBUFFERED=1

OUT=logs/qwen2.5-32b_math500_gsm8k
ARGS=(--model_name Qwen/Qwen2.5-32B-Instruct --dataset math500,gsm8k --test_samples 500
      --tp 2 --base_on_cpu --gpu_memory_utilization 0.85 --max_num_seqs 256 --max_model_len 4096
      --cuda_graphs --out_dir "$OUT")
# arguments given to sbatch override the defaults above (argparse keeps the last occurrence)
ARGS+=("$@")
for ((i = 1; i <= $#; i++)); do   # find a custom --out_dir for the progress counter
  if [[ "${!i}" == "--out_dir" ]]; then j=$((i + 1)); OUT="${!j}"; fi
done

MAX_FAILS=${MAX_FAILS:-5}
STALL_MIN=${STALL_MIN:-60}
POLL_S=${POLL_S:-60}          # how often the watchdog looks at the attempt
RETRY_S=${RETRY_S:-60}        # pause before a restart
KILL_WAIT_S=${KILL_WAIT_S:-15}
PID=""

count_done() { find "$OUT" -path '*/seeds/seed_*.json' 2>/dev/null | wc -l | tr -d ' '; }
newest_seed_time() { find "$OUT" -path '*/seeds/seed_*.json' -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1; }
kill_group() {
  [[ -z "$PID" ]] && return
  local pgid
  pgid=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
  [[ -z "$pgid" ]] && pgid=$PID
  kill -TERM -- "-$pgid" 2>/dev/null
  sleep "$KILL_WAIT_S"
  kill -KILL -- "-$pgid" 2>/dev/null
}
trap 'echo "[night] got a termination signal"; kill_group; exit 143' TERM INT

fails=0
attempt=0
while true; do
  attempt=$((attempt + 1))
  before=$(count_done)
  echo "[night] $(date '+%F %T') attempt $attempt, $before seed logs so far"
  setsid python evaluate.py "${ARGS[@]}" &
  PID=$!
  start=$(date +%s)
  while kill -0 "$PID" 2>/dev/null; do
    sleep "$POLL_S"
    last=$(newest_seed_time); last=${last:-0}
    ref=$(( last > start ? last : start ))
    if (( $(date +%s) - ref > STALL_MIN * 60 )); then
      echo "[night] no new seed for ${STALL_MIN} min, killing the attempt"
      kill_group
      break
    fi
  done
  wait "$PID" 2>/dev/null
  code=$?
  kill_group   # leftovers (vLLM workers) of this attempt
  PID=""
  if (( code == 0 )); then
    echo "[night] finished normally"
    exit 0
  fi
  after=$(count_done)
  if (( after > before )); then fails=0; else fails=$((fails + 1)); fi
  echo "[night] attempt failed (exit $code), $((after - before)) new seed logs, $fails/$MAX_FAILS failures in a row"
  if (( fails >= MAX_FAILS )); then
    echo "[night] giving up"
    exit 1
  fi
  sleep "$RETRY_S"
done
