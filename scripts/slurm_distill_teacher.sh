#!/bin/bash
# Step 1 of the scaled-up distillation: label MATH train + GSM8K train with Qwen2.5-32B (base model first, then the 10
# best perturbed models on the still unsolved problems). 2 GPUs, supervised (restart on crash/hang), resumable:
# after a kill or the time limit submit the same command again.
#   Before: python scripts/prepare_math_train.py && python scripts/prepare_gsm8k.py   (login node, needs internet)
#   sbatch scripts/slurm_distill_teacher.sh
#   then:   sbatch --dependency=afterok:<this job id> scripts/slurm_distill_rest.sh
# Extra arguments go to lora_distill/label_teacher.py (e.g. --top_k 5 --chunk 500).
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH -o slurm_distill_teacher_%j.log

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
source .venv/bin/activate
export VLLM_NO_USAGE_STATS=1
export PYTHONUNBUFFERED=1

OUT=logs/distill_big/teacher
ARGS=(lora_distill/label_teacher.py --out_dir "$OUT" --top_k 10 --chunk 1000
      --tp 2 --base_on_cpu --gpu_memory_utilization 0.85 --max_num_seqs 256 --max_model_len 4096)
ARGS+=("$@")
for ((i = 1; i <= $#; i++)); do
  if [[ "${!i}" == "--out_dir" ]]; then j=$((i + 1)); OUT="${!j}"; fi
done

PROGRESS_DIR="$OUT" PROGRESS_GLOB='*/labels/s*.jsonl' STALL_MIN=${STALL_MIN:-60} \
  bash scripts/supervise.sh python "${ARGS[@]}"
