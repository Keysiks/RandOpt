#!/bin/bash
# Fig. 2-style heatmaps (2D random-direction slice) for Qwen2.5-32B-Instruct on MATH-500 and GSM8K.
# Both datasets share the plane and are evaluated in one pass; one PNG per dataset. 2 GPUs. From the repo root:
#   sbatch scripts/slurm_slice_32b.sh                                   # 9x9 grid, about 3 h
#   sbatch scripts/slurm_slice_32b.sh --step 0.0005                     # 17x17 grid, about 8 h (use -t 12:00:00)
#   sbatch scripts/slurm_slice_32b.sh --extent 0.002 --out_dir logs/x   # extra arguments go to random_slice.py
# Resumable and supervised (restart on crash/hang, see scripts/supervise.sh): rerun the same command to continue.
# matplotlib is not in requirements.txt: `uv pip install matplotlib` here, or plot locally with --plot_only.
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 6:00:00
#SBATCH -o slurm_slice32b_%j.log

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
source .venv/bin/activate
export VLLM_NO_USAGE_STATS=1
export PYTHONUNBUFFERED=1

OUT=logs/slice_qwen2.5-32b
ARGS=(analysis/random_slice.py --dataset math500,gsm8k --test_samples 500
      --logs logs/qwen2.5-32b_math500_gsm8k --model_name Qwen/Qwen2.5-32B-Instruct
      --tp 2 --base_on_cpu --gpu_memory_utilization 0.85 --max_num_seqs 256 --max_model_len 4096
      --cuda_graphs --est_point_s 100 --out_dir "$OUT")
ARGS+=("$@")   # later occurrences override the defaults above
for ((i = 1; i <= $#; i++)); do
  if [[ "${!i}" == "--out_dir" ]]; then j=$((i + 1)); OUT="${!j}"; fi
done

PROGRESS_DIR="$OUT" PROGRESS_GLOB='*/points/p_*.json' STALL_MIN=${STALL_MIN:-60} \
  bash scripts/supervise.sh python "${ARGS[@]}"
