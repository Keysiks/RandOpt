#!/bin/bash
# Seed transfer MATH-500 -> GSM8K for Qwen2.5-3B-Instruct, one GPU. From the repo root:
#   sbatch scripts/slurm_transfer.sh
# 1) the 50 best seeds of the MATH-500 run (by TRAIN reward) + 50 random control seeds, each with its own sigma;
# 2) evaluate.py on GSM8K with exactly the settings of the source run (bf16, greedy, 1024 tokens, train 200 problems,
#    full GSM8K test) for these seeds only (--population_file); resumable and supervised;
# 3) analysis/transfer_analysis.py: do the seeds selected on MATH-500 do better on GSM8K than the control seeds?
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -o slurm_transfer_%j.log

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
source .venv/bin/activate
export VLLM_NO_USAGE_STATS=1
export PYTHONUNBUFFERED=1

SRC=${SRC:-logs/math500_qwen2.5-3b-instruct_n500}
OUT=${OUT:-logs/transfer_math_to_gsm8k}
SEEDS=data/transfer/math_top50_control50.json

python analysis/select_seeds.py --logs "$SRC" --dataset math500 --top_k 50 --control 50 --out "$SEEDS" || exit 1

PROGRESS_DIR="$OUT" PROGRESS_GLOB='*/seeds/seed_*.json' STALL_MIN=${STALL_MIN:-45} \
  bash scripts/supervise.sh python evaluate.py --dataset gsm8k --population_file "$SEEDS" \
    --model_name Qwen/Qwen2.5-3B-Instruct --train_samples 200 --max_tokens 1024 --precision bfloat16 \
    --global_seed 42 --cuda_graphs --max_num_seqs 512 --out_dir "$OUT" || exit 1

python analysis/transfer_analysis.py --target_logs "$OUT" --seeds "$SEEDS" --source_logs "$SRC" --target_dataset gsm8k
