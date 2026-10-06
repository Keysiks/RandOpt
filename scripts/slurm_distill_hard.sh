#!/bin/bash
# Hard-problem distillation, one GPU: MATH-500 train problems that the 3B model mostly fails, every correct answer
# of the 50 best perturbed 32B models on them -> LoRA on Qwen2.5-3B-Instruct (2 epochs; r=16, alpha=32, all-linear,
# lr 2e-4, batch 32), evaluated on the MATH-500 test split and GSM8K (3 prompt orders per model).
# Needs peft (uv pip install peft) and the logs of the 32B and 3B runs.
#   sbatch scripts/slurm_distill_hard.sh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 3:00:00
#SBATCH -o slurm_distill_hard_%j.log

set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source .venv/bin/activate
export VLLM_NO_USAGE_STATS=1
export PYTHONUNBUFFERED=1

OUT=${OUT:-logs/distill_hard}

python lora_distill/build_hard_traces_dataset.py --out_dir "$OUT"
python lora_distill/train_lora.py --train_file "$OUT/train.jsonl" --out_dir "$OUT/lora" \
    --lr 2e-4 --epochs 2 --batch_size 32 --r 16 --alpha 32
python lora_distill/eval_lora.py --build_dir "$OUT" --repeats 3 \
    --adapters "$OUT/lora/epoch_1,$OUT/lora/epoch_2"
