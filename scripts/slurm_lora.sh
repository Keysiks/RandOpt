#!/bin/bash
# LoRA distillation: perturbed Qwen2.5-32B answers (evaluate.py logs) -> Qwen2.5-3B-Instruct, one LoRA for
# MATH-500 + GSM8K, evaluated on each test set. One GPU. From the repo root:
#   sbatch scripts/slurm_lora.sh
# Needs peft:  uv pip install peft   (in the .venv, once)
# Steps: 1) build_dataset.py  2) train_lora.py (r=16, alpha=32, all-linear, lr 2e-4, 3 epochs, batch 32)
#        3) eval_lora.py: 3B without LoRA and after every epoch, next to the 32B numbers.
# Extra arguments are not forwarded; edit the variables below (or the scripts' defaults) instead.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -o slurm_lora_%j.log

set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source .venv/bin/activate
export VLLM_NO_USAGE_STATS=1
export PYTHONUNBUFFERED=1

LOGS=${LOGS:-logs/qwen2.5-32b_math500_gsm8k}   # evaluate.py logs of the 32B run
OUT=${OUT:-logs/lora_32b_to_3b}

python lora_distill/build_dataset.py --logs "$LOGS" --out_dir "$OUT"
python lora_distill/train_lora.py --train_file "$OUT/train.jsonl" --out_dir "$OUT/lora" \
    --lr 2e-4 --epochs 3 --batch_size 32 --r 16 --alpha 32
python lora_distill/eval_lora.py --build_dir "$OUT" \
    --adapters "$OUT/lora/epoch_1,$OUT/lora/epoch_2,$OUT/lora/epoch_3"
