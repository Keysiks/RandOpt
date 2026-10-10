#!/bin/bash
# Paper-style distillation of the RandOpt ensemble into ONE model (Neural Thickets, Table 2), here Qwen2.5-3B-Instruct on
# MATH-500, from the logs of the 3B run (no new inference for the data):
#   top-50 perturbed models (by train reward) -> all their correct answers on the HARD train problems (more than half
#   of 8 candidate answers wrong) -> LoRA SFT of the base model, 2 epochs -> evaluation on the 300 test problems next to the
#   base model and the 3B RandOpt ensemble itself (3 prompt orders per model).
# Hyperparameters of the SFT = the authors' example in distillation/README.md (lr 1e-4, LoRA r=64 alpha=128 dropout 0.05,
# batch 40, warmup 10%) with the paper's 2 epochs. One GPU. Needs peft (uv pip install peft).
#   sbatch scripts/slurm_distill_paper.sh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 3:00:00
#SBATCH -o slurm_distill_paper_%j.log

set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source .venv/bin/activate
export VLLM_NO_USAGE_STATS=1
export PYTHONUNBUFFERED=1

LOGS=${LOGS:-logs/math500_qwen2.5-3b-instruct_n500}   # evaluate.py logs of the 3B run
OUT=${OUT:-logs/distill_paper_3b}

python lora_distill/build_paper_distill_dataset.py --logs "$LOGS" --out_dir "$OUT" --top_k 50
python lora_distill/train_lora.py --train_file "$OUT/train.jsonl" --out_dir "$OUT/lora" \
    --lr 1e-4 --epochs 2 --batch_size 40 --r 64 --alpha 128 --dropout 0.05
python lora_distill/eval_lora.py --build_dir "$OUT" --repeats 3 --teacher_label Qwen2.5-3B --max_lora_rank 64 \
    --adapters "$OUT/lora/epoch_1,$OUT/lora/epoch_2"
