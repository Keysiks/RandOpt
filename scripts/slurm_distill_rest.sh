#!/bin/bash
# Step 2 of the scaled-up distillation, one GPU: the small model labels what the teacher solved (to find the hard
# problems), build the SFT set, train one LoRA (r=16, alpha=32, all-linear, lr 2e-4, 3 epochs, batch 32), evaluate
# on every test set (+ all 500 MATH-500 problems) next to the 32B numbers. Resumable: labelling continues
# at the first missing chunk. Needs peft (uv pip install peft).
#   sbatch --dependency=afterok:<teacher job id> scripts/slurm_distill_rest.sh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o slurm_distill_rest_%j.log

set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source .venv/bin/activate
export VLLM_NO_USAGE_STATS=1
export PYTHONUNBUFFERED=1

OUT=${OUT:-logs/distill_big}

python lora_distill/label_student.py --teacher_dir "$OUT/teacher" --out_dir "$OUT/student"
python lora_distill/build_big_dataset.py --out_dir "$OUT"
python lora_distill/train_lora.py --train_file "$OUT/train.jsonl" --out_dir "$OUT/lora" \
    --lr 2e-4 --epochs 3 --batch_size 32 --r 16 --alpha 32
python lora_distill/eval_lora.py --build_dir "$OUT" --full_math500 \
    --adapters "$OUT/lora/epoch_1,$OUT/lora/epoch_2,$OUT/lora/epoch_3"
