#!/bin/bash
# RandOpt on MATH-500 with the protocol of the paper (Neural Thickets): N perturbations, only sigma = 0.001 here,
# selection on the 200 train problems, top-K majority vote on the 300 test problems; one engine per GPU.
#   sbatch scripts/slurm_7b_math.sh                     # Qwen2.5-7B-Instruct, N=5000, K=50, 4 GPUs
#   sbatch --gres=gpu:8 scripts/slurm_7b_math.sh        # the number of GPUs is taken from the allocation
#   N=1000 sbatch scripts/slurm_7b_math.sh              # a smaller population (also K=, SIGMA=, MODEL=, OUT=)
#   MODEL=allenai/Olmo-3-7B-Instruct sbatch scripts/slurm_7b_math.sh   # the paper's 7B model
# Phase 1: every seed on the 200 train problems only (--train_only), supervised and resumable (rerun the same command
#          after a kill: finished seeds are skipped; the best seeds of a partial run are already usable).
# Phase 2: the K best seeds (by train reward) on all 500 problems -> base accuracy, top-K votes on the test problems.
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH -t 12:00:00
#SBATCH -o slurm_7b_math_%j.log

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
source .venv/bin/activate
export VLLM_NO_USAGE_STATS=1
export PYTHONUNBUFFERED=1

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
TAG=$(basename "$MODEL" | tr 'A-Z' 'a-z')
N=${N:-5000}
K=${K:-50}
SIGMA=${SIGMA:-0.001}
GPUS=${GPUS:-${SLURM_GPUS_ON_NODE:-$(echo "${CUDA_VISIBLE_DEVICES:-0}" | tr ',' '\n' | grep -c .)}}
OUT=${OUT:-logs/${TAG}_math500_sigma${SIGMA}_n${N}}
echo "[7b] model=$MODEL N=$N K=$K sigma=$SIGMA GPUs=$GPUS out=$OUT"

# settings of the paper for MATH-500: bf16, greedy, max 1024 new tokens, first 200 problems = train, the rest = test
COMMON=(--dataset math500 --model_name "$MODEL" --sigma_values "$SIGMA" --train_samples 200 --max_tokens 1024
        --precision bfloat16 --global_seed 42 --cuda_graphs --max_num_seqs 512 --gpu_memory_utilization 0.6
        --num_gpus "$GPUS" --stagger_s 30)

PROGRESS_DIR="$OUT/phase1_train" PROGRESS_GLOB='*/seeds/seed_*.json' STALL_MIN=${STALL_MIN:-45} \
  bash scripts/supervise.sh python evaluate.py "${COMMON[@]}" --population_size "$N" --train_only \
    --out_dir "$OUT/phase1_train" || exit 1

python analysis/pick_top_k.py --logs "$OUT/phase1_train" --top_k "$K" --out "$OUT/top_k_seeds.json" || exit 1

PROGRESS_DIR="$OUT/phase2_topk" PROGRESS_GLOB='*/seeds/seed_*.json' STALL_MIN=${STALL_MIN:-45} \
  bash scripts/supervise.sh python evaluate.py "${COMMON[@]}" --population_file "$OUT/top_k_seeds.json" \
    --top_k "1,5,25,$K" --out_dir "$OUT/phase2_topk" || exit 1

echo
echo "[7b] results: $OUT/phase2_topk/summary.json (base accuracy and top-K majority votes on the 300 test problems)"
echo "[7b] paper, Table 4, MATH-500 (K=50, N=5000, sigma in {1,2,3}e-3, mean of 3 runs); there is NO Qwen2.5-7B in the paper:"
echo "       Qwen2.5-1.5B-Inst  base 43.2 -> RandOpt 59.7    Qwen2.5-3B-Inst  base 58.6 -> RandOpt 68.7"
echo "       OLMo3-7B-Inst      base 60.6 -> RandOpt 73.7    Llama3.1-8B-Inst base 47.0 -> RandOpt 59.5"
