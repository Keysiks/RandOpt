#!/bin/bash
# RandOpt on MATH-500, 500 seeds, one GPU. Submit from the repo root:  sbatch scripts/slurm_evaluate.sh
# Resumable: if the job is killed or hits the time limit, submit it again and finished seeds are skipped.
# Smoke test:  sbatch scripts/slurm_evaluate.sh --population_size 3 --out_dir logs/smoke
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH -o slurm_%j.log

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source .venv/bin/activate
export VLLM_NO_USAGE_STATS=1

python evaluate.py "$@"
