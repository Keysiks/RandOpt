#!/bin/bash
# 2D random-direction slice heatmaps (analysis/random_slice.py), one GPU. Submit from the repo root:
#   sbatch scripts/slurm_slice.sh --experts Geometry,Algebra --out_dir logs/slice_geo_alg --cuda_graphs --max_num_seqs 512
# Resumable: finished grid points are skipped, so submit it again after a kill or the time limit.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH -o slurm_slice_%j.log

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source .venv/bin/activate
export VLLM_NO_USAGE_STATS=1
export PYTHONUNBUFFERED=1

python analysis/random_slice.py "$@"
