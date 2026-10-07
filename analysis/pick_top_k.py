#!/usr/bin/env python3
"""
Between the two phases of the paper's protocol: pick the K seeds with the best TRAIN reward from the train-only
logs of evaluate.py (--train_only) and write them as a population file for `evaluate.py --population_file`
(each seed keeps its sigma, best first). Only the first KB of every seed log is read.
  python analysis/pick_top_k.py --logs logs/<run>/phase1_train --top_k 50 --out logs/<run>/top_k_seeds.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from random_projection_heatmap import read_head


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", required=True)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.logs, "seeds", "seed_*.json")))
    if not files:
        sys.exit(f"no seed logs in {args.logs}/seeds")
    rows = [read_head(f, ["seed", "sigma", "train_reward"]) for f in files]
    rows = [{"seed": int(r["seed"]), "sigma": r["sigma"], "train_reward": r["train_reward"]} for r in rows]
    ranked = sorted(rows, key=lambda r: -r["train_reward"])        # stable: ties keep seed order
    top = ranked[:args.top_k]
    for i, r in enumerate(top):
        r["rank"] = i + 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(top, f, indent=1)
    base = read_head(os.path.join(args.logs, "base.json"), ["train_reward"])
    tr = np.array([r["train_reward"] for r in rows])
    print(f"{len(rows)} seeds; train reward: base {base['train_reward']:.3f}, mean {tr.mean():.3f}, "
          f"best {tr.max():.3f}; above base: {int((tr > base['train_reward']).sum())}")
    print(f"top-{len(top)}: train reward {top[-1]['train_reward']:.3f} .. {top[0]['train_reward']:.3f}, "
          f"mean {np.mean([r['train_reward'] for r in top]):.3f} -> {args.out}")


if __name__ == "__main__":
    main()
