#!/usr/bin/env python3
"""
Pick the seeds for a transfer experiment from the evaluate.py logs of one dataset:
  top      the --top_k best seeds by TRAIN reward (the RandOpt selection criterion, never the test set);
  control  --control random seeds from all the others (same sigma mix), to tell "the selected seeds are special"
           from "any seed behaves like this on the other dataset".
Each seed keeps the sigma it had in the source run. The list alternates top / control so that an interrupted run
has about the same number of seeds in both groups. evaluate.py reads it with --population_file:
  python analysis/select_seeds.py --logs logs/math500_qwen2.5-3b-instruct_n500 --out data/transfer/math_top50.json
Only the first KB of every seed log is read.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from random_projection_heatmap import read_head, bench_log_dir


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", required=True)
    p.add_argument("--dataset", default="math500", help="sub-directory name when the logs hold several datasets")
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--control", type=int, default=50)
    p.add_argument("--control_seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    d = bench_log_dir(args.logs, args.dataset)
    files = sorted(glob.glob(os.path.join(d, "seeds", "seed_*.json")))
    if not files:
        sys.exit(f"no seed logs in {d}/seeds")
    rows = []
    for i, f in enumerate(files):
        h = read_head(f, ["seed", "sigma", "train_reward", "test_accuracy"])
        rows.append({"seed": int(h["seed"]), "sigma": h["sigma"], "source_train_reward": h["train_reward"],
                     "source_test_accuracy": h["test_accuracy"], "source_order": i})
    ranked = sorted(rows, key=lambda r: -r["source_train_reward"])        # stable: ties keep seed order
    for rank, r in enumerate(ranked):
        r["source_rank"] = rank + 1
    top, rest = ranked[:args.top_k], ranked[args.top_k:]
    rng = np.random.default_rng(args.control_seed)
    control = [rest[i] for i in sorted(rng.choice(len(rest), size=min(args.control, len(rest)), replace=False))]
    for r in top:
        r["group"] = "top"
    for r in control:
        r["group"] = "control"
    merged = []
    for i in range(max(len(top), len(control))):
        merged += [g[i] for g in (top, control) if i < len(g)]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(merged, f, indent=1)

    base = read_head(os.path.join(d, "base.json"), ["train_reward", "test_accuracy"])
    mean = lambda rs, k: float(np.mean([r[k] for r in rs]))
    print(f"{len(rows)} seeds in {d}; base train {base['train_reward']:.3f} test {base['test_accuracy']:.3f}")
    for name, rs in (("top", top), ("control", control), ("all", rows)):
        print(f"  {name:8s} n={len(rs):3d}  source train {mean(rs, 'source_train_reward'):.3f}  source test "
              f"{mean(rs, 'source_test_accuracy'):.3f}  sigmas {sorted({r['sigma'] for r in rs})}")
    print(f"{len(merged)} seeds -> {args.out}")


if __name__ == "__main__":
    main()
