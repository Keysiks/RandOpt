#!/usr/bin/env python3
"""
Is there a seed that beats the base model on BOTH datasets (e.g. MATH-500 and GSM8K)? Compares their number with what
chance would give: each seed is shifted by independent noise on every dataset, so with a per-dataset share p1, p2 of
seeds above the base about N*p1*p2 seeds are "good on both" even if nothing is shared. Run it per sigma (the
sigma effect alone makes the two datasets correlated). Reads only the first KB of the seed logs.
  python analysis/good_on_both.py --a_logs logs/math500_qwen2.5-3b-instruct_n500 --b_logs logs/transfer_math_to_gsm8k
"""
import argparse
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from random_projection_heatmap import read_head, bench_log_dir


def load(logs, dataset):
    d = bench_log_dir(logs, dataset)
    seeds = {}
    for f in sorted(glob.glob(os.path.join(d, "seeds", "seed_*.json"))):
        h = read_head(f, ["seed", "sigma", "train_reward", "test_accuracy"])
        seeds[int(h["seed"])] = h
    base = read_head(os.path.join(d, "base.json"), ["train_reward", "test_accuracy"])
    return seeds, base


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a_logs", required=True, help="logs of dataset A (e.g. MATH-500)")
    p.add_argument("--b_logs", required=True, help="logs of dataset B (e.g. GSM8K)")
    p.add_argument("--a_dataset", default="math500")
    p.add_argument("--b_dataset", default="gsm8k")
    p.add_argument("--margin", type=float, default=0.0, help="required gain over the base in accuracy points, e.g. 0.01")
    p.add_argument("--show", type=int, default=10)
    args = p.parse_args()

    a, a_base = load(args.a_logs, args.a_dataset)
    b, b_base = load(args.b_logs, args.b_dataset)
    common = sorted(set(a) & set(b))
    print(f"{len(common)} seeds in both logs; base test accuracy: {args.a_dataset} {a_base['test_accuracy'] * 100:.1f}%, "
          f"{args.b_dataset} {b_base['test_accuracy'] * 100:.1f}%")
    rows = [(s, a[s]["sigma"], a[s]["test_accuracy"] - a_base["test_accuracy"], b[s]["test_accuracy"] - b_base["test_accuracy"])
            for s in common]
    print(f"\n{'sigma':<8}{'n':>5}{'above on A':>12}{'above on B':>12}{'both':>6}{'expected by chance':>20}")
    tot_n = tot_both = 0
    tot_exp = 0.0
    for sg in sorted({r[1] for r in rows}) + ["all"]:
        grp = [r for r in rows if sg == "all" or r[1] == sg]
        n = len(grp)
        ga = sum(r[2] > args.margin for r in grp)
        gb = sum(r[3] > args.margin for r in grp)
        both = sum(r[2] > args.margin and r[3] > args.margin for r in grp)
        exp = ga * gb / n
        if sg != "all":
            tot_n += n; tot_both += both; tot_exp += exp
        print(f"{sg:<8}{n:>5}{ga:>12}{gb:>12}{both:>6}{exp:>20.1f}")
    var = tot_exp  # rough Poisson scale for the per-sigma pooled count
    z = (tot_both - tot_exp) / math.sqrt(var) if var > 0 else float("nan")
    print(f"\nper-sigma pooled: {tot_both} seeds good on both vs {tot_exp:.1f} expected by chance (z ~ {z:+.1f})")
    print(f"\nbest seeds by the smaller of the two gains (percentage points over the base):")
    print(f"{'seed':>12}{'sigma':>8}{'A gain':>9}{'B gain':>9}")
    for s, sg, da, db in sorted(rows, key=lambda r: -min(r[2], r[3]))[:args.show]:
        print(f"{s:>12}{sg:>8g}{da * 100:>+8.1f}{db * 100:>+9.1f}")


if __name__ == "__main__":
    main()
