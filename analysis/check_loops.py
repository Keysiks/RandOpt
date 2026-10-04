#!/usr/bin/env python3
"""
Check whether answers that hit the max_tokens limit are repetition loops.

Reads the logs written by evaluate.py (base.json + seeds/seed_*.json). A truncated answer
(finish_reason == "length") is flagged as a loop when its tail is highly repetitive:
  - zlib compression ratio of the last --tail_chars characters < --max_ratio, or
  - fraction of unique word 4-grams in that tail < --min_unique.
Normal math text has a ratio around 0.35-0.5 and unique fraction close to 1.

It also separates "this problem is long for the base model too" from "the perturbation made
the model loop": truncations on problems that the base model also truncates vs. new ones.

Usage:
  python analysis/check_loops.py --out_dir logs/math500_qwen2.5-3b-instruct_n500
  python analysis/check_loops.py --out_dir ... --max_seeds 100 --examples 5
"""

import argparse
import glob
import json
import os
import random
import zlib
from collections import defaultdict


def loop_scores(text: str, tail_chars: int):
    tail = text[-tail_chars:]
    raw = tail.encode("utf-8")
    ratio = len(zlib.compress(raw)) / max(len(raw), 1)
    words = tail.split()
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    unique = len(set(grams)) / len(grams) if grams else 1.0
    return ratio, unique


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="logs/math500_qwen2.5-3b-instruct_n500")
    p.add_argument("--max_seeds", type=int, default=None, help="only look at the first N seed logs")
    p.add_argument("--tail_chars", type=int, default=2000)
    p.add_argument("--max_ratio", type=float, default=0.2)
    p.add_argument("--min_unique", type=float, default=0.3)
    p.add_argument("--examples", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    def is_loop(text):
        ratio, unique = loop_scores(text, args.tail_chars)
        return ratio < args.max_ratio or unique < args.min_unique

    base = json.load(open(os.path.join(args.out_dir, "base.json")))
    base_trunc = {r["idx"] for r in base["records"] if r["finish_reason"] == "length"}
    base_loop = {r["idx"] for r in base["records"] if r["finish_reason"] == "length" and is_loop(r["response"])}
    print(f"base: {len(base_trunc)}/{len(base['records'])} truncated, {len(base_loop)} of them look like loops")

    files = sorted(glob.glob(os.path.join(args.out_dir, "seeds", "seed_*.json")))
    if args.max_seeds:
        files = files[:args.max_seeds]

    n_seeds = 0
    tot = {"records": 0, "trunc": 0, "loop": 0, "trunc_on_base_trunc": 0, "correct_trunc": 0, "correct_other": 0}
    by_sigma = defaultdict(lambda: {"seeds": 0, "records": 0, "trunc": 0, "loop": 0, "correct": 0})
    trunc_per_problem = defaultdict(int)
    loop_examples, nonloop_examples = [], []
    rng = random.Random(args.seed)

    for f in files:
        d = json.load(open(f))
        n_seeds += 1
        s = by_sigma[d["sigma"]]
        s["seeds"] += 1
        for r in d["records"]:
            tot["records"] += 1
            s["records"] += 1
            s["correct"] += r["correct"]
            if r["finish_reason"] != "length":
                tot["correct_other"] += r["correct"]
                continue
            tot["trunc"] += 1
            s["trunc"] += 1
            tot["correct_trunc"] += r["correct"]
            trunc_per_problem[r["idx"]] += 1
            tot["trunc_on_base_trunc"] += r["idx"] in base_trunc
            loop = is_loop(r["response"])
            tot["loop"] += loop
            s["loop"] += loop
            # reservoir-ish sampling of examples
            bucket = loop_examples if loop else nonloop_examples
            item = (os.path.basename(f), r["idx"], r["response"][-300:])
            if len(bucket) < args.examples:
                bucket.append(item)
            elif rng.random() < 0.01:
                bucket[rng.randrange(args.examples)] = item

    if not n_seeds:
        raise SystemExit("no seed logs found")
    pct = lambda a, b: 100 * a / b if b else float("nan")

    print(f"\n{n_seeds} seeds, {tot['records']} answers")
    print(f"truncated (hit max_tokens): {tot['trunc']} = {pct(tot['trunc'], tot['records']):.1f}% of answers")
    print(f"  of those, look like loops: {tot['loop']} = {pct(tot['loop'], tot['trunc']):.1f}% "
          f"(= {pct(tot['loop'], tot['records']):.1f}% of all answers)")
    print(f"  on problems the base model also truncates: {pct(tot['trunc_on_base_trunc'], tot['trunc']):.1f}%")
    print(f"accuracy of truncated answers: {pct(tot['correct_trunc'], tot['trunc']):.1f}%  | "
          f"of finished answers: {pct(tot['correct_other'], tot['records'] - tot['trunc']):.1f}%")

    print("\nby sigma:  seeds  truncated%  loops%(of all)  accuracy%")
    for sigma in sorted(by_sigma):
        s = by_sigma[sigma]
        print(f"  {sigma:<8} {s['seeds']:5d}  {pct(s['trunc'], s['records']):9.1f}  "
              f"{pct(s['loop'], s['records']):13.1f}  {pct(s['correct'], s['records']):9.1f}")

    always = sum(1 for c in trunc_per_problem.values() if c >= 0.5 * n_seeds)
    print(f"\nproblems truncated in >=50% of seeds: {always}  (base truncates {len(base_trunc)})")

    for title, bucket in (("LOOP-LIKE", loop_examples), ("TRUNCATED, NOT LOOP-LIKE", nonloop_examples)):
        print(f"\n===== {title} examples (last 300 chars) =====")
        for fname, idx, tail in bucket:
            print(f"--- {fname} problem {idx}\n{tail}\n")


if __name__ == "__main__":
    main()
