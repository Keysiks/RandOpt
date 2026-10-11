#!/usr/bin/env python3
"""
Results of the LoRA distillation runs as one tab-separated table (paste into Google Sheets), long format:
experiment, variant, dataset, accuracy %, min %, max %, test problems, truncated, training examples.
Reads <run>/eval/eval_results.json (written by lora_distill/eval_lora.py) and counts the lines of <run>/train.jsonl.
Rows of the reference model ("teacher") come from the same file (its base accuracy and top-K majority votes).
  python analysis/distill_table.py "32B->3B first=logs/lora_32b_to_3b" "32B->3B scaled=logs/distill_big" \
      "32B->3B hard traces=logs/distill_hard"
"""
import json
import os
import sys


def pct(x):
    return "" if x is None else f"{x * 100:.1f}"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print("\t".join(["experiment", "variant", "dataset", "accuracy %", "min %", "max %", "test problems", "truncated",
                     "training examples", "run dir"]))
    for arg in sys.argv[1:]:
        name, _, path = arg.partition("=")
        f = os.path.join(path, "eval", "eval_results.json")
        if not os.path.exists(f):
            print(f"{name}\tNO eval_results.json\t\t\t\t\t\t\t\t{path}")
            continue
        res = json.load(open(f))
        train = os.path.join(path, "train.jsonl")
        n_train = sum(1 for line in open(train) if line.strip()) if os.path.exists(train) else ""
        for variant, per_dataset in res.items():
            if variant.startswith("teacher"):
                continue
            for dataset, r in per_dataset.items():
                print("\t".join(str(x) for x in [
                    name, "no LoRA (base)" if variant == "base" else variant, dataset, pct(r["accuracy"]),
                    pct(r.get("accuracy_min")), pct(r.get("accuracy_max")), r.get("n", ""), r.get("truncated", ""),
                    n_train, path]))
        for key, per_dataset in res.items():
            if not key.startswith("teacher"):
                continue
            for dataset, ref in per_dataset.items():
                for k, v in ref.items():
                    label = "base" if k == "base_accuracy" else k.replace("_vote_accuracy", " vote")
                    print("\t".join(str(x) for x in [name, f"reference {label}", dataset, pct(v), "", "", "", "", "", path]))


if __name__ == "__main__":
    main()
