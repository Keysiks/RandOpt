#!/usr/bin/env python3
"""
SFT set in the style of the paper's distillation: MANY correct traces per HARD problem.

  hard problem   = a MATH-500 train problem that fewer than --hard_threshold (0.5) of the 3B answers solve
                   (the unperturbed 3B model plus all its perturbed models in the 3B logs);
  traces         = every correct, non-truncated answer to it from the --top_k (50) best perturbed 32B models
                   (ranked on the train problems only), exact duplicates removed.

Train/test split and the 32B reference numbers come from the first distillation run (meta.json), so the test
problems are never trained on:
  python lora_distill/build_hard_traces_dataset.py --out_dir logs/distill_hard
"""
import argparse
import glob
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO_ROOT, read_json, write_json, write_jsonl


def seed_dir(logs, name):
    d = os.path.join(logs, name)
    return d if os.path.isdir(os.path.join(d, "seeds")) else logs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="logs/distill_hard")
    p.add_argument("--logs32", default="logs/qwen2.5-32b_math500_gsm8k")
    p.add_argument("--logs3", default="logs/math500_qwen2.5-3b-instruct_n500")
    p.add_argument("--prev_meta", default="logs/lora_32b_to_3b/meta.json")
    p.add_argument("--dataset", default="math500")
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--hard_threshold", type=float, default=0.5)
    args = p.parse_args()
    os.chdir(REPO_ROOT)
    os.makedirs(args.out_dir, exist_ok=True)

    import evaluate as ev
    from data_handlers import get_dataset_handler

    meta = read_json(args.prev_meta)
    m = meta["datasets"][args.dataset]
    train_idx = m["train_idx"]
    handler = get_dataset_handler(args.dataset)
    datas, _ = ev.load_problems(handler, handler.default_train_path, handler.default_test_path,
                                meta["train_samples"], meta["test_samples"])
    assert len(datas) == m["n_problems"]

    # --- hardness from the 3B logs: share of 3B answers (base + perturbed) that are correct
    files3 = sorted(glob.glob(os.path.join(seed_dir(args.logs3, args.dataset), "seeds", "seed_*.json")))
    base3 = read_json(os.path.join(seed_dir(args.logs3, args.dataset), "base.json"))
    rows3 = [np.array([r["correct"] for r in base3["records"]], float)]
    for f in files3:
        rows3.append(np.array([r["correct"] for r in read_json(f)["records"]], float))
    share = np.mean(rows3, axis=0)                        # per problem, over base + perturbed 3B models
    base_fails = rows3[0] == 0
    hard = [j for j in train_idx if share[j] < args.hard_threshold]
    print(f"3B answers per problem: 1 base + {len(files3)} perturbed; train problems {len(train_idx)}: "
          f"hard (<{args.hard_threshold:.0%} of 3B answers correct) = {len(hard)}, "
          f"3B base fails = {int(base_fails[train_idx].sum())}, both = {int(sum(base_fails[j] for j in hard))}")

    if not hard:
        sys.exit(f"no hard problems at --hard_threshold {args.hard_threshold}: every train problem is solved by at "
                 f"least that share of the 3B answers; raise the threshold")

    # --- 32B: best top_k perturbed models on the TRAIN problems, all their correct traces on the hard problems
    files32 = sorted(glob.glob(os.path.join(seed_dir(args.logs32, args.dataset), "seeds", "seed_*.json")))
    models = []
    for f in files32:
        d = read_json(f)
        recs = d["records"]
        acc = float(np.mean([recs[j]["correct"] for j in train_idx]))
        models.append((acc, d["seed"], {j: recs[j] for j in hard}))
    models.sort(key=lambda t: -t[0])                       # stable: ties keep seed order
    top = models[:args.top_k]
    print(f"32B: {len(models)} perturbed models, top-{len(top)} train accuracy {top[0][0]:.3f} .. {top[-1][0]:.3f}")

    rows, per_problem, unsolved = [], {}, []
    for j in hard:
        seen, n = set(), 0
        for acc, seed, recs in top:
            r = recs[j]
            if r["correct"] and r["finish_reason"] != "length" and r["response"] not in seen:
                seen.add(r["response"])
                rows.append({"dataset": args.dataset, "idx": j, "seed": seed, "messages": datas[j]["messages"],
                             "response": r["response"], "ground_truth": str(datas[j]["ground_truth"])})
                n += 1
        per_problem[j] = n
        if n == 0:
            unsolved.append(j)
    counts = np.array(list(per_problem.values()))
    print(f"{len(rows)} traces on {int((counts > 0).sum())}/{len(hard)} hard problems "
          f"({len(unsolved)} without a correct trace); traces per problem: mean {counts.mean():.1f}, "
          f"median {np.median(counts):.0f}, max {counts.max()}")
    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), rows)
    meta["train_set"] = {"source": "hard problems x all correct top-k 32B traces", "dataset": args.dataset,
                         "top_k": args.top_k, "hard_threshold": args.hard_threshold, "n_hard": len(hard),
                         "n_examples": len(rows), "n_problems_with_traces": int((counts > 0).sum()),
                         "traces_per_problem": dict(sorted(Counter(counts.tolist()).items())),
                         "hard_idx": hard}
    write_json(os.path.join(args.out_dir, "meta.json"), meta)
    print(f"-> {args.out_dir}/train.jsonl")


if __name__ == "__main__":
    main()
