#!/usr/bin/env python3
"""
Distillation data as in the paper (Neural Thickets, Table 2 and the rebuttal), for the evaluate.py logs of one model:
  - the top-K perturbed models (K=50, ranked by TRAIN reward) have answered the train problems;
  - hard samples: for each train problem 8 candidate answers are taken and the problem is kept when MORE THAN HALF of
    them are wrong. The paper does not say where the candidates come from; here they are 8 random answers of the
    perturbed models in the logs (no new inference), which is an approximation;
  - every correct, non-truncated answer of the top-K models on a hard problem is a training sample (question,
    reasoning, answer), exact duplicates removed (the authors' pipeline dedups too).
The test problems are never used. meta.json carries what eval_lora.py needs (test problems) plus the reference
numbers of the model's own RandOpt ensemble (base accuracy and top-K majority votes on the test problems), which the
distilled model is compared with:
  python lora_distill/build_paper_distill_dataset.py --logs logs/math500_qwen2.5-3b-instruct_n500 --out_dir logs/distill_paper_3b
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import bench_dir, load_seed_logs, read_json, teacher_reference
from common import REPO_ROOT, write_json, write_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", default="logs/math500_qwen2.5-3b-instruct_n500")
    p.add_argument("--out_dir", default="logs/distill_paper_3b")
    p.add_argument("--dataset", default="math500")
    p.add_argument("--train_samples", type=int, default=200, help="as in the run: the first N problems are train")
    p.add_argument("--test_samples", type=int, default=None)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--hard_candidates", type=int, default=8)
    p.add_argument("--hard_seed", type=int, default=0)
    p.add_argument("--ref_ks", default="1,5,25,50")
    args = p.parse_args()
    os.chdir(REPO_ROOT)
    os.makedirs(args.out_dir, exist_ok=True)

    import evaluate as ev
    from data_handlers import get_dataset_handler

    handler = get_dataset_handler(args.dataset)
    datas, splits = ev.load_problems(handler, handler.default_train_path, handler.default_test_path,
                                     args.train_samples, args.test_samples)
    d = bench_dir(args.logs, args.dataset)
    problems = read_json(os.path.join(d, "problems.json"))
    if len(problems) != len(datas) or any(str(a["ground_truth"]) != str(b["ground_truth"]) for a, b in zip(problems, datas)):
        sys.exit(f"{d}/problems.json does not match the dataset (different --train_samples/--test_samples?)")
    train_idx = [j for j, s in enumerate(splits) if s == "train"]
    test_idx = [j for j, s in enumerate(splits) if s == "test"]

    seeds = load_seed_logs(d, train_idx)
    acc = np.array([s["correct"][train_idx].mean() for s in seeds])
    order = sorted(range(len(seeds)), key=lambda i: -acc[i])          # stable: ties keep seed order
    top = order[:args.top_k]
    print(f"{len(seeds)} perturbed models; train problems {len(train_idx)}, test {len(test_idx)}; top-{len(top)} "
          f"train accuracy {acc[top[-1]]:.3f} .. {acc[top[0]]:.3f} (mean of all {acc.mean():.3f})")

    # --- hard problems: > half of 8 candidate answers wrong
    rng = np.random.default_rng(args.hard_seed)
    correct = np.stack([s["correct"] for s in seeds])                  # [model, problem]
    hard = []
    for j in train_idx:
        cand = rng.choice(len(seeds), size=min(args.hard_candidates, len(seeds)), replace=False)
        wrong = int((~correct[cand, j]).sum())
        if wrong > len(cand) / 2:
            hard.append(j)
    base = read_json(os.path.join(d, "base.json"))
    base_fail = [j for j in train_idx if not base["records"][j]["correct"]]
    print(f"hard train problems (> half of {args.hard_candidates} candidates wrong): {len(hard)}/{len(train_idx)}; "
          f"for comparison the base model fails {len(base_fail)}, overlap {len(set(hard) & set(base_fail))}")
    if not hard:
        sys.exit("no hard problems")

    # --- all correct answers of the top-K models on the hard problems
    rows, per_problem = [], {}
    for j in hard:
        seen = set()
        for i in top:
            s = seeds[i]
            if s["correct"][j] and not s["truncated"][j] and s["responses"][j] not in seen:
                seen.add(s["responses"][j])
                rows.append({"dataset": args.dataset, "idx": j, "seed": s["seed"], "messages": datas[j]["messages"],
                             "response": s["responses"][j], "ground_truth": str(datas[j]["ground_truth"])})
        per_problem[j] = len(seen)
    counts = np.array(list(per_problem.values()))
    print(f"{len(rows)} training samples on {int((counts > 0).sum())}/{len(hard)} hard problems "
          f"(per problem: mean {counts.mean():.1f}, median {np.median(counts):.0f}, max {counts.max()}; "
          f"{int((counts == 0).sum())} hard problems have no correct trace)")

    ks = [int(k) for k in args.ref_ks.split(",")]
    ref = teacher_reference(handler, datas, seeds, order, base, test_idx, ks)
    print(f"RandOpt reference on the {len(test_idx)} test problems: {ref}")
    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), rows)
    write_json(os.path.join(args.out_dir, "meta.json"), {
        "logs": args.logs, "train_samples": args.train_samples, "test_samples": args.test_samples,
        "datasets": {args.dataset: {"n_problems": len(datas), "train_idx": train_idx, "test_idx": test_idx,
                                    "n_train": len(train_idx), "n_test": len(test_idx), "n_models": len(seeds),
                                    "teacher_reference_on_test": ref}},
        "train_set": {"source": "paper-style: top-K traces on hard problems", "top_k": args.top_k,
                      "hard_candidates": args.hard_candidates, "n_hard": len(hard), "n_examples": len(rows),
                      "traces_per_problem_mean": float(counts.mean()), "hard_idx": hard}})
    print(f"-> {args.out_dir}/train.jsonl, meta.json")


if __name__ == "__main__":
    main()
