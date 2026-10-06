#!/usr/bin/env python3
"""
Build the SFT set for distilling the perturbed Qwen2.5-32B models (evaluate.py logs) into a small model.

For every dataset separately: the 32B models are sorted by accuracy on the train problems; all correct
answers of the best one are taken, then from the second one the correct answers on the problems that are
still not covered, and so on, until every solvable train problem has one correct answer. Truncated answers
(finish_reason == "length") are never used.

Splits: datasets with one file (MATH-500): --single_file_test random problems are test (seed --split_seed),
the rest train. Datasets with separate files (GSM8K): as in the 32B run (first --train_samples of train.parquet
are train, --test_samples of test.parquet are test). The 32B models are ranked on the train problems only.

Writes <out_dir>/train.jsonl, meta.json (splits, coverage, reference accuracy of the 32B base model and of its
top-K majority-vote ensemble on the test problems):
  python lora_distill/build_dataset.py --logs logs/qwen2.5-32b_math500_gsm8k --out_dir logs/lora_32b_to_3b
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def bench_dir(logs, name):
    d = os.path.join(logs, name)
    return d if os.path.isdir(os.path.join(d, "seeds")) else logs


def load_seed_logs(directory, keep_idx):
    """Per seed: correctness, truncation, extracted answers; responses only for the train problems."""
    files = sorted(glob.glob(os.path.join(directory, "seeds", "seed_*.json")))
    if not files:
        sys.exit(f"no seed logs in {directory}/seeds")
    keep = set(keep_idx)
    seeds = []
    for f in files:
        d = read_json(f)
        recs = d["records"]
        seeds.append({
            "seed": d["seed"], "sigma": d["sigma"],
            "correct": np.array([r["correct"] for r in recs], bool),
            "truncated": np.array([r["finish_reason"] == "length" for r in recs], bool),
            "answers": [r["answer"] for r in recs],
            "responses": {j: recs[j]["response"] for j in keep},
        })
    return seeds


def greedy_cover(seeds, train_idx):
    """Models by train accuracy (best first, ties keep seed order); each takes the still uncovered
    problems it solved. Returns [(seed index, problem index)] and the order."""
    acc = [s["correct"][train_idx].mean() for s in seeds]
    order = sorted(range(len(seeds)), key=lambda i: -acc[i])
    covered, picks = set(), []
    for i in order:
        s = seeds[i]
        for j in train_idx:
            if j not in covered and s["correct"][j] and not s["truncated"][j]:
                covered.add(j)
                picks.append((i, j))
    return picks, order, acc


def teacher_reference(handler, datas, seeds, order, base_log, test_idx, ks):
    from evaluate import answer_correct
    ref = {"base_accuracy": float(np.mean([base_log["records"][j]["correct"] for j in test_idx]))}
    for k in ks:
        if k > len(order):
            continue
        top = order[:k]
        right = 0
        for j in test_idx:
            votes = [seeds[i]["answers"][j] for i in top if seeds[i]["answers"][j]]
            if votes:
                right += answer_correct(handler, Counter(votes).most_common(1)[0][0], datas[j]["ground_truth"])
        ref[f"top{k}_vote_accuracy"] = right / len(test_idx)
    return ref


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", default="logs/qwen2.5-32b_math500_gsm8k")
    p.add_argument("--out_dir", default="logs/lora_32b_to_3b")
    p.add_argument("--datasets", default="math500,gsm8k")
    p.add_argument("--train_samples", type=int, default=200, help="as in the 32B run")
    p.add_argument("--test_samples", type=int, default=500, help="as in the 32B run")
    p.add_argument("--single_file_test", type=int, default=200, help="random test problems for one-file datasets (MATH-500)")
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--teacher_ks", default="1,5,25,50")
    args = p.parse_args()
    os.chdir(REPO_ROOT)
    os.makedirs(args.out_dir, exist_ok=True)

    import evaluate as ev
    from data_handlers import get_dataset_handler

    meta = {"logs": args.logs, "train_samples": args.train_samples, "test_samples": args.test_samples,
            "split_seed": args.split_seed, "datasets": {}}
    rows = []
    for name in [n.strip() for n in args.datasets.split(",") if n.strip()]:
        handler = get_dataset_handler(name)
        datas, splits = ev.load_problems(handler, handler.default_train_path, handler.default_test_path,
                                         args.train_samples, args.test_samples)
        d = bench_dir(args.logs, name)
        problems = read_json(os.path.join(d, "problems.json"))
        if len(problems) != len(datas) or any(str(p_["ground_truth"]) != str(x["ground_truth"])
                                              for p_, x in zip(problems, datas)):
            sys.exit(f"{name}: the problems in {d}/problems.json do not match the dataset files "
                     f"(different --train_samples/--test_samples?)")
        if handler.default_train_path == handler.default_test_path:        # one file: random split
            rng = np.random.default_rng(args.split_seed)
            test_idx = sorted(rng.choice(len(datas), size=args.single_file_test, replace=False).tolist())
            train_idx = [j for j in range(len(datas)) if j not in set(test_idx)]
        else:
            train_idx = [j for j, s in enumerate(splits) if s == "train"]
            test_idx = [j for j, s in enumerate(splits) if s == "test"]

        seeds = load_seed_logs(d, train_idx)
        picks, order, acc = greedy_cover(seeds, train_idx)
        used = Counter(i for i, _ in picks)
        for i, j in picks:
            rows.append({"dataset": name, "idx": j, "seed": seeds[i]["seed"], "sigma": seeds[i]["sigma"],
                         "messages": datas[j]["messages"], "response": seeds[i]["responses"][j],
                         "ground_truth": str(datas[j]["ground_truth"])})
        base_log = read_json(os.path.join(d, "base.json"))
        ks = [int(k) for k in args.teacher_ks.split(",")]
        meta["datasets"][name] = {
            "n_problems": len(datas), "train_idx": train_idx if len(train_idx) <= 1000 else None,
            "test_idx": test_idx, "n_train": len(train_idx), "n_test": len(test_idx),
            "n_models": len(seeds), "n_examples": len(picks),
            "train_coverage": len(picks) / len(train_idx), "models_contributing": len(used),
            "best_model_train_accuracy": float(max(acc)),
            "teacher_reference_on_test": teacher_reference(handler, datas, seeds, order, base_log, test_idx, ks)}
        m = meta["datasets"][name]
        print(f"{name}: {len(seeds)} models, train {len(train_idx)} / test {len(test_idx)}, "
              f"{len(picks)} examples ({m['train_coverage'] * 100:.1f}% of train covered) from {len(used)} models; "
              f"best model train acc {max(acc) * 100:.1f}%")
        print(f"   32B on test: {json.dumps(m['teacher_reference_on_test'])}")

    with open(os.path.join(args.out_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"{len(rows)} training examples -> {args.out_dir}/train.jsonl")


if __name__ == "__main__":
    main()
