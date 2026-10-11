#!/usr/bin/env python3
"""
Build landscape.json (the format of fork_landscape.py in randopt_plots/) from evaluate.py logs of one dataset, so that
the same heatmaps can be drawn for Qwen2.5-3B / 7B / 32B on MATH-500. Run it on the machine that holds the logs:

  python analysis/make_landscape_json.py --logs logs/math500_qwen2.5-3b-instruct_n500 --out landscape_out/qwen2.5-3b
  python analysis/make_landscape_json.py --logs logs/qwen2.5-7b-instruct_math500_sigma0.001_n300/full --out landscape_out/qwen2.5-7b
  python analysis/make_landscape_json.py --logs logs/qwen2.5-32b_math500_gsm8k --dataset math500 --out landscape_out/qwen2.5-32b

per seed: train / test accuracy (train = the first 200 problems, test = the other 300), 2D position; per sigma and for the
pooled population: top-K majority vote (seeds ranked by train accuracy, vote on the test problems) and the share of the
most common voted answer. The 2D position of a seed is (sigma*g1, sigma*g2) with g drawn from a generator seeded by the
seed: a random projection of an isotropic Gaussian perturbation has exactly this distribution (the position carries no
information about accuracy, the colours do).
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
KS = [1, 3, 5, 10, 25, 50]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", required=True)
    p.add_argument("--dataset", default="math500", help="sub-directory when the logs hold several datasets")
    p.add_argument("--out", required=True, help="output directory, gets landscape.json")
    p.add_argument("--run", default=None, help="name written into the json (default: the output directory name)")
    p.add_argument("--proj_seed", type=int, default=0)
    args = p.parse_args()
    os.chdir(REPO_ROOT)

    import evaluate as ev
    from data_handlers import get_dataset_handler

    d = Path(args.logs) / args.dataset
    d = d if (d / "seeds").is_dir() else Path(args.logs)
    problems = json.loads((d / "problems.json").read_text())
    truths = [p_["ground_truth"] for p_ in problems]
    train = [p_["idx"] for p_ in problems if p_["split"] == "train"]
    test = [p_["idx"] for p_ in problems if p_["split"] == "test"]
    handler = get_dataset_handler(args.dataset)

    def acc(correct, idx):
        return float(np.mean([correct[i] for i in idx]))

    base_log = json.loads((d / "base.json").read_text())
    base_correct = [r["correct"] for r in base_log["records"]]
    base_answers = [r["answer"] for r in base_log["records"]]
    base_top = Counter(base_answers[i] for i in test).most_common(1)[0][1] / len(test)
    base_test = acc(base_correct, test)

    pop = []
    for f in sorted((d / "seeds").glob("seed_*.json")):
        log = json.loads(f.read_text())
        correct = [r["correct"] for r in log["records"]]
        pop.append({"seed": log["seed"], "sigma": log["sigma"], "answers": [r["answer"] for r in log["records"]],
                    "train": acc(correct, train), "test": acc(correct, test)})
    print(f"{len(pop)} seeds from {d}; base test {base_test:.3f}")

    def vote(group, k):
        correct, finals = 0, []
        for i in test:
            votes = [s["answers"][i] for s in group[:k] if s["answers"][i]]
            final = Counter(votes).most_common(1)[0][0] if votes else ""
            finals.append(final)
            correct += ev.answer_correct(handler, final, truths[i]) if final else 0
        return {"acc": correct / len(test), "top": Counter(finals).most_common(1)[0][1] / len(finals)}

    groups = {"all": pop} | {str(sg): [s for s in pop if s["sigma"] == sg] for sg in sorted({s["sigma"] for s in pop})}
    table = {}
    for name, group in groups.items():
        ranked = sorted(group, key=lambda s: s["train"], reverse=True)          # stable: ties keep seed order
        ks = [k for k in KS if k <= len(ranked)]
        table[name] = {"n": len(group), "mean_test": float(np.mean([s["test"] for s in group])),
                       "above_base": float(np.mean([s["test"] > base_test for s in group])),
                       "topk": {str(k): vote(ranked, k) for k in ks},
                       "topk_seeds": {str(k): [s["seed"] for s in ranked[:k]] for k in ks}}
        print(f"  sigma {name:>6}: n={len(group):3d} mean test {table[name]['mean_test']:.3f} "
              f"above base {table[name]['above_base']:.0%}  " + "  ".join(f"top-{k} {table[name]['topk'][str(k)]['acc']:.3f}" for k in ks))

    seeds = []
    for s in pop:
        g = np.random.default_rng([args.proj_seed, int(s["seed"])]).standard_normal(2) * s["sigma"]
        seeds.append({"seed": s["seed"], "sigma": s["sigma"], "train": s["train"], "test": s["test"], "xy": [float(g[0]), float(g[1])]})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "landscape.json").write_text(json.dumps({
        "run": args.run or out.name, "dataset": args.dataset, "n_test": len(test), "n_train": len(train),
        "base": {"test": base_test, "train": acc(base_correct, train), "top": base_top},
        "table": table, "seeds": seeds}))
    print(f"-> {out / 'landscape.json'}")


if __name__ == "__main__":
    main()
