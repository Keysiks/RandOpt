#!/usr/bin/env python3
"""
Do the seeds selected on one dataset transfer to another? Reads the evaluate.py logs of the transfer run
(--target_logs) and the seed list written by select_seeds.py, and compares the groups on the target dataset:

  top      seeds selected on the SOURCE dataset (by its train reward)
  control  random other seeds
For each group: mean target test accuracy and change vs the target base model, share of seeds better than the base,
a permutation test of the top-vs-control difference, and the majority vote of the whole group. Also the rank
correlation between the source and the target test accuracy over all seeds, and (for reference) what selection
on the target itself would give. Source numbers come from the first KB of the source seed logs.
  python analysis/transfer_analysis.py --target_logs logs/transfer_math_to_gsm8k --seeds data/transfer/math_top50.json \
      --source_logs logs/math500_qwen2.5-3b-instruct_n500 --target_dataset gsm8k
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from random_projection_heatmap import read_head


def ranks(x):
    """Average ranks (ties share a rank)."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def perm_test(x, y, n=20000, seed=0):
    """Two-sided permutation p-value for the difference of means."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    obs = abs(x.mean() - y.mean())
    pool = np.concatenate([x, y])
    hits = 0
    for _ in range(n):
        rng.shuffle(pool)
        hits += abs(pool[:len(x)].mean() - pool[len(x):].mean()) >= obs
    return (hits + 1) / (n + 1)


def stratified_perm_test(strata, n=20000, seed=0):
    """strata: [(top values, control values)] per sigma. Statistic: size-weighted mean of the per-sigma
    differences top - control; labels are permuted WITHIN each sigma, so the sigma mix cannot explain the result."""
    rng = np.random.default_rng(seed)
    usable = [(np.asarray(t, float), np.asarray(c, float)) for t, c in strata if len(t) and len(c)]
    if not usable:
        return float("nan"), float("nan")
    w = np.array([len(t) + len(c) for t, c in usable], float)
    obs = float(np.sum(w * np.array([t.mean() - c.mean() for t, c in usable])) / w.sum())
    pools = [np.concatenate([t, c]) for t, c in usable]
    hits = 0
    for _ in range(n):
        stat = 0.0
        for wi, (t, c), pool in zip(w, usable, pools):
            rng.shuffle(pool)
            stat += wi * (pool[:len(t)].mean() - pool[len(t):].mean())
        hits += abs(stat / w.sum()) >= abs(obs)
    return obs, (hits + 1) / (n + 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target_logs", required=True)
    p.add_argument("--seeds", required=True, help="json from select_seeds.py")
    p.add_argument("--source_logs", default=None, help="only for the source accuracy of seeds (already in --seeds)")
    p.add_argument("--target_dataset", default="gsm8k")
    p.add_argument("--train_samples", type=int, default=200)
    p.add_argument("--test_samples", type=int, default=None)
    args = p.parse_args()
    os.chdir(os.path.dirname(HERE))

    import evaluate as ev
    from data_handlers import get_dataset_handler

    handler = get_dataset_handler(args.target_dataset)
    datas, splits = ev.load_problems(handler, handler.default_train_path, handler.default_test_path,
                                     args.train_samples, args.test_samples)
    test_idx = [j for j, s in enumerate(splits) if s == "test"]
    listed = json.load(open(args.seeds))
    by_seed = {r["seed"]: r for r in listed}

    target = {}
    tdir = args.target_logs if os.path.isdir(os.path.join(args.target_logs, "seeds")) else os.path.join(args.target_logs, args.target_dataset)
    for f in sorted(glob.glob(os.path.join(tdir, "seeds", "seed_*.json"))):
        d = json.load(open(f))
        if d["seed"] in by_seed:
            target[d["seed"]] = {"train": d["train_reward"], "test": d["test_accuracy"],
                                 "answers": [r["answer"] for r in d["records"]]}
    base = read_head(os.path.join(tdir, "base.json"), ["train_reward", "test_accuracy"])
    if not target:
        sys.exit(f"no finished seeds of {args.seeds} in {tdir}")
    base_test = base["test_accuracy"]
    print(f"target {args.target_dataset}: {len(target)}/{len(listed)} seeds done, base test accuracy {base_test * 100:.1f}% "
          f"(n={len(test_idx)})")

    out = {"target_dataset": args.target_dataset, "base_test_accuracy": base_test, "groups": {}}
    groups = {g: [r for r in listed if r["group"] == g and r["seed"] in target] for g in ("top", "control")}
    n = min(len(groups["top"]), len(groups["control"]))   # equal sizes if the run was interrupted
    for g in groups:
        groups[g] = groups[g][:n] if n else groups[g]

    def vote(seeds):
        right = 0
        for j in test_idx:
            votes = [target[s]["answers"][j] for s in seeds if target[s]["answers"][j]]
            if votes:
                right += ev.answer_correct(handler, Counter(votes).most_common(1)[0][0], datas[j]["ground_truth"])
        return right / len(test_idx)

    print(f"\n{'group':<22}{'n':>4}{'source test':>13}{'target test':>13}{'change vs base':>16}{'> base':>9}{'best':>8}{'vote of group':>15}")
    accs = {}
    for g, rs in groups.items():
        if not rs:
            continue
        a = np.array([target[r["seed"]]["test"] for r in rs])
        accs[g] = a
        out["groups"][g] = {"n": len(rs), "source_test_mean": float(np.mean([r["source_test_accuracy"] for r in rs])),
                            "target_test_mean": float(a.mean()), "change_vs_base_pp": float((a.mean() - base_test) * 100),
                            "share_above_base": float((a > base_test).mean()), "best": float(a.max()),
                            "vote_accuracy": vote([r["seed"] for r in rs])}
        o = out["groups"][g]
        print(f"{g:<22}{o['n']:>4}{o['source_test_mean'] * 100:>12.1f}%{o['target_test_mean'] * 100:>12.1f}%"
              f"{o['change_vs_base_pp']:>+14.2f}pp{o['share_above_base'] * 100:>8.0f}%{o['best'] * 100:>7.1f}%{o['vote_accuracy'] * 100:>14.1f}%")
    if len(accs) == 2:
        diff = (accs["top"].mean() - accs["control"].mean()) * 100
        pval = perm_test(accs["top"], accs["control"])
        out["top_minus_control_pp"], out["permutation_p"] = float(diff), float(pval)
        print(f"\ntop - control on the target test: {diff:+.2f} pp, permutation p = {pval:.3f}  (raw: ignores that the "
              f"groups can differ in sigma)")

        # the same comparison inside every sigma: a mild perturbation hurts every task less, so selecting
        # low-sigma seeds on the source would "transfer" without any task knowledge
        sigmas = sorted({r["sigma"] for g in groups.values() for r in g})
        strata = []
        print(f"\n{'sigma':<8}{'n top':>7}{'n ctrl':>8}{'top target':>12}{'ctrl target':>13}{'diff':>9}")
        by_sigma = {}
        for sg in sigmas:
            t = [target[r["seed"]]["test"] for r in groups["top"] if r["sigma"] == sg]
            c = [target[r["seed"]]["test"] for r in groups["control"] if r["sigma"] == sg]
            strata.append((t, c))
            by_sigma[str(sg)] = {"n_top": len(t), "n_control": len(c),
                                 "top_mean": float(np.mean(t)) if t else None, "control_mean": float(np.mean(c)) if c else None}
            fmt = lambda v: f"{np.mean(v) * 100:>11.1f}%" if v else f"{'-':>12}"
            d = f"{(np.mean(t) - np.mean(c)) * 100:>+8.1f}pp" if t and c else f"{'-':>10}"
            print(f"{sg:<8g}{len(t):>7}{len(c):>8}{fmt(t)}{fmt(c)}{d}")
        sdiff, sp = stratified_perm_test(strata)
        out["by_sigma"], out["sigma_adjusted_top_minus_control_pp"], out["sigma_adjusted_permutation_p"] = by_sigma, sdiff * 100, sp
        print(f"sigma-adjusted top - control: {sdiff * 100:+.2f} pp, permutation within sigma p = {sp:.3f} "
              f"-> {'selection on the source transfers beyond the sigma effect' if sp < 0.05 and sdiff > 0 else 'no evidence that it transfers beyond the sigma effect'}")

    both = [r for r in listed if r["seed"] in target]
    rho = spearman([r["source_test_accuracy"] for r in both], [target[r["seed"]]["test"] for r in both])
    rho_tr = spearman([r["source_train_reward"] for r in both], [target[r["seed"]]["train"] for r in both])
    within = []                                              # rank correlation inside each sigma, size-weighted
    for sg in sorted({r["sigma"] for r in both}):
        grp = [r for r in both if r["sigma"] == sg]
        rs = spearman([r["source_test_accuracy"] for r in grp], [target[r["seed"]]["test"] for r in grp])
        if len(grp) >= 8 and rs == rs:
            within.append((len(grp), rs))
    if within:
        out["spearman_within_sigma_test"] = float(sum(n * r for n, r in within) / sum(n for n, _ in within))
        print(f"Spearman source test vs target test within a sigma (size-weighted): {out['spearman_within_sigma_test']:+.2f} "
              f"over {len(within)} sigma groups")
    out["spearman_source_vs_target_test"], out["spearman_source_vs_target_train"] = rho, rho_tr
    print(f"Spearman over {len(both)} seeds: source test vs target test {rho:+.2f}, source train vs target train {rho_tr:+.2f}")

    ranked = sorted(target, key=lambda s: -target[s]["train"])          # reference: select on the target itself
    k = min(len(groups.get("top", [])) or 50, len(ranked))
    ref = {"selection_on_target_train_top%d_vote" % k: vote(ranked[:k])}
    out["reference"] = ref
    print(f"reference (select the best {k} of the {len(ranked)} run seeds on the TARGET train problems, vote): "
          f"{list(ref.values())[0] * 100:.1f}%  (target base {base_test * 100:.1f}%)")
    with open(os.path.join(args.target_logs, "transfer_summary.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
