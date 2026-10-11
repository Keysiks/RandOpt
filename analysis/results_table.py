#!/usr/bin/env python3
"""
One table (tab-separated, paste it into Google Sheets) with the results of several evaluate.py runs: accuracy of the
base model, of single perturbed models and of the top-K majority votes, plus the time per seed. Reads <run>/summary.json.
  python analysis/results_table.py \
      "Qwen2.5-3B=logs/math500_qwen2.5-3b-instruct_n500" \
      "Qwen2.5-7B=logs/qwen2.5-7b-instruct_math500_sigma0.001_n300/full" \
      "Qwen2.5-32B=logs/qwen2.5-32b_math500_gsm8k/math500"
Time: per-seed times are measured inside each process, so with several processes/GPUs sharing work they are NOT wall-clock;
"sum of seed times" is the total busy time of all processes, wall-clock = that / (number of parallel processes).
"""
import json
import os
import sys


def pct(x):
    return "" if x is None else f"{x * 100:.1f}"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ks = [1, 5, 25, 50]
    header = (["model", "run dir", "seeds", "sigma", "base test acc %", "base full acc %", "single seed mean test %",
               "single seed best test %"] + [f"top-{k} vote test %" for k in ks] +
              ["test problems", "sec / seed (mean)", "sec / seed (median)", "sum of seed times, h", "base run, s"])
    print("\t".join(header))
    for arg in sys.argv[1:]:
        name, _, path = arg.partition("=")
        f = os.path.join(path, "summary.json")
        if not os.path.exists(f):
            print(f"{name}\t{path}\tNO summary.json (the run has not finished or the path is wrong)")
            continue
        s = json.load(open(f))
        base, t = s.get("base", {}), s.get("timing", {})
        ens = s.get("ensemble", {})
        single = s.get("single_seed_test_accuracy", {})
        sigmas = "/".join(f"{float(k):g}" for k in sorted(s.get("sigma_stats", {}), key=float))
        n_test = next((v["n_test"] for v in ens.values()), "")
        row = [name, path, s.get("n_seeds_done", ""), sigmas, pct(base.get("test_accuracy")), pct(base.get("full_accuracy")),
               pct(single.get("mean")), pct(single.get("max"))]
        row += [pct(ens[str(k)]["accuracy"]) if str(k) in ens else "" for k in ks]
        one = t.get("one_seed_total_s", {})
        row += [n_test, f"{one['mean']:.1f}" if one else "", f"{one['median']:.1f}" if one else "",
                f"{t['all_seeds_total_h']:.2f}" if t else "", f"{base['timing']['total_s']:.1f}" if base.get("timing") else ""]
        print("\t".join(str(x) for x in row))


if __name__ == "__main__":
    main()
