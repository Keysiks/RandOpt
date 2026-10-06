#!/usr/bin/env python3
"""
Label the whole train sets (MATH train, GSM8K train) with the 32B teacher, cheaply:

  stage 0   the unperturbed 32B model answers every problem;
  stage 1.. the K best perturbed 32B models (ranked on the TRAIN problems of the earlier 32B run, never on test
            problems) answer only the problems that are still unsolved, one model after the other. That is the greedy
            cover: every problem keeps the correct answer of the first model that solved it.

An answer counts as solved when it is correct and not truncated. Progress is stored in chunks
(<out_dir>/labels/sSS_cCCCC.jsonl, atomically), a rerun resumes at the first missing chunk; run it under
scripts/supervise.sh to survive crashes. Needs data/math-train/train.jsonl and data/gsm8k/train.parquet.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (REPO_ROOT, add_engine_args, build_prompts, chunks, load_train_problems, make_engine,
                    read_json, read_jsonl, write_json, write_jsonl)


def bench_dir(logs, name):
    d = os.path.join(logs, name)
    return d if os.path.isdir(os.path.join(d, "seeds")) else logs


def rank_seeds(args, out_dir):
    """Seeds of the earlier 32B run by mean accuracy on the train problems of both datasets (cached)."""
    path = os.path.join(out_dir, "seed_ranking.json")
    if os.path.exists(path):
        return read_json(path)
    meta = read_json(args.prev_meta)
    per_dataset = {}
    for name, m in meta["datasets"].items():
        if m.get("train_idx") is None:
            sys.exit(f"{args.prev_meta} has no train_idx for {name}")
        d = bench_dir(args.seed_logs, name)
        for f in sorted(glob.glob(os.path.join(d, "seeds", "seed_*.json"))):
            log = read_json(f)
            acc = float(np.mean([log["records"][j]["correct"] for j in m["train_idx"]]))
            per_dataset.setdefault(log["seed"], {"sigma": log["sigma"]})[name] = acc
    rows = [{"seed": s, "sigma": v["sigma"], "score": float(np.mean([a for k, a in v.items() if k != "sigma"])),
             "per_dataset": {k: a for k, a in v.items() if k != "sigma"}} for s, v in per_dataset.items()]
    rows.sort(key=lambda r: -r["score"])
    write_json(path, rows)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_engine_args(p, tp=2, base_on_cpu=True)
    p.set_defaults(model_name="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--out_dir", default="logs/distill_big/teacher")
    p.add_argument("--datasets", default="math,gsm8k")
    p.add_argument("--seed_logs", default="logs/qwen2.5-32b_math500_gsm8k")
    p.add_argument("--prev_meta", default="logs/lora_32b_to_3b/meta.json", help="meta.json of the first distillation run")
    p.add_argument("--top_k", type=int, default=10, help="perturbed models used after the base model")
    p.add_argument("--chunk", type=int, default=1000, help="problems per generate() call / checkpoint file")
    args = p.parse_args()
    os.chdir(REPO_ROOT)
    os.makedirs(os.path.join(args.out_dir, "labels"), exist_ok=True)

    from transformers import AutoTokenizer
    from vllm import SamplingParams

    problems = load_train_problems([n.strip() for n in args.datasets.split(",")])
    tok = AutoTokenizer.from_pretrained(args.model_name)
    prompts, dropped = build_prompts(tok, problems, args.max_prompt_tokens)
    all_ids = sorted(prompts)                       # (name, idx), deterministic order
    print(f"{len(all_ids)} problems ({dropped} dropped: prompt longer than {args.max_prompt_tokens} tokens); "
          + ", ".join(f"{n}: {len(d)}" for n, (_, d) in problems.items()), flush=True)
    ranking = rank_seeds(args, args.out_dir)[:args.top_k]
    stages = [("base", None, 0.0)] + [(f"seed {r['seed']}", r["seed"], r["sigma"]) for r in ranking]
    sp = SamplingParams(temperature=0.0, seed=42, max_tokens=args.max_tokens)

    solved, engine, applied = {}, None, None
    try:
        for s, (label, seed, sigma) in enumerate(stages):
            pool = [pid for pid in all_ids if pid not in solved]
            if not pool:
                break
            n_before = len(solved)
            for c, chunk in enumerate(chunks(pool, args.chunk)):
                path = os.path.join(args.out_dir, "labels", f"s{s:02d}_c{c:04d}.jsonl")
                if os.path.exists(path):
                    for rec in read_jsonl(path):
                        if rec["correct"]:
                            solved[(rec["dataset"], rec["idx"])] = s
                    continue
                if engine is None:
                    engine = make_engine(args)
                if applied != s:
                    if seed is None:
                        if applied is not None:
                            engine.reset()
                    else:
                        engine.perturb(seed, sigma)
                    applied = s
                t0 = time.perf_counter()
                outputs = engine.generate([prompts[pid] for pid in chunk], sp)
                rows = []
                for (name, idx), out in zip(chunk, outputs):
                    o = out.outputs[0]
                    handler, datas = problems[name]
                    ok = float(handler.compute_reward(o.text, datas[idx]["ground_truth"])) > 0 and o.finish_reason != "length"
                    rows.append({"dataset": name, "idx": idx, "correct": ok, "finish_reason": o.finish_reason,
                                 "n_tokens": len(o.token_ids), "response": o.text if ok else None})
                    if ok:
                        solved[(name, idx)] = s
                write_jsonl(path, rows)
                print(f"stage {s} ({label}) chunk {c + 1}/{-(-len(pool) // args.chunk)}: "
                      f"{sum(r['correct'] for r in rows)}/{len(rows)} solved, {time.perf_counter() - t0:.0f}s", flush=True)
            print(f"== stage {s} ({label}): pool {len(pool)}, newly solved {len(solved) - n_before}, "
                  f"solved in total {len(solved)}/{len(all_ids)}", flush=True)
    finally:
        if engine is not None:
            engine.close()

    summary = {"n_problems": len(all_ids), "n_solved": len(solved), "dropped_long_prompts": dropped,
               "stages": [{"stage": s, "label": st[0]} for s, st in enumerate(stages)],
               "solved_per_dataset": {n: sum(1 for k in solved if k[0] == n) for n in problems},
               "problems_per_dataset": {n: sum(1 for k in all_ids if k[0] == n) for n in problems},
               "solved_by_stage": {str(s): sum(1 for v in solved.values() if v == s) for s in range(len(stages))}}
    write_json(os.path.join(args.out_dir, "labels_summary.json"), summary)
    print(summary)


if __name__ == "__main__":
    main()
