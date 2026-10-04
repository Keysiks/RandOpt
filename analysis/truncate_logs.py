#!/usr/bin/env python3
"""
Re-score evaluate.py logs as if generation had been limited to a smaller max_tokens.

With greedy decoding a run with max_tokens=N is the first N tokens of a run with a larger limit,
so no GPU is needed: every response longer than N tokens is cut after its N-th token and
the answer, reward and correctness are recomputed. The result is a new log directory in the
same format (base.json, seeds/seed_XXXX.json, ...), so the usual aggregation applies:

  python analysis/truncate_logs.py --src logs/math500_qwen2.5-3b-instruct_n500 \
      --dst logs/math500_qwen2.5-3b-instruct_n500_t256 --max_tokens 256
  python evaluate.py --aggregate_only --out_dir logs/math500_qwen2.5-3b-instruct_n500_t256 --max_tokens 256

Notes:
  - Timings in the new logs are copied from the original runs (they are NOT the timings of a
    real max_tokens=N run).
  - The cut position comes from re-tokenizing the logged text, which can differ from vLLM's own
    token boundaries by a token in rare cases.
  - A real run can differ slightly from this estimate because of batch-dependent bf16 numerics.
"""

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def cut_ends(tokenizer, texts, max_tokens):
    """Character offset where each text should be cut, or None if it fits into max_tokens."""
    enc = tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True)
    return [offs[max_tokens - 1][1] if len(offs) > max_tokens else None for offs in enc["offset_mapping"]]


def rescore(d, handler, tokenizer, ground_truth, max_tokens, train_samples, extract):
    recs = d["records"]
    idxs = [i for i, r in enumerate(recs) if r["n_tokens"] > max_tokens]
    ends = cut_ends(tokenizer, [recs[i]["response"] for i in idxs], max_tokens)
    for i, end in zip(idxs, ends):
        if end is None:
            continue
        r = recs[i]
        text = r["response"][:end]
        r["response"] = text
        r["answer"] = extract(handler, text)
        r["reward"] = float(handler.compute_reward(text, ground_truth[r["idx"]]))
        r["correct"] = r["reward"] > 0
        r["n_tokens"] = max_tokens
        r["finish_reason"] = "length"
    train, test = recs[:train_samples], recs[train_samples:]
    d["train_reward"] = float(np.mean([r["reward"] for r in train]))
    d["train_accuracy"] = float(np.mean([r["correct"] for r in train]))
    d["test_accuracy"] = float(np.mean([r["correct"] for r in test])) if test else None
    d["full_accuracy"] = float(np.mean([r["correct"] for r in recs]))
    d["rescored_max_tokens"] = max_tokens
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="log dir written by evaluate.py")
    p.add_argument("--dst", required=True, help="new log dir")
    p.add_argument("--max_tokens", type=int, required=True)
    args = p.parse_args()

    import evaluate as ev
    from data_handlers import get_dataset_handler
    from transformers import AutoTokenizer

    src_args = ev.read_json(os.path.join(args.src, "args.json"))
    if args.max_tokens >= src_args["max_tokens"]:
        sys.exit(f"--max_tokens must be below the source limit ({src_args['max_tokens']})")
    problems = ev.read_json(os.path.join(args.src, "problems.json"))
    ground_truth = {p_["idx"]: p_["ground_truth"] for p_ in problems}
    handler = get_dataset_handler("math500")
    tokenizer = AutoTokenizer.from_pretrained(src_args["model_name"])

    os.makedirs(os.path.join(args.dst, "seeds"), exist_ok=True)
    shutil.copy(os.path.join(args.src, "problems.json"), os.path.join(args.dst, "problems.json"))
    ev.write_json(os.path.join(args.dst, "args.json"), {**src_args, "max_tokens": args.max_tokens})

    files = [os.path.join(args.src, "base.json")] + sorted(glob.glob(os.path.join(args.src, "seeds", "seed_*.json")))
    for n, f in enumerate(files, 1):
        d = rescore(ev.read_json(f), handler, tokenizer, ground_truth, args.max_tokens,
                    src_args["train_samples"], ev.extract)
        ev.write_json(os.path.join(args.dst, os.path.relpath(f, args.src)), d)
        if n % 20 == 0 or n == len(files):
            print(f"{n}/{len(files)} logs rescored", flush=True)
    print(f"done: {args.dst}")


if __name__ == "__main__":
    main()
