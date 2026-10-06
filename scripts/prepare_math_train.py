#!/usr/bin/env python3
"""Download the MATH train split (EleutherAI/hendrycks_math, 7500 problems) and write data/math-train/train.jsonl
in the format data_handlers/math500.py reads (problem, answer, subject, level). The final answer is the last
\\boxed{...} of the reference solution. Problems that also occur in MATH-500 (data/math-500/test.jsonl) are removed.
  python scripts/prepare_math_train.py [--out data/math-train/train.jsonl]
"""
import argparse
import json
import os

CONFIGS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
           "number_theory", "prealgebra", "precalculus"]


def last_boxed(text: str):
    """Content of the last \\boxed{...} (brace matched) or None."""
    start = text.rfind("\\boxed")
    if start < 0:
        return None
    i = text.find("{", start)
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
        j += 1
    return None


def norm(s: str) -> str:
    return " ".join(s.split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/math-train/train.jsonl")
    p.add_argument("--math500", default="data/math-500/test.jsonl", help="problems to exclude")
    args = p.parse_args()

    import datasets
    excluded = set()
    if os.path.exists(args.math500):
        excluded = {norm(json.loads(l)["problem"]) for l in open(args.math500, encoding="utf-8") if l.strip()}
    rows, no_answer, overlap = [], 0, 0
    for cfg in CONFIGS:
        for ex in datasets.load_dataset("EleutherAI/hendrycks_math", cfg, split="train"):
            answer = last_boxed(ex["solution"])
            if not answer:
                no_answer += 1
                continue
            if norm(ex["problem"]) in excluded:
                overlap += 1
                continue
            level = ex.get("level", "")
            level = int(level.split()[-1]) if isinstance(level, str) and level.split()[-1].isdigit() else 0
            rows.append({"problem": ex["problem"], "answer": answer, "subject": ex.get("type", cfg), "level": level})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{args.out}: {len(rows)} problems ({no_answer} without a boxed answer, {overlap} also in MATH-500 removed)")


if __name__ == "__main__":
    main()
