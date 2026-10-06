#!/usr/bin/env python3
"""
SFT set from the scaled-up labelling: for every problem the teacher solved, the answer of the FIRST teacher model
that solved it (base 32B, then the best perturbed models), restricted to the problems the small model fails
(hard samples), optionally plus a fraction of the easy ones. The test sets stay exactly those of the first
distillation run (meta.json is copied), so the numbers are comparable with it:
  python lora_distill/build_big_dataset.py --out_dir logs/distill_big
"""
import argparse
import glob
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO_ROOT, load_train_problems, read_json, read_jsonl, write_json, write_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="logs/distill_big")
    p.add_argument("--teacher_dir", default=None, help="default: <out_dir>/teacher")
    p.add_argument("--student_dir", default=None, help="default: <out_dir>/student")
    p.add_argument("--prev_meta", default="logs/lora_32b_to_3b/meta.json")
    p.add_argument("--datasets", default="math,gsm8k")
    p.add_argument("--keep_easy_fraction", type=float, default=0.0, help="share of the easy problems added back")
    p.add_argument("--max_per_dataset", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.chdir(REPO_ROOT)
    teacher_dir = args.teacher_dir or os.path.join(args.out_dir, "teacher")
    student_dir = args.student_dir or os.path.join(args.out_dir, "student")
    rng = random.Random(args.seed)

    first = {}                                     # (dataset, idx) -> (stage, response); lowest stage wins
    for f in sorted(glob.glob(os.path.join(teacher_dir, "labels", "s*_c*.jsonl"))):
        stage = int(os.path.basename(f)[1:3])
        for rec in read_jsonl(f):
            if rec["correct"] and (rec["dataset"], rec["idx"]) not in first:
                first[(rec["dataset"], rec["idx"])] = (stage, rec["response"])
    student = {}
    for f in sorted(glob.glob(os.path.join(student_dir, "s_c*.jsonl"))):
        for rec in read_jsonl(f):
            student[(rec["dataset"], rec["idx"])] = rec["correct"]
    missing = [k for k in first if k not in student]
    if missing:
        sys.exit(f"{len(missing)} teacher-solved problems have no student result in {student_dir}; run label_student.py")

    problems = load_train_problems([n.strip() for n in args.datasets.split(",")])
    rows, stats = [], {}
    for name in problems:
        ids = sorted(k for k in first if k[0] == name)
        hard = [k for k in ids if not student[k]]
        easy = [k for k in ids if student[k]]
        chosen = hard + rng.sample(easy, int(round(args.keep_easy_fraction * len(easy))))
        if args.max_per_dataset and len(chosen) > args.max_per_dataset:
            chosen = rng.sample(chosen, args.max_per_dataset)
        for k in sorted(chosen):
            stage, response = first[k]
            _, datas = problems[name]
            rows.append({"dataset": name, "idx": k[1], "stage": stage, "messages": datas[k[1]]["messages"],
                         "response": response, "ground_truth": str(datas[k[1]]["ground_truth"])})
        stats[name] = {"teacher_solved": len(ids), "student_fails_(hard)": len(hard), "student_solves_(easy)": len(easy),
                       "used": len(chosen), "from_stage": dict(sorted(Counter(first[k][0] for k in chosen).items()))}
        print(f"{name}: teacher solved {len(ids)}, hard {len(hard)}, easy {len(easy)} -> {len(chosen)} examples")
    rng.shuffle(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), rows)
    meta = read_json(args.prev_meta)               # same test sets and 32B reference numbers as the first run
    meta["train_set"] = {"source": "scaled-up labelling", "keep_easy_fraction": args.keep_easy_fraction,
                         "per_dataset": stats, "n_examples": len(rows)}
    write_json(os.path.join(args.out_dir, "meta.json"), meta)
    print(f"{len(rows)} training examples -> {args.out_dir}/train.jsonl")


if __name__ == "__main__":
    main()
