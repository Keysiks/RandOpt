#!/usr/bin/env python3
"""
Run the small model (no LoRA, greedy) on every problem the teacher solved, to find out which problems it already
solves itself. build_big_dataset.py then keeps the hard ones (as in the paper, which distils on hard samples).
Resumable in chunks (<out_dir>/s_cCCCC.jsonl):
  python lora_distill/label_student.py --teacher_dir logs/distill_big/teacher --out_dir logs/distill_big/student
"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (REPO_ROOT, add_engine_args, build_prompts, chunks, load_train_problems, make_engine,
                    read_jsonl, write_jsonl)


def teacher_solved_ids(teacher_dir):
    ids = set()
    for f in sorted(glob.glob(os.path.join(teacher_dir, "labels", "s*_c*.jsonl"))):
        for rec in read_jsonl(f):
            if rec["correct"]:
                ids.add((rec["dataset"], rec["idx"]))
    return sorted(ids)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_engine_args(p, tp=1, base_on_cpu=False)
    p.set_defaults(model_name="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--teacher_dir", default="logs/distill_big/teacher")
    p.add_argument("--out_dir", default="logs/distill_big/student")
    p.add_argument("--datasets", default="math,gsm8k")
    p.add_argument("--chunk", type=int, default=2000)
    args = p.parse_args()
    os.chdir(REPO_ROOT)
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoTokenizer
    from vllm import SamplingParams

    problems = load_train_problems([n.strip() for n in args.datasets.split(",")])
    tok = AutoTokenizer.from_pretrained(args.model_name)
    prompts, _ = build_prompts(tok, problems, args.max_prompt_tokens)
    ids = [pid for pid in teacher_solved_ids(args.teacher_dir) if pid in prompts]
    print(f"{len(ids)} problems solved by the teacher", flush=True)
    sp = SamplingParams(temperature=0.0, seed=42, max_tokens=args.max_tokens)
    engine = None
    try:
        for c, chunk in enumerate(chunks(ids, args.chunk)):
            path = os.path.join(args.out_dir, f"s_c{c:04d}.jsonl")
            if os.path.exists(path):
                continue
            if engine is None:
                engine = make_engine(args)
            t0 = time.perf_counter()
            outputs = engine.generate([prompts[pid] for pid in chunk], sp)
            rows = []
            for (name, idx), out in zip(chunk, outputs):
                o = out.outputs[0]
                handler, datas = problems[name]
                ok = float(handler.compute_reward(o.text, datas[idx]["ground_truth"])) > 0 and o.finish_reason != "length"
                rows.append({"dataset": name, "idx": idx, "correct": ok, "finish_reason": o.finish_reason})
            write_jsonl(path, rows)
            print(f"chunk {c + 1}/{-(-len(ids) // args.chunk)}: student solves {sum(r['correct'] for r in rows)}/{len(rows)}, "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
    finally:
        if engine is not None:
            engine.close()
    print("done")


if __name__ == "__main__":
    main()
