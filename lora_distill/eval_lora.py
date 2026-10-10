#!/usr/bin/env python3
"""
Evaluate the base small model and its LoRA adapters on every test set (greedy, bf16, max_tokens 1024, the same
handlers/prompts as evaluate.py), next to the 32B reference numbers from build_dataset.py:
  python lora_distill/eval_lora.py --build_dir logs/lora_32b_to_3b \
      --adapters logs/lora_32b_to_3b/lora/epoch_1,logs/lora_32b_to_3b/lora/epoch_2,logs/lora_32b_to_3b/lora/epoch_3
Writes <build_dir>/eval/eval_results.json and generations/<variant>_<dataset>.jsonl (every response).
"""
import argparse
import json
import os
import random
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build_dir", required=True, help="dir with meta.json from build_dataset.py")
    p.add_argument("--adapters", default="", help="comma list of LoRA adapter dirs")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--max_lora_rank", type=int, default=16)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--teacher_label", default="Qwen2.5-32B",
                   help="name of the reference model in meta.json's teacher_reference_on_test, only for the printed table")
    p.add_argument("--repeats", type=int, default=1,
                   help="evaluate every variant this many times with differently ordered prompts: the batch "
                        "composition changes bf16 greedy results by +-2-3 points, so report the mean and the range")
    p.add_argument("--full_math500", action="store_true",
                   help="also evaluate on all 500 MATH-500 problems (use when none of them was used for training)")
    args = p.parse_args()
    os.chdir(REPO_ROOT)
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")

    import evaluate as ev
    from data_handlers import get_dataset_handler
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    with open(os.path.join(args.build_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    handlers, tests = {}, {}
    for name, m in meta["datasets"].items():
        handler = get_dataset_handler(name)
        datas, _ = ev.load_problems(handler, handler.default_train_path, handler.default_test_path,
                                    meta["train_samples"], meta["test_samples"])
        handlers[name] = handler
        tests[name] = [(j, datas[j]) for j in m["test_idx"]]
    if args.full_math500:
        handler = get_dataset_handler("math500")
        datas, _ = ev.load_problems(handler, handler.default_train_path, handler.default_test_path, 0, None)
        handlers["math500_all"], tests["math500_all"] = handler, list(enumerate(datas))
    tok = AutoTokenizer.from_pretrained(args.model_name)
    prompts = [tok.apply_chat_template(d["messages"], add_generation_prompt=True, tokenize=False)
               for name in tests for _, d in tests[name]]
    sp = SamplingParams(temperature=0.0, seed=42, max_tokens=args.max_tokens)

    adapters = [a.strip() for a in args.adapters.split(",") if a.strip()]
    llm = LLM(model=args.model_name, dtype="bfloat16", enable_lora=bool(adapters), max_lora_rank=args.max_lora_rank,
              max_model_len=4096, gpu_memory_utilization=args.gpu_memory_utilization, disable_log_stats=True)
    variants = [("base", None)] + [(os.path.basename(a.rstrip("/")), LoRARequest(os.path.basename(a.rstrip("/")), i + 1, a))
                                   for i, a in enumerate(adapters)]

    out_dir = os.path.join(args.build_dir, "eval")
    os.makedirs(os.path.join(out_dir, "generations"), exist_ok=True)
    results = {}
    for vname, lora in variants:
        results[vname] = {}
        per_repeat = {name: [] for name in tests}
        for rep in range(args.repeats):
            order = list(range(len(prompts)))
            if rep:                                   # repeat 0 keeps the original order
                random.Random(rep).shuffle(order)
            shuffled = llm.generate([prompts[i] for i in order], sp, lora_request=lora, use_tqdm=False)
            outputs = [None] * len(prompts)
            for pos, i in enumerate(order):
                outputs[i] = shuffled[pos]
            start = 0
            for name in tests:
                datas = [d for _, d in tests[name]]
                recs = ev.build_records(handlers[name], outputs[start:start + len(datas)], datas)
                start += len(datas)
                per_repeat[name].append(float(np.mean([r["correct"] for r in recs])))
                if rep == 0:
                    with open(os.path.join(out_dir, "generations", f"{vname}_{name}.jsonl"), "w", encoding="utf-8") as f:
                        for (j, _), r in zip(tests[name], recs):
                            f.write(json.dumps({"idx": j, **r}, ensure_ascii=False) + "\n")
                    results[vname][name] = {"n": len(recs), "truncated": int(sum(r["finish_reason"] == "length" for r in recs))}
        for name in tests:
            accs = per_repeat[name]
            results[vname][name].update(accuracy=float(np.mean(accs)), accuracy_repeats=accs,
                                        accuracy_min=min(accs), accuracy_max=max(accs))
    results["teacher_32b"] = {name: m["teacher_reference_on_test"] for name, m in meta["datasets"].items()}
    with open(os.path.join(out_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)

    names = list(tests)
    print("\n" + "=" * 70)
    width = 22 if args.repeats > 1 else 14
    print(f"{'model':<30}" + "".join(f"{n:>{width}}" for n in names) + (f"   (mean of {args.repeats} orders, min-max)" if args.repeats > 1 else ""))
    for vname, _ in variants:
        cells = []
        for n in names:
            r = results[vname][n]
            cells.append(f"{r['accuracy'] * 100:5.1f}% ({r['accuracy_min'] * 100:4.1f}-{r['accuracy_max'] * 100:4.1f})".rjust(width)
                         if args.repeats > 1 else f"{r['accuracy'] * 100:>13.1f}%")
        print(f"{vname if vname != 'base' else args.model_name.split('/')[-1] + ' (no LoRA)':<30}" + "".join(cells))
    ref = results["teacher_32b"]
    cell = lambda n, key: f"{ref[n][key] * 100:>{width - 1}.1f}%" if key in ref.get(n, {}) else f"{'-':>{width}}"
    print(f"{args.teacher_label + ' base':<30}" + "".join(cell(n, "base_accuracy") for n in names))
    for key in sorted({k for n in ref for k in ref[n] if k.startswith("top")}, key=lambda s: int(s[3:].split('_')[0])):
        print(f"{args.teacher_label + ' ' + key.replace('_vote_accuracy', ' vote'):<30}" + "".join(cell(n, key) for n in names))
    print("test sizes: " + ", ".join(f"{n}={len(tests[n])}" for n in names))


if __name__ == "__main__":
    main()
