"""Evaluate RandOpt weight perturbations: python evaluate.py --config configs/X.yaml

data_handlers/ and utils/ are copied from RandOpt (github.com/sunrainyg/RandOpt);
prepared datasets go in data/ (see data/README.md).
The model loads once. Each seed resets weights to a stored base copy and adds
sigma * noise(seed), the same noise as RandOpt's perturb_self_weights.
Finished seeds (seed_N/metrics.json) are skipped, so re-running resumes.
Under a Slurm job array (sbatch --array=0-3 run.sh X.yaml) each task takes every
n-th seed, one GPU per task, all writing to the same output_dir.
procs_per_gpu: k > 1 runs k processes per task on the same GPU (small models leave it idle),
each with every k-th of the task's seeds and 1/k of the vLLM memory.
sigma: 0 evaluates the base model.
"""

import argparse
import json
import os
from pathlib import Path

import yaml

DEFAULTS = {
    "sigma": 0.001,             # paper: 0.001, 0.002, 0.003
    "dtype": "bfloat16",        # any vLLM dtype: bfloat16, float16, float32, auto
    "max_tokens": None,         # default: benchmark's own (RandOpt), e.g. gsm8k 1024, uspto50k 64
    "train_samples": 200,       # gsm8k paper runs used 7473 (full train)
    "train_offset": 0,          # skip the first N train questions, e.g. ones used for selection
    "samples": None,            # optional per-split limit
    "splits": ["train", "test"],
    "train_data_path": None,
    "test_data_path": None,
    "output_dir": None,         # default: predictions/<config name>
    "vllm": {},                 # extra vllm.LLM kwargs, e.g. tensor_parallel_size
    "procs_per_gpu": 1,         # throughput only, results are the same
}
REQUIRED = {"model", "benchmark", "seeds"}


def parse_seeds(spec):
    """Turn 5, "0-99", or [0, 3, "10-19"] into a list of seeds; ranges include both ends."""
    seeds = []
    for item in spec if isinstance(spec, list) else [spec]:
        if isinstance(item, str) and "-" in item:
            low, high = map(int, item.split("-"))
            seeds += range(low, high + 1)
        else:
            seeds.append(int(item))
    return list(dict.fromkeys(seeds))


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if missing := REQUIRED - cfg.keys():
        raise SystemExit(f"missing config keys: {sorted(missing)}")
    if unknown := cfg.keys() - REQUIRED - DEFAULTS.keys():
        raise SystemExit(f"unknown config keys: {sorted(unknown)}")
    return {**DEFAULTS, **cfg}


def load_eval_data(handler, split, path, train_samples, train_offset, samples):
    shared = handler.default_train_path == handler.default_test_path
    if path is None and shared:
        if train_offset:
            raise SystemExit("train_offset would shift train into this benchmark's test questions")
        rows = handler.load_data(handler.default_train_path, split="train")
        rows = rows[:train_samples] if split == "train" else rows[train_samples:]
    else:
        path = path or getattr(handler, f"default_{split}_path")
        rows = handler.load_data(path, split=split)
        if split == "train":
            rows = rows[train_offset:train_offset + train_samples]
    return rows if samples is None else rows[:samples]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    config_path = parser.parse_args().config
    cfg = load_config(config_path)
    procs = cfg.pop("procs_per_gpu")
    if procs > 1 and "EVAL_SHARD" not in os.environ:
        import subprocess
        import sys
        import time
        children = []
        for k in range(procs):
            children.append(subprocess.Popen([sys.executable, *sys.argv],
                                             env={**os.environ, "EVAL_SHARD": f"{k}/{procs}"}))
            time.sleep(60)  # ponytail: staggered start so vLLM memory profiling doesn't race
        raise SystemExit(max(child.wait() for child in children))
    seeds = parse_seeds(cfg["seeds"])
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        task = int(os.environ["SLURM_ARRAY_TASK_ID"]) - int(os.environ["SLURM_ARRAY_TASK_MIN"])
        seeds = seeds[task::int(os.environ["SLURM_ARRAY_TASK_COUNT"])]
    if "EVAL_SHARD" in os.environ:
        k, procs = map(int, os.environ["EVAL_SHARD"].split("/"))
        seeds = seeds[k::procs]
    out = Path(cfg["output_dir"] or Path("predictions") / config_path.stem)

    # Same directory may only hold one setting; adding seeds later is fine.
    settings = {k: v for k, v in cfg.items() if k != "seeds"}
    saved = out / "config.yaml"
    defaults = {k: v for k, v in DEFAULTS.items() if k != "procs_per_gpu"}
    if saved.exists() and {**defaults, **yaml.safe_load(saved.read_text())} != settings:
        raise SystemExit(f"{out} holds runs with other settings; change output_dir")
    todo = [s for s in seeds if not (out / f"seed_{s}" / "metrics.json").exists()]
    print(f"{len(seeds) - len(todo)} seeds done, {len(todo)} to run -> {out}")
    if not todo:
        return
    out.mkdir(parents=True, exist_ok=True)
    saved.write_text(yaml.safe_dump(settings))

    # FlashInfer's sampler JIT-compiles with the system nvcc, which fails on older
    # CUDA toolkits; greedy decoding doesn't need it.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from data_handlers import get_dataset_handler
    from vllm import LLM, SamplingParams

    handler = get_dataset_handler(cfg["benchmark"])
    data = {split: load_eval_data(handler, split, cfg[f"{split}_data_path"],
                                  cfg["train_samples"], cfg["train_offset"], cfg["samples"])
            for split in cfg["splits"]}
    for split, rows in data.items():
        if not rows:
            raise SystemExit(f"{split} split is empty")

    llm = LLM(model=cfg["model"], dtype=cfg["dtype"], seed=42,
              worker_extension_cls="utils.worker_extn.WorkerExtension",
              **{"enforce_eager": True, "gpu_memory_utilization": 0.6,
                 "enable_prefix_caching": True, **cfg["vllm"]}
              | ({"gpu_memory_utilization": cfg["vllm"].get("gpu_memory_utilization", 0.6) / procs}
                 if procs > 1 else {}))
    llm.collective_rpc("store_base_weights")  # ponytail: base copy costs 1x weights of GPU memory
    params = SamplingParams(temperature=0.0, seed=42, max_tokens=cfg["max_tokens"] or handler.default_max_tokens)

    tokenizer = llm.get_tokenizer()
    instruct = tokenizer.chat_template and any(
        tag in cfg["model"].lower() for tag in ["instruct", "chat", "it"])
    rows = [row for split_rows in data.values() for row in split_rows]
    prompts = [
        tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
        if instruct else "\n".join(m["content"] for m in row["messages"]) + "\n"
        for row in rows
    ]

    for n, seed in enumerate(todo, 1):
        llm.collective_rpc("apply_perturbation", args=(seed, cfg["sigma"]))
        llm.reset_prefix_cache()  # cached KV came from the previous weights
        # All splits in one call so vLLM keeps its batch full.
        outputs = llm.generate(prompts, params)
        directory = out / f"seed_{seed}"
        directory.mkdir(exist_ok=True)
        metrics, start = {}, 0
        for split, split_rows in data.items():
            end = start + len(split_rows)
            with (directory / f"{split}.jsonl").open("w", encoding="utf-8") as file:
                for index in range(start, end):
                    pred = outputs[index].outputs[0]
                    file.write(json.dumps(
                        {"index": index - start, "task": rows[index], "prompt": prompts[index],
                         "response": pred.text, "token_ids": list(pred.token_ids),
                         "finish_reason": pred.finish_reason},
                        ensure_ascii=False, default=lambda value: value.tolist()) + "\n")
            metrics[split] = {"samples": len(split_rows),
                              "mean_reward": handler.postprocess_outputs(outputs[start:end], split_rows)}
            start = end
        # Written last: marks the seed complete.
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"[{n}/{len(todo)}] seed {seed}: {json.dumps(metrics)}")


if __name__ == "__main__":
    main()
