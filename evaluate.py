#!/usr/bin/env python3
"""
RandOpt on MATH-500 with per-seed timing and per-seed answer logs (single GPU, vLLM).

Protocol (Neural Thickets, arXiv:2603.12228):
  - N random perturbations theta' = theta + sigma * eps(seed), eps ~ N(0, I) on ALL parameters,
    sigma sampled uniformly from {1e-3, 2e-3, 3e-3} (Table 3), bf16, max length 1024.
  - MATH-500: first 200 problems = train (selection), remaining 300 = test.
  - Top-K seeds by train reward, majority vote over their test answers.

Every seed is evaluated on ALL 500 problems and the full model responses are written to
<out_dir>/seeds/seed_XXXX.json together with timings, so selection / voting for any K can be
recomputed from the logs without generating again. Re-running the same command resumes:
seeds that already have a log are skipped.

Output layout:
  <out_dir>/args.json            run configuration
  <out_dir>/problems.json        problems, ground truths, train/test split
  <out_dir>/run.log              console log
  <out_dir>/base.json            sigma=0 (base model) run, same format as a seed log
  <out_dir>/seeds/seed_XXXX.json one log per perturbation (timings + all responses)
  <out_dir>/summary.json         timings (one seed / all seeds) and ensemble accuracy

Usage (from the repo root):
  python evaluate.py          # on a shared cluster run it through Slurm, see scripts/slurm_evaluate.sh
  python evaluate.py --aggregate_only        # recompute summary.json from existing logs
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
log = logging.getLogger("evaluate")

# Keys of args.json that must match when resuming into an existing out_dir.
RESUME_KEYS = ("model_name", "population_size", "sigma_values", "global_seed",
               "max_tokens", "train_samples", "precision")


# -----------------------------------------------------------------------------
# Args / setup
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="RandOpt MATH-500 evaluation with timing and logs",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--data_path", default="data/math-500/test.jsonl")
    p.add_argument("--train_samples", type=int, default=200, help="first N problems are used for selection")
    p.add_argument("--population_size", type=int, default=500, help="number of seeds")
    p.add_argument("--sigma_values", default="0.001,0.002,0.003", help="sigma set from the paper (Table 3)")
    p.add_argument("--top_k", default="1,5,25,50", help="ensemble sizes for the summary")
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--precision", choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.75)
    p.add_argument("--global_seed", type=int, default=42)
    p.add_argument("--cuda_devices", default=None,
                   help="default: keep CUDA_VISIBLE_DEVICES from the environment (e.g. set by Slurm), else \"0\"")
    p.add_argument("--out_dir", default="logs/math500_qwen2.5-3b-instruct_n500")
    p.add_argument("--cuda_graphs", action="store_true",
                   help="enable torch.compile + CUDA graphs in vLLM (faster decoding; the repo default is eager)")
    p.add_argument("--max_new_seeds", type=int, default=None,
                   help="stop after running this many new seeds (for quick checks)")
    p.add_argument("--aggregate_only", action="store_true",
                   help="do not launch vLLM, only rebuild summary.json from existing logs")
    args = p.parse_args()
    args.sigma_list = [float(s) for s in args.sigma_values.split(",")]
    args.top_k_list = sorted({int(k) for k in args.top_k.split(",")})
    return args


def setup_logging(out_dir: str):
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(out_dir, "run.log"))):
        h.setFormatter(fmt)
        log.addHandler(h)


def write_json(path: str, obj):
    """Atomic write so a killed run never leaves a half-written log that resume would trust."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_population(n: int, sigmas: List[float], global_seed: int):
    """Same sampling as randopt.py: unique seeds, sigma uniform over the sigma set."""
    rng = np.random.default_rng(seed=global_seed)
    seeds = rng.choice(2**31, size=n, replace=False).tolist()
    sigma_per_seed = rng.choice(sigmas, size=n).tolist()
    return [(int(s), float(sg)) for s, sg in zip(seeds, sigma_per_seed)]


def load_problems(handler, args):
    datas = handler.load_data(args.data_path, split="train", max_samples=None)
    return datas, [("train" if i < args.train_samples else "test") for i in range(len(datas))]


# -----------------------------------------------------------------------------
# Scoring helpers (same logic as randopt.py ensemble evaluation)
# -----------------------------------------------------------------------------

def extract(handler, response: str) -> str:
    return handler.extract_answer_for_voting(response) or ""


def answer_correct(handler, answer: str, ground_truth) -> bool:
    if not answer:
        return False
    if hasattr(handler, "is_voted_answer_correct"):
        return bool(handler.is_voted_answer_correct(answer, ground_truth))
    return bool(handler.is_answer_correct(handler.format_answer_for_check(answer), ground_truth))


def build_records(handler, outputs, datas) -> List[dict]:
    records = []
    for i, (out, data) in enumerate(zip(outputs, datas)):
        o = out.outputs[0]
        reward = float(handler.compute_reward(o.text, data["ground_truth"]))
        records.append({
            "idx": i,
            "response": o.text,
            "answer": extract(handler, o.text),
            "reward": reward,
            "correct": reward > 0,
            "n_tokens": len(o.token_ids),
            "finish_reason": o.finish_reason,
        })
    return records


# -----------------------------------------------------------------------------
# Engine wrapper (one vLLM engine on one GPU, launched through core.engine)
# -----------------------------------------------------------------------------

class SingleEngine:
    def __init__(self, args):
        import ray
        from core import launch_engines

        self.ray = ray
        if args.cuda_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
        else:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ["VLLM_NO_USAGE_STATS"] = "1"
        # Ray workers must be able to import utils.worker_extn from the repo root.
        os.environ["PYTHONPATH"] = REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
        # Without num_cpus Ray starts one idle worker per core of the whole node, which starves
        # a Slurm job that was given only a few cores.
        num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or min(os.cpu_count() or 1, 8))
        ray.init(address="local", num_cpus=num_cpus, ignore_reinit_error=True)
        log.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
                 f"ray resources={ray.cluster_resources()}")
        t0 = time.perf_counter()
        self.engines, self.pgs = launch_engines(
            1, args.model_name, precision=args.precision, tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=not args.cuda_graphs)
        self.launch_s = time.perf_counter() - t0
        self.engine = self.engines[0]

    def perturb(self, seed: int, sigma: float):
        # apply_perturbation resets to the stored base weights first, so bf16 rounding
        # error cannot accumulate over hundreds of perturb/restore cycles.
        self.ray.get(self.engine.collective_rpc.remote("apply_perturbation", args=(seed, sigma)))

    def reset(self):
        self.ray.get(self.engine.collective_rpc.remote("reset_to_base_weights", args=()))

    def generate(self, prompts, sampling_params):
        return self.ray.get(self.engine.generate.remote(prompts, sampling_params, use_tqdm=False))

    def close(self):
        from core import cleanup_engines
        cleanup_engines(self.engines, self.pgs)


# -----------------------------------------------------------------------------
# One run = one seed (or the base model) on all problems
# -----------------------------------------------------------------------------

def run_one(engine, handler, prompts, datas, sampling_params, train_samples: int,
            seed: Optional[int], sigma: float) -> dict:
    t0 = time.perf_counter()
    if seed is not None:
        engine.perturb(seed, sigma)
    t1 = time.perf_counter()
    outputs = engine.generate(prompts, sampling_params)
    t2 = time.perf_counter()
    if seed is not None:
        engine.reset()
    t3 = time.perf_counter()

    records = build_records(handler, outputs, datas)
    n_tokens = sum(r["n_tokens"] for r in records)
    train = [r["correct"] for r in records[:train_samples]]
    test = [r["correct"] for r in records[train_samples:]]
    return {
        "seed": seed,
        "sigma": sigma,
        "train_reward": float(np.mean([r["reward"] for r in records[:train_samples]])),
        "train_accuracy": float(np.mean(train)),
        "test_accuracy": float(np.mean(test)) if test else None,
        "full_accuracy": float(np.mean([r["correct"] for r in records])),
        "timing": {
            "perturb_s": t1 - t0,
            "generate_s": t2 - t1,
            "reset_s": t3 - t2,
            "total_s": t3 - t0,
            "n_tokens": n_tokens,
            "tokens_per_s": n_tokens / (t2 - t1) if t2 > t1 else None,
        },
        "records": records,
    }


def seed_path(out_dir: str, i: int) -> str:
    return os.path.join(out_dir, "seeds", f"seed_{i:04d}.json")


def log_is_valid(path: str, seed: Optional[int], sigma: float) -> bool:
    if not os.path.exists(path):
        return False
    try:
        d = read_json(path)
    except (json.JSONDecodeError, OSError):
        return False
    return d.get("seed") == seed and d.get("sigma") == sigma


# -----------------------------------------------------------------------------
# Aggregation from logs: timing + top-K majority vote (selection on train, vote on test)
# -----------------------------------------------------------------------------

def stats(xs: List[float]) -> dict:
    return {"mean": statistics.fmean(xs), "median": statistics.median(xs), "min": min(xs), "max": max(xs)}


def aggregate(handler, args, datas, out_dir: str) -> dict:
    ts = args.train_samples
    seeds = []
    for i in range(args.population_size):
        path = seed_path(out_dir, i)
        if os.path.exists(path):
            d = read_json(path)
            seeds.append({k: d[k] for k in ("seed", "sigma", "train_reward", "test_accuracy", "timing")}
                         | {"answers": [r["answer"] for r in d["records"]]})
    summary = {"n_seeds_done": len(seeds), "n_seeds_total": args.population_size}
    if not seeds:
        return summary

    base = read_json(os.path.join(out_dir, "base.json")) if os.path.exists(os.path.join(out_dir, "base.json")) else None
    if base:
        summary["base"] = {k: base[k] for k in ("train_reward", "test_accuracy", "full_accuracy", "timing")}

    totals = [s["timing"]["total_s"] for s in seeds]
    summary["timing"] = {
        "one_seed_total_s": stats(totals),
        "one_seed_generate_s": stats([s["timing"]["generate_s"] for s in seeds]),
        "one_seed_perturb_s": stats([s["timing"]["perturb_s"] for s in seeds]),
        "all_seeds_total_s": sum(totals),
        "all_seeds_total_h": sum(totals) / 3600,
    }
    summary["single_seed_test_accuracy"] = stats([s["test_accuracy"] for s in seeds])
    summary["sigma_stats"] = {
        str(sg): {"n": len(v), "mean_train_reward": statistics.fmean(v)}
        for sg in args.sigma_list
        if (v := [s["train_reward"] for s in seeds if s["sigma"] == sg])
    }

    # Selection by train reward (stable sort, ties keep seed order), majority vote on test.
    ranked = sorted(seeds, key=lambda s: s["train_reward"], reverse=True)
    ensemble = {}
    for k in args.top_k_list:
        if k > len(ranked):
            continue
        top = ranked[:k]
        correct = 0
        for idx in range(ts, len(datas)):
            votes = [s["answers"][idx] for s in top if s["answers"][idx]]
            if votes:
                final = Counter(votes).most_common(1)[0][0]
                correct += answer_correct(handler, final, datas[idx]["ground_truth"])
        n_test = len(datas) - ts
        ensemble[str(k)] = {"accuracy": correct / n_test, "correct": correct, "n_test": n_test,
                            "seeds": [s["seed"] for s in top]}
    summary["ensemble"] = ensemble
    return summary


def print_summary(summary: dict):
    log.info("=" * 60)
    log.info(f"SUMMARY: {summary['n_seeds_done']}/{summary['n_seeds_total']} seeds")
    if "base" in summary:
        b = summary["base"]
        log.info(f"base: train_reward={b['train_reward']:.4f} test_acc={b['test_accuracy']:.4f} "
                 f"time={b['timing']['total_s']:.1f}s")
    if "timing" in summary:
        t = summary["timing"]["one_seed_total_s"]
        log.info(f"one seed: mean={t['mean']:.1f}s median={t['median']:.1f}s min={t['min']:.1f}s max={t['max']:.1f}s")
        log.info(f"all seeds: {summary['timing']['all_seeds_total_s']:.0f}s ({summary['timing']['all_seeds_total_h']:.2f}h)")
        s = summary["single_seed_test_accuracy"]
        log.info(f"single-seed test acc: mean={s['mean']:.4f} max={s['max']:.4f}")
    for k, v in summary.get("ensemble", {}).items():
        log.info(f"  top-{k} majority vote: test acc={v['accuracy']*100:.2f}% ({v['correct']}/{v['n_test']})")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(args):
    os.chdir(REPO_ROOT)
    from data_handlers import get_dataset_handler

    out_dir = args.out_dir
    os.makedirs(os.path.join(out_dir, "seeds"), exist_ok=True)
    setup_logging(out_dir)

    args_path = os.path.join(out_dir, "args.json")
    cfg = {k: v for k, v in vars(args).items() if k not in ("aggregate_only",)}
    if os.path.exists(args_path):
        prev = read_json(args_path)
        diff = [k for k in RESUME_KEYS if prev.get(k) != cfg.get(k)]
        if prev.get("cuda_graphs", False) != args.cuda_graphs:
            log.warning(f"{out_dir} was started with cuda_graphs={prev.get('cuda_graphs', False)}, "
                        f"now cuda_graphs={args.cuda_graphs}: timings in summary mix both modes")
        if diff:
            sys.exit(f"{out_dir} was created with different {diff}; use another --out_dir")
    else:
        write_json(args_path, cfg)

    handler = get_dataset_handler("math500")
    datas, splits = load_problems(handler, args)
    log.info(f"{len(datas)} problems: {splits.count('train')} train / {splits.count('test')} test")
    write_json(os.path.join(out_dir, "problems.json"), [
        {"idx": i, "split": splits[i], "problem": d["problem"], "ground_truth": d["ground_truth"],
         "subject": d["subject"], "level": d["level"]} for i, d in enumerate(datas)])

    population = make_population(args.population_size, args.sigma_list, args.global_seed)
    base_path = os.path.join(out_dir, "base.json")
    pending = [i for i, (s, sg) in enumerate(population) if not log_is_valid(seed_path(out_dir, i), s, sg)]
    need_base = not log_is_valid(base_path, None, 0.0)
    if args.max_new_seeds is not None:
        pending = pending[:args.max_new_seeds]
    log.info(f"{len(population) - len(pending)} seeds already done, {len(pending)} to run")

    if not args.aggregate_only and (pending or need_base):
        from transformers import AutoTokenizer
        from vllm import SamplingParams

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        prompts = [tokenizer.apply_chat_template(d["messages"], add_generation_prompt=True, tokenize=False)
                   for d in datas]
        sampling_params = SamplingParams(temperature=0.0, seed=args.global_seed, max_tokens=args.max_tokens)

        t_start = time.perf_counter()
        engine = SingleEngine(args)
        log.info(f"engine launched in {engine.launch_s:.1f}s")
        try:
            # The base run also warms the engine up, so it is not charged to seed 0.
            if need_base:
                res = run_one(engine, handler, prompts, datas, sampling_params, args.train_samples, None, 0.0)
                res["cuda_graphs"] = args.cuda_graphs
                write_json(base_path, res)
                log.info(f"base: train_reward={res['train_reward']:.4f} test_acc={res['test_accuracy']:.4f} "
                         f"time={res['timing']['total_s']:.1f}s")

            loop_start = time.perf_counter()
            for n_done, i in enumerate(pending):
                seed, sigma = population[i]
                res = run_one(engine, handler, prompts, datas, sampling_params, args.train_samples, seed, sigma)
                res["index"] = i
                res["cuda_graphs"] = args.cuda_graphs
                write_json(seed_path(out_dir, i), res)
                elapsed = time.perf_counter() - loop_start
                eta = elapsed / (n_done + 1) * (len(pending) - n_done - 1)
                t = res["timing"]
                log.info(f"[{i + 1}/{len(population)}] seed={seed} sigma={sigma} "
                         f"train={res['train_reward']:.3f} test={res['test_accuracy']:.3f} "
                         f"time={t['total_s']:.1f}s (perturb {t['perturb_s']:.1f} / gen {t['generate_s']:.1f}) "
                         f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m")
        finally:
            engine.close()
        wall = time.perf_counter() - t_start
        log.info(f"this invocation: {wall:.0f}s wall ({wall / 3600:.2f}h) incl. engine launch")
    else:
        wall = None

    summary = aggregate(handler, args, datas, out_dir)
    summary["this_invocation_wall_s"] = wall
    write_json(os.path.join(out_dir, "summary.json"), summary)
    print_summary(summary)


if __name__ == "__main__":
    main(parse_args())
