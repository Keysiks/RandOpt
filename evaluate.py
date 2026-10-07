#!/usr/bin/env python3
"""
RandOpt with per-seed timing and per-seed answer logs (vLLM, one or several GPUs).

Protocol (Neural Thickets, arXiv:2603.12228):
  - N random perturbations theta' = theta + sigma * eps(seed), eps ~ N(0, I) on ALL parameters,
    sigma sampled uniformly from {1e-3, 2e-3, 3e-3} (Table 3), bf16, max length 1024.
  - Selection on the train problems, evaluation on the test problems. MATH-500 (one file): the first
    200 problems are train, the rest (300) test. Datasets with separate files (GSM8K): the first
    --train_samples of the train file are train, --test_samples of the test file are test.
  - Top-K seeds by train reward, majority vote over their test answers.

Every seed is evaluated on ALL problems of every dataset (--dataset a,b runs several datasets in one
pass with the same seeds) and the full model responses are written to <dir>/seeds/seed_XXXX.json
together with timings, so selection / voting for any K can be recomputed from the logs without
generating again. Re-running the same command resumes: seeds that already have a log are skipped.

Output layout (<dir> is out_dir for one dataset, out_dir/<dataset> for several):
  <out_dir>/args.json            run configuration
  <out_dir>/run.log              console log
  <dir>/problems.json            problems, ground truths, train/test split
  <dir>/base.json                sigma=0 (base model) run, same format as a seed log
  <dir>/seeds/seed_XXXX.json     one log per perturbation (timings + all responses)
  <dir>/summary.json             timings (one seed / all seeds) and ensemble accuracy

Usage (from the repo root; on a shared cluster run through Slurm, see scripts/):
  python evaluate.py                              # Qwen2.5-3B-Instruct, MATH-500, 500 seeds
  python evaluate.py --aggregate_only             # recompute summary.json from existing logs
  python evaluate.py --model_name Qwen/Qwen2.5-32B-Instruct --tp 2 --base_on_cpu \
      --dataset math500,gsm8k --test_samples 500
"""

import argparse
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
log = logging.getLogger("evaluate")

# Keys of args.json that must match when resuming into an existing out_dir.
RESUME_KEYS = ("model_name", "population_size", "sigma_values", "global_seed",
               "max_tokens", "train_samples", "precision", "dataset", "tp", "test_samples", "population_file")
# values of keys that older args.json files do not have
RESUME_DEFAULTS = {"dataset": "math500", "tp": 1, "test_samples": None, "population_file": None}


# -----------------------------------------------------------------------------
# Args / setup
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="RandOpt MATH-500 evaluation with timing and logs",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--dataset", default="math500",
                   help="dataset name from data_handlers, or a comma list (e.g. math500,gsm8k) evaluated in one pass")
    p.add_argument("--train_data_path", default=None, help="override the handler's train file (one dataset only)")
    p.add_argument("--test_data_path", default=None, help="override the handler's test file (one dataset only)")
    p.add_argument("--train_samples", type=int, default=200, help="first N problems are used for selection")
    p.add_argument("--test_samples", type=int, default=None,
                   help="cap on test problems (None = all; for MATH-500 all means the 300 after the train ones)")
    p.add_argument("--population_size", type=int, default=500, help="number of seeds")
    p.add_argument("--population_file", default=None,
                   help="json list of {\"seed\": .., \"sigma\": ..} to run INSTEAD of the generated population "
                        "(e.g. the best seeds of another dataset, with their original sigmas); order is kept")
    p.add_argument("--sigma_values", default="0.001,0.002,0.003", help="sigma set from the paper (Table 3)")
    p.add_argument("--top_k", default="1,5,25,50", help="ensemble sizes for the summary")
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--precision", choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--tp", type=int, default=1, help="tensor parallel size (GPUs per engine); implies --engine direct")
    p.add_argument("--base_on_cpu", action="store_true",
                   help="keep the copy of the base weights in host RAM instead of on the GPU (needed when the model "
                        "barely fits, e.g. 32B on 2x80GB); implies --engine direct")
    p.add_argument("--max_model_len", type=int, default=None, help="vLLM max_model_len (direct engine)")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.75)
    p.add_argument("--global_seed", type=int, default=42)
    p.add_argument("--cuda_devices", default=None,
                   help="default: keep CUDA_VISIBLE_DEVICES from the environment (e.g. set by Slurm), else \"0\"")
    p.add_argument("--out_dir", default="logs/math500_qwen2.5-3b-instruct_n500")
    p.add_argument("--max_num_seqs", type=int, default=None,
                   help="max concurrently decoded sequences in vLLM (default: vLLM default, 256 on A100); "
                        "set >= number of problems so all of them run in one wave")
    p.add_argument("--cuda_graphs", action="store_true",
                   help="enable torch.compile + CUDA graphs in vLLM (faster decoding; the repo default is eager)")
    p.add_argument("--max_new_seeds", type=int, default=None,
                   help="stop after running this many new seeds, per process (for quick checks)")
    p.add_argument("--engine", choices=["ray", "direct"], default="ray",
                   help="ray: engine through core.engine (original). direct: plain vllm.LLM without Ray, "
                        "the engine core runs in its own process")
    p.add_argument("--procs_per_gpu", type=int, default=1,
                   help="k>1 runs k direct-engine processes on the same GPU, process j takes the seeds with "
                        "index %% k == j and gets gpu_memory_utilization/k (the copy of the base weights comes "
                        "on top of that); implies --engine direct")
    p.add_argument("--prefix_caching", action="store_true", help="direct engine only: vLLM prefix caching")
    p.add_argument("--stagger_s", type=float, default=60.0,
                   help="delay between starting processes when procs_per_gpu > 1 (vLLM memory profiling races otherwise)")
    p.add_argument("--shard", default=None, help=argparse.SUPPRESS)  # \"j/k\", set for child processes
    p.add_argument("--aggregate_only", action="store_true",
                   help="do not launch vLLM, only rebuild summary.json from existing logs")
    args = p.parse_args()
    if args.procs_per_gpu > 1 and args.tp > 1:
        p.error("--procs_per_gpu > 1 cannot be combined with --tp > 1")
    if args.procs_per_gpu > 1 or args.tp > 1 or args.base_on_cpu:
        args.engine = "direct"
    args.dataset_list = [n.strip() for n in args.dataset.split(",") if n.strip()]
    args.population = None
    if args.population_file:
        args.population = load_population(args.population_file)
        args.population_size = len(args.population)
        args.sigma_list = sorted({sg for _, sg in args.population})
    args.sigma_list = [float(s) for s in args.sigma_values.split(",")]
    args.top_k_list = sorted({int(k) for k in args.top_k.split(",")})
    return args


def setup_logging(out_dir: str, shard: Optional[str] = None):
    log.setLevel(logging.INFO)
    tag = f"[{shard}] " if shard else ""
    fmt = logging.Formatter(f"%(asctime)s {tag}%(message)s", "%H:%M:%S")
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


def load_population(path: str):
    """[(seed, sigma)] from a json list of {"seed", "sigma"} dicts or [seed, sigma] pairs."""
    items = read_json(path)
    pop = [(int(x["seed"]), float(x["sigma"])) if isinstance(x, dict) else (int(x[0]), float(x[1])) for x in items]
    if len({s for s, _ in pop}) != len(pop):
        sys.exit(f"{path} lists a seed twice")
    return pop


def make_population(n: int, sigmas: List[float], global_seed: int):
    """Same sampling as randopt.py: unique seeds, sigma uniform over the sigma set."""
    rng = np.random.default_rng(seed=global_seed)
    seeds = rng.choice(2**31, size=n, replace=False).tolist()
    sigma_per_seed = rng.choice(sigmas, size=n).tolist()
    return [(int(s), float(sg)) for s, sg in zip(seeds, sigma_per_seed)]


def load_problems(handler, train_path, test_path, train_samples, test_samples=None):
    """Train problems first, then test problems (same rules as randopt.py). Returns (datas, splits)."""
    if train_path == test_path:  # one file (MATH-500): split by index
        all_data = handler.load_data(train_path, split="train", max_samples=None)
        train = all_data[:train_samples]
        test = all_data[train_samples:] if test_samples is None else all_data[train_samples:train_samples + test_samples]
    else:
        train = handler.load_data(train_path, split="train", max_samples=train_samples)
        test = handler.load_data(test_path, split="test", max_samples=test_samples)
    return train + test, ["train"] * len(train) + ["test"] * len(test)


class Bench:
    """One dataset of a run: its problems, the train/test split and where its logs go."""

    def __init__(self, name, handler, datas, splits, directory):
        self.name, self.handler, self.datas, self.splits, self.dir = name, handler, datas, splits, directory
        self.n_train = splits.count("train")


def load_bench(name: str, args, root: str, multi: bool) -> Bench:
    from data_handlers import get_dataset_handler

    handler = get_dataset_handler(name)
    train_path = (None if multi else args.train_data_path) or handler.default_train_path
    test_path = (None if multi else args.test_data_path) or handler.default_test_path
    datas, splits = load_problems(handler, train_path, test_path, args.train_samples, args.test_samples)
    directory = os.path.join(root, name) if multi else root
    os.makedirs(os.path.join(directory, "seeds"), exist_ok=True)
    return Bench(name, handler, datas, splits, directory)


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
            gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=not args.cuda_graphs,
            max_num_seqs=args.max_num_seqs)
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


class DirectEngine:
    """Plain vllm.LLM without Ray (like evaluate_oleg_version.py): the engine core runs in its own
    process, so tokenization and scheduling overlap with GPU work."""

    def __init__(self, args):
        from vllm import LLM

        if args.cuda_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
        else:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ["VLLM_NO_USAGE_STATS"] = "1"
        os.environ["PYTHONPATH"] = REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
        # FlashInfer's sampler JIT-compiles with the system nvcc and can fail on old toolkits;
        # greedy decoding does not need it.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        extra = {} if args.max_num_seqs is None else {"max_num_seqs": args.max_num_seqs}
        if getattr(args, "max_model_len", None) is not None:
            extra["max_model_len"] = args.max_model_len
        t0 = time.perf_counter()
        self.llm = LLM(
            model=args.model_name, dtype=args.precision, tensor_parallel_size=getattr(args, "tp", 1),
            worker_extension_cls="utils.worker_extn.WorkerExtension",
            enforce_eager=not args.cuda_graphs,
            gpu_memory_utilization=args.gpu_memory_utilization / args.procs_per_gpu,
            enable_prefix_caching=args.prefix_caching, disable_log_stats=True, **extra)
        self.llm.collective_rpc("store_base_weights", args=(bool(getattr(args, "base_on_cpu", False)),))
        self.prefix_caching = args.prefix_caching
        self.launch_s = time.perf_counter() - t0

    def perturb(self, seed: int, sigma: float):
        self.llm.collective_rpc("apply_perturbation", args=(seed, sigma))
        if self.prefix_caching:
            self.llm.reset_prefix_cache()  # cached KV belongs to the previous weights

    def reset(self):
        self.llm.collective_rpc("reset_to_base_weights")

    def generate(self, prompts, sampling_params):
        return self.llm.generate(prompts, sampling_params, use_tqdm=False)

    def close(self):
        import gc
        del self.llm
        gc.collect()


def make_engine(args):
    return DirectEngine(args) if args.engine == "direct" else SingleEngine(args)


def shard_pending(pending: List[int], shard: Optional[str]) -> List[int]:
    """Process j of k takes the seeds with index % k == j. Depends only on the index, so the
    split stays consistent although the processes start at different times."""
    if not shard:
        return pending
    j, k = map(int, shard.split("/"))
    return [i for i in pending if i % k == j]


def run_children(args) -> List[int]:
    """Start procs_per_gpu copies of this script (one shard each) and wait for them."""
    children = []
    for j in range(args.procs_per_gpu):
        cmd = [sys.executable, os.path.join(REPO_ROOT, "evaluate.py"), *sys.argv[1:],
               "--shard", f"{j}/{args.procs_per_gpu}"]
        children.append(subprocess.Popen(cmd))
        log.info(f"started process {j + 1}/{args.procs_per_gpu}")
        if j + 1 < args.procs_per_gpu:
            time.sleep(args.stagger_s)
    return [c.wait() for c in children]


def log_throughput(out_dir: str, indices: List[int]):
    """Effective seconds per seed from the seed logs themselves (per-seed times are inflated when
    processes share a GPU): window from the earliest seed start to the latest seed end."""
    spans = []
    for i in indices:
        path = seed_path(out_dir, i)
        if os.path.exists(path):
            d = read_json(path)
            if "finished_at" in d:
                spans.append((d["finished_at"] - d["timing"]["total_s"], d["finished_at"]))
    if len(spans) > 1:
        window = max(e for _, e in spans) - min(s for s, _ in spans)
        log.info(f"throughput: {len(spans)} new seeds in {window:.0f}s = {window / len(spans):.1f}s per seed "
                 f"(from first seed start to last seed end, engine launch excluded)")


# -----------------------------------------------------------------------------
# One run = one seed (or the base model) on all problems
# -----------------------------------------------------------------------------

def run_one(engine, benches, prompts, sampling_params, seed: Optional[int], sigma: float) -> dict:
    """One generate() call over the prompts of all datasets; returns {dataset name: result log}.

    apply_perturbation always starts from the stored base weights, so no reset is needed afterwards."""
    t0 = time.perf_counter()
    if seed is not None:
        engine.perturb(seed, sigma)
    t1 = time.perf_counter()
    outputs = engine.generate(prompts, sampling_params)
    t2 = time.perf_counter()

    per_bench, start = {}, 0
    for b in benches:
        per_bench[b.name] = build_records(b.handler, outputs[start:start + len(b.datas)], b.datas)
        start += len(b.datas)
    n_tokens_all = sum(r["n_tokens"] for recs in per_bench.values() for r in recs)

    results = {}
    for b in benches:
        records, ts = per_bench[b.name], b.n_train
        train = [r["correct"] for r in records[:ts]]
        test = [r["correct"] for r in records[ts:]]
        n_tokens = sum(r["n_tokens"] for r in records)
        results[b.name] = {
            "seed": seed,
            "sigma": sigma,
            "finished_at": time.time(),
            "train_reward": float(np.mean([r["reward"] for r in records[:ts]])),
            "train_accuracy": float(np.mean(train)),
            "test_accuracy": float(np.mean(test)) if test else None,
            "full_accuracy": float(np.mean([r["correct"] for r in records])),
            "timing": {
                "perturb_s": t1 - t0,
                "generate_s": t2 - t1,  # one call for all datasets of the run
                "total_s": t2 - t0,
                "n_tokens": n_tokens,
                "tokens_per_s": n_tokens_all / (t2 - t1) if t2 > t1 else None,
                "datasets_in_call": [x.name for x in benches],
            },
            "records": records,
        }
    return results


def run_meta(args, name: str) -> dict:
    return dict(cuda_graphs=args.cuda_graphs, engine=args.engine, procs_per_gpu=args.procs_per_gpu,
                tp=args.tp, dataset=name)


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


def aggregate(handler, args, datas, out_dir: str, n_train: int) -> dict:
    ts = n_train
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


def print_summary(summary: dict, name: str = ""):
    log.info("=" * 60)
    log.info(f"SUMMARY{' ' + name if name else ''}: {summary['n_seeds_done']}/{summary['n_seeds_total']} seeds")
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

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    setup_logging(out_dir, args.shard)
    names = args.dataset_list
    multi = len(names) > 1
    if multi and (args.train_data_path or args.test_data_path):
        sys.exit("--train_data_path / --test_data_path work with a single --dataset only")

    args_path = os.path.join(out_dir, "args.json")
    cfg = {k: v for k, v in vars(args).items() if k not in ("aggregate_only", "dataset_list", "population")}
    if os.path.exists(args_path):
        prev = read_json(args_path)
        diff = [k for k in RESUME_KEYS if prev.get(k, RESUME_DEFAULTS.get(k)) != cfg.get(k)]
        if prev.get("cuda_graphs", False) != args.cuda_graphs:
            log.warning(f"{out_dir} was started with cuda_graphs={prev.get('cuda_graphs', False)}, "
                        f"now cuda_graphs={args.cuda_graphs}: timings in summary mix both modes")
        if diff:
            sys.exit(f"{out_dir} was created with different {diff}; use another --out_dir")
    else:
        write_json(args_path, cfg)

    benches = [load_bench(n, args, out_dir, multi) for n in names]
    for b in benches:
        log.info(f"{b.name}: {len(b.datas)} problems: {b.n_train} train / {len(b.datas) - b.n_train} test")
        if not args.shard:  # child processes must not race on this file
            write_json(os.path.join(b.dir, "problems.json"), [
                {"idx": i, "split": b.splits[i],
                 "problem": d.get("problem") or d["messages"][-1]["content"],
                 "ground_truth": d["ground_truth"], "subject": d.get("subject", ""), "level": d.get("level", "")}
                for i, d in enumerate(b.datas)])

    population = args.population or make_population(args.population_size, args.sigma_list, args.global_seed)

    def seed_ok(b, i):
        return log_is_valid(seed_path(b.dir, i), *population[i])

    def base_ok(b):
        return log_is_valid(os.path.join(b.dir, "base.json"), None, 0.0)

    pending = [i for i in range(len(population)) if not all(seed_ok(b, i) for b in benches)]
    need_base = not all(base_ok(b) for b in benches)
    spawn_children = args.procs_per_gpu > 1 and not args.shard and not args.aggregate_only
    if args.shard:
        pending = shard_pending(pending, args.shard)
        need_base = need_base and args.shard.startswith("0/")
    if args.max_new_seeds is not None:  # per process
        pending = pending[:args.max_new_seeds]
    log.info(f"{len(population) - len(pending)} seeds already done, {len(pending)} to run")

    child_failed = False
    if spawn_children and (pending or need_base):
        t_start = time.perf_counter()
        codes = run_children(args)
        child_failed = any(codes)
        wall = time.perf_counter() - t_start
        log.info(f"all {args.procs_per_gpu} processes finished (exit codes {codes}) in {wall:.0f}s wall")
        log_throughput(benches[0].dir, pending)
        if any(codes):
            log.warning("some processes failed; the summary covers the finished seeds, rerun to continue")
    elif not args.aggregate_only and (pending or need_base):
        from transformers import AutoTokenizer
        from vllm import SamplingParams

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        prompts = [tokenizer.apply_chat_template(d["messages"], add_generation_prompt=True, tokenize=False)
                   for b in benches for d in b.datas]
        sampling_params = SamplingParams(temperature=0.0, seed=args.global_seed, max_tokens=args.max_tokens)

        t_start = time.perf_counter()
        engine = make_engine(args)
        log.info(f"engine launched in {engine.launch_s:.1f}s")
        try:
            # The base run also warms the engine up, so it is not charged to seed 0.
            if need_base:
                results = run_one(engine, benches, prompts, sampling_params, None, 0.0)
                for b in benches:
                    res = results[b.name]
                    res.update(run_meta(args, b.name))
                    if not base_ok(b):  # keep logs that are already valid
                        write_json(os.path.join(b.dir, "base.json"), res)
                    log.info(f"base {b.name}: train_reward={res['train_reward']:.4f} "
                             f"test_acc={res['test_accuracy']:.4f} time={res['timing']['total_s']:.1f}s")

            loop_start = time.perf_counter()
            for n_done, i in enumerate(pending):
                seed, sigma = population[i]
                results = run_one(engine, benches, prompts, sampling_params, seed, sigma)
                for b in benches:
                    if seed_ok(b, i):  # keep logs that are already valid
                        continue
                    res = results[b.name]
                    res["index"] = i
                    res.update(run_meta(args, b.name))
                    write_json(seed_path(b.dir, i), res)
                elapsed = time.perf_counter() - loop_start
                eta = elapsed / (n_done + 1) * (len(pending) - n_done - 1)
                t = results[benches[0].name]["timing"]
                scores = " | ".join(f"{b.name} train={results[b.name]['train_reward']:.3f} "
                                    f"test={results[b.name]['test_accuracy']:.3f}" for b in benches)
                log.info(f"[{i + 1}/{len(population)}] seed={seed} sigma={sigma} {scores} "
                         f"time={t['total_s']:.1f}s (perturb {t['perturb_s']:.1f} / gen {t['generate_s']:.1f}) "
                         f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m")
        finally:
            engine.close()
        wall = time.perf_counter() - t_start
        log.info(f"this invocation: {wall:.0f}s wall ({wall / 3600:.2f}h) incl. engine launch")
    else:
        wall = None

    if args.shard:  # child process: the parent writes the summary
        return

    for b in benches:
        summary = aggregate(b.handler, args, b.datas, b.dir, b.n_train)
        summary["dataset"] = b.name
        summary["this_invocation_wall_s"] = wall
        write_json(os.path.join(b.dir, "summary.json"), summary)
        print_summary(summary, b.name)
    if child_failed:  # a supervisor must see the failure; a rerun skips the finished seeds
        sys.exit("some worker processes failed")


if __name__ == "__main__":
    main(parse_args())
