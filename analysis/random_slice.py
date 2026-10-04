#!/usr/bin/env python3
"""
2D random-direction slice of the weight space (heatmaps like Fig. 2 of Neural Thickets).

The plane through the base weights is spanned by two noise directions eps(seed A) and eps(seed B),
the very same directions evaluate.py perturbs with (independent seeds are almost orthogonal). Every
grid point (a, b) is the model  theta + a*eps(A) + b*eps(B)  and is evaluated on the benchmark.
Axes are in sigma units: a RandOpt perturbation with scale sigma is at distance sigma from the
origin, so the dashed circles at sigma = 0.001 / 0.002 / 0.003 mark where the perturbations live.
Colour = relative accuracy change vs. the base model (red: better, blue: worse); stars = best grid
points. Choosing A and B as the best seeds for two subjects (--experts Geometry,Algebra) shows in
which direction of the plane the weights should move for each of them.

Seeds A, B come from the logs of evaluate.py (--logs). They are picked on the train problems and the
heatmaps are drawn on the test problems (--pick_on / --color_on), so the "best" seeds are not chosen
on the same problems they are judged on.

Steps (evaluation needs a GPU, plotting only numpy + matplotlib and can run anywhere):
  python analysis/random_slice.py --experts Geometry,Algebra --out_dir logs/slice_geo_alg
  python analysis/random_slice.py --plot_only --out_dir logs/slice_geo_alg
Several processes on one GPU: --shard 0/3 --shard 1/3 --shard 2/3 (same command, memory split by 3).
The grid point (sigma_A, 0) is the exact model of seed A, so it must reproduce that seed's log.
"""

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def make_axis(extent: float, step: float):
    n = int(round(extent / step))
    return [round(i * step, 10) for i in range(-n, n + 1)]


def point_path(out_dir, ia, ib):
    return os.path.join(out_dir, "points", f"p_{ia:03d}_{ib:03d}.json")


def grid_order(axis):
    """All (ia, ib), the origin first, then by distance from it."""
    n = len(axis)
    pts = [(ia, ib) for ia in range(n) for ib in range(n)]
    return sorted(pts, key=lambda p: (axis[p[0]] ** 2 + axis[p[1]] ** 2, p))


def pick_seeds(logs, experts, seeds, pick_on):
    """Return [(seed, sigma, label), (seed, sigma, label)] from the evaluate.py logs."""
    files = sorted(glob.glob(os.path.join(logs, "seeds", "seed_*.json")))
    if not files:
        sys.exit(f"no seed logs in {logs}/seeds")
    problems = read_json(os.path.join(logs, "problems.json"))
    subject = np.array([p["subject"] for p in problems])
    split = np.array([p["split"] for p in problems])
    in_pick = np.ones(len(problems), bool) if pick_on == "all" else split == pick_on
    rows = []  # (seed, sigma, correct vector, train_reward)
    for f in files:
        d = read_json(f)
        rows.append((d["seed"], d["sigma"], np.array([r["correct"] for r in d["records"]], float), d["train_reward"]))
    by_seed = {r[0]: r for r in rows}

    if seeds:
        chosen = []
        for s in seeds:
            if s not in by_seed:
                sys.exit(f"seed {s} not found in {logs}")
            chosen.append((s, by_seed[s][1], f"seed {s}"))
        return chosen

    chosen, used = [], set()
    for label in experts or [None, None]:
        if label is None:
            order = sorted(rows, key=lambda r: -r[3])           # best by overall train reward
            tag = "best overall"
        else:
            mask = in_pick & (subject == label)
            if not mask.any():
                sys.exit(f"no problems of subject {label!r} in the {pick_on} split; "
                         f"subjects: {sorted(set(subject.tolist()))}")
            order = sorted(rows, key=lambda r: (-r[2][mask].mean(), -r[3]))
            tag = f"best on {label}"
        best = next(r for r in order if r[0] not in used)
        used.add(best[0])
        chosen.append((best[0], best[1], f"{tag} (seed {best[0]})"))
    return chosen


# -----------------------------------------------------------------------------
# evaluation (GPU)
# -----------------------------------------------------------------------------

def evaluate_grid(args):
    import types
    import evaluate as ev
    from data_handlers import get_dataset_handler
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    os.makedirs(os.path.join(args.out_dir, "points"), exist_ok=True)
    axis = make_axis(args.extent, args.step)
    cfg_path = os.path.join(args.out_dir, "config.json")
    if os.path.exists(cfg_path):
        cfg = read_json(cfg_path)
        same = (cfg["axis"] == axis and cfg["model"] == args.model_name and cfg["max_tokens"] == args.max_tokens)
        if not same:
            sys.exit(f"{args.out_dir} was created with other settings; use another --out_dir")
        chosen = [tuple(c) for c in cfg["directions"]]
    else:
        chosen = pick_seeds(args.logs, args.experts.split(",") if args.experts else None,
                            [int(s) for s in args.seeds.split(",")] if args.seeds else None, args.pick_on)
        write_json(cfg_path, {"axis": axis, "model": args.model_name, "max_tokens": args.max_tokens,
                              "directions": chosen, "logs": args.logs, "train_samples": args.train_samples,
                              "sigmas": [float(s) for s in args.sigma_values.split(",")]})
    seeds = [c[0] for c in chosen]
    print(f"plane spanned by eps(seed {seeds[0]}) [{chosen[0][2]}] and eps(seed {seeds[1]}) [{chosen[1][2]}]")

    shard_j, shard_k = (map(int, args.shard.split("/")) if args.shard else (0, 1))
    todo = [p for i, p in enumerate(grid_order(axis))
            if i % shard_k == shard_j and not os.path.exists(point_path(args.out_dir, *p))]
    print(f"{len(axis)}x{len(axis)} grid, {len(todo)} points to run in this process "
          f"(~{len(todo) * args.est_point_s / 60:.0f} min at {args.est_point_s:.0f}s per point)")
    if not todo:
        return

    handler = get_dataset_handler("math500")
    datas, _ = ev.load_problems(handler, args.data_path, args.data_path, args.train_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    prompts = [tokenizer.apply_chat_template(d["messages"], add_generation_prompt=True, tokenize=False)
               for d in datas]
    sampling_params = SamplingParams(temperature=0.0, seed=42, max_tokens=args.max_tokens)

    class SliceEngine(ev.DirectEngine):
        def set_point(self, seeds, coeffs):
            self.llm.collective_rpc("apply_linear_combination", args=(seeds, coeffs))

    engine = SliceEngine(types.SimpleNamespace(
        cuda_devices=args.cuda_devices, model_name=args.model_name, precision="bfloat16",
        max_num_seqs=args.max_num_seqs, cuda_graphs=args.cuda_graphs, prefix_caching=False,
        gpu_memory_utilization=args.gpu_memory_utilization, procs_per_gpu=shard_k,
        tp=args.tp, base_on_cpu=args.base_on_cpu, max_model_len=None))
    try:
        for n, (ia, ib) in enumerate(todo, 1):
            a, b = axis[ia], axis[ib]
            t0 = time.perf_counter()
            engine.set_point(seeds, [a, b])
            outputs = engine.generate(prompts, sampling_params)
            records = ev.build_records(handler, outputs, datas)
            write_json(point_path(args.out_dir, ia, ib), {
                "ia": ia, "ib": ib, "a": a, "b": b, "seeds": seeds,
                "correct": [int(r["correct"]) for r in records],
                "truncated": sum(r["finish_reason"] == "length" for r in records),
                "time_s": time.perf_counter() - t0, "finished_at": time.time()})
            acc = float(np.mean([r["correct"] for r in records]))
            print(f"[{n}/{len(todo)}] a={a:+.4f} b={b:+.4f} acc={acc:.3f} {time.perf_counter() - t0:.0f}s", flush=True)
    finally:
        engine.close()


# -----------------------------------------------------------------------------
# plotting (CPU)
# -----------------------------------------------------------------------------

def load_grid(out_dir):
    cfg = read_json(os.path.join(out_dir, "config.json"))
    n = len(cfg["axis"])
    n_prob = None
    pts = {}
    for f in glob.glob(os.path.join(out_dir, "points", "p_*.json")):
        d = read_json(f)
        pts[(d["ia"], d["ib"])] = np.array(d["correct"], float)
        n_prob = len(d["correct"])
    if not pts:
        sys.exit("no evaluated points found")
    grid = np.full((n, n, n_prob), np.nan)
    for (ia, ib), c in pts.items():
        grid[ia, ib] = c
    return cfg, grid


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg, grid = load_grid(args.out_dir)
    axis = np.array(cfg["axis"])
    n, h = len(axis), (axis[1] - axis[0]) / 2
    problems = read_json(os.path.join(cfg["logs"], "problems.json"))
    subject = np.array([p["subject"] for p in problems])
    split = np.array([p["split"] for p in problems])
    in_color = np.ones(len(problems), bool) if args.color_on == "all" else split == args.color_on

    panels = [("all subjects", in_color)]
    for s in sorted(set(subject.tolist())):
        m = in_color & (subject == s)
        if m.sum() >= args.min_problems:
            panels.append((s, m))

    mid = n // 2                                   # grid index of the origin
    sigmas = cfg["sigmas"]
    seed_a, seed_b = cfg["directions"][0], cfg["directions"][1]
    cols = min(4, len(panels))
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.3 * rows), squeeze=False)
    summary = []

    for ax, (title, mask) in zip(axes.ravel(), panels):
        acc = np.nanmean(grid[:, :, mask], axis=2)             # [ia, ib]; NaN where not evaluated
        base = acc[mid, mid]
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = (acc - base) / base * 100 if base > 0 else np.full_like(acc, np.nan)
        # white = no change; the red side is stretched to its own maximum so that small gains stay
        # visible next to the strongly degraded far-away points
        finite = rel[np.isfinite(rel)]
        vmin = min(np.percentile(finite, 2), -1e-3) if finite.size else -1.0
        vmax = max(finite.max(), 1e-3) if finite.size else 1.0
        cmap = matplotlib.colormaps["RdBu_r"].copy()
        cmap.set_bad("#eeeeee")
        im = ax.imshow(rel.T, origin="lower", extent=(axis[0] - h, axis[-1] + h, axis[0] - h, axis[-1] + h),
                       cmap=cmap, norm=matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax), interpolation="bilinear" if args.smooth else "nearest")
        for s in sigmas:
            ax.add_patch(plt.Circle((0, 0), s, fill=False, ls="--", lw=0.9, color="k", alpha=0.7))
        ax.plot(0, 0, "k+", ms=9)
        # the two seeds the plane is built from: eps(A) at (sigma_A, 0), eps(B) at (0, sigma_B)
        ax.plot(seed_a[1], 0, "o", mfc="none", mec="k", ms=8)
        ax.annotate("A", (seed_a[1], 0), textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.plot(0, seed_b[1], "o", mfc="none", mec="k", ms=8)
        ax.annotate("B", (0, seed_b[1]), textcoords="offset points", xytext=(6, 4), fontsize=9)
        masked = rel.copy()
        masked[mid, mid] = np.nan
        if np.isfinite(masked).any():
            ia, ib = np.unravel_index(np.nanargmax(masked), masked.shape)
            ax.plot(axis[ia], axis[ib], "*", color="gold", mec="k", ms=15, zorder=5)
            best = {"a": float(axis[ia]), "b": float(axis[ib]), "accuracy": float(acc[ia, ib]),
                    "relative_change_pct": float(rel[ia, ib])}
        else:
            best = None
        ax.set_title(f"{title}  (n={int(mask.sum())}, base {base * 100:.1f}%)", fontsize=10)
        ax.set_xlabel("coefficient on eps(A), sigma units")
        ax.set_ylabel("coefficient on eps(B)")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="acc change vs base, %")
        summary.append({"panel": title, "n_problems": int(mask.sum()), "base_accuracy": float(base), "best_point": best})

    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"A: {seed_a[2]} | B: {seed_b[2]} | dashed circles: sigma {', '.join(f'{s:g}' for s in sigmas)}; "
                 f"star: best point; colour on {args.color_on} problems", fontsize=10)
    fig.tight_layout()
    png = os.path.join(args.out_dir, "slice_heatmaps.png")
    fig.savefig(png, dpi=150)
    write_json(os.path.join(args.out_dir, "slice_summary.json"),
               {"directions": cfg["directions"], "color_on": args.color_on, "panels": summary})
    done = int(np.isfinite(grid[:, :, 0]).sum())
    print(f"{png}  ({done}/{n * n} grid points evaluated)")
    for p in summary:
        b = p["best_point"]
        print(f"  {p['panel']:<24} n={p['n_problems']:<4} base={p['base_accuracy'] * 100:5.1f}%  " +
              (f"best at (a={b['a']:+.4f}, b={b['b']:+.4f}) -> {b['accuracy'] * 100:5.1f}% ({b['relative_change_pct']:+.1f}%)"
               if b else "no best point"))


# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--plot_only", action="store_true")
    # plane
    p.add_argument("--logs", default="logs/math500_qwen2.5-3b-instruct_n500", help="evaluate.py log dir")
    p.add_argument("--experts", default=None, help="two subjects, e.g. Geometry,Algebra: A and B = best seed for each")
    p.add_argument("--seeds", default=None, help="two seeds A,B explicitly (overrides --experts)")
    p.add_argument("--pick_on", choices=["train", "test", "all"], default="train")
    p.add_argument("--extent", type=float, default=0.004, help="grid covers [-extent, extent] on both axes")
    p.add_argument("--step", type=float, default=0.0005)
    p.add_argument("--sigma_values", default="0.001,0.002,0.003")
    # evaluation
    p.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--data_path", default="data/math-500/test.jsonl")
    p.add_argument("--train_samples", type=int, default=200)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--cuda_graphs", action="store_true")
    p.add_argument("--max_num_seqs", type=int, default=None)
    p.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    p.add_argument("--base_on_cpu", action="store_true", help="keep the base-weights copy in host RAM")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.75)
    p.add_argument("--cuda_devices", default=None)
    p.add_argument("--shard", default=None, help="j/k: this process evaluates every k-th grid point")
    p.add_argument("--est_point_s", type=float, default=25.0, help="only for the printed time estimate")
    # plotting
    p.add_argument("--color_on", choices=["train", "test", "all"], default="test")
    p.add_argument("--min_problems", type=int, default=15, help="skip subjects with fewer problems in the coloured split")
    p.add_argument("--smooth", action="store_true", help="bilinear interpolation instead of blocks")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.chdir(REPO_ROOT)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.plot_only:
        plot(args)
    else:
        evaluate_grid(args)
        if not args.shard:
            try:
                plot(args)
            except ImportError:
                print("matplotlib is not installed here; run with --plot_only where it is (the points are small)")
