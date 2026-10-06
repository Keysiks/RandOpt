#!/usr/bin/env python3
"""
2D random-direction slice of the weight space (heatmaps like Fig. 2 of Neural Thickets).

The plane through the base weights is spanned by two noise directions eps(seed A) and eps(seed B),
the very same directions evaluate.py perturbs with (independent seeds are almost orthogonal), by default
two random seeds taken from the logs ("random projection"). Every grid point (a, b) is the model
theta + a*eps(A) + b*eps(B); it is evaluated on the benchmark(s) in ONE generate call, so several
datasets (--dataset math500,gsm8k) share the plane and cost one pass. Axes are in sigma units: a RandOpt
perturbation with scale sigma sits at distance sigma from the origin, so the dashed circles at
sigma = 0.001 / 0.002 / 0.003 mark where the perturbations live. Colour = relative accuracy change vs. the
base model (red: better, blue: worse); stars = best grid points. One figure per dataset.

Other ways to choose the plane: --seeds A,B explicitly, --best (two best seeds by train reward),
--experts Geometry,Algebra (best seed per MATH-500 subject, picked on train, drawn on test).
The grid point (sigma_A, 0) is the exact model of seed A, so it reproduces that seed's log.

Evaluation needs a GPU; plotting only numpy + matplotlib and can run anywhere:
  python analysis/random_slice.py --dataset math500,gsm8k --logs logs/qwen2.5-32b_math500_gsm8k \
      --model_name Qwen/Qwen2.5-32B-Instruct --tp 2 --base_on_cpu --out_dir logs/slice_32b
  python analysis/random_slice.py --plot_only --out_dir logs/slice_32b
Several processes on one GPU (small models): --shard 0/3 --shard 1/3 --shard 2/3.
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


def bench_log_dir(logs, name):
    """evaluate.py logs: <logs>/<name>/ when several datasets were run together, else <logs> itself."""
    d = os.path.join(logs, name)
    return d if os.path.isdir(os.path.join(d, "seeds")) else logs


def choose_directions(args, first_dataset):
    """[(seed, sigma, label), (seed, sigma, label)] taken from the evaluate.py logs."""
    log_dir = bench_log_dir(args.logs, first_dataset)
    files = sorted(glob.glob(os.path.join(log_dir, "seeds", "seed_*.json")))
    if not files:
        sys.exit(f"no seed logs in {log_dir}/seeds")

    if args.seeds:                                    # explicit seeds: look their sigma up
        wanted = [int(s) for s in args.seeds.split(",")]
        found = {}
        for f in files:
            d = read_json(f)
            if d["seed"] in wanted:
                found[d["seed"]] = d["sigma"]
            if len(found) == len(wanted):
                break
        missing = [s for s in wanted if s not in found]
        if missing:
            sys.exit(f"seeds {missing} not found in {log_dir}")
        return [(s, found[s], f"seed {s}") for s in wanted]

    if not args.experts and not args.best:            # random plane: two random logged seeds
        rng = np.random.default_rng(args.plane_seed)
        chosen = []
        for i in rng.choice(len(files), size=2, replace=False):
            d = read_json(files[i])
            chosen.append((d["seed"], d["sigma"], f"random seed {d['seed']}"))
        return chosen

    rows = []                                         # (seed, sigma, correct vector, train_reward)
    for f in files:
        d = read_json(f)
        rows.append((d["seed"], d["sigma"], np.array([r["correct"] for r in d["records"]], float), d["train_reward"]))
    problems = read_json(os.path.join(log_dir, "problems.json"))
    subject = np.array([p["subject"] for p in problems])
    split = np.array([p["split"] for p in problems])
    in_pick = np.ones(len(problems), bool) if args.pick_on == "all" else split == args.pick_on

    chosen, used = [], set()
    for label in (args.experts.split(",") if args.experts else [None, None]):
        if label is None:
            order = sorted(rows, key=lambda r: -r[3])
            tag = "best overall"
        else:
            mask = in_pick & (subject == label)
            if not mask.any():
                sys.exit(f"no problems of subject {label!r} in the {args.pick_on} split; "
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
    names = [n.strip() for n in args.dataset.split(",") if n.strip()]
    axis = make_axis(args.extent, args.step)
    sigmas = [float(s) for s in args.sigma_values.split(",")]
    cfg_path = os.path.join(args.out_dir, "config.json")
    settings = {"axis": axis, "model": args.model_name, "max_tokens": args.max_tokens,
                "datasets": names, "eval_on": args.eval_on, "train_samples": args.train_samples,
                "test_samples": args.test_samples}
    if os.path.exists(cfg_path):
        cfg = read_json(cfg_path)
        if any(cfg.get(k) != v for k, v in settings.items()):
            sys.exit(f"{args.out_dir} was created with other settings; use another --out_dir")
        chosen = [tuple(c) for c in cfg["directions"]]
    else:
        chosen = choose_directions(args, names[0])
        write_json(cfg_path, {**settings, "directions": chosen, "logs": args.logs, "sigmas": sigmas})
    seeds = [c[0] for c in chosen]
    print(f"plane spanned by eps(seed {seeds[0]}) [{chosen[0][2]}] and eps(seed {seeds[1]}) [{chosen[1][2]}]")

    # problems of every dataset; only the test problems unless --eval_on all
    handlers, bench_data = {}, {}
    for name in names:
        handler = get_dataset_handler(name)
        datas, splits = ev.load_problems(handler, handler.default_train_path, handler.default_test_path,
                                         args.train_samples, args.test_samples)
        keep = [i for i, s in enumerate(splits) if args.eval_on == "all" or s == "test"]
        handlers[name] = handler
        bench_data[name] = [datas[i] for i in keep]
        info_path = os.path.join(args.out_dir, f"problems_{name}.json")
        if not os.path.exists(info_path):
            write_json(info_path, [{"idx": j, "split": splits[i], "subject": datas[i].get("subject", ""),
                                    "level": datas[i].get("level", "")} for j, i in enumerate(keep)])
        print(f"{name}: {len(bench_data[name])} problems evaluated at every grid point")

    shard_j, shard_k = (map(int, args.shard.split("/")) if args.shard else (0, 1))
    todo = [p for i, p in enumerate(grid_order(axis))
            if i % shard_k == shard_j and not os.path.exists(point_path(args.out_dir, *p))]
    print(f"{len(axis)}x{len(axis)} grid, {len(todo)} points to run in this process "
          f"(~{len(todo) * args.est_point_s / 60:.0f} min at {args.est_point_s:.0f}s per point)")
    if not todo:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    prompts = [tokenizer.apply_chat_template(d["messages"], add_generation_prompt=True, tokenize=False)
               for name in names for d in bench_data[name]]
    sampling_params = SamplingParams(temperature=0.0, seed=42, max_tokens=args.max_tokens)

    class SliceEngine(ev.DirectEngine):
        def set_point(self, seeds, coeffs):
            self.llm.collective_rpc("apply_linear_combination", args=(seeds, coeffs))

    engine = SliceEngine(types.SimpleNamespace(
        cuda_devices=args.cuda_devices, model_name=args.model_name, precision="bfloat16",
        max_num_seqs=args.max_num_seqs, cuda_graphs=args.cuda_graphs, prefix_caching=False,
        gpu_memory_utilization=args.gpu_memory_utilization, procs_per_gpu=shard_k,
        tp=args.tp, base_on_cpu=args.base_on_cpu, max_model_len=args.max_model_len))
    try:
        for n, (ia, ib) in enumerate(todo, 1):
            a, b = axis[ia], axis[ib]
            t0 = time.perf_counter()
            engine.set_point(seeds, [a, b])
            outputs = engine.generate(prompts, sampling_params)
            correct, truncated, start, accs = {}, {}, 0, []
            for name in names:
                recs = ev.build_records(handlers[name], outputs[start:start + len(bench_data[name])], bench_data[name])
                start += len(bench_data[name])
                correct[name] = [int(r["correct"]) for r in recs]
                truncated[name] = sum(r["finish_reason"] == "length" for r in recs)
                accs.append(f"{name}={np.mean(correct[name]):.3f}")
            write_json(point_path(args.out_dir, ia, ib), {
                "ia": ia, "ib": ib, "a": a, "b": b, "seeds": seeds, "correct": correct, "truncated": truncated,
                "time_s": time.perf_counter() - t0, "finished_at": time.time()})
            print(f"[{n}/{len(todo)}] a={a:+.4f} b={b:+.4f} acc: {' '.join(accs)} {time.perf_counter() - t0:.0f}s",
                  flush=True)
    finally:
        engine.close()


# -----------------------------------------------------------------------------
# plotting (CPU)
# -----------------------------------------------------------------------------

def load_grid(out_dir):
    cfg = read_json(os.path.join(out_dir, "config.json"))
    n = len(cfg["axis"])
    pts = {}
    for f in glob.glob(os.path.join(out_dir, "points", "p_*.json")):
        d = read_json(f)
        pts[(d["ia"], d["ib"])] = d["correct"]
    if not pts:
        sys.exit("no evaluated points found")
    grids = {}
    for name in cfg["datasets"]:
        n_prob = len(next(iter(pts.values()))[name])
        g = np.full((n, n, n_prob), np.nan)
        for (ia, ib), c in pts.items():
            g[ia, ib] = np.array(c[name], float)
        grids[name] = g
    return cfg, grids


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg, grids = load_grid(args.out_dir)
    axis = np.array(cfg["axis"])
    n, h = len(axis), (axis[1] - axis[0]) / 2
    mid = n // 2                                   # grid index of the origin
    sigmas = cfg["sigmas"]
    seed_a, seed_b = cfg["directions"][0], cfg["directions"][1]
    summary = {"directions": cfg["directions"], "color_on": args.color_on, "datasets": {}}

    for name in cfg["datasets"]:
        grid = grids[name]
        problems = read_json(os.path.join(args.out_dir, f"problems_{name}.json"))
        subject = np.array([p["subject"] for p in problems])
        split = np.array([p["split"] for p in problems])
        in_color = np.ones(len(problems), bool) if args.color_on == "all" else split == args.color_on
        panels = [(f"{name}", in_color)]
        if args.by_subject:
            for s in sorted(set(subject.tolist()) - {""}):
                m = in_color & (subject == s)
                if m.sum() >= args.min_problems:
                    panels.append((f"{name}: {s}", m))

        cols = min(3, len(panels))
        rows = math.ceil(len(panels) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 5.0 * rows), squeeze=False)
        out_panels = []
        for ax, (title, mask) in zip(axes.ravel(), panels):
            acc = np.nanmean(grid[:, :, mask], axis=2)          # [ia, ib]; NaN where not evaluated
            base = acc[mid, mid]
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = (acc - base) / base * 100 if base > 0 else np.full_like(acc, np.nan)
            # white = no change; the red side is stretched to its own maximum so that small gains stay
            # visible next to strongly degraded far-away points
            finite = rel[np.isfinite(rel)]
            vmin = min(np.percentile(finite, 2), -1e-3) if finite.size else -1.0
            vmax = max(finite.max(), 1e-3) if finite.size else 1.0
            cmap = matplotlib.colormaps["RdBu_r"].copy()
            cmap.set_bad("#eeeeee")
            im = ax.imshow(rel.T, origin="lower",
                           extent=(axis[0] - h, axis[-1] + h, axis[0] - h, axis[-1] + h), cmap=cmap,
                           norm=matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax),
                           interpolation="nearest" if args.no_smooth else "bilinear")
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
            best = None
            if np.isfinite(masked).any():
                ia, ib = np.unravel_index(np.nanargmax(masked), masked.shape)
                ax.plot(axis[ia], axis[ib], "*", color="gold", mec="k", ms=15, zorder=5)
                best = {"a": float(axis[ia]), "b": float(axis[ib]), "accuracy": float(acc[ia, ib]),
                        "relative_change_pct": float(rel[ia, ib])}
            ax.set_title(f"{title}  (n={int(mask.sum())}, base {base * 100:.1f}%)", fontsize=10)
            ax.set_xlabel("coefficient on eps(A), sigma units")
            ax.set_ylabel("coefficient on eps(B)")
            ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="acc change vs base, %")
            out_panels.append({"panel": title, "n_problems": int(mask.sum()), "base_accuracy": float(base),
                               "best_point": best})
        for ax in axes.ravel()[len(panels):]:
            ax.axis("off")
        fig.suptitle(f"{cfg['model']}  |  {args.color_on} problems\nA: {seed_a[2]}   B: {seed_b[2]}\n"
                     f"dashed circles: sigma {', '.join(f'{s:g}' for s in sigmas)}   star: best point", fontsize=9)
        fig.tight_layout()
        png = os.path.join(args.out_dir, f"slice_{name}.png")
        fig.savefig(png, dpi=150)
        plt.close(fig)
        done = int(np.isfinite(grid[:, :, 0]).sum())
        print(f"{png}  ({done}/{n * n} grid points evaluated)")
        for p in out_panels:
            b = p["best_point"]
            print(f"  {p['panel']:<28} n={p['n_problems']:<4} base={p['base_accuracy'] * 100:5.1f}%  " +
                  (f"best at (a={b['a']:+.4f}, b={b['b']:+.4f}) -> {b['accuracy'] * 100:5.1f}% "
                   f"({b['relative_change_pct']:+.1f}%)" if b else "no best point"))
        summary["datasets"][name] = out_panels
    write_json(os.path.join(args.out_dir, "slice_summary.json"), summary)


# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--plot_only", action="store_true")
    # plane
    p.add_argument("--logs", default="logs/math500_qwen2.5-3b-instruct_n500", help="evaluate.py log dir (source of seeds A, B)")
    p.add_argument("--seeds", default=None, help="two seeds A,B explicitly")
    p.add_argument("--best", action="store_true", help="A and B = the two best seeds by train reward")
    p.add_argument("--experts", default=None, help="two MATH-500 subjects, e.g. Geometry,Algebra: A and B = best seed for each")
    p.add_argument("--pick_on", choices=["train", "test", "all"], default="train")
    p.add_argument("--plane_seed", type=int, default=0, help="which random pair of logged seeds spans the plane")
    p.add_argument("--extent", type=float, default=0.004, help="grid covers [-extent, extent] on both axes")
    p.add_argument("--step", type=float, default=0.001)
    p.add_argument("--sigma_values", default="0.001,0.002,0.003", help="circles to draw")
    # evaluation
    p.add_argument("--dataset", default="math500", help="one or several datasets, e.g. math500,gsm8k")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--train_samples", type=int, default=200)
    p.add_argument("--test_samples", type=int, default=None, help="cap on test problems (e.g. 500 for GSM8K)")
    p.add_argument("--eval_on", choices=["test", "all"], default="test", help="problems evaluated at every grid point")
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--cuda_graphs", action="store_true")
    p.add_argument("--max_num_seqs", type=int, default=None)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    p.add_argument("--base_on_cpu", action="store_true", help="keep the base-weights copy in host RAM")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.75)
    p.add_argument("--cuda_devices", default=None)
    p.add_argument("--shard", default=None, help="j/k: this process evaluates every k-th grid point")
    p.add_argument("--est_point_s", type=float, default=100.0, help="only for the printed time estimate")
    # plotting
    p.add_argument("--color_on", choices=["train", "test", "all"], default="test")
    p.add_argument("--by_subject", action="store_true", help="extra panels per subject (MATH-500)")
    p.add_argument("--min_problems", type=int, default=15, help="skip subjects with fewer problems")
    p.add_argument("--no_smooth", action="store_true", help="blocks instead of bilinear interpolation")
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
