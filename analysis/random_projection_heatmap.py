#!/usr/bin/env python3
"""
Fast Fig.-2-style heatmaps from existing evaluate.py logs: random 2D projection of the perturbed models,
colour = relative accuracy change vs. the base model. No GPU, no new inference (seconds for 500 seeds).

Every seed is a perturbation sigma * eps(seed). Projecting it on two fixed random unit directions gives
the 2D position (sigma*g1, sigma*g2) with g ~ N(0, I): the projection of an isotropic Gaussian on any
fixed direction is N(0, 1), independent of everything else, so instead of the 3e10-dimensional dot
products the two numbers are drawn directly from a generator seeded by the seed (same distribution,
deterministic, --proj_seed changes the projection). Consequence: the position of a point carries no
information about its accuracy, the information is in the colours (how many perturbations are better
than the base model and by how much). The smooth map is a kernel average of the points.

Dashed circles: mean projected distance of a perturbation of each sigma (sigma * sqrt(pi/2)).
Stars: the best-performing perturbations. One PNG per dataset:
  python analysis/random_projection_heatmap.py --logs logs/qwen2.5-32b_math500_gsm8k --datasets math500,gsm8k
Only the first KB of each seed log is read, so it stays fast with 1 MB logs.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np


def read_head(path, keys, nbytes=4096):
    """Scalar keys that evaluate.py writes before the (huge) 'records' list; falls back to full json."""
    with open(path, "rb") as f:
        head = f.read(nbytes).decode("utf-8", "ignore")
    out = {}
    for k in keys:
        m = re.search(r'"%s":\s*(null|-?[0-9.]+(?:[eE][-+]?\d+)?)' % k, head)
        if m is None:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            return {k2: d[k2] for k2 in keys}
        out[k] = None if m.group(1) == "null" else float(m.group(1))
    return out


def bench_log_dir(logs, name):
    d = os.path.join(logs, name)
    return d if os.path.isdir(os.path.join(d, "seeds")) else logs


def project(seed, sigma, proj_seed):
    g = np.random.default_rng([proj_seed, int(seed)]).standard_normal(2)
    return sigma * g[0], sigma * g[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", required=True, help="evaluate.py log dir")
    p.add_argument("--datasets", default="math500", help="comma list; each one is read from <logs>/<name>/ or <logs>/")
    p.add_argument("--out_dir", default=None, help="default: the log dir")
    p.add_argument("--metric", default="test_accuracy", choices=["test_accuracy", "train_reward", "full_accuracy"])
    p.add_argument("--proj_seed", type=int, default=0)
    p.add_argument("--bandwidth", type=float, default=0.5, help="kernel width of the smooth map, in units of the median sigma")
    p.add_argument("--min_mass", type=float, default=0.5, help="hide map regions with fewer points than this nearby")
    p.add_argument("--stars", type=int, default=3, help="how many best perturbations to mark")
    p.add_argument("--no_map", action="store_true", help="scatter only, no smooth map")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name in [n.strip() for n in args.datasets.split(",") if n.strip()]:
        d = bench_log_dir(args.logs, name)
        files = sorted(glob.glob(os.path.join(d, "seeds", "seed_*.json")))
        if not files:
            sys.exit(f"no seed logs in {d}/seeds")
        keys = ["seed", "sigma", args.metric]
        rows = [read_head(f, keys) for f in files]
        base = read_head(os.path.join(d, "base.json"), [args.metric])[args.metric]
        model = args.logs
        if os.path.exists(os.path.join(args.logs, "args.json")):
            model = json.load(open(os.path.join(args.logs, "args.json"))).get("model_name", args.logs)

        sig = np.array([r["sigma"] for r in rows])
        acc = np.array([r[args.metric] for r in rows])
        xy = np.array([project(r["seed"], r["sigma"], args.proj_seed) for r in rows])
        rel = (acc - base) / base * 100

        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        lim = max(np.abs(xy).max() * 1.1, 1e-9)
        vmin, vmax = min(np.percentile(rel, 1), -1e-3), max(rel.max(), 1e-3)
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cmap = matplotlib.colormaps["RdBu_r"].copy()
        cmap.set_bad("#f2f2f2")
        if not args.no_map:
            g = np.linspace(-lim, lim, 220)
            gx, gy = np.meshgrid(g, g)
            h = args.bandwidth * np.median(sig)
            w = np.exp(-(((gx[..., None] - xy[:, 0]) ** 2 + (gy[..., None] - xy[:, 1]) ** 2) / (2 * h * h)))
            mass = w.sum(-1)
            field = (w * rel).sum(-1) / np.maximum(mass, 1e-12)
            field[mass < args.min_mass] = np.nan
            ax.imshow(field, origin="lower", extent=(-lim, lim, -lim, lim), cmap=cmap, norm=norm, alpha=0.9)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=rel, cmap=cmap, norm=norm, s=16, edgecolors="k", linewidths=0.3, zorder=3)
        for s in sorted(set(sig.tolist())):
            ax.add_patch(plt.Circle((0, 0), s * np.sqrt(np.pi / 2), fill=False, ls="--", lw=0.9, color="k", alpha=0.7))
            ax.annotate(f"σ={s:g}", (s * np.sqrt(np.pi / 2) * 0.72, s * np.sqrt(np.pi / 2) * 0.72), fontsize=7, alpha=0.8)
        top = np.argsort(-rel)[:args.stars]
        ax.scatter(xy[top, 0], xy[top, 1], marker="*", s=230, c="gold", edgecolors="k", zorder=5)
        ax.plot(0, 0, "k+", ms=10, zorder=4)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xlabel("random projection 1"); ax.set_ylabel("random projection 2")
        better = int((rel > 0).sum())
        ax.set_title(f"{os.path.basename(model)} | {name} | {args.metric}\n{len(rows)} perturbations, base {base * 100:.1f}%, "
                     f"{better} ({better / len(rows) * 100:.0f}%) better than base, best {rel.max():+.1f}%", fontsize=9)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03, label="accuracy change vs base, %")
        fig.tight_layout()
        out_dir = args.out_dir or args.logs
        os.makedirs(out_dir, exist_ok=True)
        png = os.path.join(out_dir, f"heatmap_{name}.png")
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"{png}: {len(rows)} seeds, base {base * 100:.1f}%, better than base: {better} ({better / len(rows) * 100:.0f}%), "
              f"best {rel.max():+.1f}%, median {np.median(rel):+.1f}%")
        for s in sorted(set(sig.tolist())):
            m = sig == s
            print(f"   sigma={s:g}: n={int(m.sum())}, mean change {rel[m].mean():+.1f}%, better than base {int((rel[m] > 0).sum())}")


if __name__ == "__main__":
    main()
