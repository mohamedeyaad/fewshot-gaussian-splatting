"""Plot the sweep from runs/*/results.json.

Three figures:

  curves_paired.png   - THE headline. One panel per strategy, x = synthetic
                        ratio, y = paired delta vs the same seed's own
                        baseline, one line per subset size. The crossover from
                        positive to negative as k grows is the main finding.
  scaling.png         - PSNR vs number of real views, with the best and worst
                        augmentation effects drawn at the same scale so the
                        size of the two effects can be compared directly.
  curves_absolute.png - raw PSNR per subset size, floor and ceiling marked.

NOTE ON A PAST BUG: this script previously hardcoded k=10 when computing
ratios and when looking up the paired baseline. That was correct while k=10
was the only sweep, but silently merged all three subset sizes once k=5 and
k=20 existed. Every k is now carried through explicitly.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scene_key import scene_of

ROOT = Path(os.path.expanduser("~/fewshot_gs"))
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

K_COLORS = {5: "#c2453d", 10: "#e08a1e", 20: "#3b7dd8"}
STRATEGIES = ("inpaint", "outpaint", "guided")
KS = (5, 10, 20)


def agg(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, 0.0
    return mean(v), (stdev(v) if len(v) > 1 else 0.0)


def load(scene="truck"):
    # One scene only: (k, seed) is unique within a scene but not across them,
    # so mixing drjohnson in would pair truck runs against drjohnson floors.
    base, runs, full = {}, defaultdict(list), None
    for p in sorted((ROOT / "runs").glob("*/results.json")):
        r = json.loads(p.read_text())
        if scene_of(r) != scene:
            continue
        pr = r["provenance"]
        if pr.get("method") == "full":
            full = r["metrics"]
            continue
        k, seed, nf = pr.get("k"), pr.get("seed"), pr.get("n_synthetic", 0)
        if k not in KS:
            continue
        if nf == 0:
            base[(k, seed)] = r["metrics"]
        else:
            runs[(k, pr.get("strategy"), nf)].append((seed, r["metrics"]))
    return base, runs, full


def paired_series(base, runs, k, strat, key="psnr"):
    """(ratios, mean deltas, stds) for one subset size and strategy."""
    pts = sorted({nf for (kk, s, nf) in runs if kk == k and s == strat})
    xs, ys, es = [0.0], [0.0], [0.0]
    for nf in pts:
        ds = [m[key]["mean"] - base[(k, seed)][key]["mean"]
              for seed, m in runs[(k, strat, nf)] if (k, seed) in base]
        mm, ss = agg(ds)
        if mm is None:
            continue
        xs.append(100.0 * nf / k)
        ys.append(mm)
        es.append(ss)
    return xs, ys, es


def fig_paired(base, runs):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), sharey=True)
    for ax, strat in zip(axes, STRATEGIES):
        ax.axhline(0, color="#444", lw=1.3)
        # Noise floor measured over 3 identical repeats: sigma = 0.039 dB,
        # so a paired difference of two runs has sigma ~ 0.039*sqrt(2).
        ax.axhspan(-0.055, 0.055, color="#999", alpha=0.18, zorder=0)
        for k in KS:
            xs, ys, es = paired_series(base, runs, k, strat)
            if len(xs) < 2:
                continue
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, lw=1.9,
                        color=K_COLORS[k], label=f"k={k} real views")
        ax.set_title(strat, fontsize=11)
        ax.set_xlabel("synthetic : real ratio (%)")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Δ PSNR vs same-seed baseline (dB)")
    fig.suptitle("Augmentation helps only when real views are scarce  "
                 "(grey band = measured noise floor, ±0.055 dB)")
    fig.tight_layout()
    fig.savefig(OUT / "curves_paired.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT/'curves_paired.png'}")


def fig_scaling(base, runs, full):
    """Real views buy far more than synthetic ones. Same axis, same units."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    xs, ys, es = [], [], []
    for k in KS:
        mm, ss = agg([m["psnr"]["mean"] for (kk, _), m in base.items() if kk == k])
        xs.append(k); ys.append(mm); es.append(ss)
    ax.errorbar(xs, ys, yerr=es, marker="o", lw=2, capsize=4, color="#2a6f97",
                label="real photographs")
    if full:
        ax.plot([219], [full["psnr"]["mean"]], marker="*", ms=15,
                color="#2a9d4a", label="full data (219 views)")
        ax.axhline(full["psnr"]["mean"], color="#2a9d4a", ls=":", lw=1.3)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(6, -12), fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 219])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("number of real training views")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("What real photographs buy", fontsize=11)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8)

    # Same vertical scale, so the comparison is honest rather than rhetorical.
    gains = [("+5 real\nviews (5→10)", ys[1] - ys[0], "#2a6f97"),
             ("+10 real\nviews (10→20)", ys[2] - ys[1], "#2a6f97")]
    best = max((agg([m["psnr"]["mean"] - base[(k, s)]["psnr"]["mean"]
                     for s, m in v if (k, s) in base])[0], k, st, nf)
               for (k, st, nf), v in runs.items())
    gains.append((f"best synthetic\n({best[2]}, k={best[1]})", best[0], "#e08a1e"))
    worst = min((agg([m["psnr"]["mean"] - base[(k, s)]["psnr"]["mean"]
                      for s, m in v if (k, s) in base])[0], k, st, nf)
                for (k, st, nf), v in runs.items())
    gains.append((f"worst synthetic\n({worst[2]}, k={worst[1]})", worst[0], "#c2453d"))

    ax2.bar([g[0] for g in gains], [g[1] for g in gains],
            color=[g[2] for g in gains])
    ax2.axhline(0, color="#444", lw=1.2)
    for i, g in enumerate(gains):
        ax2.annotate(f"{g[1]:+.2f}", (i, g[1]), ha="center", fontsize=9,
                     textcoords="offset points",
                     xytext=(0, 4 if g[1] >= 0 else -13))
    ax2.set_ylabel("Δ PSNR (dB)")
    ax2.set_title("Real vs synthetic, same scale", fontsize=11)
    ax2.grid(alpha=0.25, lw=0.5, axis="y")
    ax2.tick_params(axis="x", labelsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "scaling.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT/'scaling.png'}")


def fig_absolute(base, runs, full):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), sharey=True)
    for ax, k in zip(axes, KS):
        bm, bs = agg([m["psnr"]["mean"] for (kk, _), m in base.items() if kk == k])
        ax.axhline(bm, color="#888", ls="--", lw=1.4, label=f"{k}-view floor ({bm:.2f})")
        ax.axhspan(bm - bs, bm + bs, color="#888", alpha=0.15)
        if full:
            ax.axhline(full["psnr"]["mean"], color="#2a9d4a", ls=":", lw=1.5,
                       label=f"ceiling ({full['psnr']['mean']:.2f})")
        for strat in STRATEGIES:
            pts = sorted({nf for (kk, s, nf) in runs if kk == k and s == strat})
            xs, ys, es = [0.0], [bm], [bs]
            for nf in pts:
                mm, ss = agg([m["psnr"]["mean"] for _, m in runs[(k, strat, nf)]])
                xs.append(100.0 * nf / k); ys.append(mm); es.append(ss)
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, lw=1.7,
                        label=strat)
        ax.set_title(f"k = {k} real views", fontsize=11)
        ax.set_xlabel("synthetic : real ratio (%)")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("PSNR (dB)")
    fig.suptitle("Absolute held-out quality per subset size (truck, 3 seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "curves_absolute.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT/'curves_absolute.png'}")


def main():
    base, runs, full = load()
    if not base:
        print("no results")
        return
    fig_paired(base, runs)
    fig_scaling(base, runs, full)
    fig_absolute(base, runs, full)


if __name__ == "__main__":
    main()
