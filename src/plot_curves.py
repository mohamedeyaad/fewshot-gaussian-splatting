"""Plot metric-vs-synthetic-ratio curves from runs/*/results.json.

Two figures:
  curves_absolute.png - raw metric values with the few-shot floor and the
                        full-data ceiling drawn in for scale
  curves_paired.png   - paired within-seed deltas, which is the honest view
                        of whether augmentation helped
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
import numpy as np

ROOT = Path(os.path.expanduser("~/fewshot_gs"))
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {"inpaint": "#3b7dd8", "outpaint": "#e08a1e",
          "guided": "#c2453d", "mixed": "#5a9e5a", "none": "#888888"}
METRICS = [("psnr", "PSNR (dB)", True),
           ("ssim", "SSIM", True),
           ("lpips", "LPIPS", False)]   # bool = higher is better


def load():
    recs = []
    for p in sorted((ROOT / "runs").glob("*/results.json")):
        recs.append(json.loads(p.read_text()))
    return recs


def agg(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, 0.0
    return mean(v), (stdev(v) if len(v) > 1 else 0.0)


def main():
    recs = load()
    if not recs:
        print("no results")
        return

    fewshot, full, baselines = defaultdict(list), None, {}
    by_strategy = defaultdict(lambda: defaultdict(list))

    for r in recs:
        p = r["provenance"]
        m = r["metrics"]
        if p.get("method") == "full":
            full = m
            continue
        nf = p.get("n_synthetic", 0)
        k, seed = p.get("k"), p.get("seed")
        if nf == 0:
            baselines[(k, seed)] = m
            fewshot[0].append(m)
        else:
            by_strategy[p.get("strategy")][nf].append((seed, m))

    ratios_of = lambda k, nf: 100.0 * nf / k

    # ---------- absolute ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (key, label, higher_better) in zip(axes, METRICS):
        base_m, base_s = agg([m[key]["mean"] for m in fewshot[0]])
        ax.axhline(base_m, color="#888", ls="--", lw=1.4,
                   label=f"10-photo floor ({base_m:.3f})")
        ax.axhspan(base_m - base_s, base_m + base_s, color="#888", alpha=0.15)
        if full:
            ax.axhline(full[key]["mean"], color="#2a9d4a", ls=":", lw=1.6,
                       label=f"219-photo ceiling ({full[key]['mean']:.3f})")

        for strat, per_n in sorted(by_strategy.items()):
            xs, ys, es = [0], [base_m], [base_s]
            for nf in sorted(per_n):
                vals = [m[key]["mean"] for _, m in per_n[nf]]
                mm, ss = agg(vals)
                xs.append(ratios_of(10, nf)); ys.append(mm); es.append(ss)
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, lw=1.8,
                        color=COLORS.get(strat, "#333"), label=strat)
        ax.set_xlabel("synthetic : real ratio (%)")
        ax.set_ylabel(label)
        ax.set_title(f"{label}  ({'higher' if higher_better else 'lower'} is better)",
                     fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=7)
    fig.suptitle("Held-out view quality vs synthetic data ratio  (truck, k=10, 3 seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "curves_absolute.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT/'curves_absolute.png'}")

    # ---------- paired deltas ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (key, label, higher_better) in zip(axes, METRICS):
        ax.axhline(0, color="#444", lw=1.4)
        for strat, per_n in sorted(by_strategy.items()):
            xs, ys, es = [0], [0.0], [0.0]
            for nf in sorted(per_n):
                ds = []
                for seed, m in per_n[nf]:
                    b = baselines.get((10, seed))
                    if b:
                        ds.append(m[key]["mean"] - b[key]["mean"])
                mm, ss = agg(ds)
                if mm is None:
                    continue
                xs.append(ratios_of(10, nf)); ys.append(mm); es.append(ss)
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, lw=1.8,
                        color=COLORS.get(strat, "#333"), label=strat)
        good = "better ↑" if higher_better else "better ↓"
        ax.set_xlabel("synthetic : real ratio (%)")
        ax.set_ylabel(f"Δ {label}  ({good})")
        ax.set_title(f"Δ{label} vs same-seed baseline", fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=8)
    fig.suptitle("Paired change vs each seed's own 0-synthetic baseline "
                 "(error bars = std across 3 seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "curves_paired.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT/'curves_paired.png'}")


if __name__ == "__main__":
    main()
