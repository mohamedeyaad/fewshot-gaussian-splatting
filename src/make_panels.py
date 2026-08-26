"""Build the qualitative figures: renders vs ground truth, side by side.

Three figures:
  panel_strategies.png - GT | floor | each strategy at 100% ratio
  panel_ratio_<s>.png  - GT | 0 | 2 | 5 | 10 | 20 fakes, one strategy
  panel_training_data.png - the real source view vs what each strategy fed
                            the optimiser (this is where the hallucinations
                            are actually visible)

Test-view ordering is identical across runs because the held-out split is
frozen, so index i is the same camera everywhere.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(os.path.expanduser("~/fewshot_gs"))
RUNS = ROOT / "runs"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
ITER = 7000
SEED = 0


def render_dir(tag):
    return RUNS / tag / "test" / f"ours_{ITER}"


def load(tag, idx, which="renders"):
    p = render_dir(tag) / which / f"{idx:05d}.png"
    return Image.open(p).convert("RGB") if p.exists() else None


def psnr_of(tag, idx):
    f = RUNS / tag / "results.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    pv = r["metrics"].get("per_view", {})
    ps = pv.get("psnr")
    return ps[idx] if ps and idx < len(ps) else None


def grid(rows, col_titles, path, suptitle, row_labels=None):
    nr, nc = len(rows), len(rows[0])
    fig, axes = plt.subplots(nr, nc, figsize=(2.9 * nc, 1.85 * nr), squeeze=False)
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            ax = axes[i][j]
            ax.axis("off")
            img, sub = cell
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        fontsize=8, color="#b00")
            if i == 0:
                ax.set_title(col_titles[j], fontsize=9, pad=4)
            if sub:
                # Inside the image, not below it - a caption under each axes
                # collides with the next row once there are several rows.
                ax.text(0.015, 0.98, sub, transform=ax.transAxes,
                        ha="left", va="top", fontsize=7.5, color="white",
                        bbox=dict(facecolor="black", alpha=0.55,
                                  edgecolor="none", pad=1.6))
        if row_labels:
            axes[i][0].text(-0.04, 0.5, row_labels[i], transform=axes[i][0].transAxes,
                            rotation=90, va="center", ha="right", fontsize=8)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=135)
    plt.close(fig)
    print(f"wrote {path}")


def representative_views(pairs, n=3, pool=32):
    """Pick views that behave like the average, and say so in the caption.

    Per-view effects scatter far more widely than the mean: at k=20 the 200%
    condition is worth -0.618 dB averaged over the held-out set, yet individual
    cameras range from roughly -3 to +2. Three views chosen arbitrarily can
    therefore contradict the very finding the figure is illustrating, which is
    worse than useless in a report.

    `pairs` is a list of (augmented_tag, baseline_tag). For each candidate view
    we score how far its deltas sit from the mean delta of each pair, and keep
    the views with the smallest total deviation - the least misleading
    illustration of a result that is itself an average.
    """
    means = []
    for aug, base in pairs:
        ds = [psnr_of(aug, i) - psnr_of(base, i) for i in range(pool)
              if psnr_of(aug, i) is not None and psnr_of(base, i) is not None]
        means.append(sum(ds) / len(ds) if ds else 0.0)

    scored = []
    for i in range(pool):
        err = 0.0
        ok = True
        for (aug, base), mu in zip(pairs, means):
            pa, pb = psnr_of(aug, i), psnr_of(base, i)
            if pa is None or pb is None:
                ok = False
                break
            err += abs((pa - pb) - mu)
        if ok:
            scored.append((err, i))
    scored.sort()
    return sorted(i for _, i in scored[:n])


def _cells(cols, views, base):
    """One row per view, one column per condition, delta printed on each cell.

    A bare PSNR invites the reader to rank columns by eye across rows, which
    is exactly how three views can be read against the finding they
    illustrate. The delta against the baseline is the number that matters.
    """
    rows, labels = [], []
    for v in views:
        row = []
        for title, tag in cols:
            if tag is None:
                row.append((load(base, v, "gt"), "reference"))
                continue
            p = psnr_of(tag, v)
            sub = f"PSNR {p:.2f}" if p is not None else ""
            if tag != base and p is not None:
                pb = psnr_of(base, v)
                if pb is not None:
                    sub += f"   {p - pb:+.2f}"
            row.append((load(tag, v), sub))
        rows.append(row)
        # index into the 32 held-out cameras, NOT a photograph number
        labels.append(f"held-out {v + 1}/32")
    return rows, labels


def panel_strategies():
    base = f"truck_k10_seed{SEED}_fps_fake0"
    cols = [("ground truth", None), ("10 real only", base)]
    for s in ("inpaint", "outpaint", "guided"):
        cols.append((s + "  +10 fake", f"truck_k10_seed{SEED}_fps_{s}_fake10"))

    views = representative_views([(c[1], base) for c in cols[2:]])
    rows, labels = _cells(cols, views, base)
    grid(rows, [c[0] for c in cols], OUT / "panel_strategies.png",
         f"Held-out renderings at 100% synthetic ratio (truck, k=10, seed {SEED}). "
         f"Views closest to the mean effect;\nthe number under each cell is "
         f"that view's difference from the 10-real baseline.",
         labels)


def panel_ratio(strategy):
    base = f"truck_k10_seed{SEED}_fps_fake0"
    cols = [("ground truth", None), ("0 fake", base)]
    for n in (2, 5, 10, 20):
        cols.append((f"{n} fake ({n*10}%)",
                     f"truck_k10_seed{SEED}_fps_{strategy}_fake{n}"))
    views = representative_views([(c[1], base) for c in cols[2:]])
    rows, labels = _cells(cols, views, base)
    grid(rows, [c[0] for c in cols], OUT / f"panel_ratio_{strategy}.png",
         f"{strategy}: increasing synthetic ratio (truck, k=10, seed {SEED})",
         labels)


def panel_training_data():
    """What the optimiser was actually shown - where hallucinations live."""
    src = ROOT / "data/tandt/truck/images"
    syn = ROOT / "synthetic"
    rows, labels = [], []
    for stem in ("000044", "000067"):
        row = [(Image.open(src / f"{stem}.jpg").convert("RGB"), "REAL")]
        for s in ("inpaint", "outpaint", "guided"):
            d = syn / f"truck_k10_seed{SEED}_fps_{s}" / "images"
            f = d / f"synth_{s}_{stem}_v00.jpg"
            row.append((Image.open(f).convert("RGB") if f.exists() else None, s))
        rows.append(row)
        labels.append(stem)
    grid(rows, ["real training view", "inpaint", "outpaint", "pose-guided"],
         OUT / "panel_training_data.png",
         "Synthetic training images vs their real source view", labels)


if __name__ == "__main__":
    panel_strategies()
    for s in ("inpaint", "outpaint", "guided"):
        panel_ratio(s)
    panel_training_data()
