"""Schematic figures for the deck: the pipeline, and the experiment grid.

The deck described its own experimental design in prose - four bullet points
about splits, pairing and seeds - and never showed the grid it was describing.
A factorial is a shape, and a shape should be drawn.

Rendered as PNG rather than as CSS blocks on purpose: export_office.py turns
each slide into a PowerPoint slide and copies the first <img> it finds, so a
raster figure survives the conversion while a div-and-border diagram would be
silently dropped, exactly as the bar chart was before draw_bars() existed.

  python src/make_diagrams.py   ->  results/diagram_pipeline.png
                                    results/diagram_grid.png
                                    results/heatmap_grid.png
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as R  # noqa: E402

ROOT = Path(os.path.expanduser("~/fewshot_gs"))
OUT = ROOT / "results"

# Matched to the deck's light palette so the figures do not look imported.
INK = "#16202A"
MUTED = "#5C6E78"
LINE = "#D2DDE1"
ACCENT = "#0B6E7F"
GAIN = "#1C7C54"
LOSS = "#B3452C"
SURFACE = "#FFFFFF"
GROUND = "#EFF3F4"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.edgecolor": LINE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def box(ax, x, y, w, h, title, sub="", fc=SURFACE, ec=LINE, tc=INK, lw=1.4,
        title_size=11, sub_size=9):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), title,
            ha="center", va="center", fontsize=title_size, color=tc,
            fontweight="bold", zorder=3)
    if sub:
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                fontsize=sub_size, color=MUTED, zorder=3, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=1.5, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
        linewidth=lw, color=color, zorder=1,
        shrinkA=2, shrinkB=2))


def pipeline():
    """One pass of the experiment, end to end.

    Two lanes - real-only and augmented - that converge on the same trainer and
    the same held-out set, because that convergence is the whole design: the
    only difference between the arms is what went into training.
    """
    fig, ax = plt.subplots(figsize=(14.2, 4.6))
    ax.set_xlim(0, 116)
    ax.set_ylim(0, 33)
    ax.axis("off")

    ytop, h = 21.0, 7.4
    ylow, hl = 7.5, 7.4

    box(ax, 1, ytop, 13.5, h, "219 photographs", "Tanks & Temples\ntruck",
        fc=GROUND)
    arrow(ax, 14.7, ytop + h / 2, 18.3, ytop + h / 2)
    box(ax, 18.5, ytop, 13.5, h, "COLMAP",
        "camera poses +\nsparse 3D points")
    arrow(ax, 32.2, ytop + h / 2, 35.8, ytop + h / 2)
    box(ax, 36, ytop, 16, h, "Frozen split",
        "187 train pool\n32 held-out, never trained on", ec=ACCENT, lw=2)
    arrow(ax, 52.2, ytop + h / 2, 55.8, ytop + h / 2)
    box(ax, 56, ytop, 16, h, "Subset selection",
        "farthest-point sampling\nk = 5 / 10 / 20, 3 seeds")

    # upper lane: the unaugmented control
    arrow(ax, 72.2, ytop + h / 2, 75.8, ytop + h / 2)
    box(ax, 76, ytop, 14, h, "Baseline", "k real images", fc=GROUND)

    # lower lane: everything synthetic
    arrow(ax, 64, ytop - 0.3, 64, ylow + hl + 0.4)
    box(ax, 40, ylow, 24, hl, "Diffusion augmentation",
        "inpaint / outpaint / pose-guided\nSD 1.5  ·  25-200 % ratio",
        ec=ACCENT, lw=2)
    arrow(ax, 64.2, ylow + hl / 2, 67.8, ylow + hl / 2)
    box(ax, 68, ylow, 22, hl, "Training set",
        "k real + n synthetic", fc=GROUND)

    # both lanes converge
    arrow(ax, 90.2, ytop + h / 2 - 1.0, 95.8, 22.0)
    arrow(ax, 90.2, ylow + hl / 2 + 1.0, 95.8, 19.6)
    box(ax, 96, 17.4, 18, 6.4, "3DGS training", "7,000 iterations",
        ec=ACCENT, lw=2, title_size=10.5, sub_size=8.5)
    arrow(ax, 105, 17.2, 105, 14.2)
    box(ax, 96, 7.0, 18, 7.0, "Evaluate on 32 held-out",
        "PSNR · SSIM · LPIPS", title_size=9.5, sub_size=8.5)

    ax.text(58, 30.6, "One condition, end to end",
            ha="center", va="center", fontsize=13.5, fontweight="bold",
            color=INK)
    ax.text(58, 1.6,
            "Synthetic images are added to TRAINING only. The 32 held-out "
            "photographs are identical in every run, verified by hash.",
            ha="center", va="center", fontsize=9.5, color=MUTED)

    fig.tight_layout()
    p = OUT / "diagram_pipeline.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    print(f"wrote {p}")


def grid():
    """The factorial, drawn as the shape it is."""
    fig, ax = plt.subplots(figsize=(13.6, 5.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    strategies = [("Inpainting", "same pose"),
                  ("Outpainting", "same centre, wider frustum"),
                  ("Pose-guided", "a new camera")]
    ratios = ["25 %", "50 %", "100 %", "200 %"]
    ks = ["k = 5", "k = 10", "k = 20"]

    x0, y0 = 15.5, 5.0
    cw, ch = 5.2, 6.6
    gapx, gapy = 0.9, 1.5
    blockw = len(ratios) * cw + (len(ratios) - 1) * gapx

    # column group headers: subset size
    for j, kk in enumerate(ks):
        bx = x0 + j * (blockw + 4.6)
        ax.text(bx + blockw / 2, y0 + 3 * (ch + gapy) + 2.4, kk,
                ha="center", va="center", fontsize=11.5, fontweight="bold",
                color=ACCENT)
        for r, lab in enumerate(ratios):
            ax.text(bx + r * (cw + gapx) + cw / 2,
                    y0 + 3 * (ch + gapy) + 0.4, lab,
                    ha="center", va="center", fontsize=8.5, color=MUTED)

    for i, (name, note) in enumerate(strategies):
        yy = y0 + (2 - i) * (ch + gapy)
        ax.text(14.2, yy + ch / 2, name, ha="right", va="center",
                fontsize=11, fontweight="bold", color=INK)
        ax.text(14.2, yy + ch / 2 - 1.9, note, ha="right", va="center",
                fontsize=8, color=MUTED, style="italic")
        for j in range(len(ks)):
            bx = x0 + j * (blockw + 4.6)
            for r in range(len(ratios)):
                cx = bx + r * (cw + gapx)
                ax.add_patch(FancyBboxPatch(
                    (cx, yy), cw, ch,
                    boxstyle="round,pad=0.01,rounding_size=0.02",
                    facecolor=SURFACE, edgecolor=LINE, linewidth=1.1))
                # three seeds, drawn as three ticks
                for s in range(3):
                    ax.add_patch(plt.Rectangle(
                        (cx + 0.9 + s * 1.15, yy + 1.5), 0.75, ch - 3.0,
                        facecolor=ACCENT, alpha=0.30, edgecolor="none"))

    ax.text(50, 37.6,
            "3 strategies  ×  4 synthetic ratios  ×  3 subset sizes  "
            "×  3 seeds  =  108 augmented runs",
            ha="center", va="center", fontsize=12.5, fontweight="bold",
            color=INK)
    ax.text(50, 34.8,
            "plus 9 unaugmented baselines, 4 control conditions and a "
            "full-data ceiling",
            ha="center", va="center", fontsize=9.5, color=MUTED)
    ax.text(50, 1.4,
            "Each tick is one training run. Every augmented cell is paired "
            "against the baseline built from the SAME seed's subset.",
            ha="center", va="center", fontsize=9.5, color=MUTED)

    fig.tight_layout()
    p = OUT / "diagram_grid.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    print(f"wrote {p}")


def heatmap():
    """The same grid, filled in with what it measured."""
    recs = R.load_runs("truck")
    _, rows = R.build_tables(recs)
    by = {(r["strategy"], r["k"], r["ratio"]): r for r in rows}

    def get(strategy, k, ratio):
        for (s, kk, rt), r in by.items():
            if s == strategy and kk == k and abs(rt - ratio) <= 10:
                return r
        return None

    strategies = [("inpaint", "Inpainting"), ("outpaint", "Outpainting"),
                  ("guided", "Pose-guided")]
    ratios = [25, 50, 100, 200]
    ks = [5, 10, 20]

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4), sharey=True)
    # Clipped well below pose-guided's worst cell: at vmax = 1.9 the
    # outpainting crossover, which is what this figure is FOR, rendered as
    # two nearly identical pale tints. Pose-guided saturates instead, which
    # costs nothing - it is uniformly bad and the annotations carry it.
    vmax = 1.0
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "gl", [LOSS, "#E8B4A6", "#F4F6F6", "#A8CFBC", GAIN])

    for ax, k in zip(axes, ks):
        data = []
        for key, _ in strategies:
            data.append([(get(key, k, rt) or {}).get("d_psnr", float("nan"))
                         for rt in ratios])
        im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(ratios)))
        ax.set_xticklabels([f"{r} %" for r in ratios], fontsize=9.5)
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels([n for _, n in strategies], fontsize=10)
        ax.set_title(f"k = {k}", fontsize=12, fontweight="bold", color=ACCENT,
                     pad=8)
        for i in range(len(strategies)):
            for j in range(len(ratios)):
                v = data[i][j]
                if v != v:
                    continue
                r = get(strategies[i][0], k, ratios[j])
                star = "*" if r and r.get("sig_psnr") else ""
                ax.text(j, i, f"{v:+.2f}{star}", ha="center", va="center",
                        fontsize=9.5,
                        color="#FFFFFF" if abs(v) > 0.62 else INK,
                        fontweight="bold" if abs(v) > 0.25 else "normal")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)

    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015)
    cb.set_label("ΔPSNR vs same-seed baseline (dB), clipped at ±1", fontsize=9.5,
                 color=MUTED)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8.5, length=0)

    fig.suptitle("The whole grid, in one picture: green helps, red harms",
                 fontsize=13, fontweight="bold", color=INK, y=0.99)
    p = OUT / "heatmap_grid.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline()
    grid()
    heatmap()
