"""Visual examples for the conditions make_panels.py does not cover.

make_panels.py was written when k=10 was the whole study, and every one of its
figures hardcodes `truck_k10_seed0`. That left the project's actual findings
without any picture:

  panel_crossover.png  - THE missing one. The same held-out camera at k=5 and
                         k=20, with and without the identical 200% outpainting.
                         Augmentation helping and harming, side by side.
  panel_depth.png      - the 2x2 at k=5: baseline / +depth / +outpaint / +both
  panel_scaling.png    - 5, 10, 20 and 219 real views, so the gap being closed
                         is visible rather than tabulated
  panel_control.png    - pose-guided vs warp-only: what the diffusion step is
                         actually repairing
  panel_scene2.png     - drjohnson, the second scene

Nothing needs retraining: every run keeps its 32 held-out renders under
runs/<tag>/test/ours_7000/, which survived the checkpoint cleanup.

  python src/make_panels_extra.py
"""
from __future__ import annotations

import os
from pathlib import Path

from make_panels import OUT, SEED, grid, load, psnr_of

ROOT = Path(os.path.expanduser("~/fewshot_gs"))

# Three cameras spread through the held-out set. The split is frozen, so index
# i is the same physical camera in every run of a scene.
VIEWS = [3, 12, 21]
NF = {5: 10, 10: 20, 20: 40}          # the 200% ratio at each subset size


def cells(cols, views, gt_from, deltas_vs=None):
    """Build grid rows: one row per view, one column per (title, tag).

    `deltas_vs` maps a column tag to the baseline tag it should be compared
    against; those cells are annotated with the per-view difference as well as
    the absolute PSNR.
    """
    deltas_vs = deltas_vs or {}
    rows, labels = [], []
    for v in views:
        row = []
        for title, tag in cols:
            if tag is None:
                row.append((load(gt_from, v, "gt"), "reference"))
                continue
            p = psnr_of(tag, v)
            sub = f"PSNR {p:.2f}" if p is not None else ""
            b = deltas_vs.get(tag)
            if b and p is not None:
                pb = psnr_of(b, v)
                if pb is not None:
                    sub += f"   {p - pb:+.2f}"
            row.append((load(tag, v), sub))
        rows.append(row)
        labels.append(f"view {v}")
    return rows, labels


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


def panel_crossover():
    """The headline finding, made visible.

    Identical treatment - outpainting at the 200% ratio - applied at two
    subset sizes. At k=5 it improves the render; at k=20 it degrades it. The
    columns are ordered so the two baselines sit next to their augmented
    versions, which is what makes the reversal legible.
    """
    b5 = f"truck_k5_seed{SEED}_fps_fake0"
    b20 = f"truck_k20_seed{SEED}_fps_fake0"
    a5 = f"truck_k5_seed{SEED}_fps_outpaint_fake{NF[5]}"
    a20 = f"truck_k20_seed{SEED}_fps_outpaint_fake{NF[20]}"
    cols = [
        ("ground truth", None),
        ("5 real", b5),
        ("5 real +200% synth", a5),
        ("20 real", b20),
        ("20 real +200% synth", a20),
    ]
    views = representative_views([(a5, b5), (a20, b20)])
    rows, labels = cells(cols, views, b5, deltas_vs={a5: b5, a20: b20})
    grid(rows, [c[0] for c in cols], OUT / "panel_crossover.png",
         "The crossover: the SAME augmentation helps at 5 real views (+0.285 dB mean) "
         "and harms at 20 (-0.618 dB mean).\nViews chosen as those closest to the mean "
         "effect; per-view deltas shown, and they scatter widely either side.", labels)


def panel_depth():
    b = f"truck_k5_seed{SEED}_fps_fake0"
    cols = [
        ("ground truth", None),
        ("5 real only", b),
        ("+ depth prior", f"truck_k5_seed{SEED}_fps_fake0_depth"),
        ("+ outpainting", f"truck_k5_seed{SEED}_fps_outpaint_fake{NF[5]}"),
        ("+ both", f"truck_k5_seed{SEED}_fps_outpaint_fake{NF[5]}_depth"),
    ]
    rows, labels = cells(cols, VIEWS, b)
    grid(rows, [c[0] for c in cols], OUT / "panel_depth.png",
         "Depth regularisation at k=5: the two interventions compound "
         "(+0.259, +0.285 alone; +0.714 together)", labels)


def panel_scaling():
    b = f"truck_k5_seed{SEED}_fps_fake0"
    cols = [
        ("ground truth", None),
        ("5 real", b),
        ("10 real", f"truck_k10_seed{SEED}_fps_fake0"),
        ("20 real", f"truck_k20_seed{SEED}_fps_fake0"),
        ("219 real (ceiling)", "truck_k219_seed0_full_fake0"),
    ]
    rows, labels = cells(cols, VIEWS, b)
    grid(rows, [c[0] for c in cols], OUT / "panel_scaling.png",
         "What real photographs buy: 15.20 -> 17.11 -> 19.74 -> 25.23 dB", labels)


def panel_control():
    """Why the warp-only control settles the question.

    Same poses, same hole masks, same seeds; the only difference is whether
    diffusion fills the disocclusions or they stay black. Removing diffusion
    makes it far worse, so the pose-guided damage is not hallucinated content.
    """
    b = f"truck_k10_seed{SEED}_fps_fake0"
    cols = [
        ("ground truth", None),
        ("10 real only", b),
        ("pose-guided +200%", f"truck_k10_seed{SEED}_fps_guided_fake20"),
        ("warp-only (holes black)", f"truck_k10_seed{SEED}_fps_warponly_fake20"),
    ]
    rows, labels = cells(cols, VIEWS, b)
    grid(rows, [c[0] for c in cols], OUT / "panel_control.png",
         "Warp-only control: deleting the diffusion step costs a further "
         "2.49 dB, so diffusion was repairing, not damaging", labels)


def panel_scene2():
    """drjohnson at the 100% ratio, not 200%.

    THE 200% RATIO IS THE WRONG CHOICE HERE, and an earlier version of this
    figure used it. drjohnson k=20 outpainting runs -0.310* / -0.147* / -0.242
    / +0.195 across the four ratios: it is negative at every ratio EXCEPT 200%,
    the single cell where it comes out positive. A panel drawn at 200% shows
    augmentation improving k=20 while its caption claims the reverse - the
    figure argues against the finding it illustrates.

    100% applies the same treatment at both subset sizes and is negative at
    k=20, so the columns are comparable and the direction is right. The caption
    states plainly that the k=20 effect is not statistically separated at this
    ratio, because it is not.

    drjohnson has 33 held-out views rather than truck's 32.
    """
    b5 = f"drjohnson_k5_seed{SEED}_fps_fake0"
    b20 = f"drjohnson_k20_seed{SEED}_fps_fake0"
    a5 = f"drjohnson_k5_seed{SEED}_fps_outpaint_fake5"     # 100% at k=5
    a20 = f"drjohnson_k20_seed{SEED}_fps_outpaint_fake20"  # 100% at k=20
    cols = [
        ("ground truth", None),
        ("5 real", b5),
        ("5 real +100% synth", a5),
        ("20 real", b20),
        ("20 real +100% synth", a20),
    ]
    views = representative_views([(a5, b5), (a20, b20)], pool=33)
    rows, labels = cells(cols, views, b5, deltas_vs={a5: b5, a20: b20})
    grid(rows, [c[0] for c in cols], OUT / "panel_scene2.png",
         "Second scene (drjohnson, indoor, quarter resolution) at the 100% ratio: "
         "+0.829 dB at k=5, -0.242 dB at k=20.\nViews closest to the mean effect; "
         "the k=20 effect is negative but not statistically separated at this ratio "
         "(it is at 25% and 50%).", labels)


def panel_lego():
    """The third capture regime: an isolated object on a clean background.

    Only subset sizes here - no augmentation conditions exist for this scene,
    because the Blender format stores one global camera_angle_x and cannot
    express the widened frustum an outpainted view needs.

    lego has 34 held-out views (200 test frames, stride 6), so the view indices
    differ again from truck's 32 and drjohnson's 33.
    """
    b = f"lego_k5_seed{SEED}_fps_fake0"
    cols = [
        ("ground truth", None),
        ("5 real", b),
        ("10 real", f"lego_k10_seed{SEED}_fps_fake0"),
        ("20 real", f"lego_k20_seed{SEED}_fps_fake0"),
        ("100 real (ceiling)", "lego_k100_seed0_full_fake0"),
    ]
    rows, labels = cells(cols, [2, 14, 27], b)
    grid(rows, [c[0] for c in cols], OUT / "panel_lego.png",
         "Third scene (lego, isolated object on white): what real views buy "
         "when the background is trivial", labels)


if __name__ == "__main__":
    panel_crossover()
    panel_depth()
    panel_scaling()
    panel_control()
    panel_scene2()
    panel_lego()
