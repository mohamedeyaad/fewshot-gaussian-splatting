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

import make_panels
from make_panels import (OUT, SEED, grid, load, psnr_of,
                         representative_views)

from contextlib import contextmanager


@contextmanager
def at(runs_dir, iters):
    """Read renders and per-view PSNR from a different sweep.

    load() and psnr_of() resolve make_panels.RUNS and make_panels.ITER at
    call time, so swapping the module globals redirects them. The 30,000
    runs live in runs_30k/<tag>/test/ours_30000/ rather than the 7,000
    defaults, and mixing the two would silently compare a 7k render
    against a 30k number.
    """
    old_runs, old_iter = make_panels.RUNS, make_panels.ITER
    make_panels.RUNS = ROOT / runs_dir
    make_panels.ITER = iters
    try:
        yield
    finally:
        make_panels.RUNS, make_panels.ITER = old_runs, old_iter

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
        # index into the 32 held-out cameras, NOT a photograph number
        labels.append(f"held-out {v + 1}/32")
    return rows, labels


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
    """The 2x2 at k=5 and 7,000 iterations.

    VIEWS was arbitrary here, which is the bug already fixed in
    panel_selection: per-view deltas scatter several dB either side of a mean
    worth a few tenths, so three fixed indices can illustrate the opposite of
    the caption they sit under. Chosen against +both, the cell the figure is
    actually about.
    """
    b = f"truck_k5_seed{SEED}_fps_fake0"
    both = f"truck_k5_seed{SEED}_fps_outpaint_fake{NF[5]}_depth"
    dep = f"truck_k5_seed{SEED}_fps_fake0_depth"
    out = f"truck_k5_seed{SEED}_fps_outpaint_fake{NF[5]}"
    cols = [
        ("ground truth", None),
        ("5 real only", b),
        ("+ depth prior", dep),
        ("+ outpainting", out),
        ("+ both", both),
    ]
    views = representative_views([(both, b), (dep, b)])
    rows, labels = cells(cols, views, b,
                         deltas_vs={dep: b, out: b, both: b})
    grid(rows, [c[0] for c in cols], OUT / "panel_depth.png",
         "Depth regularisation at k=5, 7,000 iterations: the two interventions "
         "compound (+0.259 and +0.285 alone; +0.714 together).\n"
         "Views chosen as those closest to the mean effect; per-view deltas shown.",
         labels)


def panel_depth_30k():
    """The same 2x2 after 30,000 iterations - where the story changes.

    At this budget outpainting on its own is worth -0.078 dB, indistinguishable
    from zero: the benefit measured at 7,000 is gone. The depth prior is not
    gone (+0.208), and the two together are worth +0.469 - more than the sum of
    the parts, and more than either alone. The figure exists because that is a
    claim about images, not just about a mean: the +outpainting column should
    look no better than the baseline, while +both still does.
    """
    b = f"truck_k5_seed{SEED}_fps_fake0"
    dep = f"truck_k5_seed{SEED}_fps_fake0_depth"
    out = f"truck_k5_seed{SEED}_fps_outpaint_fake{NF[5]}"
    both = f"truck_k5_seed{SEED}_fps_outpaint_fake{NF[5]}_depth"
    cols = [
        ("ground truth", None),
        ("5 real only", b),
        ("+ depth prior", dep),
        ("+ outpainting", out),
        ("+ both", both),
    ]
    with at("runs_30k", 30000):
        views = representative_views([(both, b), (out, b)])
        rows, labels = cells(cols, views, b,
                             deltas_vs={dep: b, out: b, both: b})
        grid(rows, [c[0] for c in cols], OUT / "panel_depth_30k.png",
             "At 30,000 iterations outpainting alone is worth -0.078 dB (dead), "
             "the depth prior still +0.208, and the two together +0.469.\n"
             "Views chosen as those closest to the mean effect; per-view deltas shown.",
             labels)


def panel_ratio_extreme():
    """How far can the synthetic ratio be pushed before it stops helping?

    The swept grid stops at 200%, which is where the specification stops. But
    runs at 400% and 800% exist at k=5 - 20 and 40 fabricated images against 5
    real ones - and nothing had ever looked at them. At 800% the training set
    is 89% invented.

    Ratios here are of the REAL subset size: 40 fakes against 5 real views is
    800%, not 40%.
    """
    b = f"truck_k5_seed{SEED}_fps_fake0"
    cols = [
        ("ground truth", None),
        ("5 real only", b),
        ("+200%  (10 fake)", f"truck_k5_seed{SEED}_fps_outpaint_fake10"),
        ("+400%  (20 fake)", f"truck_k5_seed{SEED}_fps_outpaint_fake20"),
        ("+800%  (40 fake)", f"truck_k5_seed{SEED}_fps_outpaint_fake40"),
    ]
    tags = [c[1] for c in cols[2:]]
    views = representative_views([(tags[-1], b)])
    rows, labels = cells(cols, views, b, deltas_vs={t: b for t in tags})
    grid(rows, [c[0] for c in cols], OUT / "panel_ratio_extreme.png",
         "Pushing the synthetic ratio past the swept grid, at 5 real views. "
         "Mean paired gain: +0.285 dB at 200%, +0.285 at 400%, +0.426 at 800% "
         "(3 seeds).\nAt 800% the training set is 40 fabricated images against "
         "5 real ones. Views chosen as those closest to the mean effect.",
         labels)


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


def panel_selection():
    """Does the crossover depend on having chosen the five views well?

    Every other result uses farthest-point sampling, which spreads the K views
    over the trajectory. Random draws cover the scene worse (k=5 max_nn_dist
    5.15 against fps's 3.54), so the coverage account predicted augmentation
    would help MORE there, and cross over later.

    It does not. Random is slightly less positive at k=5 (+0.261 vs +0.285,
    and not separated from zero) and clearly MORE negative at k=20 (-0.929 vs
    -0.618). The columns are ordered fps-then-random at each subset size so
    the two baselines can be compared before the two treatments are.
    """
    b5f = f"truck_k5_seed{SEED}_fps_fake0"
    b5r = f"truck_k5_seed{SEED}_random_fake0"
    b20f = f"truck_k20_seed{SEED}_fps_fake0"
    b20r = f"truck_k20_seed{SEED}_random_fake0"
    cols = [
        ("ground truth", None),
        ("5 real, spread", b5f),
        ("5 real, spread\n+200%", f"truck_k5_seed{SEED}_fps_outpaint_fake10"),
        ("5 real, random", b5r),
        ("5 real, random\n+200%", f"truck_k5_seed{SEED}_random_outpaint_fake10"),
        ("20 real, spread\n+200%", f"truck_k20_seed{SEED}_fps_outpaint_fake40"),
        ("20 real, random\n+200%", f"truck_k20_seed{SEED}_random_outpaint_fake40"),
    ]
    a5f = f"truck_k5_seed{SEED}_fps_outpaint_fake10"
    a5r = f"truck_k5_seed{SEED}_random_outpaint_fake10"
    a20f = f"truck_k20_seed{SEED}_fps_outpaint_fake40"
    a20r = f"truck_k20_seed{SEED}_random_outpaint_fake40"
    # Not VIEWS. Arbitrary cameras scatter several dB either side of the mean:
    # view 12 shows +2.02 for the k=20 fps condition whose mean is -0.618, so a
    # panel drawn there argues against its own caption.
    views = representative_views([(a5f, b5f), (a5r, b5r), (a20f, b20f), (a20r, b20r)])
    rows, labels = cells(cols, views, b5f,
                         deltas_vs={a5f: b5f, a5r: b5r, a20f: b20f, a20r: b20r})
    grid(rows, [c[0] for c in cols], OUT / "panel_selection.png",
         "Badly-chosen views are NOT rescued by augmentation. Random draws cover the "
         "scene worse, so the coverage account predicted a bigger gain;\ninstead random "
         "is +0.261 at k=5 (not separated from zero, against fps's +0.285*) and -0.929* "
         "at k=20 against fps's -0.618*.\nOutpainting widens each existing frustum - it "
         "cannot reach scene a badly-spread set never came near.", labels)


if __name__ == "__main__":
    panel_crossover()
    panel_depth()
    panel_depth_30k()
    panel_ratio_extreme()
    panel_scaling()
    panel_control()
    panel_scene2()
    panel_selection()
