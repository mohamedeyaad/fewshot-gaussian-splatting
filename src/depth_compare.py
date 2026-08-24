"""The depth-regularisation 2x2, paired within seed.

Four cells at k=5, each compared against the SAME seed's plain baseline:

                        no depth          + depth prior
    real only           0 (reference)     effect of the prior alone
    + outpainting 200%  known             do the two gains add?

The bottom-right cell is the point. If a depth prior and synthetic views fix
the same deficiency, their gains overlap and the combined cell falls short of
the sum. If they fix different ones - constraint versus coverage - it meets or
beats it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(os.path.expanduser("~/fewshot_gs"))
K = int(os.environ.get("K", 5))
NF = int(os.environ.get("NF", 10))
SEEDS = (0, 1, 2)

# Which scene, and which iteration budget. Both were hardcoded to truck/7k,
# so run_depth_reg.sh with SCENE=drjohnson trained six drjohnson runs and
# then printed the TRUCK table underneath them - a summary that looked
# entirely correct because nothing in it named a scene. Hence SUBJECT below.
SCENE = os.environ.get("SCENE", "truck")
OUTDIR = os.environ.get("OUTDIR", "runs")
SUBJECT = f"{SCENE}, {OUTDIR}"


def psnr(tag):
    p = ROOT / OUTDIR / tag / "results.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["metrics"]


def agg(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, 0.0
    return mean(v), (stdev(v) if len(v) > 1 else 0.0)


def main():
    cells = {
        "baseline":      "{SCENE}_k{K}_seed{s}_fps_fake0",
        "+depth":        "{SCENE}_k{K}_seed{s}_fps_fake0_depth",
        "+outpaint":     "{SCENE}_k{K}_seed{s}_fps_outpaint_fake{NF}",
        "+both":         "{SCENE}_k{K}_seed{s}_fps_outpaint_fake{NF}_depth",
    }

    got = {}
    for label, pat in cells.items():
        got[label] = {s: psnr(pat.format(SCENE=SCENE, K=K, s=s, NF=NF))
                      for s in SEEDS}

    missing = [f"{l} seed{s}" for l, d in got.items()
               for s, m in d.items() if m is None]
    if missing:
        print("missing runs: " + ", ".join(missing))
        print("(run src/run_depth_reg.sh first)\n")

    print(f"=== depth regularisation: {SUBJECT}, "
          f"k={K}, {len(SEEDS)} seeds ===\n")
    print(f"{'condition':<14} {'PSNR':>16} {'vs baseline':>16} {'SSIM':>9} {'LPIPS':>9}")
    print("-" * 70)

    deltas = {}
    for label in cells:
        vals = [got[label][s]["psnr"]["mean"] for s in SEEDS if got[label][s]]
        if not vals:
            print(f"{label:<14} {'--':>16}")
            continue
        m, sd = agg(vals)
        ds = [got[label][s]["psnr"]["mean"] - got["baseline"][s]["psnr"]["mean"]
              for s in SEEDS if got[label][s] and got["baseline"][s]]
        dm, dsd = agg(ds)
        deltas[label] = (dm, dsd)
        sm = agg([got[label][s]["ssim"]["mean"] for s in SEEDS if got[label][s]])[0]
        lm = agg([got[label][s]["lpips"]["mean"] for s in SEEDS if got[label][s]])[0]
        star = "*" if dsd > 0 and abs(dm) > dsd else " "
        d_txt = "reference" if label == "baseline" else f"{dm:+.3f} ± {dsd:.3f}{star}"
        print(f"{label:<14} {m:9.3f} ± {sd:.3f} {d_txt:>16} {sm:9.4f} {lm:9.4f}")

    print("-" * 70)
    if all(k in deltas for k in ("+depth", "+outpaint", "+both")):
        a = deltas["+depth"][0]
        b = deltas["+outpaint"][0]
        both = deltas["+both"][0]
        print(f"\n  depth alone      {a:+.3f} dB")
        print(f"  outpaint alone   {b:+.3f} dB")
        print(f"  sum if additive  {a + b:+.3f} dB")
        print(f"  measured (+both) {both:+.3f} dB")
        gap = both - (a + b)
        print(f"  interaction      {gap:+.3f} dB")
        if abs(gap) < 0.055:
            verdict = ("additive - the two interventions fix different "
                       "deficiencies (constraint vs coverage)")
        elif gap < 0:
            verdict = ("sub-additive - the gains overlap, so both are partly "
                       "fixing the same under-determined geometry")
        else:
            verdict = "super-additive - the two reinforce each other"
        print(f"\n  -> {verdict}")
        print("  (0.055 dB = the measured paired noise floor)")


if __name__ == "__main__":
    main()
