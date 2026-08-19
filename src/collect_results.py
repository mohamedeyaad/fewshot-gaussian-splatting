"""Aggregate every runs/*/results.json into tables (markdown + CSV).

Groups by condition (k, method, n_fake) and reports mean +/- std across seeds,
which is the form the report needs.

  python src/collect_results.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from scene_key import is_depth, scene_of

ROOT = Path(os.path.expanduser("~/fewshot_gs"))


def load(runs_dir: Path):
    recs = []
    for p in sorted(runs_dir.glob("*/results.json")):
        try:
            recs.append(json.loads(p.read_text()))
        except Exception as e:
            print(f"  ! skipping {p}: {e}")
    return recs


def key_of(r):
    prov = r.get("provenance", {})
    # Depth runs share a scene - and therefore a provenance - with their
    # non-depth twin, so they must be labelled apart or they merge into it.
    method = (prov.get("method") or "") + ("+depth" if is_depth(r) else "")
    return (scene_of(r), prov.get("k"), method, prov.get("n_synthetic", 0))


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return mean(vals), (stdev(vals) if len(vals) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    runs_dir, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    recs = load(runs_dir)
    if not recs:
        print("no results yet")
        return
    print(f"loaded {len(recs)} runs\n")

    groups = defaultdict(list)
    for r in recs:
        groups[key_of(r)].append(r)

    # ---- per-run CSV (everything, ungrouped) --------------------------
    csv_path = out / "runs_raw.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "scene", "k", "method", "seed", "n_fake", "iterations",
                    "psnr", "ssim", "lpips", "train_s", "peak_vram_mib",
                    "n_gaussians"])
        for r in sorted(recs, key=lambda x: x["tag"]):
            p, c, m = r["provenance"], r["config"], r["metrics"]
            w.writerow([r["tag"], scene_of(r), p.get("k"), p.get("method"), p.get("seed"),
                        p.get("n_synthetic", 0), c["iterations"],
                        f"{m['psnr']['mean']:.4f}", f"{m['ssim']['mean']:.4f}",
                        f"{m['lpips']['mean']:.4f}",
                        r["cost"].get("train_seconds"),
                        r["cost"].get("peak_vram_mib"),
                        r["cost"].get("n_gaussians")])
    print(f"wrote {csv_path}")

    # ---- grouped markdown table ---------------------------------------
    lines = ["| scene | k | method | fake | seeds | PSNR | SSIM | LPIPS | Gaussians | train s |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for scene, k, method, nfake in sorted(
            groups, key=lambda t: (str(t[0]), t[1] or 0, str(t[2]), t[3] or 0)):
        rs = groups[(scene, k, method, nfake)]
        pm, ps = agg([r["metrics"]["psnr"]["mean"] for r in rs])
        sm, ss = agg([r["metrics"]["ssim"]["mean"] for r in rs])
        lm, ls = agg([r["metrics"]["lpips"]["mean"] for r in rs])
        gm, _ = agg([r["cost"].get("n_gaussians") for r in rs])
        tm, _ = agg([r["cost"].get("train_seconds") for r in rs])
        lines.append(
            f"| {scene} | {k} | {method} | {nfake} | {len(rs)} | "
            f"{pm:.2f} ± {ps:.2f} | {sm:.4f} ± {ss:.4f} | {lm:.4f} ± {ls:.4f} | "
            f"{gm:,.0f} | {tm:.0f} |" if tm is not None else
            f"| {scene} | {k} | {method} | {nfake} | {len(rs)} | "
            f"{pm:.2f} ± {ps:.2f} | {sm:.4f} ± {ss:.4f} | {lm:.4f} ± {ls:.4f} | "
            f"{gm:,.0f} | - |")

    table = "\n".join(lines)

    # ---- paired within-seed deltas -------------------------------------
    # The grouped table above can mislead when conditions have different
    # numbers of seeds. Pairing each augmented run against the SAME seed's
    # 0-fake baseline removes seed-to-seed luck entirely.
    # The baseline key MUST carry the scene as well as (k, seed): drjohnson
    # k=5 seed0 and truck k=5 seed0 are different runs, and without the scene
    # one overwrites the other, pairing every augmented run of one scene
    # against the other scene's floor.
    # A depth-regularised 0-fake run is NOT a baseline: it is a treatment that
    # happens to add no synthetic views. Letting it in here overwrites the real
    # baseline and shifts every delta in the table.
    baselines = {}
    for r in recs:
        p = r["provenance"]
        if (p.get("n_synthetic", 0) == 0 and p.get("method") != "full"
                and not is_depth(r)):
            baselines[(scene_of(r), p.get("k"), p.get("seed"))] = r["metrics"]

    paired = defaultdict(list)
    for r in recs:
        p = r["provenance"]
        nf = p.get("n_synthetic", 0)
        if p.get("method") == "full":
            continue
        dep = is_depth(r)
        if nf == 0 and not dep:
            continue                      # a plain baseline is the reference
        base = baselines.get((scene_of(r), p.get("k"), p.get("seed")))
        if not base:
            continue
        strat = p.get("strategy") or "none"
        if dep:
            strat = "depth" if nf == 0 else f"{strat}+depth"
        paired[(scene_of(r), p.get("k"), strat, nf)].append({
            "seed": p.get("seed"),
            "d_psnr": r["metrics"]["psnr"]["mean"] - base["psnr"]["mean"],
            "d_ssim": r["metrics"]["ssim"]["mean"] - base["ssim"]["mean"],
            "d_lpips": r["metrics"]["lpips"]["mean"] - base["lpips"]["mean"],
        })

    plines = ["", "### Paired deltas vs same-seed 0-fake baseline",
              "(positive PSNR/SSIM = better; negative LPIPS = better)", "",
              "| scene | k | strategy | fake | seeds | dPSNR | dSSIM | dLPIPS |",
              "|---|---|---|---|---|---|---|---|"]
    for scene, k, strat, nf in sorted(
            paired, key=lambda t: (str(t[0]), t[1] or 0, str(t[2]), t[3])):
        ds = paired[(scene, k, strat, nf)]
        pm, ps = agg([d["d_psnr"] for d in ds])
        sm, ss = agg([d["d_ssim"] for d in ds])
        lm, ls = agg([d["d_lpips"] for d in ds])
        # Flag deltas whose +/-1 std interval excludes zero - i.e. every seed
        # agreed on the sign with margin. Not a formal test at n=3, but it
        # separates "consistent" from "noise".
        sig = lambda m, s: "*" if s > 0 and abs(m) > s else " "
        plines.append(f"| {scene} | {k} | {strat} | {nf} | {len(ds)} | "
                      f"{pm:+.3f} ± {ps:.3f}{sig(pm,ps)} | "
                      f"{sm:+.4f} ± {ss:.4f}{sig(sm,ss)} | "
                      f"{lm:+.4f} ± {ls:.4f}{sig(lm,ls)} |")
    ptable = "\n".join(plines)

    (out / "summary.md").write_text(table + "\n" + ptable + "\n")
    print(f"wrote {out/'summary.md'}\n")
    print(table)
    if len(plines) > 6:
        print(ptable)


if __name__ == "__main__":
    main()
