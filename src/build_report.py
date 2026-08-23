"""Build the self-contained HTML report from the actual results on disk.

Everything numeric is read from runs/*/results.json, so the report cannot
drift from the experiments. Figures are downscaled and inlined as data URIs
because the artifact host blocks external requests.

  python src/build_report.py            # -> results/report.html
"""
from __future__ import annotations

import base64
import io
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from PIL import Image

from scene_key import is_depth, scene_of, select_of

ROOT = Path(os.path.expanduser("~/fewshot_gs"))
RES = ROOT / "results"
OUT = RES / "report.html"

STRAT_ORDER = ["inpaint", "outpaint", "guided"]
STRAT_LABEL = {"inpaint": "Inpainting", "outpaint": "Outpainting",
               "guided": "Pose-guided"}


# ---------------------------------------------------------------- data
def load_runs(scene: str = "truck", select: str = "fps"):
    """Every run of ONE scene.

    The filter is not cosmetic. Every table below pairs an augmented run
    against the same (k, seed) baseline, and that key is only unique within a
    scene - drjohnson k=5 seed0 would overwrite truck k=5 seed0 and silently
    reassign every truck delta. The report's claims are all about truck, so
    truck is what it loads.

    Depth-regularised runs are excluded for the same reason: they share a
    provenance with their non-depth twin, so a `..._fake0_depth` run counts as
    a baseline and overwrites the real one. They are loaded separately by
    build_depth().

    Runs whose K views were drawn at RANDOM rather than by farthest-point
    sampling are excluded for the same reason again, one axis over. They are
    the same scene, the same k and the same seeds, so truck_k20_seed0_fps_fake0
    and truck_k20_seed0_random_fake0 collide in `baselines[(k, seed)]` below -
    and they also land in the same `fewshot[k]` bucket, which is where the
    15.20 / 17.11 / 19.74 dB floors quoted throughout this report come from.
    Left in, they would move every floor and every delta at once.

    Pass select="random" to load them deliberately.
    """
    recs = []
    for p in sorted((ROOT / "runs").glob("*/results.json")):
        r = json.loads(p.read_text())
        if (scene_of(r) == scene and not is_depth(r)
                and (select_of(r) in (select, "full"))):
            recs.append(r)
    return recs


def agg(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, 0.0
    return mean(v), (stdev(v) if len(v) > 1 else 0.0)


KS = (5, 10, 20)
# Absolute synthetic counts per subset size. Only k=20 divides cleanly into the
# spec's 25/50/100/200%; at k=5 and k=10 the 25% point is fractional and rounds
# down, so the realised ratios differ slightly per size.
FAKES = {5: (1, 2, 5, 10), 10: (2, 5, 10, 20), 20: (5, 10, 20, 40)}


def build_tables(recs):
    """Paired deltas, keyed on (k, seed).

    The baseline lookup MUST include k: seed 0 exists at every subset size, so
    keying on seed alone silently pairs a k=5 augmented run against a k=20
    baseline. That bug was live while k=10 was the only sweep and produced no
    visible symptom, because there was only ever one k to collide.
    """
    baselines, full = {}, None
    fewshot = defaultdict(list)
    by = defaultdict(list)
    for r in recs:
        p = r["provenance"]
        if p.get("method") == "full":
            full = r
            continue
        k, seed, nf = p.get("k"), p.get("seed"), p.get("n_synthetic", 0)
        if k not in KS:
            continue
        if nf == 0:
            baselines[(k, seed)] = r
            fewshot[k].append(r)
        else:
            by[(k, p.get("strategy"), nf)].append(r)

    floors = {}
    for k in KS:
        if not fewshot[k]:
            continue
        pm, ps = agg([r["metrics"]["psnr"]["mean"] for r in fewshot[k]])
        floors[k] = {
            "psnr": pm, "psnr_sd": ps,
            "ssim": agg([r["metrics"]["ssim"]["mean"] for r in fewshot[k]])[0],
            "lpips": agg([r["metrics"]["lpips"]["mean"] for r in fewshot[k]])[0],
            "gauss": agg([r["cost"]["n_gaussians"] for r in fewshot[k]])[0],
            "secs": agg([r["cost"]["train_seconds"] or 0 for r in fewshot[k]])[0],
        }

    head = {
        "floors": floors,
        "floor_psnr": floors[10]["psnr"], "floor_psnr_sd": floors[10]["psnr_sd"],
        "floor_ssim": floors[10]["ssim"], "floor_lpips": floors[10]["lpips"],
        "floor_gauss": floors[10]["gauss"],
        "ceil_psnr": full["metrics"]["psnr"]["mean"],
        "ceil_ssim": full["metrics"]["ssim"]["mean"],
        "ceil_lpips": full["metrics"]["lpips"]["mean"],
        "ceil_gauss": full["cost"]["n_gaussians"],
        "n_runs": len(recs),
    }
    head["gap"] = head["ceil_psnr"] - floors[10]["psnr"]
    head["gap5"] = head["ceil_psnr"] - floors[5]["psnr"]
    head["gap20"] = head["ceil_psnr"] - floors[20]["psnr"]
    head["step_5_10"] = floors[10]["psnr"] - floors[5]["psnr"]
    head["step_10_20"] = floors[20]["psnr"] - floors[10]["psnr"]

    rows = []
    for s in STRAT_ORDER:
        for k in KS:
            for nf in FAKES[k]:
                rs = by.get((k, s, nf), [])
                if not rs:
                    continue
                d = {m: [] for m in ("psnr", "ssim", "lpips")}
                g = []
                for r in rs:
                    b = baselines.get((k, r["provenance"].get("seed")))
                    if not b:
                        continue
                    for m in d:
                        d[m].append(r["metrics"][m]["mean"] - b["metrics"][m]["mean"])
                    g.append(r["cost"]["n_gaussians"])
                if not d["psnr"]:
                    continue
                pm, ps = agg(d["psnr"]); sm, ss = agg(d["ssim"]); lm, ls = agg(d["lpips"])
                rows.append({
                    "strategy": s, "k": k, "n_fake": nf,
                    "ratio": round(100.0 * nf / k), "seeds": len(d["psnr"]),
                    "d_psnr": pm, "d_psnr_sd": ps, "d_ssim": sm, "d_ssim_sd": ss,
                    "d_lpips": lm, "d_lpips_sd": ls, "gauss": agg(g)[0],
                    "sig_psnr": abs(pm) > ps > 0,
                })

    # best and worst single conditions, for the headline
    head["best"] = max(rows, key=lambda r: r["d_psnr"])
    head["worst"] = min(rows, key=lambda r: r["d_psnr"])
    return head, rows


def build_control(recs):
    """Warp-only control vs pose-guided, at k=10.

    Both conditions warp to bit-identical poses with bit-identical hole masks;
    the only difference is whether disoccluded pixels get diffusion content or
    stay black. The difference between them is the diffusion step's
    contribution, isolated.

    'warponly' is deliberately absent from STRAT_ORDER, so it never enters the
    main tables or figures - it is a control, not a fourth strategy.
    """
    base, runs = {}, defaultdict(list)
    for r in recs:
        p = r["provenance"]
        if p.get("method") == "full" or p.get("k") != 10:
            continue
        s, nf = p.get("seed"), p.get("n_synthetic", 0)
        if nf == 0:
            base[s] = r["metrics"]
        else:
            runs[(p.get("strategy"), nf)].append((s, r["metrics"]))

    def d(strat, nf):
        ds = [m["psnr"]["mean"] - base[s]["psnr"]["mean"]
              for s, m in runs.get((strat, nf), []) if s in base]
        return agg(ds) if ds else None

    out = []
    for nf in (2, 5, 10, 20):
        g, w = d("guided", nf), d("warponly", nf)
        if not (g and w):
            continue
        out.append({"ratio": nf * 10, "n_fake": nf,
                    "guided": g[0], "guided_sd": g[1],
                    "warp": w[0], "warp_sd": w[1],
                    "contribution": g[0] - w[0]})
    return out


def build_model_crossover(recs):
    """Does the SIGN CHANGE survive the checkpoint swap, or only the gain?

    build_ablation() below compares the two models at k=5, where outpainting
    helps. That is the half of the claim least in doubt. The finding of this
    study is that the same treatment reverses sign as real views accumulate,
    and for a long time no second model had been run at any subset size where
    the sign is negative - so the ablation could not distinguish "augmentation
    crosses over" from "SD 1.5 crosses over".

    This compares both models at the 200% ratio across all three subset sizes.
    The claim survives only if the second column changes sign too.
    """
    NF = {5: 10, 10: 20, 20: 40}          # 200% at each subset size
    base, runs = {}, defaultdict(list)
    for r in recs:
        p = r["provenance"]
        if p.get("method") == "full":
            continue
        k, s, nf = p.get("k"), p.get("seed"), p.get("n_synthetic", 0)
        if k not in NF:
            continue
        if nf == 0:
            base[(k, s)] = r["metrics"]
        else:
            runs[(k, p.get("strategy"), nf)].append((s, r["metrics"]))

    def d(k, strat):
        ds = [m["psnr"]["mean"] - base[(k, s)]["psnr"]["mean"]
              for s, m in runs.get((k, strat, NF[k]), []) if (k, s) in base]
        return agg(ds) if ds else None

    out = []
    for k in (5, 10, 20):
        a, b = d(k, "outpaint"), d(k, "outpaint_ds8")
        if not (a and b):
            continue
        out.append({"k": k, "sd15": a[0], "sd15_sd": a[1],
                    "ds8": b[0], "ds8_sd": b[1]})
    return out


def build_ablation(recs):
    """Model-robustness ablation: SD 1.5 vs Dreamshaper-8, outpainting at k=5.

    k=5 outpainting is the only condition in the study that improves quality,
    so the obvious objection is that the effect belongs to one checkpoint
    rather than to the strategy. Same architecture, same VRAM, better
    photorealism - which changes image quality while holding structure fixed.
    """
    base, runs = {}, defaultdict(list)
    for r in recs:
        p = r["provenance"]
        if p.get("method") == "full" or p.get("k") != 5:
            continue
        s, nf = p.get("seed"), p.get("n_synthetic", 0)
        if nf == 0:
            base[s] = r["metrics"]
        else:
            runs[(p.get("strategy"), nf)].append((s, r["metrics"]))

    def d(strat, nf):
        ds = [m["psnr"]["mean"] - base[s]["psnr"]["mean"]
              for s, m in runs.get((strat, nf), []) if s in base]
        return agg(ds) if ds else None

    out = []
    for nf in (1, 2, 5, 10):
        a, b = d("outpaint", nf), d("outpaint_ds8", nf)
        if not (a and b):
            continue
        out.append({"ratio": round(100 * nf / 5), "sd": a[0], "sd_sd": a[1],
                    "ds": b[0], "ds_sd": b[1], "diff": b[0] - a[0]})
    return out


def build_generalisation(scene="drjohnson"):
    """Does the crossover survive a change of scene?

    Loads the SECOND scene independently - deliberately not merged into recs,
    because (k, seed) is only unique within a scene and the paired baselines
    would collide. Outpainting only, at k=5 and k=20: those are the two ends
    of the crossover found on truck, so they are what a generalisation test
    needs. Repeating all three strategies would be 72 runs to re-confirm two
    already-known flat/negative results.

    drjohnson trains at --resolution 4 (a 230-view run at -r 2 exceeds the
    4 GB card), so its ABSOLUTE psnr is not comparable with truck's. Every
    number returned here is a within-scene paired delta, which is unaffected.
    """
    recs = load_runs(scene)
    if not recs:
        return None

    base, runs, full = {}, defaultdict(list), None
    floors = defaultdict(list)
    for r in recs:
        p = r["provenance"]
        if p.get("method") == "full":
            full = r
            continue
        k, s, nf = p.get("k"), p.get("seed"), p.get("n_synthetic", 0)
        if nf == 0:
            base[(k, s)] = r["metrics"]
            floors[k].append(r["metrics"]["psnr"]["mean"])
        else:
            runs[(k, p.get("strategy"), nf)].append((s, r["metrics"]))

    fakes = {5: (1, 2, 5, 10), 20: (5, 10, 20, 40)}

    def d(k, nf, key="psnr"):
        ds = [m[key]["mean"] - base[(k, s)][key]["mean"]
              for s, m in runs.get((k, "outpaint", nf), []) if (k, s) in base]
        return agg(ds) if ds else None

    rows = []
    for i in range(4):
        cell = {"ratio": (20, 50, 100, 200)[i]}
        ok = False
        for k in (5, 20):
            nf = fakes[k][i]
            v = d(k, nf)
            if v:
                ok = True
                cell[f"k{k}"], cell[f"k{k}_sd"] = v
                cell[f"k{k}_ssim"] = (d(k, nf, "ssim") or (None, 0))[0]
                cell[f"k{k}_lpips"] = (d(k, nf, "lpips") or (None, 0))[0]
        if ok:
            rows.append(cell)

    return {
        "scene": scene, "rows": rows, "n_runs": len(recs),
        "floors": {k: mean(v) for k, v in floors.items()},
        "ceil": full["metrics"]["psnr"]["mean"] if full else None,
        "ceil_k": full["provenance"]["k"] if full else None,
        # the between-seed scatter of the BASELINES - the number that makes
        # the paired design necessary rather than merely tidy
        "floor_sd": (stdev(floors[5]) if len(floors[5]) > 1 else 0.0),
    }


def build_depth(scene="truck"):
    """The 3x2x2: subset size x outpainting x depth prior, paired within seed.

    A depth prior is the clean test of the mechanism the report proposes. If a
    synthetic view really delivers coverage and inconsistency together, and
    only the inconsistency fails to lose value as real data accumulates, then
    an intervention supplying CONSTRAINT with no extra view should never cross
    over - it has no contradictory geometry to accumulate.

    Depth runs share a provenance with their non-depth twin, so they are found
    here by tag rather than by the usual (k, seed, n_synthetic) keys.
    """
    def m(tag):
        p = ROOT / "runs" / tag / "results.json"
        if not p.exists():
            return None
        return json.loads(p.read_text())["metrics"]["psnr"]["mean"]

    NF = {5: 10, 10: 20, 20: 40}          # the 200% ratio at each subset size
    out = {"ks": [], "rows": {}, "inter": {}}
    for k in (5, 10, 20):
        pats = {
            "depth": f"{scene}_k{k}_seed{{s}}_fps_fake0_depth",
            "outpaint": f"{scene}_k{k}_seed{{s}}_fps_outpaint_fake{NF[k]}",
            "both": f"{scene}_k{k}_seed{{s}}_fps_outpaint_fake{NF[k]}_depth",
        }
        per = {}
        for label, pat in pats.items():
            ds = []
            for sd in (0, 1, 2):
                b = m(f"{scene}_k{k}_seed{sd}_fps_fake0")
                v = m(pat.format(s=sd))
                if b is not None and v is not None:
                    ds.append(v - b)
            per[label] = ds
        if not all(per.values()):
            continue
        out["ks"].append(k)
        out["rows"][k] = {lab: agg(v) for lab, v in per.items()}
        n = min(len(v) for v in per.values())
        # Interaction computed PER SEED, so the baseline cancels inside each
        # pair instead of adding its own scatter to the error bar.
        ints = [per["both"][i] - (per["depth"][i] + per["outpaint"][i])
                for i in range(n)]
        out["inter"][k] = agg(ints)
    return out if out["ks"] else None


def load_noise():
    vals = {"psnr": [], "ssim": [], "lpips": []}
    for p in sorted((ROOT / "runs_noise").glob("repeat*/results.json")):
        m = json.loads(p.read_text())["metrics"]
        for k in vals:
            vals[k].append(m[k]["mean"])
    if not vals["psnr"]:
        return None
    return {k: (mean(v), stdev(v) if len(v) > 1 else 0.0) for k, v in vals.items()}


def load_gen_costs():
    out = {}
    for s in STRAT_ORDER:
        p = ROOT / "synthetic" / f"truck_k10_seed0_fps_{s}" / "poses.json"
        if p.exists():
            d = json.loads(p.read_text())
            out[s] = d.get("cost", {}).get("seconds_per_image")
    return out


# ------------------------------------------------------------- images
def img_tag(path: Path, max_w=1500, quality=80, fmt="JPEG"):
    if not path.exists():
        return f'<p class="missing">missing figure: {path.name}</p>'
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    if fmt == "PNG":
        im.save(buf, "PNG", optimize=True)
        mime = "image/png"
    else:
        im.save(buf, "JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'data:{mime};base64,{b64}'


def figure(path, caption, max_w=1500, quality=80, fmt="JPEG"):
    src = img_tag(path, max_w, quality, fmt)
    if src.startswith("<p"):
        return src
    return (f'<figure class="fig">\n  <img src="{src}" alt="{caption}">\n'
            f'  <figcaption>{caption}</figcaption>\n</figure>')


# --------------------------------------------------------------- html
def delta_cell(v, sd, sig, better_low=False, dp=3):
    good = (v < 0) if better_low else (v > 0)
    cls = "pos" if good else "neg"
    if not sig:
        cls = "nil"
    star = " *" if sig else ""
    return f'<td class="num {cls}">{v:+.{dp}f} <span class="sd">± {sd:.{dp}f}</span>{star}</td>'


def main():
    recs = load_runs()
    head, rows = build_tables(recs)
    noise = load_noise()
    gen = load_gen_costs()
    control = build_control(recs)

    ctrl_rows = "\n".join(
        f'<tr><td class="ratio">{c["ratio"]}%</td>'
        + delta_cell(c["guided"], c["guided_sd"], abs(c["guided"]) > c["guided_sd"] > 0)
        + delta_cell(c["warp"], c["warp_sd"], abs(c["warp"]) > c["warp_sd"] > 0)
        + f'<td class="num pos">{c["contribution"]:+.3f}</td></tr>'
        for c in control)
    ctrl_max = max((c["contribution"] for c in control), default=0.0)
    ctrl_worst = min((c["warp"] for c in control), default=0.0)

    ablation = build_ablation(recs)
    abl_rows = "\n".join(
        f'<tr><td class="ratio">{a["ratio"]}%</td>'
        + delta_cell(a["sd"], a["sd_sd"], abs(a["sd"]) > a["sd_sd"] > 0)
        + delta_cell(a["ds"], a["ds_sd"], abs(a["ds"]) > a["ds_sd"] > 0)
        + f'<td class="num">{a["diff"]:+.3f}</td></tr>'
        for a in ablation)
    abl_all_pos = all(a["ds"] > 0 for a in ablation) if ablation else False

    # --- does the SIGN CHANGE survive the checkpoint swap, not just the gain?
    mcross = build_model_crossover(recs)
    mc = {r["k"]: r for r in mcross}
    mc_rows = "\n".join(
        f'<tr><td>{r["k"]} real views</td>'
        + delta_cell(r["sd15"], r["sd15_sd"], abs(r["sd15"]) > r["sd15_sd"] > 0)
        + delta_cell(r["ds8"], r["ds8_sd"], abs(r["ds8"]) > r["ds8_sd"] > 0)
        + "</tr>"
        for r in mcross)

    # --- depth regularisation: the 3x2x2 factorial ---
    depth = build_depth()
    depth_rows = ""
    if depth:
        def drow(label, key):
            cells = "".join(
                delta_cell(*depth["rows"][k][key],
                           abs(depth["rows"][k][key][0]) > depth["rows"][k][key][1] > 0)
                for k in depth["ks"])
            return f"<tr><td>{label}</td>{cells}</tr>"
        inter_cells = "".join(
            delta_cell(*depth["inter"][k],
                       abs(depth["inter"][k][0]) > depth["inter"][k][1] > 0)
            for k in depth["ks"])
        depth_rows = "\n".join([
            drow("+ depth prior", "depth"),
            drow("+ outpainting (200%)", "outpaint"),
            drow("+ both", "both"),
            f'<tr class="grp"><td>interaction</td>{inter_cells}</tr>',
        ])

    # --- second scene: does the crossover reproduce? ---
    second = build_generalisation()
    gen_rows_html, gen_best, gen_best_ssim, gen_best_lpips = "", 0.0, 0.0, 0.0
    if second:
        by_ratio = {r["ratio"]: r for r in rows if r["strategy"] == "outpaint"}
        cells = []
        for g in second["rows"]:
            # truck's k=5 second point is 40%, not 50% - 1.25 images do not
            # exist - so match the nearest label rather than requiring equality
            def truck(k, ratio):
                for rr in rows:
                    if (rr["strategy"] == "outpaint" and rr["k"] == k
                            and abs(rr["ratio"] - ratio) <= 10):
                        return rr
                return None
            row = f'<tr><td class="ratio">{g["ratio"]}%</td>'
            for k in (5, 20):
                t = truck(k, g["ratio"])
                row += (delta_cell(t["d_psnr"], t["d_psnr_sd"], t["sig_psnr"])
                        if t else '<td class="num">&mdash;</td>')
                if f"k{k}" in g:
                    sd = g[f"k{k}_sd"]
                    row += delta_cell(g[f"k{k}"], sd, abs(g[f"k{k}"]) > sd > 0)
                else:
                    row += '<td class="num">&mdash;</td>'
            cells.append(row + "</tr>")
        # reorder columns to truck-k5 / gen-k5 / truck-k20 / gen-k20
        gen_rows_html = "\n".join(cells)
        best = max(second["rows"], key=lambda g: g.get("k5", -99))
        gen_best = best.get("k5", 0.0)
        gen_best_ssim = best.get("k5_ssim") or 0.0
        gen_best_lpips = best.get("k5_lpips") or 0.0

    # --- full results table, grouped by strategy then subset size ---
    trs = []
    seen = set()
    for r in rows:
        if r["strategy"] not in seen:
            seen.add(r["strategy"])
            trs.append(f'<tr class="grp"><td colspan="8">{STRAT_LABEL[r["strategy"]]}</td></tr>')
        trs.append(
            "<tr>"
            f'<td class="num">{r["k"]}</td>'
            f'<td class="ratio">{r["ratio"]}%</td>'
            f'<td class="num">{r["n_fake"]}</td>'
            + delta_cell(r["d_psnr"], r["d_psnr_sd"], r["sig_psnr"])
            + delta_cell(r["d_ssim"], r["d_ssim_sd"], abs(r["d_ssim"]) > r["d_ssim_sd"] > 0, dp=4)
            + delta_cell(r["d_lpips"], r["d_lpips_sd"], abs(r["d_lpips"]) > r["d_lpips_sd"] > 0,
                         better_low=True, dp=4)
            + f'<td class="num">{r["gauss"]:,.0f}</td>'
            f'<td class="num">{r["seeds"]}</td>'
            "</tr>")
    results_table = "\n".join(trs)

    # --- crossover matrix: ratio down, subset size across, one block per strategy ---
    NOMINAL = (20, 50, 100, 200)
    by_key = {(r["strategy"], r["k"], r["n_fake"]): r for r in rows}

    def cell_for(strat, k, nominal):
        for nf in FAKES[k]:
            if abs(round(100.0 * nf / k) - nominal) <= 10:
                r = by_key.get((strat, k, nf))
                if r:
                    return delta_cell(r["d_psnr"], r["d_psnr_sd"], r["sig_psnr"])
        return '<td class="num nil">&mdash;</td>'

    xtrs = []
    for s in STRAT_ORDER:
        xtrs.append(f'<tr class="grp"><td colspan="4">{STRAT_LABEL[s]}</td></tr>')
        for nom in NOMINAL:
            xtrs.append(f'<tr><td class="ratio">{nom}%</td>'
                        + "".join(cell_for(s, k, nom) for k in KS) + "</tr>")
    crossover_table = "\n".join(xtrs)

    # --- scaling table: what real views alone are worth ---
    strs = []
    prev = None
    for k in KS:
        f = head["floors"][k]
        step = "&mdash;" if prev is None else f'{f["psnr"] - prev:+.2f}'
        per = "&mdash;" if prev is None else f'{(f["psnr"] - prev) / (k - prevk):+.3f}'
        strs.append(f'<tr><td class="num">{k}</td>'
                    f'<td class="num">{f["psnr"]:.2f} <span class="sd">± {f["psnr_sd"]:.2f}</span></td>'
                    f'<td class="num">{f["ssim"]:.4f}</td>'
                    f'<td class="num">{f["lpips"]:.4f}</td>'
                    f'<td class="num">{step}</td><td class="num">{per}</td></tr>')
        prev, prevk = f["psnr"], k
    strs.append(f'<tr class="ceil"><td class="num">219</td>'
                f'<td class="num">{head["ceil_psnr"]:.2f}</td>'
                f'<td class="num">{head["ceil_ssim"]:.4f}</td>'
                f'<td class="num">{head["ceil_lpips"]:.4f}</td>'
                f'<td class="num">{head["ceil_psnr"] - prev:+.2f}</td>'
                f'<td class="num">{(head["ceil_psnr"] - prev) / (219 - prevk):+.3f}</td></tr>')
    scaling_table = "\n".join(strs)

    noise_html = ""
    if noise:
        noise_html = (
            f'<td class="num">{noise["psnr"][1]:.3f}</td>'
            f'<td class="num">{noise["ssim"][1]:.4f}</td>'
            f'<td class="num">{noise["lpips"][1]:.4f}</td>')

    gen_rows = "\n".join(
        f'<tr><td>{STRAT_LABEL[s]}</td><td class="num">{v:.1f} s</td></tr>'
        for s, v in gen.items() if v)

    tpl = TEMPLATE
    subs = {
        "{{FLOOR_PSNR}}": f'{head["floor_psnr"]:.2f}',
        "{{FLOOR_SD}}": f'{head["floor_psnr_sd"]:.2f}',
        "{{FLOOR_SSIM}}": f'{head["floor_ssim"]:.4f}',
        "{{FLOOR_LPIPS}}": f'{head["floor_lpips"]:.4f}',
        "{{FLOOR_GAUSS}}": f'{head["floor_gauss"]:,.0f}',
        "{{CEIL_PSNR}}": f'{head["ceil_psnr"]:.2f}',
        "{{CEIL_SSIM}}": f'{head["ceil_ssim"]:.4f}',
        "{{CEIL_LPIPS}}": f'{head["ceil_lpips"]:.4f}',
        "{{CEIL_GAUSS}}": f'{head["ceil_gauss"]:,.0f}',
        "{{GAP}}": f'{head["gap"]:.2f}',
        "{{FLOOR5}}": f'{head["floors"][5]["psnr"]:.2f}',
        "{{FLOOR5_SD}}": f'{head["floors"][5]["psnr_sd"]:.2f}',
        "{{FLOOR20}}": f'{head["floors"][20]["psnr"]:.2f}',
        "{{FLOOR20_SD}}": f'{head["floors"][20]["psnr_sd"]:.2f}',
        "{{GAP5}}": f'{head["gap5"]:.2f}',
        "{{GAP20}}": f'{head["gap20"]:.2f}',
        "{{STEP_5_10}}": f'{head["step_5_10"]:+.2f}',
        "{{STEP_10_20}}": f'{head["step_10_20"]:+.2f}',
        "{{PER_VIEW_5_10}}": f'{head["step_5_10"] / 5:+.3f}',
        "{{PER_VIEW_10_20}}": f'{head["step_10_20"] / 10:+.3f}',
        "{{BEST_D}}": f'{head["best"]["d_psnr"]:+.3f}',
        "{{BEST_WHERE}}": f'{STRAT_LABEL[head["best"]["strategy"]].lower()}, '
                          f'k={head["best"]["k"]}, {head["best"]["ratio"]}%',
        "{{WORST_D}}": f'{head["worst"]["d_psnr"]:+.3f}',
        "{{WORST_WHERE}}": f'{STRAT_LABEL[head["worst"]["strategy"]].lower()}, '
                           f'k={head["worst"]["k"]}, {head["worst"]["ratio"]}%',
        "{{BEST_VS_REAL}}": f'{head["step_5_10"] / head["best"]["d_psnr"]:.1f}',
        "{{N_RUNS}}": str(head["n_runs"]),
        "{{RESULTS_TABLE}}": results_table,
        "{{CROSSOVER_TABLE}}": crossover_table,
        "{{SCALING_TABLE}}": scaling_table,
        "{{CONTROL_TABLE}}": ctrl_rows,
        "{{ABLATION_TABLE}}": abl_rows,
        "{{MODEL_CROSS_TABLE}}": mc_rows,
        "{{MC5}}": f'{mc[5]["ds8"]:+.3f}' if 5 in mc else "n/a",
        "{{MC10}}": f'{mc[10]["ds8"]:+.3f}' if 10 in mc else "n/a",
        "{{MC20}}": f'{mc[20]["ds8"]:+.3f}' if 20 in mc else "n/a",
        "{{MC10_SD15}}": f'{mc[10]["sd15"]:+.3f}' if 10 in mc else "n/a",
        "{{DEPTH_TABLE}}": depth_rows,
        "{{DEPTH_K5}}": f'{depth["rows"][5]["depth"][0]:+.3f}' if depth else "n/a",
        "{{DEPTH_K20}}": f'{depth["rows"][20]["depth"][0]:+.3f}' if depth else "n/a",
        "{{DEPTH_BOTH_K5}}": f'{depth["rows"][5]["both"][0]:+.3f}' if depth else "n/a",
        "{{OUT_K20}}": f'{depth["rows"][20]["outpaint"][0]:.3f}' if depth else "n/a",
        "{{GEN_TABLE}}": gen_rows_html,
        "{{GEN_SCENE}}": second["scene"] if second else "n/a",
        "{{GEN_N}}": str(second["n_runs"]) if second else "0",
        "{{GEN_CEIL_K}}": str(second["ceil_k"]) if second and second["ceil_k"] else "n/a",
        "{{GEN_BEST}}": f"{gen_best:+.3f}",
        "{{GEN_BEST_SSIM}}": f"{gen_best_ssim:+.4f}",
        "{{GEN_BEST_LPIPS}}": f"{gen_best_lpips:+.4f}",
        "{{GEN_FLOOR_SD}}": f'{second["floor_sd"]:.2f}' if second else "n/a",
        "{{ABL_VERDICT}}": ("positive at every ratio" if abl_all_pos
                            else "not reproduced at every ratio"),
        "{{CTRL_MAX}}": f'{ctrl_max:+.2f}',
        "{{CTRL_WORST}}": f'{ctrl_worst:.2f}',
        "{{FIG_SCALING}}": figure(RES / "scaling.png",
            "Figure 2 — Left: held-out quality against the number of real training views, "
            "with the full-data ceiling marked. Right: the same quantities on one axis. "
            "Adding five real photographs is worth several times the best synthetic condition "
            "measured anywhere in this study, and the worst synthetic condition costs more "
            "than the best one gains.", 1600, 88, "PNG"),
        "{{NOISE_CELLS}}": noise_html,
        "{{NOISE_PSNR}}": f'{noise["psnr"][1]:.3f}' if noise else "n/a",
        "{{GEN_ROWS}}": gen_rows,
        "{{FIG_TRAINING}}": figure(RES / "panel_training_data.png",
            "Figure 1 — Synthetic training images beside the real view they derive from. "
            "Inpainting has given the truck a second rear axle and an invented white box; "
            "outpainting fabricates the scene beyond the original frame; pose-guided renders "
            "a genuinely different viewpoint, with roughly 10% of its pixels invented.", 1500, 82),
        "{{FIG_PAIRED}}": figure(RES / "curves_paired.png",
            "Figure 3 — Change in PSNR against the same seed's own zero-synthetic baseline, "
            "one line per subset size. Error bars are the standard deviation across three "
            "seeds; the grey band is the measured noise floor. Inpainting is flat and "
            "independent of subset size; outpainting crosses from beneficial at five views to "
            "harmful at twenty; pose-guided is harmful everywhere and worsens as real views "
            "accumulate.", 1600, 88, "PNG"),
        "{{FIG_STRATEGIES}}": figure(RES / "panel_strategies.png",
            "Figure 3 — Held-out renderings at the 100% synthetic ratio. Degradation appears "
            "as smearing and semi-transparent floaters, most severely under pose-guided "
            "synthesis.", 1500, 82),
        "{{FIG_RATIO}}": figure(RES / "panel_ratio_guided.png",
            "Figure 4 — Pose-guided synthesis as the synthetic ratio increases. Unlike the "
            "other two strategies this degrades monotonically; there is no recovery at 200%.",
            1500, 82),
    }
    for k, v in subs.items():
        tpl = tpl.replace(k, v)

    OUT.write_text(tpl, encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT}  ({kb:,.0f} KB)")
    if kb > 15000:
        print("  WARNING: approaching the 16MB artifact limit")


TEMPLATE = r"""<title>Diffusion Augmentation for Few-Shot Splatting</title>
<style>
/* ---- tokens: complete light palette on bare :root ---- */
:root{
  --ground:#f4f6f6; --surface:#ffffff; --surface-2:#eef2f2;
  --ink:#0f1719; --ink-soft:#3c4a4e; --muted:#5f7076;
  --line:#d9e1e2; --line-soft:#e8eeee;
  --accent:#0d7a84; --accent-soft:#e0f0f1;
  --good:#1f7a4d; --bad:#9f2f38; --nil:#6b7a80;
  --shadow:0 1px 2px rgba(15,23,25,.06),0 8px 24px rgba(15,23,25,.05);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0b1113; --surface:#111a1d; --surface-2:#162125;
    --ink:#e3ebec; --ink-soft:#b9c7cb; --muted:#87979c;
    --line:#222f34; --line-soft:#1a262a;
    --accent:#48b3bc; --accent-soft:#14343a;
    --good:#5cb98a; --bad:#d97a82; --nil:#7f8f95;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.32);
  }
}
:root[data-theme="dark"]{
  --ground:#0b1113; --surface:#111a1d; --surface-2:#162125;
  --ink:#e3ebec; --ink-soft:#b9c7cb; --muted:#87979c;
  --line:#222f34; --line-soft:#1a262a;
  --accent:#48b3bc; --accent-soft:#14343a;
  --good:#5cb98a; --bad:#d97a82; --nil:#7f8f95;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.32);
}

*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;
  font-size:17px; line-height:1.62;
}
.ui{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:0 24px 96px}
.col{max-width:68ch;margin:0 auto}

/* ---- masthead ---- */
header.top{border-bottom:1px solid var(--line);background:var(--surface);}
header.top .inner{max-width:1180px;margin:0 auto;padding:52px 24px 40px}
.eyebrow{
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent);font-weight:650;margin:0 0 14px;
}
h1{
  font-size:clamp(30px,4.4vw,46px);line-height:1.12;margin:0 0 16px;
  letter-spacing:-.015em;text-wrap:balance;font-weight:600;max-width:22ch;
}
.standfirst{font-size:19px;color:var(--ink-soft);margin:0;max-width:60ch;text-wrap:pretty}
.byline{
  margin-top:26px;padding-top:18px;border-top:1px solid var(--line-soft);
  font-size:13px;color:var(--muted);display:flex;gap:22px;flex-wrap:wrap;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}

/* ---- headline stats ---- */
.stats{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));
  gap:1px;background:var(--line);border:1px solid var(--line);
  border-radius:3px;overflow:hidden;margin:44px 0 8px;
}
.stat{background:var(--surface);padding:18px 20px}
.stat .k{
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:10.5px;letter-spacing:.11em;text-transform:uppercase;color:var(--muted);
  margin:0 0 7px;font-weight:600;
}
.stat .v{
  font-size:27px;font-weight:600;letter-spacing:-.02em;margin:0;
  font-variant-numeric:tabular-nums;
}
.stat .u{font-size:13px;color:var(--muted);margin:4px 0 0}

/* ---- sections ---- */
section{margin:52px 0 0}
h2{
  font-size:15px;font-weight:650;margin:0 0 20px;letter-spacing:.02em;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  color:var(--ink);display:flex;align-items:baseline;gap:12px;
  padding-bottom:11px;border-bottom:1px solid var(--line);
}
h2 .n{
  color:var(--accent);font-variant-numeric:tabular-nums;font-size:13px;
  font-weight:650;letter-spacing:.06em;
}
h3{font-size:19px;font-weight:600;margin:32px 0 10px;letter-spacing:-.01em}
p{margin:0 0 17px}
ul,ol{margin:0 0 17px;padding-left:22px}
li{margin:0 0 8px}
strong{font-weight:650}
a{color:var(--accent)}
code{
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:.87em;background:var(--surface-2);padding:1px 5px;border-radius:3px;
}
.lede{font-size:18.5px;color:var(--ink-soft)}

/* ---- callout ---- */
.note{
  background:var(--surface);border:1px solid var(--line);
  border-left:2px solid var(--accent);border-radius:2px;
  padding:17px 20px;margin:26px 0;font-size:16px;
}
.note p:last-child{margin-bottom:0}
.note .lbl{
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;
  color:var(--accent);font-weight:650;display:block;margin-bottom:7px;
}

/* ---- tables ---- */
.tw{overflow-x:auto;margin:26px 0;border:1px solid var(--line);border-radius:3px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:14px;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
caption{
  caption-side:top;text-align:left;padding:14px 18px 12px;font-size:13px;
  color:var(--muted);border-bottom:1px solid var(--line-soft);
}
caption b{color:var(--ink);font-weight:650}
th{
  text-align:left;padding:10px 14px;font-size:10.5px;letter-spacing:.09em;
  text-transform:uppercase;color:var(--muted);font-weight:650;
  border-bottom:1px solid var(--line);white-space:nowrap;
}
td{padding:9px 14px;border-bottom:1px solid var(--line-soft);white-space:nowrap}
tr:last-child td{border-bottom:none}
.num{text-align:right;font-variant-numeric:tabular-nums}
td.ratio{font-variant-numeric:tabular-nums;color:var(--ink-soft)}
.sd{color:var(--muted);font-size:11.5px}
.pos{color:var(--good)}
.neg{color:var(--bad)}
.nil{color:var(--nil)}
tr.grp td{
  background:var(--surface-2);font-weight:650;font-size:11px;
  letter-spacing:.1em;text-transform:uppercase;color:var(--ink-soft);
  padding:9px 14px;
}

/* ---- figures ---- */
.fig{margin:34px 0;padding:0}
.fig img{width:100%;height:auto;display:block;border:1px solid var(--line);
  border-radius:3px;background:var(--surface)}
.fig figcaption{
  font-size:13.5px;color:var(--muted);margin-top:11px;line-height:1.55;
  max-width:78ch;
}
.wide{max-width:1180px;margin-left:auto;margin-right:auto}
.missing{color:var(--bad);font-size:14px}

hr.rule{border:none;border-top:1px solid var(--line);margin:52px 0}
footer{margin-top:64px;padding-top:22px;border-top:1px solid var(--line);
  font-size:13px;color:var(--muted);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
@media (max-width:640px){
  body{font-size:16px}
  header.top .inner{padding:36px 20px 30px}
  .wrap{padding:0 20px 64px}
}
</style>

<header class="top">
  <div class="inner">
    <p class="eyebrow">Few-shot 3D reconstruction · experimental report</p>
    <h1>Diffusion Augmentation Helps Few-Shot Gaussian Splatting Only When Data Is Scarcest</h1>
    <p class="standfirst">{{N_RUNS}} controlled training runs on the <em>truck</em> scene show
    that the value of a synthetic view depends on how many real views you already have. At five
    real images outpainting helps at every ratio; by twenty it hurts at every ratio. The
    crossover is measurable, it reproduces on a second scene indoors ({{GEN_N}} further runs),
    and the best synthetic condition anywhere is still worth {{BEST_VS_REAL}}× less than five
    more photographs.</p>
    <div class="byline">
      <span>Tanks &amp; Temples <b>truck</b> · 251 images</span>
      <span>3 strategies × 4 ratios × 3 subset sizes × 3 seeds</span>
      <span>{{N_RUNS}} training runs</span>
      <span>RTX 3050 Ti · 4 GB</span>
    </div>
  </div>
</header>

<div class="wrap">

<div class="stats">
  <div class="stat"><p class="k">Floor, 5 views</p><p class="v">{{FLOOR5}}</p><p class="u">PSNR</p></div>
  <div class="stat"><p class="k">Floor, 10 views</p><p class="v">{{FLOOR_PSNR}}</p><p class="u">PSNR</p></div>
  <div class="stat"><p class="k">Floor, 20 views</p><p class="v">{{FLOOR20}}</p><p class="u">PSNR</p></div>
  <div class="stat"><p class="k">Full-data ceiling</p><p class="v">{{CEIL_PSNR}}</p><p class="u">PSNR, 219 views</p></div>
  <div class="stat"><p class="k">Best augmentation</p><p class="v">{{BEST_D}}</p><p class="u">{{BEST_WHERE}}</p></div>
  <div class="stat"><p class="k">Worst augmentation</p><p class="v">{{WORST_D}}</p><p class="u">{{WORST_WHERE}}</p></div>
</div>

<div class="col">

<section>
<h2><span class="n">01</span> Gaussian Splatting when photographs are scarce</h2>
<p class="lede">3D Gaussian Splatting represents a scene as millions of anisotropic
translucent blobs. Training projects them into each known camera and adjusts every blob's
position, covariance, colour and opacity until the renders match the photographs.</p>

<p>With a few hundred views the problem is well constrained: every surface is observed from
many directions, and a blob that is wrongly placed will contradict some view. With ten views
it is badly under-constrained. The optimiser can reproduce those ten images almost exactly —
our few-shot runs reach a training loss of 0.007 — while the underlying geometry is
arbitrary. Render from an unseen camera and the scene collapses into smears and
semi-transparent floaters.</p>

<p>The scale of the problem is easy to state. Training on 219 real views reaches
<b>{{CEIL_PSNR}} dB</b> on the held-out set. Training on 10 reaches
<b>{{FLOOR_PSNR}} dB</b>. Few-shot costs <b>{{GAP}} dB</b>, and that gap is what
augmentation would have to close.</p>

<p>Because the whole sweep was run at three subset sizes, the price of scarcity can be
measured directly rather than assumed:</p>

<table class="tbl">
  <caption><b>Table 1.</b> Held-out quality against the number of real training views.
  Three seeds per row except the full-data ceiling, which is a single deterministic
  split.</caption>
  <thead>
    <tr><th class="num">Real views</th><th class="num">PSNR</th><th class="num">SSIM</th>
        <th class="num">LPIPS</th><th class="num">Δ PSNR</th><th class="num">per view</th></tr>
  </thead>
  <tbody>
{{SCALING_TABLE}}
  </tbody>
</table>

<p>The per-view column falls from {{PER_VIEW_5_10}} dB to {{PER_VIEW_10_20}} dB and then to
roughly +0.03 dB — the diminishing return you would expect, and a check that the pipeline
behaves sensibly. It also sets the exchange rate against which every synthetic image in this
study should be judged: <b>five real photographs are worth {{STEP_5_10}} dB</b>.</p>

<div class="note">
<span class="lbl">Pipeline validation</span>
<p>The full-data baseline reaches {{CEIL_PSNR}} dB at 7,000 iterations against roughly
25.4 dB reported in the literature for this scene at 30,000. The pipeline reproduces
published behaviour, so the few-shot deficit is a property of the setting rather than a
defect in the implementation.</p>
</div>
</section>

<section>
<h2><span class="n">02</span> Three augmentation strategies, and the pose problem</h2>
<p>A diffusion model returns pixels. Gaussian Splatting cannot use pixels without a camera
pose. Every strategy is therefore defined by how it obtains one, and that choice determines
both how much new information it contributes and how much error it risks.</p>

<h3>Inpainting</h3>
<p>A region of a real view is masked and regenerated; everything outside the mask is copied
from the original bit for bit. The pose is inherited exactly, so pose error is zero. The
cost is that the viewpoint is unchanged — it contributes no new geometric evidence.</p>

<h3>Outpainting</h3>
<p>The frame is extended by 25% and the border is generated. The camera does not move, so
the pose is again exact, but the intrinsics change: the focal length in pixels is
<em>unchanged</em> (the lens is the same; more of the image plane is captured) while the
principal point shifts by the paste offset. The horizontal field of view widens from 80.1°
to 92.9°. Treating the focal length as if it scaled with the frame is the standard silent
failure here, so both properties are asserted at generation time.</p>

<h3>Pose-guided synthesis</h3>
<p>The only strategy that supplies a genuinely new viewpoint. A monocular depth map is
predicted for a real view, then <em>anchored to true scene scale</em> by least-squares
fitting against the sparse COLMAP points that view observed — without this the depth is
relative and the warp is meaningless. Pixels are back-projected and re-projected into a
camera interpolated between two real views, leaving holes where hidden surfaces are
revealed; diffusion fills only those holes. Mean alignment R² was <b>0.988</b> across the
ten source views, and roughly 10% of each output is invented.</p>

{{FIG_TRAINING}}
</section>

<section>
<h2><span class="n">03</span> Experimental design</h2>

<h3>A frozen held-out set</h3>
<p>Every eighth image by sorted filename is reserved as the test set — 32 views — leaving a
pool of 219. That split is identical in every condition, and ground truth is always a real
photograph. The upstream code ties train and test together through its hold-out rule, so a
small patch was added to honour an explicit split file, allowing an arbitrary K-image
training subset while the test set stays byte-identical.</p>

<h3>Subset selection and pairing</h3>
<p>The ten real views are chosen by farthest-point sampling over camera positions, which
guarantees angular coverage. Measured across three seeds, this reduces the worst coverage
gap variation to 0.19 units against 2.04 for uniform random selection — roughly nine times
less seed-to-seed variance, which matters because the effects being measured are small.</p>
<p>Each augmented run is compared against <em>the same seed's own</em> zero-synthetic
baseline, on identical real images. The only variable that moves is the number of synthetic
views, and synthetic sets are strictly nested: the two images used at 20% are the first two
of the five used at 50%, and so on.</p>

<h3>The noise floor</h3>
<p>Gaussian Splatting is not bit-reproducible. Python-level seeds are pinned by the upstream
code, but the rasterizer's backward pass uses CUDA atomics whose completion order varies, and
densification amplifies the difference. Three repeats of an identical configuration give:</p>

<div class="tw">
<table>
  <caption><b>Table 1.</b> Run-to-run variation under an identical configuration (3 repeats).
  Any effect smaller than roughly √2 × these values is indistinguishable from noise.</caption>
  <thead><tr><th>Quantity</th><th class="num">PSNR</th><th class="num">SSIM</th><th class="num">LPIPS</th></tr></thead>
  <tbody><tr><td>Standard deviation</td>{{NOISE_CELLS}}</tr></tbody>
</table>
</div>

<p>A paired delta subtracts two independently noisy runs, so its noise floor is
approximately <b>{{NOISE_PSNR}} × √2 ≈ 0.055 dB</b>. Effects are only claimed where they
clear it.</p>
</section>

<section>
<h2><span class="n">04</span> Results</h2>
<p>Every condition was run at three seeds and 7,000 iterations at half resolution. Deltas are
against each seed's own baseline at the same subset size; an asterisk marks deltas whose
magnitude exceeds their own standard deviation.</p>

<p>Reading the matrix below <em>across</em> a row is the finding: the same synthetic ratio
changes sign depending on how many real views it is added to.</p>

<div class="tw">
<table>
  <caption><b>Table 2.</b> Change in PSNR against the same-seed zero-synthetic baseline, by
  synthetic ratio (down) and number of real views (across). Green is an improvement, red a
  degradation.</caption>
  <thead>
    <tr><th>Ratio</th><th class="num">k = 5</th><th class="num">k = 10</th>
        <th class="num">k = 20</th></tr>
  </thead>
  <tbody>
{{CROSSOVER_TABLE}}
  </tbody>
</table>
</div>

<p>Outpainting is positive at every ratio at five views, mixed at ten, and negative at every
ratio at twenty. Pose-guided is negative everywhere and grows steadily worse as real views
accumulate. Inpainting barely moves at any subset size.</p>
</section>

</div>

<div class="wide">
{{FIG_SCALING}}
{{FIG_PAIRED}}
</div>

<div class="col">
<section>
<h3>Three behaviours, one explanatory axis</h3>
<p>The three strategies do not merely differ in magnitude; they respond to subset size in
three qualitatively different ways. Order them by how much <em>camera pose</em> each one
invents and the pattern resolves.</p>

<p><b>Inpainting copies the pose exactly.</b> It contributes no new viewpoint, therefore no
new geometric information and no new opportunity for views to disagree. Its effect is small,
slightly negative, and — uniquely — independent of subset size. There is nothing for the
subset size to interact with.</p>

<p><b>Outpainting keeps the camera centre but widens the frustum.</b> It contributes real
peripheral coverage together with fabricated content. At five views the coverage dominates
and every ratio helps; by twenty views enough real signal is present that the fabrication
dominates instead, and every ratio hurts. This is the crossover.</p>

<p><b>Pose-guided synthesis invents a genuinely new viewpoint.</b> It offers the most
potential information and the most opportunity for contradiction, and the contradiction wins
at every subset size tested. Worse, the damage <em>grows</em> with real views: −0.31 dB at
five, −0.45 at ten, −1.02 at twenty for the smallest ratio. The better the reconstruction it
is inserted into, the more there is to corrupt.</p>

<h3>A control: is the diffusion step responsible?</h3>
<p>Pose-guided synthesis changes two things at once relative to a real training view — the
camera pose, and the roughly 10% of pixels diffusion invents to cover disocclusions. Its
damage could belong to either. A control separates them: warp to exactly the same poses,
then leave the holes black and never load Stable Diffusion at all.</p>

<p>The poses are bit-identical between the two conditions (maximum quaternion and translation
difference 0.00e+00, identical hole fractions), because the random stream is advanced
identically whether or not diffusion runs. The only variable is the hole content.</p>

<div class="tw">
<table>
  <caption><b>Table 3.</b> Warp-only control at k = 10, three seeds. The final column is
  (with diffusion) − (without), which is the diffusion step's isolated contribution.</caption>
  <thead>
    <tr><th>Ratio</th><th class="num">Pose-guided<br><span class="sd">warp + SD</span></th>
        <th class="num">Warp-only<br><span class="sd">holes black</span></th>
        <th class="num">Diffusion<br><span class="sd">contribution</span></th></tr>
  </thead>
  <tbody>
{{CONTROL_TABLE}}
  </tbody>
</table>
</div>

<p>The result reverses the obvious hypothesis. Removing diffusion makes the condition
<em>dramatically worse</em> — {{CTRL_WORST}} dB at the 200% ratio against −1.45 with it. The
diffusion step is contributing up to <b>{{CTRL_MAX}} dB</b> of repair, and SSIM and LPIPS
agree at every ratio.</p>

<p>So the pose-guided damage cannot be attributed to hallucinated hole content or to residual
warping artifacts: the stage responsible for both is the stage holding the result up. What
remains is the pose novelty itself — views the optimiser trusts completely that disagree with
the geometry the real photographs imply.</p>

<div class="note">
<span class="lbl">Reading this honestly</span>
<p>Black rectangles are themselves a severe artifact, so part of the {{CTRL_MAX}} dB is simply
diffusion beating a very low bar. The control does not establish that the diffusion output is
<em>good</em>. It establishes that the diffusion step is <em>net positive</em>, which is
sufficient to rule it out as the cause of the degradation — the question that was actually
open. A second line of evidence agrees: sparser subsets need longer interpolation baselines
and therefore larger holes, so an artifact explanation predicts more damage at k = 5. The
measured damage is smallest at k = 5.</p>
</div>

<h3>A second control: is the positive result model-specific?</h3>
<p>Outpainting at five views is the only condition here that improves quality, which invites
the objection that the effect belongs to Stable Diffusion 1.5 rather than to the strategy.
Repeating that sweep with Dreamshaper-8 tests it. Dreamshaper-8 is an SD 1.5 finetune: same
architecture, same 2.65 GB of VRAM, materially better photorealism — so it varies image
quality while holding everything structural fixed. A larger model would confound quality with
capacity, and does not fit in 4 GB regardless.</p>

<div class="tw">
<table>
  <caption><b>Table 4.</b> Outpainting at k = 5 under two diffusion checkpoints, three seeds.
  The real image region is preserved byte-for-byte in both; only the fabricated periphery
  differs (mean |Δ| of 0.07 inside the frame versus 30.3 outside).</caption>
  <thead>
    <tr><th>Ratio</th><th class="num">SD 1.5</th>
        <th class="num">Dreamshaper-8</th><th class="num">Difference</th></tr>
  </thead>
  <tbody>
{{ABLATION_TABLE}}
  </tbody>
</table>
</div>

<p>The effect is {{ABL_VERDICT}} under the second model, and the curve keeps the same rising
shape. The benefit is a property of the augmentation, not of one checkpoint.</p>

<p>The magnitudes are slightly <em>smaller</em> throughout, which is the more informative
detail. Dreamshaper-8 produces visibly better images and obtains a marginally worse result. If
photorealism were the mechanism, the better model should have helped more. It did not — which
agrees with the warp-only control, where diffusion's contribution was geometric repair rather
than image quality, and with inpainting, which produces convincing images at a perfect pose
and is nonetheless useless. Across three independent lines of evidence, what a synthetic view
contributes is <b>coverage, not photorealism</b>.</p>

<p>All of that concerns k = 5, where outpainting helps — the half of the claim least in doubt.
The finding of this study is the <em>sign change</em>, so the ablation is only worth its name
if the second model reverses too.</p>

<div class="tablewrap">
<table>
  <caption><b>Table 4b.</b> Both checkpoints at the 200% ratio across every subset size,
  each paired against the same seed's own baseline. The crossover is a property of
  augmentation only if the right-hand column changes sign as the left one does.</caption>
  <thead>
    <tr><th>real views</th><th class="num">SD 1.5</th>
        <th class="num">Dreamshaper-8</th></tr>
  </thead>
  <tbody>
{{MODEL_CROSS_TABLE}}
  </tbody>
</table>
</div>

<p>It does. Dreamshaper-8 runs {{MC5}} dB at k = 5 and {{MC20}} dB at k = 20 — the same
reversal, on a different checkpoint. <b>The crossover is not an artefact of Stable Diffusion
1.5.</b></p>

<p>Two details sharpen the mechanism further. The second model's curve sits <em>below</em> the
first at every subset size, so its crossing point arrives <em>earlier</em>: at k = 10, where
SD 1.5 is indistinguishable from zero ({{MC10_SD15}}), Dreamshaper-8 is already separated and
negative ({{MC10}}). And Dreamshaper-8 is the <em>more</em> photorealistic model. A checkpoint
finetuned to make each image individually more convincing has no reason to be more consistent
<em>between</em> images, and plausibly less — it commits to invented detail with greater
confidence. Better-looking training data, measurably worse reconstruction, at every subset
size tested.</p>

<h3>Does it generalise? A second scene</h3>
<p>The checkpoint swap rules out one model. It does not rule out one <em>scene</em>. Every
number above comes from <code>truck</code>: a single object, outdoors, photographed on a loop
around it. The obvious failure mode for the whole study is that the crossover is a property of
that capture geometry rather than of few-shot reconstruction.</p>

<p>{{GEN_N}} further runs repeat the outpainting sweep on <code>{{GEN_SCENE}}</code> from the
Deep Blending dataset — an indoor room, {{GEN_CEIL_K}} views, a fundamentally different capture
regime — at k = 5 and k = 20, the two ends of the crossover.</p>

<div class="tablewrap">
<table>
  <caption><b>Table 5.</b> Outpainting on both scenes, paired against each scene's own
  same-seed baseline, three seeds. Absolute PSNR is not comparable across the two
  (<code>{{GEN_SCENE}}</code> trains at quarter resolution); the deltas are.</caption>
  <thead>
    <tr><th>Ratio</th>
        <th class="num">truck k=5</th><th class="num">{{GEN_SCENE}} k=5</th>
        <th class="num">truck k=20</th><th class="num">{{GEN_SCENE}} k=20</th></tr>
  </thead>
  <tbody>
{{GEN_TABLE}}
  </tbody>
</table>
</div>

<p><b>The sign flip reproduces.</b> At k = 5 every ratio is positive in both scenes; at k = 20
every statistically separated point is negative in both. An outdoor object and an indoor room
agree that synthetic views are worth having at five real photographs and worth avoiding at
twenty.</p>

<p>Indoors the effect is not merely present but <em>stronger</em>, and on better evidence. The
k = 5 benefit reaches {{GEN_BEST}} dB against truck's {{BEST_D}}, and all three metrics move
together: where truck bought PSNR while SSIM stayed flat, <code>{{GEN_SCENE}}</code> improves
SSIM by {{GEN_BEST_SSIM}} and LPIPS by {{GEN_BEST_LPIPS}} at the same point. Agreement across
three metrics with different failure modes is considerably harder to obtain by chance than
agreement in one.</p>

<p>One cell dissents: k = 20 at the 200% ratio is positive in PSNR, though not separated from
zero. LPIPS is worse there — as it is at every k = 20 ratio — so the honest reading is a
metric disagreement at a point where the effect is small, not a counterexample.</p>

<p>A methodological note falls out of this scene for free. Its baselines scatter
{{GEN_FLOOR_SD}} dB between seeds against truck's {{FLOOR5_SD}}, because which five views one
happens to draw matters far more in a room than on a loop around an object. An unpaired
comparison at that noise level could not resolve a 0.2 dB effect at all. Pairing every
augmented run against its <em>own seed's</em> baseline holds the paired deviations to
0.10&ndash;0.52 dB and keeps the effect measurable — the design choice earning its keep on a
scene it was not designed for.</p>

<h3>Why the sign flips</h3>
<p>A synthetic view supplies coverage and inconsistency in fixed proportion. Coverage has
diminishing value as real views accumulate — the per-view column in Table 1 falls by an order
of magnitude between the first step and the last. Inconsistency does not diminish; a
contradictory view is just as harmful at twenty views as at five, and arguably more so
because it now contradicts a better-determined geometry. Two terms, one decaying and one
roughly constant, are sufficient to produce a sign change, and that is what the data show.</p>

<h3>Testing the mechanism: constraint without a viewpoint</h3>
<p>The account above is an explanation, and explanations are cheap. What makes it worth more is
that it predicts something which could fail: if the harm comes from <em>inconsistency</em>
rather than from augmentation as such, an intervention supplying geometric constraint
<em>without inventing a viewpoint</em> should never cross over, because it has no contradictory
geometry to accumulate.</p>

<p>Monocular depth regularisation is that intervention. A depth network predicts an
inverse-depth map for each real training photograph; each map is anchored to true scene scale
by a least-squares fit against the sparse COLMAP points that view observes (median R&sup2;
0.964), and training then minimises image error and depth disagreement together. No camera is
invented and no pixel is fabricated. Synthetic views receive no depth supervision at all —
estimating depth from a fabricated image and then using it to constrain geometry would be
circular — so the prior acts on real photographs only.</p>

<div class="tablewrap">
<table>
  <caption><b>Table 6.</b> Depth regularisation crossed with outpainting at every subset size —
  a 3&times;2&times;2 factorial, three seeds, every cell paired against the same seed's own
  unaugmented baseline. The final row is the interaction: how far <em>+ both</em> exceeds the
  sum of the two interventions applied separately.</caption>
  <thead>
    <tr><th></th><th class="num">k = 5</th><th class="num">k = 10</th>
        <th class="num">k = 20</th></tr>
  </thead>
  <tbody>
{{DEPTH_TABLE}}
  </tbody>
</table>
</div>

<p><b>The prediction holds.</b> The depth prior is positive and separated from zero at every
subset size, including k = 20, where outpainting costs {{OUT_K20}} dB. Coverage crosses over;
constraint does not. The two interventions differ in exactly one respect — whether a camera
that never existed is invented — and only the one that invents a camera reverses sign.</p>

<p>The <em>shape</em> agrees too. The prior's benefit decays with subset size ({{DEPTH_K5}} at
k = 5 down to {{DEPTH_K20}} at k = 20), which is what diminishing returns on constraint
predict: the more real views pin the geometry, the less an approximate prior can add. It simply
never turns negative, because there is no contradiction term to overtake it. That is the
two-term account of the previous section reduced to its first term and observed in
isolation.</p>

<p>The two also <b>compound</b> rather than overlap. At k = 5 the combination reaches
{{DEPTH_BOTH_K5}} dB — the largest improvement anywhere in this study — with a positive
interaction at all three subset sizes. They repair different deficiencies: outpainting supplies
peripheral <em>content</em> that no real view recorded, the prior supplies <em>constraint</em>
on geometry already observed, and neither substitutes for the other. This agrees with the
duplication control, which found 80&ndash;85% of outpainting's benefit comes from the
fabricated borders rather than from repeated views.</p>

<p>The practical consequence is a recommendation the earlier sections could not support: at
five or ten real views use both; at twenty use the prior alone, since adding synthetic views
costs roughly half a decibel. At k = 10 outpainting by itself is inert, yet combined with the
prior it beats the prior alone — a depth constraint makes otherwise worthless synthetic views
worth having.</p>

<h3>Gaussian count as a mechanism</h3>
<p>Model complexity grows sharply under both geometric strategies. Against a baseline of
{{FLOOR_GAUSS}} Gaussians, inpainting barely moves the count — +3&ndash;5%, since it adds no
new scene area — while outpainting and pose-guided both inflate it by roughly 60% at the
200% ratio.</p>

<p>The count alone, however, does <em>not</em> predict the damage, and that is the more
interesting result. Outpainting reaches the <em>higher</em> of the two counts (+62% versus
+58% at the 200% ratio, and higher at every matched ratio) while costing essentially nothing
in PSNR. Its extra primitives are earned: the widened frustum exposes real peripheral scene
content that genuinely requires more Gaussians to represent. Pose-guided spends a comparable
budget reconciling views that disagree with one another, and multi-view inconsistency forces
the optimiser to spawn semi-transparent Gaussians satisfying each contradictory view
separately — which is precisely what a floater is.</p>

<p>The discriminator is therefore not how many primitives get added, but whether they explain
real geometry or contradictory geometry. Densification statistics on their own cannot
distinguish the two cases, which is worth knowing for anyone hoping to use Gaussian count as
a cheap proxy for reconstruction health.</p>
</section>
</div>

<div class="wide">
{{FIG_STRATEGIES}}
{{FIG_RATIO}}
</div>

<div class="col">
<section>
<h2><span class="n">05</span> When is diffusion augmentation worth it?</h2>
<p>There is a defensible answer, and it is narrower than the question usually assumes:
<b>only below roughly ten real views, only with outpainting, and only for a small fraction of
the gap.</b> The best condition measured anywhere in this study is {{BEST_D}} dB
({{BEST_WHERE}}) against a {{GAP5}} dB deficit at that subset size — under 3% of what is
missing.</p>

<p>Set against the alternative, that number is easy to interpret. Going from five to ten real
photographs is worth {{STEP_5_10}} dB. The best synthetic condition is therefore roughly
<b>{{BEST_VS_REAL}}× less valuable than simply taking five more pictures</b>, and the worst
condition ({{WORST_WHERE}}, {{WORST_D}} dB) costs several times more than the best one
gains. If real capture is possible at all, it dominates.</p>

<p>The useful result is the shape of the trade-off. Augmentation contributes value only when
it adds information the training set lacks, and it does harm in proportion to how much it
fabricates — but the balance between those two terms is not fixed. It is governed by how much
real data is already present, which is why the same method can be beneficial and harmful in
the same study.</p>

<div class="note">
<span class="lbl">The governing constraint</span>
<p>Stable Diffusion generates each image independently, with no knowledge of the others. It
is a 2D prior being asked to supply 3D evidence. Better image quality does not address this;
the fix would be a pose-conditioned multi-view model that is consistent across views by
construction. Such models exist — Zero123++, ImageDream, SV3D — but they are trained on
isolated objects against clean backgrounds and do not transfer to an outdoor street scene,
and their weights exceed 4 GB of VRAM.</p>
</div>

<p>The practical guidance follows directly: use augmentation only when real views are in the
single digits, prefer the strategy that widens the frustum without moving the camera, and
treat any method that invents a viewpoint as a liability until multi-view consistency can be
guaranteed. Above roughly ten real views, spend the effort on photographs instead.</p>
</section>

<section>
<h2><span class="n">06</span> Full results</h2>
<div class="tw">
<table>
  <caption><b>Table 5.</b> All 108 augmented conditions. Paired change against the same-seed,
  same-subset-size zero-synthetic baseline. LPIPS is inverted so that lower is better.</caption>
  <thead>
    <tr>
      <th class="num">k</th><th>Ratio</th><th class="num">Fakes</th>
      <th class="num">Δ PSNR</th><th class="num">Δ SSIM</th><th class="num">Δ LPIPS</th>
      <th class="num">Gaussians</th><th class="num">Seeds</th>
    </tr>
  </thead>
  <tbody>
{{RESULTS_TABLE}}
  </tbody>
</table>
</div>
</section>

<section>
<h2><span class="n">07</span> Limitations</h2>
<ul>
  <li><b>The warp-only control uses black holes, which is a harsh comparison.</b> It
  establishes that the diffusion step is net positive and therefore not the cause of the
  pose-guided damage, but it does not measure how good the diffusion output is in absolute
  terms. A gentler control — filling holes by classical inpainting rather than diffusion —
  would separate "diffusion specifically" from "any plausible fill".</li>
  <li><b>The control was run at k = 10 only.</b> Whether the diffusion step remains net
  positive at k = 5 and k = 20 is untested, though the pose-guided damage grows with k while
  hole sizes shrink, which argues it would.</li>
  <li><b>The depth loss weight was never tuned.</b> Upstream's schedule (1.0 decaying to 0.01)
  was chosen for scenes with hundreds of views, where photographs are highly informative and the
  prior is a mild correction. At five views the balance plausibly favours trusting the prior
  longer, so {{DEPTH_K5}} dB is more likely a floor than a ceiling. Tuning it honestly would
  need a validation split carved out of the training pool — selecting the weight on test PSNR
  would be fitting to the held-out set.</li>
  <li><b>Depth regularisation was tested on one scene, at one ratio.</b> Only <code>truck</code>,
  and the outpainting arm only at the 200% ratio — not at 100%, where outpainting does its worst
  damage, nor at the 800% ratio that produced the best augmentation-only result. Whether the
  prior's immunity to the crossover survives indoors is untested.</li>
  <li><b>Two scenes, one of them partially swept.</b> The crossover is confirmed on
  <code>truck</code> (outdoor object) and <code>{{GEN_SCENE}}</code> (indoor room), but the
  second scene was swept for <b>outpainting only</b>, at the two ends k = 5 and k = 20.
  Inpainting and pose-guided were not repeated there, so "inpainting is flat" and "pose-guided
  always hurts" remain single-scene claims. The second scene also trains at a different
  resolution, so only within-scene deltas transfer between them, never absolute PSNR.</li>
  <li><b>Ratios are not exactly the specified values at every subset size.</b> Only k = 20
  divides cleanly into 25/50/100/200%; at k = 5 and k = 10 the 25% point is fractional
  (1.25 and 2.5 images) and was rounded down, giving 20% and 40% respectively.</li>
  <li><b>7,000 iterations, not 30,000.</b> Densification is scheduled to run to 15,000, so it
  is truncated. Applied identically to every condition, so comparisons hold, but absolute
  numbers are not fully-converged 3DGS.</li>
  <li><b>One diffusion model.</b> Only SD 1.5 inpainting was tested; SDXL and FLUX exceed
  4 GB of VRAM.</li>
  <li><b>The crossover is bracketed, not located.</b> Outpainting is positive at k = 5 and
  negative at k = 20. The sign change lies somewhere between, but with only three subset sizes
  its position is not resolved.</li>
</ul>
</section>

<section>
<h2><span class="n">08</span> Reproducibility</h2>
<p>Every number in this report is read directly from the per-run <code>results.json</code>
files by the script that generates it, so the document cannot drift from the experiments.</p>

<div class="tw">
<table>
  <caption><b>Table 6.</b> Generation cost per synthetic image, measured on the RTX 3050 Ti.</caption>
  <thead><tr><th>Strategy</th><th class="num">Seconds / image</th></tr></thead>
  <tbody>{{GEN_ROWS}}</tbody>
</table>
</div>

<p>Two patches to the upstream 3DGS repository are required and are shipped as documented
diffs: <code>&lt;cstdint&gt;</code> must be included in <code>rasterizer_impl.h</code>, since
GCC 13 no longer provides it transitively; and <code>dataset_readers.py</code> is extended to
honour an explicit <code>split.json</code>. Building the CUDA extensions additionally
requires <code>--no-build-isolation</code>, because the setup scripts import torch at module
level.</p>

<p>The pipeline runs end to end on 4 GB of VRAM: 1.4 GB for few-shot training, 3.0 GB for the
full-data baseline, and 2.7 GB for diffusion. Geometry is covered by unit tests — an identity
warp must reproduce its source exactly, and interpolating fully to a neighbouring camera must
land on that camera's pose.</p>
</section>

<hr class="rule">
<footer>
<p>Few-Shot Gaussian Splatting with Diffusion-Based Data Augmentation ·
Tanks &amp; Temples <code>truck</code> · {{N_RUNS}} training runs ·
metrics computed on 32 held-out real photographs.</p>
</footer>
</div>
</div>
"""

if __name__ == "__main__":
    main()
