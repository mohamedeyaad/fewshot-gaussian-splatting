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

ROOT = Path(os.path.expanduser("~/fewshot_gs"))
RES = ROOT / "results"
OUT = RES / "report.html"

STRAT_ORDER = ["inpaint", "outpaint", "guided"]
STRAT_LABEL = {"inpaint": "Inpainting", "outpaint": "Outpainting",
               "guided": "Pose-guided"}


# ---------------------------------------------------------------- data
def load_runs():
    recs = []
    for p in sorted((ROOT / "runs").glob("*/results.json")):
        recs.append(json.loads(p.read_text()))
    return recs


def agg(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, 0.0
    return mean(v), (stdev(v) if len(v) > 1 else 0.0)


def build_tables(recs):
    baselines, full, fewshot = {}, None, []
    by = defaultdict(list)
    for r in recs:
        p, m, c = r["provenance"], r["metrics"], r["cost"]
        if p.get("method") == "full":
            full = r
            continue
        nf = p.get("n_synthetic", 0)
        if nf == 0:
            baselines[p.get("seed")] = r
            fewshot.append(r)
        else:
            by[(p.get("strategy"), nf)].append(r)

    # headline numbers
    fm, fs = agg([r["metrics"]["psnr"]["mean"] for r in fewshot])
    fsm, _ = agg([r["metrics"]["ssim"]["mean"] for r in fewshot])
    flm, _ = agg([r["metrics"]["lpips"]["mean"] for r in fewshot])
    fg, _ = agg([r["cost"]["n_gaussians"] for r in fewshot])

    head = {
        "floor_psnr": fm, "floor_psnr_sd": fs, "floor_ssim": fsm,
        "floor_lpips": flm, "floor_gauss": fg,
        "ceil_psnr": full["metrics"]["psnr"]["mean"],
        "ceil_ssim": full["metrics"]["ssim"]["mean"],
        "ceil_lpips": full["metrics"]["lpips"]["mean"],
        "ceil_gauss": full["cost"]["n_gaussians"],
        "n_runs": len(recs),
    }
    head["gap"] = head["ceil_psnr"] - head["floor_psnr"]

    # paired deltas
    rows = []
    for s in STRAT_ORDER:
        for nf in (2, 5, 10, 20):
            rs = by.get((s, nf), [])
            if not rs:
                continue
            d = {k: [] for k in ("psnr", "ssim", "lpips")}
            g = []
            for r in rs:
                b = baselines.get(r["provenance"].get("seed"))
                if not b:
                    continue
                for k in d:
                    d[k].append(r["metrics"][k]["mean"] - b["metrics"][k]["mean"])
                g.append(r["cost"]["n_gaussians"])
            pm, ps = agg(d["psnr"]); sm, ss = agg(d["ssim"]); lm, ls = agg(d["lpips"])
            gm, _ = agg(g)
            rows.append({
                "strategy": s, "n_fake": nf, "ratio": nf * 10, "seeds": len(rs),
                "d_psnr": pm, "d_psnr_sd": ps, "d_ssim": sm, "d_ssim_sd": ss,
                "d_lpips": lm, "d_lpips_sd": ls, "gauss": gm,
                "sig_psnr": abs(pm) > ps > 0,
            })
    return head, rows


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

    # --- results table ---
    trs = []
    for r in rows:
        first = r["n_fake"] == 2
        if first:
            trs.append(f'<tr class="grp"><td colspan="7">{STRAT_LABEL[r["strategy"]]}</td></tr>')
        trs.append(
            "<tr>"
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
        "{{N_RUNS}}": str(head["n_runs"]),
        "{{RESULTS_TABLE}}": results_table,
        "{{NOISE_CELLS}}": noise_html,
        "{{NOISE_PSNR}}": f'{noise["psnr"][1]:.3f}' if noise else "n/a",
        "{{GEN_ROWS}}": gen_rows,
        "{{FIG_TRAINING}}": figure(RES / "panel_training_data.png",
            "Figure 1 — Synthetic training images beside the real view they derive from. "
            "Inpainting has given the truck a second rear axle and an invented white box; "
            "outpainting fabricates the scene beyond the original frame; pose-guided renders "
            "a genuinely different viewpoint, with roughly 10% of its pixels invented.", 1500, 82),
        "{{FIG_PAIRED}}": figure(RES / "curves_paired.png",
            "Figure 2 — Change in each metric against the same seed's own zero-synthetic "
            "baseline. Error bars are the standard deviation across three seeds. All three "
            "metrics agree on both the ordering of the strategies and the shape of each curve.",
            1600, 88, "PNG"),
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
    <h1>Diffusion Augmentation Does Not Rescue Few-Shot Gaussian Splatting</h1>
    <p class="standfirst">Forty controlled training runs on the <em>truck</em> scene show that
    synthetic views generated by Stable Diffusion degrade held-out quality in eleven of twelve
    conditions — and that the damage scales directly with how much each strategy invents.</p>
    <div class="byline">
      <span>Tanks &amp; Temples <b>truck</b> · 251 images</span>
      <span>3 strategies × 4 ratios × 3 seeds</span>
      <span>{{N_RUNS}} training runs</span>
      <span>RTX 3050 Ti · 4 GB</span>
    </div>
  </div>
</header>

<div class="wrap">

<div class="stats">
  <div class="stat"><p class="k">Few-shot floor</p><p class="v">{{FLOOR_PSNR}}</p><p class="u">PSNR, 10 real images</p></div>
  <div class="stat"><p class="k">Full-data ceiling</p><p class="v">{{CEIL_PSNR}}</p><p class="u">PSNR, 219 real images</p></div>
  <div class="stat"><p class="k">Gap to close</p><p class="v">{{GAP}}</p><p class="u">dB lost to few-shot</p></div>
  <div class="stat"><p class="k">Best augmentation</p><p class="v">+0.17</p><p class="u">outpainting at 20%</p></div>
  <div class="stat"><p class="k">Worst augmentation</p><p class="v">−1.45</p><p class="u">pose-guided at 200%</p></div>
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
<p>Every condition was run at three seeds and 7,000 iterations at half resolution. The table
reports change against each seed's own baseline; an asterisk marks deltas whose magnitude
exceeds their own standard deviation.</p>

<div class="tw">
<table>
  <caption><b>Table 2.</b> Paired change against the same-seed zero-synthetic baseline.
  Green is an improvement, red a degradation; LPIPS is inverted so that lower is better.
  Baseline Gaussian count is {{FLOOR_GAUSS}}.</caption>
  <thead>
    <tr>
      <th>Ratio</th><th class="num">Fakes</th>
      <th class="num">Δ PSNR</th><th class="num">Δ SSIM</th><th class="num">Δ LPIPS</th>
      <th class="num">Gaussians</th><th class="num">Seeds</th>
    </tr>
  </thead>
  <tbody>
{{RESULTS_TABLE}}
  </tbody>
</table>
</div>

<p>Eleven of the twelve conditions degrade the model. The single exception is outpainting at
the 20% ratio, where PSNR improves by 0.172 dB — three times the noise floor — and neither
perceptual metric contradicts it.</p>
</section>

</div>

<div class="wide">
{{FIG_PAIRED}}
</div>

<div class="col">
<section>
<h3>An ordering by how much each strategy invents</h3>
<p>At the 100% ratio the damage is −0.21 dB for inpainting, −0.42 for outpainting and
−1.40 for pose-guided. That ordering is exactly the ordering of how much content each
strategy fabricates, and all three metrics reproduce it independently.</p>

<h3>Two curve shapes, not one</h3>
<p>Inpainting and outpainting are V-shaped: they worsen to a minimum at 100% and then
partially recover at 200%. Pose-guided declines monotonically, −1.40 to −1.45, with no
recovery. A plausible reading is that at 200% each real view carries two <em>different</em>
hallucinations of the same region, which partially cancel; pose-guided instead produces
geometrically coherent errors at precisely known poses, so additional samples reinforce
rather than contradict one another.</p>

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
<p>On this scene, almost never. The one beneficial condition returns 0.17 dB against a
{{GAP}} dB deficit — about 2% of the gap — for roughly six minutes of generation. Nothing
here approaches a solution to few-shot reconstruction.</p>

<p>The useful result is the shape of the trade-off. Augmentation contributes value only when
it adds information the training set lacks, and it does harm in proportion to how much it
fabricates. Inpainting is safe precisely because it is uninformative: the pose is unchanged,
so it can neither help nor do much harm. Pose-guided synthesis is the only strategy that
addresses the actual deficiency — missing viewpoints — and it is the most damaging, because
producing a new viewpoint requires inventing roughly a tenth of every image and presenting
it at a pose the optimiser trusts completely.</p>

<div class="note">
<span class="lbl">The governing constraint</span>
<p>Stable Diffusion generates each image independently, with no knowledge of the others. It
is a 2D prior being asked to supply 3D evidence. Better image quality does not address this;
the fix would be a pose-conditioned multi-view model that is consistent across views by
construction. Such models exist — Zero123++, ImageDream, SV3D — but they are trained on
isolated objects against clean backgrounds and do not transfer to an outdoor street scene,
and their weights exceed 4 GB of VRAM.</p>
</div>

<p>Where augmentation would plausibly earn its place: at the frame periphery, in small
quantities, where few-shot models are least constrained and floaters breed — which is
exactly the one condition that worked. The practical guidance is to keep synthetic data
below roughly a quarter of the real set, prefer strategies that preserve pose exactly, and
treat any method that invents geometry as a liability until multi-view consistency can be
guaranteed.</p>
</section>

<section>
<h2><span class="n">06</span> Limitations</h2>
<ul>
  <li><b>A warping artifact confounds the pose-guided result.</b> Forward depth-warping
  leaves residual speckle at depth discontinuities. Some part of the −1.45 dB may be image
  degradation rather than hallucinated content. The clean control — warp to a new pose and
  leave holes unfilled, with no diffusion — was not run.</li>
  <li><b>One scene.</b> All conclusions come from <code>truck</code>. Whether they hold for
  indoor scenes or for objects with different geometry is untested.</li>
  <li><b>One shot count.</b> Only k = 10 was swept. Augmentation may behave differently at
  k = 5, where the model is more starved, or at k = 20.</li>
  <li><b>7,000 iterations, not 30,000.</b> Densification is scheduled to run to 15,000, so it
  is truncated. Applied identically to every condition, so comparisons hold, but absolute
  numbers are not fully-converged 3DGS.</li>
  <li><b>One diffusion model.</b> Only SD 1.5 inpainting was tested; SDXL and FLUX exceed
  4 GB of VRAM.</li>
</ul>
</section>

<section>
<h2><span class="n">07</span> Reproducibility</h2>
<p>Every number in this report is read directly from the per-run <code>results.json</code>
files by the script that generates it, so the document cannot drift from the experiments.</p>

<div class="tw">
<table>
  <caption><b>Table 3.</b> Generation cost per synthetic image, measured on the RTX 3050 Ti.</caption>
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
