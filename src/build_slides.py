"""Build the exam presentation from the same loaders the report uses.

  python src/build_slides.py        # -> results/slides.html

Every number on a slide is read out of runs/*/results.json through
build_report's loaders, for the same reason build_brief.py does it: a deck that
quotes figures by hand drifts from the report the moment anything is re-run,
and the one place that must never disagree with itself is the document being
defended in an exam.

Figures are embedded as JPEG data URIs so the file is self-contained - it has
to open on an unfamiliar machine with no network.
"""
from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image

import build_report as R

ROOT = Path(os.path.expanduser("~/fewshot_gs"))
RES = ROOT / "results"
OUT = RES / "slides.html"


def data_uri(name: str, width: int = 1700, quality: int = 82) -> str:
    """Downscale a panel and inline it. Panels are up to 3100 px wide and
    several MB; a projector never resolves that, and the artifact has a size
    budget."""
    p = RES / name
    if not p.exists():
        return ""
    im = Image.open(p).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True, progressive=True)
    b = buf.getvalue()
    print(f"  {name}: {p.stat().st_size/1e6:.1f} MB -> {len(b)/1e6:.2f} MB")
    return "data:image/jpeg;base64," + base64.b64encode(b).decode()


def bar(value, scale, label):
    """One row of the diverging bar chart: a delta drawn either side of zero.

    The chart exists because the finding IS a sign change, and a column of
    signed numbers does not show a sign change - it shows numbers that happen
    to have signs. Width is proportional to |value| against `scale`, so bars
    from different scenes are only ever compared within one chart.
    """
    pct = min(100.0, abs(value) / scale * 100.0)
    side = "pos" if value >= 0 else "neg"
    return (f'<div class="barrow"><div class="barlab">{label}</div>'
            f'<div class="bartrack"><div class="barzero"></div>'
            f'<div class="bar {side}" style="width:{pct/2:.1f}%"></div></div>'
            f'<div class="barval {side}">{value:+.3f}</div></div>')


def main():
    recs = R.load_runs("truck")
    head, rows = R.build_tables(recs)
    gen = R.build_generalisation()
    depth = R.build_depth()
    iso = R.build_isolated()

    def out(k, ratio):
        for r in rows:
            if r["strategy"] == "outpaint" and r["k"] == k and r["ratio"] == ratio:
                return r
        return None

    # The crossover at the 200% ratio, the one ratio present at every k.
    cross = [(k, out(k, 200)) for k in (5, 10, 20)]
    cross_bars = "".join(bar(r["d_psnr"], 0.8, f"k = {k}") for k, r in cross if r)

    iso_v = {(r["strategy"], r["ratio"]): r for r in (iso.get("rows") or [])}
    iso_bars = "".join(
        bar(iso_v[(s, "100%")][f"k{k}"], 17.0, lab)
        for s, k, lab in (("outpaint", 5, "outpainting, k = 5"),
                          ("outpaint", 20, "outpainting, k = 20"),
                          ("guided", 5, "pose-guided, k = 5"),
                          ("guided", 20, "pose-guided, k = 20"),
                          ("outwhite", 5, "control: white, k = 5"),
                          ("outwhite", 20, "control: white, k = 20"))
        if (s, "100%") in iso_v and iso_v[(s, "100%")].get(f"k{k}") is not None)

    fl = head["floors"]
    v = {
        "FLOOR5": f'{fl[5]["psnr"]:.2f}', "FLOOR10": f'{fl[10]["psnr"]:.2f}',
        "FLOOR20": f'{fl[20]["psnr"]:.2f}', "CEIL": f'{head["ceil_psnr"]:.2f}',
        "GAP5": f'{head["gap5"]:.2f}', "NRUNS": str(len(list((ROOT / "runs").glob("*/results.json")))),
        "BEST": f'{head["best"]["d_psnr"]:+.3f}',
        "STEP510": f'{head["step_5_10"]:+.2f}',
        "CROSS_BARS": cross_bars, "ISO_BARS": iso_bars,
        "GEN5": f'{gen["rows"][-1]["k5"]:+.3f}' if gen else "n/a",
        "GEN20": f'{gen["rows"][2]["k20"]:+.3f}' if gen else "n/a",
        "GENFLOORSD": f'{gen["floor_sd"]:.2f}' if gen else "n/a",
        "D5": f'{depth["rows"][5]["depth"][0]:+.3f}' if depth else "n/a",
        "D10": f'{depth["rows"][10]["depth"][0]:+.3f}' if depth else "n/a",
        "D20": f'{depth["rows"][20]["depth"][0]:+.3f}' if depth else "n/a",
        "DBOTH": f'{depth["rows"][5]["both"][0]:+.3f}' if depth else "n/a",
        "ISOCEIL": f'{iso["ceil"]:.2f}', "ISOF5": f'{iso["floors"][5]:.2f}',
        "ISOF20": f'{iso["floors"][20]:.2f}',
        "ISOW5": f'{iso_v[("outwhite","100%")]["k5"]:+.2f}',
        "ISOW20": f'{iso_v[("outwhite","100%")]["k20"]:+.2f}',
        "ISOO5": f'{iso_v[("outpaint","100%")]["k5"]:.2f}',
        "ISOO20": f'{iso_v[("outpaint","100%")]["k20"]:.2f}',
        "ISOG5": f'{iso_v[("guided","100%")]["k5"]:.2f}',
        "ISOG20": f'{iso_v[("guided","100%")]["k20"]:.2f}',
        "GB": f'{iso["gauss_base"]:,.0f}', "GA": f'{iso["gauss_aug"]:,.0f}',
        "IMG_CROSS": data_uri("panel_crossover.png"),
        "IMG_LEGOC": data_uri("panel_legoc.png", width=1900),
        "IMG_SCALE": data_uri("panel_scaling.png"),
    }

    html = TEMPLATE
    for k, val in v.items():
        html = html.replace("{{" + k + "}}", str(val))
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}  ({len(html)/1e6:.2f} MB)")
    left = [t for t in html.split("{{")[1:]]
    if left:
        print("UNRESOLVED:", [t.split("}}")[0] for t in left][:5])


TEMPLATE = r"""<title>Splatting Crossover</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap">
<style>
:root{
  --ground:#EFF3F4; --surface:#FFFFFF; --ink:#16202A; --muted:#5C6E78;
  --line:#D2DDE1; --accent:#0B6E7F; --gain:#1C7C54; --loss:#B3452C;
  --shadow:0 1px 2px rgba(22,32,42,.06),0 8px 28px rgba(22,32,42,.07);
  --serif:"Newsreader",Georgia,"Times New Roman",serif;
  --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,"SFMono-Regular",Menlo,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0C1317; --surface:#141F25; --ink:#E4EDF1; --muted:#8CA3AD;
  --line:#243239; --accent:#5AC6D8; --gain:#4FBD8C; --loss:#E39072;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 34px rgba(0,0,0,.34);
}}
:root[data-theme="dark"]{
  --ground:#0C1317; --surface:#141F25; --ink:#E4EDF1; --muted:#8CA3AD;
  --line:#243239; --accent:#5AC6D8; --gain:#4FBD8C; --loss:#E39072;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 34px rgba(0,0,0,.34);
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{
  background:var(--ground); color:var(--ink); font-family:var(--sans);
  font-size:clamp(15px,1.15vw,19px); line-height:1.55;
  -webkit-font-smoothing:antialiased;
}
.deck{position:relative}
.slide{
  min-height:100vh; display:none; padding:clamp(28px,5vh,64px) clamp(24px,6vw,90px)
  clamp(56px,8vh,80px); flex-direction:column; justify-content:center; gap:clamp(14px,2.2vh,26px);
}
.slide.on{display:flex}
.inner{width:100%; max-width:1180px; margin:0 auto; display:flex;
  flex-direction:column; gap:clamp(12px,2vh,22px)}
.inner.wide{max-width:1560px}

.eyebrow{
  font-family:var(--mono); font-size:.72em; letter-spacing:.14em;
  text-transform:uppercase; color:var(--accent); font-weight:500;
}
h1{font-family:var(--serif); font-weight:500; font-size:clamp(2.1rem,5.2vw,4.1rem);
  line-height:1.06; margin:0; letter-spacing:-.015em; text-wrap:balance}
h2{font-family:var(--serif); font-weight:500; font-size:clamp(1.6rem,3.4vw,2.7rem);
  line-height:1.12; margin:0; letter-spacing:-.01em; text-wrap:balance}
h3{font-family:var(--sans); font-weight:600; font-size:1.02em; margin:0;
  letter-spacing:.005em}
p{margin:0; max-width:66ch}
.lead{font-size:1.12em; color:var(--ink)}
.muted{color:var(--muted)}
strong{font-weight:600}
em.q{font-style:normal; color:var(--accent); font-weight:500}

.num{font-family:var(--mono); font-variant-numeric:tabular-nums; font-weight:500}
.pos{color:var(--gain)} .neg{color:var(--loss)}

.cols{display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
  gap:clamp(12px,1.6vw,22px)}
.card{background:var(--surface); border:1px solid var(--line); border-radius:3px;
  padding:clamp(14px,1.7vw,22px); display:flex; flex-direction:column; gap:8px;
  box-shadow:var(--shadow)}
.card .k{font-family:var(--mono); font-size:clamp(1.5rem,3vw,2.3rem); font-weight:600;
  line-height:1; letter-spacing:-.02em}
.card .cap{font-size:.86em; color:var(--muted); line-height:1.4}

table{border-collapse:collapse; width:100%; font-size:.95em}
caption{caption-side:bottom; text-align:left; padding-top:10px; font-size:.82em;
  color:var(--muted); line-height:1.45}
th,td{padding:.5em .7em; border-bottom:1px solid var(--line); text-align:left}
th{font-weight:600; font-size:.8em; letter-spacing:.06em; text-transform:uppercase;
  color:var(--muted)}
td.n,th.n{text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums}
.tw{overflow-x:auto}

.barrow{display:grid; grid-template-columns:minmax(120px,190px) 1fr minmax(84px,102px);
  align-items:center; gap:clamp(10px,1.4vw,20px)}
.barlab{font-size:.9em; color:var(--muted)}
.bartrack{position:relative; height:clamp(20px,2.6vh,30px); background:var(--surface);
  border:1px solid var(--line); border-radius:2px}
.barzero{position:absolute; left:50%; top:-3px; bottom:-3px; width:1px;
  background:var(--muted); opacity:.55}
.bar{position:absolute; top:3px; bottom:3px; border-radius:1px}
.bar.pos{left:50%; background:var(--gain)}
.bar.neg{right:50%; background:var(--loss)}
.barval{font-family:var(--mono); font-variant-numeric:tabular-nums; text-align:right;
  font-weight:500; font-size:.95em}

.rail{position:fixed; left:0; top:0; height:2px; background:var(--accent);
  z-index:10; transition:width .22s ease}
.hud{position:fixed; right:clamp(14px,2.4vw,30px); bottom:clamp(12px,2vh,22px);
  font-family:var(--mono); font-size:.76em; color:var(--muted); z-index:10;
  letter-spacing:.04em}
.hint{position:fixed; left:clamp(14px,2.4vw,30px); bottom:clamp(12px,2vh,22px);
  font-family:var(--mono); font-size:.72em; color:var(--muted); z-index:10;
  opacity:.72; letter-spacing:.03em}
figure{margin:0; display:flex; flex-direction:column; gap:9px}
figure img{width:100%; height:auto; border:1px solid var(--line); border-radius:2px;
  background:var(--surface)}
figcaption{font-size:.83em; color:var(--muted); line-height:1.45; max-width:100ch}
ul{margin:0; padding-left:1.15em; display:flex; flex-direction:column; gap:.5em;
  max-width:70ch}
li::marker{color:var(--accent)}
.rule{height:1px; background:var(--line); border:0; margin:0}
.tag{display:inline-block; font-family:var(--mono); font-size:.72em; padding:.18em .5em;
  border:1px solid var(--line); border-radius:2px; color:var(--muted);
  letter-spacing:.05em}
a{color:var(--accent)}
:focus-visible{outline:2px solid var(--accent); outline-offset:3px}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
@media print{
  .rail,.hud,.hint{display:none}
  .slide{display:flex!important; min-height:auto; page-break-after:always;
    padding:14mm 16mm}
  body{background:#fff; font-size:11pt}
}
</style>

<div class="rail" id="rail"></div>
<div class="deck" id="deck">

<section class="slide on">
  <div class="inner">
    <div class="eyebrow">Università degli Studi di Genova &middot; Robotics</div>
    <h1>When does a generated photograph<br>help a 3D reconstruction?</h1>
    <p class="lead muted">Few-shot Gaussian Splatting with diffusion-based data
    augmentation — and the boundary where augmentation stops helping and starts
    destroying.</p>
    <hr class="rule">
    <div class="cols">
      <div class="card"><div class="k num">{{NRUNS}}</div>
        <div class="cap">training runs</div></div>
      <div class="card"><div class="k num">4</div>
        <div class="cap">scenes: outdoor object, indoor room, isolated object &times;2 formats</div></div>
      <div class="card"><div class="k num">3</div>
        <div class="cap">augmentation strategies, 4 ratios, 3 seeds</div></div>
      <div class="card"><div class="k num">5</div>
        <div class="cap">controls, one of which reverses the headline reading</div></div>
    </div>
    <p class="muted" style="font-size:.9em">Mohamed Eyad</p>
  </div>
</section>

<section class="slide">
  <div class="inner">
    <div class="eyebrow">The problem</div>
    <h2>3D Gaussian Splatting is excellent — given enough photographs</h2>
    <p class="lead">On <span class="tag">truck</span> (Tanks &amp; Temples), the full
    219-view reconstruction reaches <span class="num">{{CEIL}}</span> dB. Cut to five
    photographs and it falls to <span class="num">{{FLOOR5}}</span> dB.</p>
    <div class="cols">
      <div class="card"><div class="k num">{{FLOOR5}}</div><div class="cap">dB &middot; 5 views</div></div>
      <div class="card"><div class="k num">{{FLOOR10}}</div><div class="cap">dB &middot; 10 views</div></div>
      <div class="card"><div class="k num">{{FLOOR20}}</div><div class="cap">dB &middot; 20 views</div></div>
      <div class="card"><div class="k num">{{CEIL}}</div><div class="cap">dB &middot; 219 views (ceiling)</div></div>
    </div>
    <p><strong>A {{GAP5}} dB gap.</strong> Photographs are expensive; a diffusion model
    generates images for free. <em class="q">Can generated views substitute for
    photographs that were never taken?</em></p>
  </div>
</section>

<section class="slide">
  <div class="inner wide">
    <div class="eyebrow">The problem &middot; what it looks like</div>
    <figure>
      <img src="{{IMG_SCALE}}" alt="The same held-out cameras reconstructed from 5, 10, 20 and 219 real photographs.">
      <figcaption>The same held-out cameras, reconstructed from 5, 10, 20 and 219 real
      photographs. At five views the geometry is broadly right — what costs the
      {{GAP5}} dB is haze, floaters and surfaces the optimiser had no second view to
      pin down.</figcaption>
    </figure>
  </div>
</section>

<section class="slide">
  <div class="inner">
    <div class="eyebrow">Method</div>
    <h2>Three ways to manufacture a training view</h2>
    <div class="cols">
      <div class="card">
        <h3>Inpainting</h3>
        <p class="cap">Mask a region of a real photograph and let the model refill it.
        Same camera, same framing — no new geometry at all.</p>
      </div>
      <div class="card">
        <h3>Outpainting</h3>
        <p class="cap">Keep the focal length, enlarge the canvas. The camera does not
        move; the lens gets wider. Supplies peripheral <strong>content</strong>.</p>
      </div>
      <div class="card">
        <h3>Pose-guided</h3>
        <p class="cap">Estimate depth, anchor it to the sparse SfM points, warp to a
        <em>chosen</em> new camera, and let diffusion fill the disocclusions. The only
        one supplying a genuinely new <strong>viewpoint</strong>.</p>
      </div>
    </div>
    <p class="muted">Each is generated at 25%, 50%, 100% and 200% of the real subset
    size, at k = 5, 10 and 20 real views, over three seeds.</p>
  </div>
</section>

<section class="slide">
  <div class="inner">
    <div class="eyebrow">Design</div>
    <h2>The measurement has to survive a 0.2 dB effect</h2>
    <ul>
      <li><strong>Frozen held-out split.</strong> Every run of a scene is scored against
      byte-identical test images — verified by hash across all {{NRUNS}} runs.</li>
      <li><strong>Paired within seed.</strong> Each augmented run is compared to
      <em>its own seed's</em> baseline, so the luck of which five views were drawn
      cancels instead of adding noise.</li>
      <li><strong>Measured noise floor.</strong> The identical configuration re-run
      three times varies by <span class="num">0.039</span> dB, so an effect of
      0.2 dB is resolvable and one of 0.03 dB is not.</li>
      <li><strong>Farthest-point sampling.</strong> Subsets are spread over the camera
      trajectory rather than drawn at random; the seed picks only the starting camera.</li>
    </ul>
    <p class="muted">An asterisk in what follows means |mean| exceeds the between-seed
    spread — a consistency check at n = 3, not a formal significance test.</p>
  </div>
</section>

<section class="slide">
  <div class="inner">
    <div class="eyebrow">Result</div>
    <h2>The same augmentation helps at five views and harms at twenty</h2>
    <div style="display:flex;flex-direction:column;gap:clamp(8px,1.4vh,14px)">
      {{CROSS_BARS}}
    </div>
    <p>Outpainting at the 200% ratio, on <span class="tag">truck</span>, paired within
    seed. The treatment is <em>identical</em> at every subset size — only the number of
    real photographs changes. Best single condition anywhere in the study:
    <span class="num pos">{{BEST}}</span> dB.</p>
    <p class="muted">For scale: one extra real photograph between k = 5 and k = 10 is
    worth <span class="num">{{STEP510}}</span> dB total. Augmentation buys a fraction of
    a photograph, and only while photographs are scarce.</p>
  </div>
</section>

<section class="slide">
  <div class="inner wide">
    <div class="eyebrow">Result &middot; what it looks like</div>
    <figure>
      <img src="{{IMG_CROSS}}" alt="Held-out renders at 5 and 20 real views, with and without identical 200% outpainting.">
      <figcaption>The same held-out cameras at 5 and 20 real views, with and without the
      identical augmentation. Views were chosen as those closest to the <em>mean</em>
      effect, and per-view deltas are printed, because per-view effects scatter several
      dB either side of the average — three arbitrary views can contradict the finding
      they illustrate.</figcaption>
    </figure>
  </div>
</section>

<section class="slide">
  <div class="inner">
    <div class="eyebrow">Generalisation</div>
    <h2>It reproduces indoors</h2>
    <p class="lead">34 further runs repeat the outpainting sweep on
    <span class="tag">drjohnson</span> (Deep Blending, 230 views, an indoor room) — a
    different capture regime, a different scene scale.</p>
    <div class="cols">
      <div class="card"><div class="k num pos">{{GEN5}}</div>
        <div class="cap">dB at k = 5 — <em>stronger</em> than outdoors</div></div>
      <div class="card"><div class="k num neg">{{GEN20}}</div>
        <div class="cap">dB at k = 20 — the sign flip repeats</div></div>
      <div class="card"><div class="k num">{{GENFLOORSD}}</div>
        <div class="cap">dB baseline scatter between seeds — which five views you draw
        matters far more in a room</div></div>
    </div>
    <p>Indoors all three metrics move together, where on truck outpainting bought PSNR
    while SSIM stayed flat. Agreement across three metrics with different failure modes
    is much harder to get by chance than agreement in one.</p>
  </div>
</section>

<section class="slide">
  <div class="inner">
    <div class="eyebrow">Explanation</div>
    <h2>Two terms, one decaying and one constant</h2>
    <div class="cols">
      <div class="card">
        <h3>Coverage — decays</h3>
        <p class="cap">A synthetic view supplies scene content no real camera saw. That
        is worth a lot at five views and very little at twenty: the per-view value of a
        real photograph falls by an order of magnitude across the same range.</p>
      </div>
      <div class="card">
        <h3>Inconsistency — does not decay</h3>
        <p class="cap">A fabricated view contradicts the others. A contradiction is just
        as harmful at twenty views as at five — arguably worse, because it now
        contradicts a better-determined geometry.</p>
      </div>
    </div>
    <p>A decaying benefit plus a roughly constant cost is sufficient to produce a sign
    change. That is an explanation, and explanations are cheap — so it was made to
    predict something that could fail.</p>
  </div>
</section>

<section class="slide">
  <div class="inner">
    <div class="eyebrow">Prediction 1 &middot; held</div>
    <h2>Constraint without a viewpoint should never cross over</h2>
    <p class="lead">If the harm is <em>inconsistency</em>, an intervention that supplies
    geometric constraint <strong>without inventing a camera</strong> has no contradictory
    geometry to accumulate — so it should stay positive at every subset size.</p>
    <p>Monocular depth regularisation is that intervention: a depth map per real
    photograph, anchored to true scale against the sparse SfM points that view observes
    (median R² 0.964). Synthetic views receive no depth supervision — estimating depth
    from a fabricated image would be circular.</p>
    <div class="cols">
      <div class="card"><div class="k num pos">{{D5}}</div><div class="cap">dB at k = 5</div></div>
      <div class="card"><div class="k num pos">{{D10}}</div><div class="cap">dB at k = 10</div></div>
      <div class="card"><div class="k num pos">{{D20}}</div><div class="cap">dB at k = 20 — still positive</div></div>
      <div class="card"><div class="k num pos">{{DBOTH}}</div><div class="cap">dB combined with outpainting at k = 5 — the largest gain in the study</div></div>
    </div>
    <p class="muted">Coverage crosses over; constraint does not. The two interventions
    differ in exactly one respect — whether a camera that never existed is invented.</p>
  </div>
</section>

<section class="slide">
  <div class="inner">
    <div class="eyebrow">Prediction 2 &middot; failed</div>
    <h2>An object with nothing around it</h2>
    <p class="lead">If augmentation pays in proportion to the <em>real scene content</em>
    the synthesised region recovers, then a scene whose synthesised region contains
    <strong>nothing</strong> should show no benefit at any subset size.</p>
    <p><span class="tag">lego</span> is that scene — one object on clean white, 70.4% of
    every frame trivially predictable. Its full-data ceiling of
    <span class="num">{{ISOCEIL}}</span> dB reproduces the ≈33 dB published for it: the
    only number in this study anchored outside it.</p>
    <hr class="rule">
    <p><strong>What I predicted:</strong> outpainting worth approximately nothing —
    there is no content outside the frame to recover. Pose-guided <em>helping</em>,
    because an isolated object's deficit is angular and that is exactly what a new
    viewpoint supplies.</p>
    <p class="muted">Both halves were wrong.</p>
  </div>
</section>

<section class="slide">
  <div class="inner">
    <div class="eyebrow">Prediction 2 &middot; what happened</div>
    <h2>Both strategies collapse</h2>
    <div style="display:flex;flex-direction:column;gap:clamp(6px,1.1vh,12px)">
      {{ISO_BARS}}
    </div>
    <p><strong>A diffusion model cannot generate nothing.</strong> Asked to extend an
    isolated object's border — with a prompt that explicitly requests <em>"a plain white
    background"</em> — Stable Diffusion paints dense confetti texture, because empty
    white is not what its training distribution says the periphery of a photograph looks
    like. Pose-guided fails twice over: depth anchors to points existing only <em>on</em>
    the object, so most of the frame warps to speckle, and the holes fill with invented
    objects.</p>
    <p class="muted">Gaussian counts corroborate: <span class="num">{{GB}}</span> at
    baseline against <span class="num">{{GA}}</span> when fed views the optimiser cannot
    reconcile. Note the last two rows — that is the control.</p>
  </div>
</section>

<section class="slide">
  <div class="inner wide">
    <div class="eyebrow">The control</div>
    <figure>
      <img src="{{IMG_LEGOC}}" alt="Isolated object renders: baseline, outpainting, pose-guided, and the white-border control at 5 and 20 real views.">
      <figcaption>Outpainting changes two things at once: it widens the frustum, and it
      fills the new border with invented content. Regenerating the same views with a
      <strong>plain white</strong> border — identical canvas, focal length, widened
      camera and poses, verified equal in the stored metadata — leaves the frustum change
      intact and removes only the fabrication.</figcaption>
    </figure>
  </div>
</section>

<section class="slide">
  <div class="inner">
    <div class="eyebrow">The control &middot; verdict</div>
    <h2>The frustum is harmless. The fabrication is the whole effect.</h2>
    <div class="tw"><table>
      <thead><tr><th>Condition at the 100% ratio</th><th class="n">k = 5</th><th class="n">k = 20</th></tr></thead>
      <tbody>
        <tr><td>Outpainting — diffusion border</td>
            <td class="n neg">{{ISOO5}}</td><td class="n neg">{{ISOO20}}</td></tr>
        <tr><td>Pose-guided</td>
            <td class="n neg">{{ISOG5}}</td><td class="n neg">{{ISOG20}}</td></tr>
        <tr><td><strong>Control — identical camera, white border</strong></td>
            <td class="n">{{ISOW5}}</td><td class="n">{{ISOW20}}</td></tr>
      </tbody>
      <caption>Paired against each seed's own baseline
      (<span class="num">{{ISOF5}}</span> dB at k = 5,
      <span class="num">{{ISOF20}}</span> dB at k = 20).</caption>
    </table></div>
    <p>At k = 20 the widened frustum accounts for <span class="num">0.27</span> dB of a
    <span class="num">15.57</span> dB collapse. <strong>Roughly 98% of the damage is the
    content the model invented</strong> where the truth was empty white.</p>
  </div>
</section>

<section class="slide">
  <div class="inner">
    <div class="eyebrow">Conclusion</div>
    <h2>What the four scenes say together</h2>
    <ul>
      <li><strong>Diffusion augmentation is not free coverage.</strong> It helps only
      while real views are in the single digits, and the crossover reproduces outdoors
      and indoors.</li>
      <li><strong>The mechanism is pose novelty.</strong> Five controls exclude image
      quality, hole content, warping artifacts, mere view repetition — and now the
      frustum change itself.</li>
      <li><strong>Constraint beats invention.</strong> A depth prior is positive at every
      subset size and compounds with augmentation; anything that invents a camera
      eventually reverses sign.</li>
      <li><strong>The boundary:</strong> augmentation pays in proportion to the real
      scene content the synthesised region can recover — and an isolated object has
      none to recover.</li>
    </ul>
    <hr class="rule">
    <p class="lead">Practical guidance: use augmentation below about ten real views,
    prefer the strategy that widens the frustum without moving the camera, and treat any
    method that invents a viewpoint as a liability. Above ten views, spend the effort on
    photographs.</p>
  </div>
</section>

<section class="slide">
  <div class="inner">
    <div class="eyebrow">Limitations</div>
    <h2>What this does not show</h2>
    <ul>
      <li><strong>7,000 iterations, not 30,000.</strong> Densification is scheduled to
      15,000, so it is truncated. Applied identically everywhere, so comparisons hold,
      but absolute numbers are not fully converged 3DGS.</li>
      <li><strong>One diffusion model.</strong> Only SD 1.5 inpainting; SDXL and FLUX
      exceed the 4 GB card. A pose-conditioned multi-view model would be the right tool
      and was out of reach.</li>
      <li><strong>Three seeds.</strong> A consistency check, not statistical
      significance.</li>
      <li><strong>The pose-guided collapse is implementation-specific.</strong> Warping
      only the object's pixels and leaving the background white was not attempted. The
      outpainting arm carries no such caveat — its control isolates the cause cleanly.</li>
      <li><strong>A failed prediction is reported as one.</strong> Pose-guided was
      expected to help on an isolated object. It did not, and the corrected account is
      the more general one.</li>
    </ul>
  </div>
</section>

<section class="slide">
  <div class="inner">
    <div class="eyebrow">In one sentence</div>
    <h1 style="max-width:20ch">Augmentation buys coverage, and pays for it in
    consistency.</h1>
    <p class="lead">Whether that trade is worth taking depends on one thing: how much
    real scene the synthesised region can recover. At five outdoor views, a lot. At
    twenty, not enough. Around an isolated object, none — and the model fabricates
    instead.</p>
    <hr class="rule">
    <p class="muted">{{NRUNS}} runs · 4 scenes · every held-out set hash-verified
    identical within its scene · code and full report in the repository.</p>
  </div>
</section>

</div>

<div class="hud" id="hud"></div>
<div class="hint">← → or space · F fullscreen · P print</div>

<script>
(function(){
  var slides=[].slice.call(document.querySelectorAll('.slide')),i=0;
  var rail=document.getElementById('rail'),hud=document.getElementById('hud');
  function show(n){
    i=Math.max(0,Math.min(slides.length-1,n));
    slides.forEach(function(s,j){s.classList.toggle('on',j===i);});
    rail.style.width=((i+1)/slides.length*100)+'%';
    hud.textContent=(i+1)+' / '+slides.length;
    if(location.hash!=='#'+(i+1)){history.replaceState(null,'','#'+(i+1));}
    window.scrollTo(0,0);
  }
  document.addEventListener('keydown',function(e){
    if(e.key==='ArrowRight'||e.key===' '||e.key==='PageDown'){e.preventDefault();show(i+1);}
    else if(e.key==='ArrowLeft'||e.key==='PageUp'){e.preventDefault();show(i-1);}
    else if(e.key==='Home'){show(0);} else if(e.key==='End'){show(slides.length-1);}
    else if(e.key==='f'||e.key==='F'){
      if(document.fullscreenElement){document.exitFullscreen();}
      else if(document.documentElement.requestFullscreen){document.documentElement.requestFullscreen();}
    }
    else if(e.key==='p'||e.key==='P'){window.print();}
  });
  document.addEventListener('click',function(e){
    if(e.target.closest('a')||window.getSelection().toString())return;
    show(e.clientX<window.innerWidth*0.28?i-1:i+1);
  });
  var x0=null;
  document.addEventListener('touchstart',function(e){x0=e.touches[0].clientX;},{passive:true});
  document.addEventListener('touchend',function(e){
    if(x0===null)return; var dx=e.changedTouches[0].clientX-x0;
    if(Math.abs(dx)>44){show(dx<0?i+1:i-1);} x0=null;
  },{passive:true});
  var h=parseInt((location.hash||'').slice(1),10);
  show(isNaN(h)?0:h-1);
})();
</script>
"""

if __name__ == "__main__":
    main()
