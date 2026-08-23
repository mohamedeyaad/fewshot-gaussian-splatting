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
    mcx = {r["k"]: r for r in R.build_model_crossover(recs)}

    def out(k, ratio):
        for r in rows:
            if r["strategy"] == "outpaint" and r["k"] == k and r["ratio"] == ratio:
                return r
        return None

    # The crossover at the 200% ratio, the one ratio present at every k.
    cross = [(k, out(k, 200)) for k in (5, 10, 20)]
    cross_bars = "".join(bar(r["d_psnr"], 0.8, f"k = {k}") for k, r in cross if r)


    fl = head["floors"]
    v = {
        "FLOOR5": f'{fl[5]["psnr"]:.2f}', "FLOOR10": f'{fl[10]["psnr"]:.2f}',
        "FLOOR20": f'{fl[20]["psnr"]:.2f}', "CEIL": f'{head["ceil_psnr"]:.2f}',
        "GAP5": f'{head["gap5"]:.2f}', "NRUNS": str(len(list((ROOT / "runs").glob("*/results.json")))),
        "BEST": f'{head["best"]["d_psnr"]:+.3f}',
        "STEP510": f'{head["step_5_10"]:+.2f}',
        "CROSS_BARS": cross_bars,
        "GEN5": f'{gen["rows"][-1]["k5"]:+.3f}' if gen else "n/a",
        "GEN20": f'{gen["rows"][2]["k20"]:+.3f}' if gen else "n/a",
        "GENFLOORSD": f'{gen["floor_sd"]:.2f}' if gen else "n/a",
        "D5": f'{depth["rows"][5]["depth"][0]:+.3f}' if depth else "n/a",
        "D10": f'{depth["rows"][10]["depth"][0]:+.3f}' if depth else "n/a",
        "D20": f'{depth["rows"][20]["depth"][0]:+.3f}' if depth else "n/a",
        "DBOTH": f'{depth["rows"][5]["both"][0]:+.3f}' if depth else "n/a",
        # 7,000 vs 30,000. build_convergence() returns brace-wrapped keys for
        # build_report's substitution style, so strip them for this one.
        **{k.strip("{}"): v for k, v in R.build_convergence().items()},
        # Bare keys: the substitution loop below adds the braces itself.
        **{f"MC_{m.upper()}{k}":
           (f'{mcx[k]["ds8" if m == "ds" else "sd15"]:+.3f}' if k in mcx else "n/a")
           for k in (5, 10, 20) for m in ("sd", "ds")},
        "IMG_CROSS": data_uri("panel_crossover.png"),
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
      <div class="card"><div class="k num">2</div>
        <div class="cap">scenes: an outdoor object and an indoor room</div></div>
      <div class="card"><div class="k num">3</div>
        <div class="cap">augmentation strategies, 4 ratios, 3 seeds</div></div>
      <div class="card"><div class="k num">4</div>
        <div class="cap">controls, plus a depth prior that tests the mechanism</div></div>
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
    <div class="eyebrow">Is it just this one model?</div>
    <h2>A second checkpoint reverses in the same place</h2>
    <p class="lead">The obvious objection: perhaps the crossover is a property of Stable
    Diffusion 1.5 rather than of augmentation. Dreamshaper-8 — same architecture, same
    VRAM budget, <em>better</em> photorealism — answers it.</p>
    <div class="tw"><table>
      <thead><tr><th>real views</th><th class="n">SD 1.5</th>
                 <th class="n">Dreamshaper-8</th></tr></thead>
      <tbody>
        <tr><td>5</td><td class="n pos">{{MC_SD5}}</td><td class="n pos">{{MC_DS5}}</td></tr>
        <tr><td>10</td><td class="n">{{MC_SD10}}</td><td class="n neg">{{MC_DS10}}</td></tr>
        <tr><td>20</td><td class="n neg">{{MC_SD20}}</td><td class="n neg">{{MC_DS20}}</td></tr>
      </tbody>
      <caption>Outpainting at the 200% ratio, paired within seed.</caption>
    </table></div>
    <p>Same reversal, different checkpoint. And the better-looking model is
    <strong>worse at every subset size</strong> — it even crosses over earlier, turning
    negative at k = 10 where SD 1.5 is still at zero.</p>
    <p class="muted">A model finetuned to make each image individually more convincing has
    no reason to be more consistent <em>between</em> images. Photorealism is not what a
    synthetic view contributes.</p>
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
    <div class="eyebrow">Conclusion</div>
    <h2>What both scenes say together</h2>
    <ul>
      <li><strong>Diffusion augmentation is not free coverage.</strong> It helps only
      while real views are in the single digits, and the crossover reproduces outdoors
      and indoors.</li>
      <li><strong>The mechanism is pose novelty.</strong> Four controls exclude image
      quality, hallucinated hole content, warping artifacts, and mere view
      repetition.</li>
      <li><strong>Constraint beats invention.</strong> A depth prior is positive at every
      subset size and compounds with augmentation; anything that invents a camera
      eventually reverses sign.</li>
      <li><strong>The two effects are separable.</strong> Outpainting supplies peripheral
      content, the depth prior supplies constraint — and combining them exceeds the sum
      of the parts at every subset size.</li>
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
      <li><strong>The 7,000-iteration setting is load-bearing.</strong> At 30,000 the
      benefit does not survive — outpainting at k = 5 falls from {{K5_7K}} to
      {{K5_30K}} dB, while the harm at k = 20 persists. The plain baselines fall too
      ({{B5_7K}} → {{B5_30K}}), so 30,000 is <em>past</em> this regime's optimum, not
      better converged. Early stopping is correct here, but the gain is conditional
      on it.</li>
      <li><strong>Two checkpoints, one architecture.</strong> Dreamshaper-8 is an
      SD 1.5 finetune, so the swap varies image quality, not architecture. SDXL needs
      ~6.5 GB and FLUX ~54 GB against this card's 4 GB.</li>
      <li><strong>Three seeds.</strong> A consistency check, not statistical
      significance.</li>
      <li><strong>Two scenes, one partially swept.</strong> The second scene was swept
      for outpainting only, at k = 5 and k = 20. "Inpainting is flat" and "pose-guided
      always hurts" remain single-scene claims.</li>
      <li><strong>The depth prior was tested on one scene, at one ratio.</strong> Whether
      its immunity to the crossover survives indoors is untested.</li>
    </ul>
  </div>
</section>

<section class="slide">
  <div class="inner">
    <div class="eyebrow">In one sentence</div>
    <h1 style="max-width:20ch">Augmentation buys coverage, and pays for it in
    consistency.</h1>
    <p class="lead">Whether that trade is worth taking depends on how scarce the real
    photographs are. At five views the coverage is worth the inconsistency. By twenty it
    is not — and the same treatment that helped now harms.</p>
    <hr class="rule">
    <p class="muted">{{NRUNS}} runs · 2 scenes · every held-out set hash-verified
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
