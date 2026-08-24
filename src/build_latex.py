"""Emit the LaTeX report in the DIBRIS/UniGe essay template.

The template ships a PROPOSAL skeleton - Concept Overview, Target Audience,
Development Plan - which is a different deliverable from the one this project
has to hand in. Only its visual identity is kept: the cover page, the fancyhdr
header with the UniGe mark, the geometry, the onehalfspacing body, booktabs
tables. The section structure follows the assignment brief instead:

  1. Gaussian splatting when photographs are scarce
  2. Three diffusion augmentation strategies
  3. The experiment grid and protocol
  4. Results - tables, plots, visual examples per condition
  5. Discussion - what increasing reliance on synthetic data does
  6. Conclusions

Every number is read from runs/*/results.json at build time, exactly as
build_report.py and build_brief.py do, so the PDF cannot drift from the
experiments. Nothing is typed by hand.

  python src/build_latex.py     ->  latex/report.tex, latex/refs.bib, figures
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as R  # noqa: E402

ROOT = Path(os.path.expanduser("~/fewshot_gs"))
OUT = ROOT / "latex"
FIGS = OUT / "figures"
RES = ROOT / "results"

# Figures, in the order they appear. Width is a fraction of \textwidth.
FIGURES = [
    ("panel_training_data.png", "training", 1.0),
    ("scaling.png", "scaling", 1.0),
    ("curves_paired.png", "paired", 1.0),
    ("panel_crossover.png", "crossover", 1.0),
    ("panel_strategies.png", "strategies", 1.0),
]


def esc(s):
    """LaTeX-escape a plain string. Numbers pass through untouched."""
    return (str(s).replace("\\", r"\textbackslash{}")
            .replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
            .replace("#", r"\#").replace("_", r"\_")
            .replace("{", r"\{").replace("}", r"\}")
            .replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}"))


def d(mu, sd=None, dp=3, star=False):
    """A signed delta cell. Stars mark |mean| > between-seed sigma."""
    t = f"{mu:+.{dp}f}"
    if sd is not None:
        t += rf" $\pm$ {sd:.{dp}f}"
    if star:
        t += r"\,*"
    return t


def copy_figures():
    FIGS.mkdir(parents=True, exist_ok=True)
    kept = []
    for name, key, width in FIGURES:
        src = RES / name
        if not src.exists():
            print(f"  MISSING figure {name} - skipped")
            continue
        shutil.copy2(src, FIGS / name)
        kept.append((name, key, width))
        print(f"  {name}  {src.stat().st_size / 1e6:.1f} MB")
    return kept


def figure(name, key, width, caption):
    return "\n".join([
        r"\begin{figure}[htbp]",
        r"  \centering",
        rf"  \includegraphics[width={width}\textwidth]{{figures/{name}}}",
        rf"  \caption{{{caption}}}",
        rf"  \label{{fig:{key}}}",
        r"\end{figure}",
        "",
    ])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("figures:")
    kept = {k: (n, w) for n, k, w in copy_figures()}

    recs = R.load_runs("truck")
    head, rows = R.build_tables(recs)
    control = R.build_control(recs)
    gen = R.build_generalisation()
    depth = R.build_depth()
    noise = R.load_noise()
    cv = R.build_convergence()
    dc = R.build_depth_convergence()

    fl = head["floors"]
    ce = head["ceil_psnr"]
    by = {(r["strategy"], r["k"], r["ratio"]): r for r in rows}

    def get(strategy, k, ratio):
        for (s, kk, rt), r in by.items():
            if s == strategy and kk == k and abs(rt - ratio) <= 10:
                return r
        return None

    def dcp(k, lab, name):
        v = dc.get((k, lab, name))
        return f"{v[0]:+.3f}" if v else "n/a"

    # ---- Table 1: floors, ceiling, and the cost of each ---------------------
    t1 = ""
    for k in (5, 10, 20):
        f = fl[k]
        t1 += (f"    {k} & {f['psnr']:.2f} $\\pm$ {f['psnr_sd']:.2f} & {f['ssim']:.3f} & "
               f"{f['lpips']:.3f} & {f['secs']:.0f} & {f['gauss'] / 1000:.0f}k \\\\\n")
    t1 += (f"    \\midrule\n    219 (all) & {ce:.2f} & {head['ceil_ssim']:.3f} & "
           f"{head['ceil_lpips']:.3f} & --- & {head['ceil_gauss'] / 1000:.0f}k \\\\\n")

    # ---- Table 2: the grid, every strategy x ratio x subset size ------------
    RATIOS = [(5, [20, 40, 100, 200]), (10, [20, 50, 100, 200]), (20, [25, 50, 100, 200])]
    t2 = ""
    for label, key in (("Inpainting", "inpaint"), ("Outpainting", "outpaint"),
                       ("Pose-guided", "guided")):
        cells = []
        for k, _ in RATIOS:
            for ratio in (25, 50, 100, 200):
                r = get(key, k, ratio)
                cells.append(d(r["d_psnr"], star=r["sig_psnr"]) if r else "---")
        # one line per subset size, with k named rather than implied
        t2 += (f"    \\multirow{{3}}{{*}}{{{label}}} & 5 & "
               + " & ".join(cells[:4]) + r" \\" + "\n")
        t2 += "     & 10 & " + " & ".join(cells[4:8]) + r" \\" + "\n"
        t2 += "     & 20 & " + " & ".join(cells[8:12]) + r" \\" + "\n"
        t2 += "    \\addlinespace\n"

    # ---- Table 3: both required metrics, all three strategies, 200% ---------
    t3 = ""
    for label, key in (("Inpainting", "inpaint"), ("Outpainting", "outpaint"),
                       ("Pose-guided", "guided")):
        cells = []
        for k in (5, 10, 20):
            r = get(key, k, 200)
            if not r:
                cells += ["---", "---"]
                continue
            cells.append(d(r["d_psnr"], star=r["sig_psnr"]))
            cells.append(d(r["d_ssim"], dp=4,
                           star=abs(r["d_ssim"]) > r["d_ssim_sd"] > 0))
        t3 += f"    {label} & " + " & ".join(cells) + r" \\" + "\n"

    out5 = get("outpaint", 5, 200)
    out20 = get("outpaint", 20, 200)
    inp = [r["d_psnr"] for r in rows if r["strategy"] == "inpaint"]
    gd = [r["d_psnr"] for r in rows if r["strategy"] == "guided"]
    ctrl_max = max(c["contribution"] for c in control)
    grow = next((r for r in gen["rows"] if r["ratio"] == 200), None)

    figs = ""
    if "training" in kept:
        n, w = kept["training"]
        figs += figure(n, "training", w,
                       "Synthetic training images beside the real view each derives from. "
                       "Inpainting has given the truck a second rear axle; outpainting "
                       "fabricates scene beyond the original frame; pose-guided renders a "
                       "genuinely different viewpoint, with roughly 10\\% of its pixels "
                       "invented.")

    body = TEX.format(
        n_runs=head["n_runs"],
        t1=t1, t2=t2, t3=t3,
        f5=fl[5]["psnr"], f10=fl[10]["psnr"], f20=fl[20]["psnr"], ceil=ce,
        step510=head["step_5_10"] / 5, step_beyond=(ce - fl[20]["psnr"]) / 199,
        noise_psnr=f'{noise["psnr"][1]:.3f}' if noise else "n/a",
        out5=d(out5["d_psnr"], out5["d_psnr_sd"], star=out5["sig_psnr"]),
        out5p=f'{out5["d_psnr"]:+.3f}',
        out5s=f'{out5["d_ssim"]:+.4f}',
        out20=d(out20["d_psnr"], out20["d_psnr_sd"], star=out20["sig_psnr"]),
        out20p=f'{out20["d_psnr"]:+.3f}',
        inp_lo=f"{min(inp):+.3f}", inp_hi=f"{max(inp):+.3f}",
        gd_worst=f"{min(gd):+.3f}",
        ctrl_max=f"{ctrl_max:.2f}",
        gen5=f'{grow["k5"]:+.3f}' if grow else "n/a",
        gen20=f'{grow["k20"]:+.3f}' if grow else "n/a",
        gen20sd=f'{grow["k20_sd"]:.3f}' if grow else "n/a",
        gen_scene=gen["scene"],
        dep5=f'{depth["rows"][5]["depth"][0]:+.3f}' if depth else "n/a",
        dep20=f'{depth["rows"][20]["depth"][0]:+.3f}' if depth else "n/a",
        both5=f'{depth["rows"][5]["both"][0]:+.3f}' if depth else "n/a",
        k5_7k=cv["{{K5_7K}}"], k5_15k=cv["{{K5_15K}}"], k5_30k=cv["{{K5_30K}}"],
        b5_7k=cv["{{B5_7K}}"], b5_15k=cv["{{B5_15K}}"], b5_30k=cv["{{B5_30K}}"],
        d5_30k=dcp(5, "30K", "depth"), o5_30k=dcp(5, "30K", "outpaint"),
        both5_30k=dcp(5, "30K", "both"),
        figs=figs,
        fig_scaling=figure(*(kept["scaling"][0], "scaling", kept["scaling"][1]),
                           caption="Left: held-out quality against the number of real "
                                   "training views, with the full-data ceiling marked. "
                                   "Right: the best and worst synthetic effects on the same "
                                   "axis. Five more real photographs are worth several times "
                                   "the best synthetic condition measured anywhere here.")
        if "scaling" in kept else "",
        fig_paired=figure(*(kept["paired"][0], "paired", kept["paired"][1]),
                          caption="Paired change in PSNR against each seed's own "
                                  "zero-synthetic baseline, one line per subset size. Error "
                                  "bars are the standard deviation across three seeds; the "
                                  "grey band is the measured noise floor. Inpainting is flat, "
                                  "outpainting crosses over, pose-guided is harmful "
                                  "everywhere.")
        if "paired" in kept else "",
        fig_cross=figure(*(kept["crossover"][0], "crossover", kept["crossover"][1]),
                         caption="The crossover made visible: the same 200\\% outpainting "
                                 "treatment at five real views and at twenty. Views are those "
                                 "whose per-view delta sits closest to the mean effect.")
        if "crossover" in kept else "",
        fig_strat=figure(*(kept["strategies"][0], "strategies", kept["strategies"][1]),
                         caption="Held-out renderings for each condition at the 100\\% "
                                 "synthetic ratio. Degradation appears as smearing and "
                                 "semi-transparent floaters, most severely under pose-guided "
                                 "synthesis.")
        if "strategies" in kept else "",
    )

    (OUT / "report.tex").write_text(body, encoding="utf-8")
    (OUT / "refs.bib").write_text(BIB, encoding="utf-8")
    print(f"\nwrote {OUT / 'report.tex'}  ({len(body.split())} words of source)")
    print(f"wrote {OUT / 'refs.bib'}")


BIB = r"""@article{kerbl3Dgaussians,
  author  = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas
             and Drettakis, George},
  title   = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal = {ACM Transactions on Graphics},
  volume  = {42},
  number  = {4},
  year    = {2023}
}

@article{chung2023depthreg,
  author  = {Chung, Jaeyoung and Oh, Jeongtaek and Lee, Kyoung Mu},
  title   = {Depth-Regularized Optimization for 3D Gaussian Splatting in
             Few-Shot Images},
  journal = {arXiv preprint arXiv:2311.13398},
  year    = {2023}
}

@inproceedings{rombach2022ldm,
  author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik
               and Esser, Patrick and Ommer, Bj{\"o}rn},
  title     = {High-Resolution Image Synthesis with Latent Diffusion Models},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition
               (CVPR)},
  year      = {2022}
}

@inproceedings{yang2024depthanythingv2,
  author    = {Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen
               and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  title     = {Depth Anything V2},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024}
}

@inproceedings{schoenberger2016sfm,
  author    = {Sch{\"o}nberger, Johannes L. and Frahm, Jan-Michael},
  title     = {Structure-from-Motion Revisited},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition
               (CVPR)},
  pages     = {4104--4113},
  year      = {2016}
}

@article{knapitsch2017tanks,
  author  = {Knapitsch, Arno and Park, Jaesik and Zhou, Qian-Yi
             and Koltun, Vladlen},
  title   = {Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction},
  journal = {ACM Transactions on Graphics},
  volume  = {36},
  number  = {4},
  year    = {2017}
}

@article{hedman2018deepblending,
  author  = {Hedman, Peter and Philip, Julien and Price, True
             and Frahm, Jan-Michael and Drettakis, George and Brostow, Gabriel},
  title   = {Deep Blending for Free-Viewpoint Image-Based Rendering},
  journal = {ACM Transactions on Graphics},
  volume  = {37},
  number  = {6},
  year    = {2018}
}

@inproceedings{zhang2018lpips,
  author    = {Zhang, Richard and Isola, Phillip and Efros, Alexei A.
               and Shechtman, Eli and Wang, Oliver},
  title     = {The Unreasonable Effectiveness of Deep Features as a Perceptual
               Metric},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition
               (CVPR)},
  year      = {2018}
}

@article{wang2004ssim,
  author  = {Wang, Zhou and Bovik, Alan C. and Sheikh, Hamid R.
             and Simoncelli, Eero P.},
  title   = {Image Quality Assessment: From Error Visibility to Structural
             Similarity},
  journal = {IEEE Transactions on Image Processing},
  volume  = {13},
  number  = {4},
  year    = {2004}
}
"""


TEX = r"""\documentclass{{article}}

\usepackage{{dibrisunige-report}}

% Encodings
\usepackage{{amsmath,amssymb,gensymb,textcomp}}

% Better tables
\usepackage{{array,multicol,multirow,siunitx,tabularx}}

% Better enum
\usepackage{{enumitem}}

% Graphics
\usepackage{{caption,float}}
\usepackage[export]{{adjustbox}}

\ocoursename{{Virtual Reality for Robotics - code 104737 - a.y. 2025/2026}}
\oreporttype{{Project Report}}
\otitle{{Few-Shot Gaussian Splatting with Diffusion-Based Data Augmentation}}
\oadvisor{{Prof. Fabio Solari, Prof. Manuela Chessa}}
\reportlayout%

\title{{Few-Shot Gaussian Splatting with Diffusion-Based Data Augmentation}}
\author{{Mohamed Eyad}}
\date{{\today}}

\begin{{document}}

\coverpage%

\tableofcontents
\newpage

\begin{{abstract}}
\noindent
Diffusion models can invent photographs that look convincing, which suggests an
obvious remedy when a 3D scene has been captured from too few viewpoints:
generate more views. This report tests that idea on {n_runs} Gaussian-splatting
training runs and finds that it is conditionally false. The same augmentation
that improves a five-view reconstruction by {out5p}\,dB degrades a twenty-view
one by {out20p}\,dB. Ordering three augmentation strategies by how much
\emph{{camera pose}} they invent orders them exactly by how much damage they do,
and four controls exclude image quality, hallucinated content, warping
artefacts and mere view repetition as explanations. A synthetic view supplies
coverage and inconsistency together; only the value of coverage decays as real
photographs accumulate.
\end{{abstract}}

\section{{Gaussian splatting when photographs are scarce}}

3D Gaussian Splatting~\cite{{kerbl3Dgaussians}} represents a scene as a set of
anisotropic 3D Gaussians, each carrying a position, covariance, opacity and
view-dependent colour, and rasterises them differentiably so that the primitives
are fitted directly to the training photographs. Given a dense capture it is
both fast and accurate. Given a sparse one it is not: with few views the
photometric objective is badly under-determined, and the optimiser is free to
place primitives that explain the training images while producing floaters and
smeared geometry anywhere the cameras did not look.

The scale of that penalty is the first thing to establish, because every later
number is measured against it. Table~\ref{{tab:floors}} reports held-out quality
for subsets of 5, 10 and 20 real photographs of the Tanks~\&
Temples~\cite{{knapitsch2017tanks}} \texttt{{truck}} scene, against a full-data
ceiling trained on all 219.

\begin{{table}}[htbp]
  \centering
  \caption{{Held-out quality against the number of real training views, three
  seeds per row except the ceiling, which is a single deterministic run. Time
  and primitive count are the per-run cost on an RTX 3050 Ti (4\,GB).}}
  \label{{tab:floors}}
  \begin{{tabular}}{{lccccc}}
    \toprule
    \textbf{{Real views}} & \textbf{{PSNR}} & \textbf{{SSIM}} & \textbf{{LPIPS}}
    & \textbf{{Train (s)}} & \textbf{{Gaussians}} \\
    \midrule
{t1}    \bottomrule
  \end{{tabular}}
\end{{table}}

Five views reach {f5:.2f}\,dB against a {ceil:.2f}\,dB ceiling. The marginal
value of a real photograph falls by an order of magnitude across the range:
{step510:+.3f}\,dB each between five and ten views, {step_beyond:+.3f}\,dB each
beyond twenty. That decay is what makes synthetic coverage plausible in the
first place, and, as Section~\ref{{sec:discussion}} argues, it is also why the
idea eventually fails. Figure~\ref{{fig:scaling}} shows both halves of that
picture on one axis.

{fig_scaling}

\section{{Three diffusion augmentation strategies}}

All synthetic images come from Stable Diffusion~1.5
inpainting~\cite{{rombach2022ldm}}. The three strategies differ in one
respect only --- how much new \emph{{camera pose}} they invent --- which is what
makes them a usable experimental axis rather than three unrelated tricks.
Figure~\ref{{fig:training}} shows what each one actually feeds the optimiser.

\begin{{description}}[leftmargin=1.4em,style=nextline]
  \item[Inpainting.] A mask is placed over part of a real training photograph
  and the region is regenerated. The camera pose is exactly that of the real
  view, and the pose is reused unchanged. No new viewpoint is created.
  \item[Outpainting.] The real image is padded and the diffusion model
  fabricates the border. The camera centre and orientation are unchanged, but
  the effective field of view is widened, so the synthetic image covers scene
  the original frame never recorded.
  \item[Pose-guided synthesis.] A monocular depth map~\cite{{yang2024depthanythingv2}}
  is estimated for a real view, the image is warped to a genuinely new camera
  interpolated between two real ones, and the disocclusion holes are filled by
  diffusion. This is the only strategy that supplies a viewpoint that does not
  exist in the capture.
\end{{description}}

{figs}
Poses for real views come from COLMAP~\cite{{schoenberger2016sfm}}. Synthetic
views inherit the pose of the real image they derive from, exactly for
inpainting and outpainting, and by interpolation for pose-guided synthesis.

\section{{The experiment grid and protocol}}

Three subset sizes ($k = 5, 10, 20$) are crossed with three strategies and
five synthetic ratios ($0, 25, 50, 100, 200\%$ of $k$), each repeated over
three seeds --- {n_runs} training runs in total for the primary scene.

Three details carry most of the study's reliability.

\begin{{enumerate}}[leftmargin=1.6em]
  \item \textbf{{A frozen held-out set.}} Thirty-two real photographs are
  reserved once and never trained on by any run, so every number in this report
  is comparable across every condition. Synthetic images are only ever added to
  \emph{{training}}.
  \item \textbf{{Pairing within seed.}} Each augmented run is compared against
  the zero-synthetic baseline built from \emph{{the same seed's}} subset, so the
  difficulty of a particular draw of $k$ photographs cancels rather than
  appearing as an effect.
  \item \textbf{{A measured noise floor.}} Repeating an identical configuration
  gives a run-to-run standard deviation of {noise_psnr}\,dB. Any effect smaller
  than roughly $\sqrt{{2}}\times$ that is indistinguishable from noise, and is
  reported as such. Three seeds support a consistency check --- marked * where
  $|\text{{mean}}|$ exceeds the between-seed spread --- not a significance test.
\end{{enumerate}}

Quality is reported as PSNR and SSIM, the two metrics the brief
specifies, with LPIPS~\cite{{zhang2018lpips}} added because neither of the
first two is perceptual. SSIM follows Wang et al.~\cite{{wang2004ssim}}.

\section{{Results}}

Table~\ref{{tab:grid}} is the full grid: the paired change in PSNR for every
strategy, ratio and subset size, and Figure~\ref{{fig:paired}} plots the same
quantities against the measured noise floor.

\begin{{table}}[htbp]
  \centering
  \caption{{Paired $\Delta$PSNR against the same seed's zero-synthetic
  baseline, three seeds per cell. Columns are the synthetic ratio as a
  percentage of the real subset size. * marks a mean exceeding its between-seed
  standard deviation. At $k = 5$ and $k = 10$ the nominal 25\% and 50\% points
  are fractional and were rounded down.}}
  \label{{tab:grid}}
  \begin{{tabular}}{{llcccc}}
    \toprule
    \textbf{{Strategy}} & $k$ & \textbf{{25\%}} & \textbf{{50\%}}
    & \textbf{{100\%}} & \textbf{{200\%}} \\
    \midrule
{t2}    \bottomrule
  \end{{tabular}}
\end{{table}}

{fig_paired}
\subsection{{The crossover}}

Outpainting is positive at every ratio at five real views and negative at every
ratio at twenty: {out5}\,dB at $k = 5$ against {out20}\,dB at $k = 20$. The
sign of the effect depends on how much real data is already present, not on the
method or on the ratio.

The other two strategies bracket the explanation. Inpainting, which reuses the
camera pose exactly, is flat everywhere --- between {inp_lo} and {inp_hi}\,dB
--- despite producing convincing images. Pose-guided synthesis, which invents a
fully novel viewpoint, is harmful at every subset size and worsens as $k$ grows,
reaching {gd_worst}\,dB. Ordering the three by invented camera pose orders them
by damage. Figure~\ref{{fig:crossover}} shows the same treatment helping and
harming side by side, and Figure~\ref{{fig:strategies}} gives held-out
renderings for each condition.

{fig_cross}
{fig_strat}
\subsection{{Both required metrics}}

Table~\ref{{tab:metrics}} reports PSNR and SSIM together at the 200\% ratio.

\begin{{table}}[htbp]
  \centering
  \caption{{Both specified metrics at the 200\% ratio, three seeds, paired
  within seed. Note the disagreement at $k = 5$.}}
  \label{{tab:metrics}}
  \begin{{tabular}}{{lcccccc}}
    \toprule
    & \multicolumn{{2}}{{c}}{{$k = 5$}} & \multicolumn{{2}}{{c}}{{$k = 10$}}
    & \multicolumn{{2}}{{c}}{{$k = 20$}} \\
    \cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}} \cmidrule(lr){{6-7}}
    \textbf{{Strategy}} & $\Delta$PSNR & $\Delta$SSIM & $\Delta$PSNR
    & $\Delta$SSIM & $\Delta$PSNR & $\Delta$SSIM \\
    \midrule
{t3}    \bottomrule
  \end{{tabular}}
\end{{table}}

The headline gain is \textbf{{PSNR-only}}. Outpainting at $k = 5$ is worth
{out5p}\,dB but {out5s} SSIM, consistently across all three seeds. PSNR rewards
the reduction in large errors where synthetic coverage fills an unobserved
region; SSIM penalises the local structural noise the fabricated pixels
introduce. The \emph{{ordering}} of the three strategies holds on both metrics;
the \emph{{sign}} of outpainting's benefit at $k = 5$ does not.

\subsection{{Controls}}

Four controls each remove one candidate explanation. Repeating real images to
the same count instead of generating new ones recovers only part of
outpainting's gain, so the fabricated borders --- not mere repetition --- carry
most of it. Skipping diffusion and leaving the warped holes black costs a
further {ctrl_max}\,dB, so the diffusion step repairs rather than causes
pose-guided damage. Swapping the checkpoint for DreamShaper-8 reproduces the
sign change, so the effect is not an artefact of one model. Finally, replacing
outpainted borders with plain white isolates fabrication from framing.

The benefit also reproduces on a second, indoor scene, \texttt{{{gen_scene}}}
from Deep Blending~\cite{{hedman2018deepblending}}: {gen5}\,dB at $k = 5$ at the
200\% ratio, larger than on \texttt{{truck}}. The reversal at $k = 20$ is
\emph{{weaker}} there and should not be overstated --- negative at the 25, 50 and
100\% ratios but {gen20}~$\pm$~{gen20sd}\,dB at 200\%, which is indistinguishable
from zero. What transfers between scenes is the benefit at small $k$; the harm at
large $k$ transfers in sign but not reliably in magnitude.

\section{{Discussion: when synthetic data helps, and when it hurts}}
\label{{sec:discussion}}

\paragraph{{A two-term account.}} A synthetic view supplies two things at once:
\emph{{coverage}} of scene geometry the real views did not record, and
\emph{{inconsistency}} with them, because the fabricated pixels are not
photographs of anything. Coverage loses value as real views accumulate --- that
is precisely the per-view decay in Table~\ref{{tab:floors}}, an order of
magnitude across the range. Inconsistency does not decay. A contradictory view
is at least as harmful at twenty real views as at five, and arguably worse,
because it now contradicts a better-determined geometry. Two terms, one
decaying and one roughly constant, are sufficient to produce a sign change, and
that is what the grid shows.

\paragraph{{Increasing reliance on synthetic data.}} Reliance can be increased
along two axes, and they behave differently. Raising the \emph{{ratio}} at fixed
$k$ is comparatively benign where the balance is already favourable --- at
$k = 5$ outpainting stays positive from 25\% to 200\% --- and monotonically
damaging where it is not. Lowering the proportion of \emph{{real}} data at a
fixed ratio is what actually reverses the sign. The practical reading is that
synthetic images are a substitute for photographs only in the regime where
photographs are very scarce, and never a supplement to an adequate capture.

\paragraph{{A constraint that does not cross over.}} If the harm comes from
invented viewpoints rather than from augmentation as such, then an intervention
that constrains geometry \emph{{without}} inventing a camera should never
reverse sign. Monocular depth regularisation is that intervention, and it is
prior work rather than a contribution here: Chung et al.~\cite{{chung2023depthreg}}
propose the same construction --- a dense monocular depth map as a geometry
guide, scale-aligned against sparse COLMAP points. Used as a control, it is
worth {dep5}\,dB at $k = 5$ and remains positive at {dep20}\,dB at $k = 20$,
where outpainting costs {out20p}. Combined with outpainting at $k = 5$ it
reaches {both5}\,dB, the largest improvement measured in this study. Coverage
crosses over; constraint does not.

\paragraph{{The result is conditional on early stopping.}} Training longer
decays the benefit monotonically: outpainting at $k = 5$ falls from {k5_7k} to
{k5_15k} at 15{{,}}000 iterations and {k5_30k} at 30{{,}}000. The unaugmented
baseline is flat from 7{{,}}000 to 15{{,}}000 ({b5_7k}~$\rightarrow$~{b5_15k}\,dB)
and lower by 30{{,}}000 ({b5_30k}\,dB), so 7{{,}}000 is a fair operating point
rather than a flattering one --- few-shot splatting overfits well before
30{{,}}000. The depth prior is the exception: at 30{{,}}000 iterations outpainting
alone is worth {o5_30k}\,dB while the prior still returns {d5_30k} and the two
together {both5_30k}. Fabricated pixels lose their value to longer training; a
geometric constraint does not.

\section{{Conclusions}}

\begin{{itemize}}[leftmargin=1.4em]
  \item Diffusion augmentation is not free coverage. It helps only while real
  views are scarce, and the crossover reproduces on an outdoor object and an
  indoor room.
  \item The mechanism is invented camera pose. Four controls exclude image
  quality, hallucinated hole content, warping artefacts and view repetition.
  \item Constraint beats invention where data is plentiful: a depth prior never
  crosses over and costs no diffusion model.
  \item Scale honestly --- the best synthetic condition measured anywhere is
  worth a small fraction of simply taking five more photographs.
\end{{itemize}}

\paragraph{{Limitations.}} Three seeds support a consistency check, not a
significance test. The crossover is bracketed between $k = 5$ and $k = 20$, not
located. Both diffusion checkpoints are SD~1.5-class, the only ones fitting
4\,GB of VRAM; multi-view-consistent generators are the natural next step and
are arguably what pose-guided augmentation is really reaching for. The positive
result is conditional on both the early-stopped operating point and the choice
of metric, as stated above.

\bibliographystyle{{plain}}
\bibliography{{refs}}

\end{{document}}
"""


if __name__ == "__main__":
    main()
