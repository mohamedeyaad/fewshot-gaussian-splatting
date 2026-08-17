# Few-Shot Gaussian Splatting with Diffusion-Based Data Augmentation

Does generating synthetic training views with a diffusion model help 3D Gaussian
Splatting when you only have a handful of real photographs?

**Short answer: only when you have very few real views, and even then barely.**
The value of a synthetic image depends on how much real data you already have —
it crosses from beneficial to harmful somewhere between 5 and 20 real views.
This repository contains the full pipeline, all 118 training runs, and the
analysis behind that claim.

University of Genoa (UNIGE) — robotics project.

---

## Headline result

Scene: Tanks & Temples `truck`, 251 images, 32 held out and frozen across every
run.

### What real photographs are worth

| real views | PSNR | SSIM | LPIPS | Δ | per view |
|---|---|---|---|---|---|
| 5 | 15.20 ± 0.33 | 0.5375 | 0.4096 | — | — |
| 10 | 17.11 ± 0.39 | 0.6584 | 0.3017 | +1.91 | **+0.382** |
| 20 | 19.74 ± 0.33 | 0.7842 | 0.2027 | +2.63 | **+0.263** |
| 219 | 25.23 | 0.9070 | 0.1227 | +5.50 | +0.028 |

### What synthetic images are worth

ΔPSNR against the same seed's own baseline at the same subset size. 3 seeds.
`*` = |mean| exceeds the between-seed standard deviation.

**Outpainting** — crosses from helpful to harmful:

| ratio | k=5 | k=10 | k=20 |
|---|---|---|---|
| 20–25 % | **+0.140** \* | **+0.172** \* | −0.283 \* |
| 40–50 % | **+0.140** \* | +0.013 | −0.542 \* |
| 100 % | **+0.182** \* | −0.423 \* | −0.983 \* |
| 200 % | **+0.285** \* | −0.003 | −0.618 \* |

**Inpainting** — flat, and the only strategy independent of subset size:

| ratio | k=5 | k=10 | k=20 |
|---|---|---|---|
| 100 % | −0.166 \* | −0.213 \* | −0.161 \* |
| 200 % | +0.029 | −0.141 | −0.102 \* |

**Pose-guided** — harmful everywhere, and worse the more real data it is added to:

| ratio | k=5 | k=10 | k=20 |
|---|---|---|---|
| 20–25 % | −0.309 \* | −0.451 \* | −1.017 \* |
| 200 % | −1.013 \* | −1.451 \* | −1.884 \* |

Measured training noise floor is σ = 0.039 dB (≈ 0.055 dB paired), so these
effects are real even where small.

### The finding

**Pose novelty explains all three behaviours.** Order the strategies by how much
new *camera pose* each invents:

| | new pose | new information | new contradiction | effect |
|---|---|---|---|---|
| inpainting | none — pose copied | none | none | ~0, regardless of k |
| outpainting | same centre, wider frustum | some | some | helps when starved, hurts when not |
| pose-guided | fully novel viewpoint | most | most | always hurts, worse as baseline improves |

A synthetic view supplies coverage and inconsistency in fixed proportion.
Coverage loses value as real views accumulate; inconsistency does not. Two
terms, one decaying and one roughly constant, produce a sign change — which is
what the data show.

**The exchange rate:** the best synthetic condition anywhere (+0.285 dB) is
**6.7× less valuable than simply taking five more photographs** (+1.91 dB).

Full write-up in [`results/report.html`](results/report.html).

---

## Repository layout

```
src/                    all pipeline code (see "Reproducing" below)
patches/                the two required edits to upstream 3DGS
subsets/                view-selection manifests + the frozen test split
synthetic/              every generated image + poses.json (source-view linkage)
runs/*/results.json     raw metrics for all 118 runs - the experimental record
results/                figures, summary tables, and the final report
build_rasterizer.sh     one-shot CUDA submodule build
requirements.txt        Python dependencies
```

Not committed (see `.gitignore`): `venv/` (7.5 GB), `data/` (1.4 GB, downloaded),
`gaussian-splatting/` (upstream clone), `runs/` checkpoints and renders (~15 GB),
and `scenes/` (symlink farms with absolute paths — regenerate them).

Every number in the report is read back out of `runs/*/results.json` at build
time by `src/build_report.py`, so the prose cannot drift away from the data.

---

## Setup

Built and tested on **WSL2 Ubuntu 24.04**, RTX 3050 Ti Laptop (**4 GB VRAM**),
7.6 GB RAM. The 4 GB budget is the binding constraint throughout and is why
several design decisions look the way they do.

### 1. CUDA toolkit

You need `nvcc` matching your PyTorch CUDA version (12.8 here).

```bash
# NVIDIA's apt repo - Ubuntu 24.04, exact 12.8 match
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update
sudo apt install -y cuda-toolkit-12-8
```

> **Two dead ends worth avoiding.** Ubuntu's own `nvidia-cuda-toolkit` package
> is CUDA 12.0, which does not compile against GCC 13 (the 24.04 default).
> And the pip wheels `nvidia-cuda-nvcc-cu12` (12.8.93 / 12.9.86) ship `ptxas`
> only — there is no `nvcc` frontend inside them, so `pip install`-ing your way
> to a compiler does not work.

### 2. Python environment

```bash
python3.12 -m venv venv
./venv/bin/pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu128
./venv/bin/pip install -r requirements.txt
```

### 3. Upstream 3DGS + patches

```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive --depth 1
cd gaussian-splatting
git apply ../patches/01-dataset_readers-explicit-split.patch
git apply --directory=submodules/diff-gaussian-rasterization \
          ../patches/02-rasterizer-cstdint.patch
cd ..
```

Verified against upstream `54c035f` (main repo) and `9c5c202`
(diff-gaussian-rasterization).

**What the patches do, and why both are mandatory:**

| patch | reason |
|---|---|
| `02-rasterizer-cstdint` | GCC 13 stopped including `<cstdint>` transitively. Without it the build dies on `'uintptr_t' is not a member of 'std'`. One line; purely a compiler-compatibility fix. |
| `01-dataset_readers-explicit-split` | Upstream picks the test set with the LLFF rule `idx % 8 == 0` and trains on *everything else*. This project needs an arbitrary K-image training subset while the **test set stays byte-identical across all 118 runs**. The patch makes a `split.json` in the scene root override the rule; with no such file, behaviour is exactly as upstream. |

### 4. Build the CUDA submodules

```bash
bash build_rasterizer.sh
```

> **`--no-build-isolation` is required.** The three submodules' `setup.py` files
> import `torch` at module level, and PEP 517 build isolation hides the venv's
> torch from them. Without the flag pip fails with a bare
> `get_requires_for_build_wheel ... exit 1` that says nothing about the cause.
> The script also sets `TORCH_CUDA_ARCH_LIST="8.6"` (build for this GPU only —
> far faster) and `MAX_JOBS=4` (each `nvcc` job is memory-hungry).

The script ends with an import test; all three of
`diff_gaussian_rasterization`, `simple_knn._C`, `fused_ssim` must load.

### 5. Data

```bash
mkdir -p data && cd data
wget https://storage.googleapis.com/gresearch/refraw360/360_v2.zip  # or:
# Tanks & Temples from the 3DGS release:
#   https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
```

Expected at `data/tandt/truck/` with `images/` (251 JPEGs, 979×546) and
`sparse/0/` (COLMAP binary model at 1957×1091, PINHOLE). Inspect a model with
`./venv/bin/python src/inspect_colmap.py`.

---

## Reproducing the experiments

Total compute ≈ **15 hours** on the 4 GB laptop GPU. Every stage is idempotent —
`run_experiment.py` skips any scene that already has a `results.json`, so an
interrupted batch can simply be re-run. This matters: the sweep was interrupted
twice during development (once by a full host disk, once by a CUDA OOM) and both
times resumed without losing completed work.

```bash
VPY=./venv/bin/python

# 1. Choose the few-shot subsets and freeze the test split.
#    Farthest-point sampling over camera centres, 3 seeds. Writes
#    subsets/*.json, test_split.json, and camera-position maps.
$VPY src/select_subsets.py --k 5 10 20 --seeds 0 1 2 --methods fps random --plot

# 2. Full-data ceiling (219 real views).
$VPY src/make_full_manifest.py
$VPY src/build_scene.py --manifest subsets/truck_k219_seed0_full.json
$VPY src/run_experiment.py --scene scenes/truck_k219_seed0_full_fake0

# 3. Few-shot floors at every subset size (k = 5, 10, 20 x 3 seeds).
KS="5 10 20" bash src/run_floors.sh

# 4. The augmentation sweep at k=10 (3 strategies x 4 ratios x 3 seeds).
STRATEGY=inpaint  bash src/run_curve.sh
STRATEGY=outpaint bash src/run_curve.sh
STRATEGY=guided   bash src/run_curve.sh

# 5. The same sweep at k=5 and k=20 (72 more runs, ~8 h).
bash src/run_all_scales.sh

# 6. Noise floor: same scene, same config, 3 repeats.
bash src/measure_train_noise.sh

# 7. Analysis and deliverables.
$VPY src/collect_results.py   # -> results/summary.md, results/runs_raw.csv
$VPY src/crossover_table.py   # -> the headline matrix, to stdout
$VPY src/plot_curves.py       # -> results/{curves_paired,scaling,curves_absolute}.png
$VPY src/make_panels.py       # -> results/panel_*.png
$VPY src/build_report.py      # -> results/report.html (self-contained)
```

Synthetic counts differ per subset size because the ratio is relative to `k`:

| k | fakes | realised ratios |
|---|---|---|
| 5 | 1 2 5 10 | 20 / 40 / 100 / 200 % |
| 10 | 2 5 10 20 | 20 / 50 / 100 / 200 % |
| 20 | 5 10 20 40 | 25 / 50 / 100 / 200 % |

Only k=20 divides cleanly into the spec's 25/50/100/200%; at k=5 and k=10 the
25% point is fractional (1.25 and 2.5 images) and rounds down.

> **Disk.** The 118 runs produce ~15 GB of Gaussian checkpoints
> (`runs/*/point_cloud`). Every metric is extracted into `results.json` during
> the run, so the checkpoints can be deleted afterwards — `rm -rf
> runs/*/point_cloud runs/*/input.ply` — without losing anything the analysis
> needs. On WSL, freeing space inside the distro does not return it to Windows;
> compact the vhdx with `diskpart` (`attach vdisk readonly` / `compact vdisk` /
> `detach vdisk`) from an Administrator prompt.

Geometry unit tests for the pose-guided warp (CPU only, no GPU needed):

```bash
$VPY src/test_warp.py
```

Test 1 warps a view onto its own pose: 99.43 % coverage, mean |diff| **0.0000**,
rotation error 1.1e-16. Test 2 checks that hole fraction grows monotonically
with baseline (4.47 % → 8.84 % → 16.65 %) and that interpolation fraction 1.0
lands on the neighbour pose to 2.2e-16.

---

## The three strategies

Each is defined by **how it obtains a camera pose for the synthetic image** —
that turns out to be what determines whether it helps or hurts.

| | pose | what is synthesised | src |
|---|---|---|---|
| **Inpainting** | copied exactly from the real view | a random 8–22 % rectangle/ellipse, composited back at native resolution so only masked pixels are synthetic | [`gen_inpaint.py`](src/gen_inpaint.py) |
| **Outpainting** | same centre, **widened intrinsics** | the border ring around the real frame; focal lengths unchanged, principal point shifted, FOV 80.1°×50.5° → 92.9°×61.1° | [`gen_outpaint.py`](src/gen_outpaint.py) |
| **Pose-guided** | a **new** pose interpolated between two real cameras | monocular depth (Depth Anything V2) anchored to sparse COLMAP points, forward-warped with a z-buffer, disocclusion holes filled by diffusion | [`gen_guided.py`](src/gen_guided.py) |

Depth anchoring solves a least-squares scale/shift in disparity space
(`pred_disp * a + b ≈ 1/z_colmap`) at the sparse keypoints, with 95th-percentile
residual trimming. Mean R² across views = **0.988**.

Filenames encode the linkage the brief asks for:
`synth_<strategy>_<source_view>_v<NN>.jpg`, with full camera parameters in each
directory's `poses.json`.

Stable Diffusion 1.5 inpainting runs at 704×392 with `variant="fp16"`,
attention slicing and VAE slicing — 2.65 GB peak, which fits 4 GB only because
diffusion and 3DGS training never run concurrently.

---

## Experimental design notes

- **Frozen test set.** 32 views chosen by the LLFF `idx % 8` rule and held
  constant across all 118 runs. No synthetic image is ever derived from a test
  view.
- **Nested synthetic sets.** The 5-image condition contains the 2-image
  condition's images, and so on, so the ratio sweep is a genuine dose-response
  curve rather than four unrelated samples.
- **Paired comparison.** Every delta is against the *same seed's* own 0-fake
  baseline, which removes the large between-seed variance (σ = 0.39 dB) that
  otherwise swamps effects an order of magnitude smaller.
- **Measured noise floor.** Python RNG is seeded, so residual variation is CUDA
  `atomicAdd` nondeterminism in the rasteriser backward pass, amplified by
  densification. Three identical repeats give σ_PSNR = 0.039 dB. Any claimed
  effect below ≈ 0.055 dB paired is not distinguishable from noise, and the
  report says so.
- **Cost accounting.** `run_experiment.py` samples VRAM from a monitor thread
  and records wall-clock training time and final Gaussian count in every
  `results.json`.

---

## Known limitations

- **The pose-guided artifact is a confound.** Residual speckle survives at depth
  discontinuities, where the warp produces thin holes that Telea inpainting
  fills imperfectly. Part of the pose-guided damage may be image degradation
  rather than pose novelty. The clean control (warp to a new pose, leave holes
  black, no diffusion at all) is designed but was not run.

  One piece of evidence argues *against* the artifact explanation: sparser
  subsets need longer interpolation baselines and therefore larger holes, so the
  artifact account predicts more damage at k=5. Measured damage is smallest at
  k=5 (−1.013 dB at 200%) and largest at k=20 (−1.884 dB) — the opposite.
- **The crossover is bracketed, not located.** Outpainting is positive at k=5
  and negative at k=20. The sign change lies somewhere between, but three subset
  sizes do not resolve where.
- **One scene.** All conclusions are from `truck`. A second scene
  (`drjohnson`, indoor room-scale, already present in `data/db/`) would test
  whether this generalises across capture regimes. Note the Stable Diffusion
  canvas is tuned to truck's 1.7930 aspect ratio; drjohnson is 1.5205 and would
  need roughly 656×432 rather than 704×392.
- **One diffusion model.** SD 1.5 inpainting only, chosen for the 4 GB budget.
  SDXL and FLUX do not fit. `src/check_alt_models.sh` enumerates the
  alternatives that were considered, including the multi-view-consistent
  generators (Zero123++, SV3D, ImageDream) that are arguably the actual fix for
  what pose-guided augmentation is trying to do.
- **7000 iterations**, not the upstream 30000, to keep 118 runs tractable. The
  full-data run reaches 25.23 dB against ≈ 25.4 dB published at full schedule,
  so the pipeline is validated, but absolute numbers are slightly below
  literature values.
- **Random-selection subsets were never trained.** `select_subsets.py` writes
  both `fps` and `random` manifests; only `fps` was swept. Comparing the two
  would isolate how much view *placement* matters relative to view *count*.

---

## Hardware and cost

| stage | peak VRAM | time |
|---|---|---|
| Diffusion generation (SD 1.5 fp16, 704×392) | 2.65 GB | ~2 min / 20 images |
| Few-shot training (10–30 views, 7000 iters) | 1.4 GB | ~5 min |
| Full-data training (219 views, 7000 iters) | 3.0 GB | ~8 min |

Peak across the whole project stayed under 3.0 GB. Diffusion and training are
never concurrent — an early baseline was contaminated by exactly that mistake
and had to be discarded and re-run.

---

## References

- Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*,
  SIGGRAPH 2023 — [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- Yang et al., *Depth Anything V2*, NeurIPS 2024
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*,
  CVPR 2022 — Stable Diffusion 1.5 inpainting
- Knapitsch et al., *Tanks and Temples*, SIGGRAPH 2017 — the `truck` scene
