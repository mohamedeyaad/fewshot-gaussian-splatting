# Few-Shot Gaussian Splatting with Diffusion-Based Data Augmentation

Does generating synthetic training views with a diffusion model help 3D Gaussian
Splatting when you only have a handful of real photographs?

**Short answer: barely, and only in a narrow regime.** The best condition
recovers **2%** of the quality lost by dropping from 219 real views to 10. The
worst actively destroys the reconstruction. This repository contains the full
pipeline, all 42 training runs, and the analysis that supports that claim.

University of Genoa (UNIGE) — robotics project.

---

## Headline result

Scene: Tanks & Temples `truck`, 251 images, 32 held out for evaluation.

| condition | train views | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| Full-data ceiling | 219 real | **25.23** | 0.9070 | 0.1227 |
| Few-shot floor | 10 real | **17.11 ± 0.39** | 0.6584 | 0.3017 |

The **8.12 dB gap** between those two rows is what augmentation has to close.
Measured against it, across 3 seeds, paired within seed:

| synthetic ratio | inpainting | outpainting | pose-guided |
|---|---|---|---|
| 20 % (2 imgs) | +0.014 | **+0.172** \* | −0.451 \* |
| 50 % (5 imgs) | −0.142 \* | +0.013 | −0.855 \* |
| 100 % (10 imgs) | −0.213 \* | −0.423 \* | −1.404 \* |
| 200 % (20 imgs) | −0.141 | −0.003 | −1.451 \* |

ΔPSNR in dB. `*` = |mean| exceeds the between-seed standard deviation.
Measured training noise floor is σ = 0.039 dB (≈ 0.055 dB paired), so these
effects are real, just small.

**The ordering is the finding:** the less a strategy pretends to know about 3D
geometry, the less damage it does. Full write-up in
[`results/report.html`](results/report.html) — open it in a browser.

---

## Repository layout

```
src/                    all pipeline code (see "Reproducing" below)
patches/                the two required edits to upstream 3DGS
subsets/                view-selection manifests + the frozen test split
synthetic/              every generated image + poses.json (source-view linkage)
runs/*/results.json     raw metrics for all 42 runs - the experimental record
results/                figures, summary tables, and the final report
build_rasterizer.sh     one-shot CUDA submodule build
requirements.txt        Python dependencies
```

Not committed (see `.gitignore`): `venv/` (7.5 GB), `data/` (1.4 GB, downloaded),
`gaussian-splatting/` (upstream clone), `runs/` checkpoints and renders (6.7 GB),
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
| `01-dataset_readers-explicit-split` | Upstream picks the test set with the LLFF rule `idx % 8 == 0` and trains on *everything else*. This project needs an arbitrary K-image training subset while the **test set stays byte-identical across all 42 runs**. The patch makes a `split.json` in the scene root override the rule; with no such file, behaviour is exactly as upstream. |

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

Total compute ≈ **7 hours** on the 4 GB laptop GPU. Every stage is idempotent —
`run_experiment.py` skips any scene that already has a `results.json`, so an
interrupted batch can simply be re-run.

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

# 3. Few-shot floor (10 real views x 3 seeds).
for s in 0 1 2; do
  $VPY src/build_scene.py --manifest subsets/truck_k10_seed${s}_fps.json
  $VPY src/run_experiment.py --scene scenes/truck_k10_seed${s}_fps_fake0
done

# 4. The three augmentation curves. Each generates 20 synthetic views per seed,
#    builds the nested 2/5/10/20 scenes, trains and evaluates all of them.
STRATEGY=inpaint  bash src/run_curve.sh
STRATEGY=outpaint bash src/run_curve.sh
STRATEGY=guided   bash src/run_curve.sh

# 5. Noise floor: same scene, same config, 3 repeats.
bash src/measure_train_noise.sh

# 6. Analysis and deliverables.
$VPY src/collect_results.py   # -> results/summary.md, results/runs_raw.csv
$VPY src/plot_curves.py       # -> results/curves_{absolute,paired}.png
$VPY src/make_panels.py       # -> results/panel_*.png
$VPY src/build_report.py      # -> results/report.html (self-contained)
```

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
  constant across all 42 runs. No synthetic image is ever derived from a test
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
  fills imperfectly. The −1.45 dB result is therefore partly attributable to
  image artifacts rather than to pose novelty per se. The clean control (warp to
  a new pose, leave holes black, no diffusion at all) is designed but was not
  run; it is the single most valuable follow-up.
- **One scene.** All conclusions are from `truck`. A second scene
  (`drjohnson`, ≈ 5 h) would establish whether the strategy ordering generalises.
- **One diffusion model.** SD 1.5 inpainting only, chosen for the 4 GB budget.
  SDXL and FLUX do not fit. `src/check_alt_models.sh` enumerates the
  alternatives that were considered, including the multi-view-consistent
  generators (Zero123++, SV3D, ImageDream) that are arguably the actual fix for
  what pose-guided augmentation is trying to do.
- **7000 iterations**, not the upstream 30000, to keep 42 runs tractable. The
  full-data run reaches 25.23 dB against ≈ 25.4 dB published at full schedule,
  so the pipeline is validated, but absolute numbers are slightly below
  literature values.
- Single scale (k = 10) for the augmentation sweep; k = 5 and k = 20 subsets
  are selected and committed but only the ceiling and floor were trained.

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
