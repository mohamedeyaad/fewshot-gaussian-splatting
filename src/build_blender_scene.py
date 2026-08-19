"""Few-shot scene directories for a Blender (NeRF-Synthetic) scene.

A third capture regime, after truck (outdoor object) and drjohnson (indoor
room): an isolated object on a clean background. This is the distribution the
multi-view-consistent generators - Zero123++, SV3D, ImageDream - are trained
on, which is exactly why the report argues they would not transfer to a street
scene. It is therefore the regime where a fabricated viewpoint should do the
LEAST harm, and so a useful third point for the crossover.

No patch to the upstream loader is needed here. scene/__init__.py picks the
Blender reader whenever transforms_train.json exists and sparse/ does not, and
the K-image subset simply IS that file - unlike the COLMAP path, where an
explicit split.json had to be patched in because the loader derives train/test
from a fixed llffhold rule.

  python src/build_blender_scene.py --k 5 10 20 --seeds 0 1 2
  python src/build_blender_scene.py --full          # the 100-view ceiling
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.path.expanduser("~/fewshot_gs"))

# The standard protocol evaluates on all 200 test frames. Every one is
# rendered and scored (PSNR/SSIM/LPIPS) after every run, which at 800x800 costs
# roughly six times truck's 32-view evaluation. Subsampling to a fixed stride
# keeps the held-out set the same size as the other two scenes, so the compute
# budget stays comparable - and, being frozen, keeps every condition of this
# scene mutually comparable, which is all the within-scene deltas require.
TEST_STRIDE = 6


def centres(frames):
    """Camera positions. transform_matrix is camera-to-world, so the
    translation column IS the camera centre - no inversion needed, unlike the
    COLMAP path where the centre is -R^T t."""
    return np.array([np.array(f["transform_matrix"], dtype=np.float64)[:3, 3]
                     for f in frames])


def fps(pts: np.ndarray, k: int, seed: int):
    """Farthest-point sampling; the seed picks only the starting camera."""
    rng = np.random.default_rng(seed)
    first = int(rng.integers(len(pts)))
    chosen = [first]
    dist = np.linalg.norm(pts - pts[first], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(pts - pts[nxt], axis=1))
    return sorted(chosen)


def link(target: Path, dst: Path):
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(target.resolve())


def build(source: Path, out: Path, train_idx, test_frames, tag, k, seed, method):
    tr = json.loads((source / "transforms_train.json").read_text())
    frames = tr["frames"]
    sel = [frames[i] for i in train_idx]

    out.mkdir(parents=True, exist_ok=True)
    # The loader resolves frame["file_path"] relative to the scene root, so the
    # image directories must be reachable from here.
    for sub in ("train", "test"):
        link(source / sub, out / sub)

    (out / "transforms_train.json").write_text(json.dumps(
        {"camera_angle_x": tr["camera_angle_x"], "frames": sel}, indent=1))
    te = json.loads((source / "transforms_test.json").read_text())
    (out / "transforms_test.json").write_text(json.dumps(
        {"camera_angle_x": te["camera_angle_x"], "frames": test_frames}, indent=1))

    # Same shape as the COLMAP scenes' split.json so run_experiment.py can read
    # counts and provenance without caring which loader will be used.
    (out / "split.json").write_text(json.dumps({
        "train": [f["file_path"] for f in sel],
        "test": [f["file_path"] for f in test_frames],
        "provenance": {
            "manifest": None, "k": k, "seed": seed, "method": method,
            "strategy": "none", "n_real": len(sel), "n_synthetic": 0,
            "ratio_pct": 0, "synthetic_dir": None, "source": str(source),
            "format": "blender", "test_stride": TEST_STRIDE,
        },
    }, indent=2))
    print(f"  built {tag}: {len(sel)} train / {len(test_frames)} test")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(ROOT / "data/blender/lego"))
    ap.add_argument("--out", default=str(ROOT / "scenes"))
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--full", action="store_true", help="also the all-views ceiling")
    args = ap.parse_args()

    source = Path(args.source)
    scene = source.name
    tr = json.loads((source / "transforms_train.json").read_text())
    te = json.loads((source / "transforms_test.json").read_text())
    frames = tr["frames"]
    test_frames = te["frames"][::TEST_STRIDE]

    print(f"scene {scene}: {len(frames)} train pool, "
          f"{len(te['frames'])} test -> {len(test_frames)} held out "
          f"(stride {TEST_STRIDE})")

    pts = centres(frames)
    for k in args.k:
        for s in args.seeds:
            idx = fps(pts, k, s)
            tag = f"{scene}_k{k}_seed{s}_fps_fake0"
            build(source, Path(args.out) / tag, idx, test_frames, tag, k, s, "fps")

    if args.full:
        n = len(frames)
        tag = f"{scene}_k{n}_seed0_full_fake0"
        build(source, Path(args.out) / tag, list(range(n)), test_frames,
              tag, n, 0, "full")


if __name__ == "__main__":
    main()
