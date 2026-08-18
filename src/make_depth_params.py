"""Fit per-image scale and offset mapping monocular disparity into scene units.

Replaces gaussian-splatting/utils/make_depth_scale.py, which is broken on the
OpenCV version this project pins (5.0.0).

THE UPSTREAM BUG, since this is not obvious. It samples the depth map at the
COLMAP keypoints with

    invmonodepth = cv2.remap(depthmap, maps[..., 0], maps[..., 1], ...)[..., 0]

`maps[..., 0]` is 1-D of shape (N,), so OpenCV treats it as a 1xN image and
returns shape (1, N). The trailing `[..., 0]` then takes the first COLUMN, not
the first channel, collapsing 4,117 sampled points to a single scalar. The mean
absolute deviation of one number is exactly 0, so every image comes out as

    scale = s_colmap / 0 = inf

and an inf poisons the median that dataset_readers.py computes across scales,
after which every image fails the 0.2x/5x reliability gate in cameras.py. The
net effect is depth regularisation silently doing nothing, with no error.

WHAT THIS DOES INSTEAD. Same idea, fixed sampling, and a least-squares fit
rather than median/MAD - the same approach already validated elsewhere in this
project (src/gen_guided.py aligns depth the same way, R^2 = 0.988). Fitting in
DISPARITY space is the right choice: the relationship is linear there, so a
plain least-squares solution is exact rather than approximate.

Also reports R^2 per image, so a bad fit is visible instead of silent.

  python src/make_depth_params.py --source data/tandt/truck
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(os.path.expanduser("~/fewshot_gs"))
sys.path.insert(0, str(ROOT / "gaussian-splatting/utils"))
from read_write_model import read_model, qvec2rotmat  # noqa: E402

MIN_POINTS = 10


def sample_at(depthmap: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear sample a 2-D map at N (x, y) locations -> shape (N,).

    Maps are reshaped to (1, N) explicitly and the single row read back, which
    is what upstream meant to do.
    """
    mx = np.ascontiguousarray(xy[:, 0].reshape(1, -1).astype(np.float32))
    my = np.ascontiguousarray(xy[:, 1].reshape(1, -1).astype(np.float32))
    out = cv2.remap(depthmap, mx, my, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    return np.asarray(out).reshape(-1)


def fit(mono: np.ndarray, colmap: np.ndarray):
    """Least squares colmap ~= scale * mono + offset. Returns (scale, offset, r2)."""
    A = np.stack([mono, np.ones_like(mono)], axis=1)
    (scale, offset), *_ = np.linalg.lstsq(A, colmap, rcond=None)
    pred = A @ np.array([scale, offset])
    ss_res = float(np.sum((colmap - pred) ** 2))
    ss_tot = float(np.sum((colmap - colmap.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(scale), float(offset), r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(ROOT / "data/tandt/truck"))
    ap.add_argument("--depths", default=None)
    ap.add_argument("--ext", default="bin")
    args = ap.parse_args()

    source = Path(args.source)
    depths = Path(args.depths) if args.depths else source / "depths"
    cams, imgs, pts3d = read_model(str(source / "sparse" / "0"), ext=f".{args.ext}")

    ids = np.array([pts3d[k].id for k in pts3d])
    xyz = np.array([pts3d[k].xyz for k in pts3d])
    ordered = np.zeros([ids.max() + 1, 3])
    ordered[ids] = xyz

    params, r2s, skipped = {}, [], []
    for key in imgs:
        meta = imgs[key]
        cam = cams[meta.camera_id]
        stem = Path(meta.name).stem

        png = depths / f"{stem}.png"
        dmap = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        if dmap is None:
            skipped.append((stem, "no depth map"))
            continue
        if dmap.ndim != 2:
            dmap = dmap[..., 0]
        dmap = dmap.astype(np.float32) / (2 ** 16)

        pid = meta.point3D_ids
        m = (pid >= 0) & (pid < len(ordered))
        pid, xys = pid[m], meta.xys[m]
        if len(pid) < MIN_POINTS:
            skipped.append((stem, f"only {len(pid)} keypoints"))
            params[stem] = {"scale": 0, "offset": 0}
            continue

        pts = ordered[pid] @ qvec2rotmat(meta.qvec).T + meta.tvec
        z = pts[..., 2]
        inv_colmap = np.divide(1.0, z, out=np.zeros_like(z), where=z != 0)

        s = dmap.shape[0] / cam.height          # depth map may be downscaled
        xy = xys * s
        ok = ((xy[:, 0] >= 0) & (xy[:, 1] >= 0)
              & (xy[:, 0] < cam.width * s) & (xy[:, 1] < cam.height * s)
              & (z > 0) & np.isfinite(inv_colmap))
        if ok.sum() < MIN_POINTS:
            skipped.append((stem, f"only {int(ok.sum())} in-frame points"))
            params[stem] = {"scale": 0, "offset": 0}
            continue

        mono = sample_at(dmap, xy[ok])
        colmap = inv_colmap[ok]
        if float(mono.max() - mono.min()) < 1e-6:
            skipped.append((stem, "flat depth map"))
            params[stem] = {"scale": 0, "offset": 0}
            continue

        scale, offset, r2 = fit(mono, colmap)
        if not (np.isfinite(scale) and np.isfinite(offset)) or scale <= 0:
            skipped.append((stem, f"degenerate fit (scale={scale:.3g})"))
            params[stem] = {"scale": 0, "offset": 0}
            continue

        params[stem] = {"scale": scale, "offset": offset}
        r2s.append(r2)

    out = source / "sparse" / "0" / "depth_params.json"
    if out.is_symlink() or out.exists():
        out.unlink()
    out.write_text(json.dumps(params, indent=2))

    good = [v["scale"] for v in params.values() if v["scale"] > 0]
    print(f"wrote {out}")
    print(f"  {len(params)} images, {len(good)} with a usable scale, "
          f"{len(params) - len(good)} masked")
    if good:
        print(f"  scale   median {np.median(good):.4f}   "
              f"range {min(good):.4f} .. {max(good):.4f}")
    if r2s:
        print(f"  fit R^2 median {np.median(r2s):.4f}   worst {min(r2s):.4f}")
    for stem, why in skipped[:8]:
        print(f"    masked {stem}: {why}")
    if len(skipped) > 8:
        print(f"    ... and {len(skipped) - 8} more")


if __name__ == "__main__":
    main()
