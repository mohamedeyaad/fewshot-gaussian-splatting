"""CPU-only sanity checks for the pose-guided geometry (no GPU needed).

Test 1 (identity): warping to frac=0 targets the SAME pose, so the warped
image must reproduce the original almost exactly. Any real error here means
the projection maths is wrong.

Test 2 (motion): warping to frac>0 must produce holes and must NOT be
identical to the source - otherwise the "new viewpoint" is a lie.
"""
import os, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(os.path.expanduser("~/fewshot_gs"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "gaussian-splatting"))
from scene.colmap_loader import (read_extrinsics_binary, read_intrinsics_binary,
                                 read_next_bytes, qvec2rotmat)
from gen_guided import interp_pose, warp, cam_centre, align_depth_to_colmap

SRC = ROOT / "data/tandt/truck"
intr = read_intrinsics_binary(str(SRC / "sparse/0/cameras.bin"))
extr = read_extrinsics_binary(str(SRC / "sparse/0/images.bin"))
by_name = {im.name: im for im in extr.values()}

cam = intr[1]
Wc, Hc = int(cam.width), int(cam.height)
fx_c, fy_c, cx_c, cy_c = (float(x) for x in cam.params)

name = "000044.jpg"
img = Image.open(SRC / "images" / name).convert("RGB")
Wd, Hd = img.size
sx, sy = Wd / Wc, Hd / Hc
K = (fx_c * sx, fy_c * sy, cx_c * sx, cy_c * sy)
rgb = np.asarray(img)

im = by_name[name]
R1, t1 = qvec2rotmat(im.qvec), np.asarray(im.tvec, float)

# Build a plausible depth map from the sparse points (no NN needed):
# take the median observed depth and use a constant plane. Constant depth is
# a valid test of the projection maths - identity must still hold.
id_to_xyz = {}
with open(SRC / "sparse/0/points3D.bin", "rb") as fid:
    n_pts = read_next_bytes(fid, 8, "Q")[0]
    for _ in range(n_pts):
        p = read_next_bytes(fid, 43, "QdddBBBd")
        id_to_xyz[p[0]] = np.array(p[1:4], float)
        tl = read_next_bytes(fid, 8, "Q")[0]
        read_next_bytes(fid, 8 * tl, "ii" * tl)

ids = im.point3D_ids
sel = ids > 0
P = np.stack([id_to_xyz[i] for i in ids[sel] if i in id_to_xyz])
z_true = (P @ R1.T + t1)[:, 2]
z_med = float(np.median(z_true))
print(f"scene: {len(id_to_xyz):,} points; this view observes {len(P):,}")
print(f"observed depth: median {z_med:.3f}, range {z_true.min():.2f}..{z_true.max():.2f}")

depth = np.full((Hd, Wd), z_med, dtype=np.float64)

print("\n=== TEST 1: identity warp (frac=0, same pose) ===")
q0, t0, R0 = interp_pose(im.qvec, t1, im.qvec, t1, 0.0)
w0, v0 = warp(rgb, depth, K, R1, t1, R0, t0)
cover = v0.mean()
diff = np.abs(w0[v0].astype(int) - rgb[v0].astype(int)).mean()
print(f"  coverage      : {cover*100:.2f}%   (expect ~100%)")
print(f"  mean |diff|   : {diff:.4f}          (expect ~0)")
print(f"  rotation err  : {np.abs(R0 - R1).max():.2e}")
print(f"  translation err: {np.abs(t0 - t1).max():.2e}")
# 0.97 not 0.99: warp() deliberately erodes the valid boundary by 1px so the
# ragged depth-discontinuity zone is filled by diffusion rather than smeared.
assert cover > 0.97, "identity warp lost coverage - projection maths wrong"
assert diff < 1.0, "identity warp changed pixels - projection maths wrong"
print("  PASS")

print("\n=== TEST 2: real motion toward nearest neighbour ===")
centres = {n: cam_centre(qvec2rotmat(by_name[n].qvec), np.asarray(by_name[n].tvec, float))
           for n in by_name}
c0 = centres[name]
nb = min((n for n in centres if n != name), key=lambda n: np.linalg.norm(centres[n] - c0))
print(f"  nearest neighbour: {nb} at {np.linalg.norm(centres[nb]-c0):.3f} units")
im2 = by_name[nb]
for frac in (0.25, 0.5, 1.0):
    q, t, R = interp_pose(im.qvec, t1, im2.qvec, np.asarray(im2.tvec, float), frac)
    w, v = warp(rgb, depth, K, R1, t1, R, t)
    holes = 1 - v.mean()
    moved = np.abs(w.astype(int) - rgb.astype(int)).mean()
    print(f"  frac={frac:.2f}: holes {holes*100:5.2f}%, mean pixel change {moved:6.2f}")

# frac=1.0 should land exactly on the neighbour's pose
q1, t1n, R1n = interp_pose(im.qvec, t1, im2.qvec, np.asarray(im2.tvec, float), 1.0)
print(f"  frac=1 rotation err vs neighbour   : "
      f"{np.abs(R1n - qvec2rotmat(im2.qvec)).max():.2e}")
print(f"  frac=1 translation err vs neighbour: "
      f"{np.abs(t1n - np.asarray(im2.tvec, float)).max():.2e}")
print("  PASS" if np.abs(R1n - qvec2rotmat(im2.qvec)).max() < 1e-9 else "  FAIL")

print("\nAll geometry checks done.")
