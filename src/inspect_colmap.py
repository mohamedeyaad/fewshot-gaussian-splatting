"""Inspect the truck COLMAP model: camera model, params, sample poses."""
import os, sys
from pathlib import Path
import numpy as np

REPO = Path(os.path.expanduser("~/fewshot_gs/gaussian-splatting"))
sys.path.insert(0, str(REPO))
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary

SP = Path(os.path.expanduser("~/fewshot_gs/data/tandt/truck/sparse/0"))
intr = read_intrinsics_binary(str(SP / "cameras.bin"))
extr = read_extrinsics_binary(str(SP / "images.bin"))

print("=== cameras ===")
for cid, c in intr.items():
    print(f"  id={cid} model={c.model} {c.width}x{c.height} params={c.params}")

print(f"\n=== images: {len(extr)} ===")
for i, (iid, im) in enumerate(sorted(extr.items())[:3]):
    print(f"  id={iid} name={im.name} camera_id={im.camera_id}")
    print(f"     qvec={np.round(im.qvec,6)}")
    print(f"     tvec={np.round(im.tvec,6)}")
    print(f"     xys shape={im.xys.shape} point3D_ids shape={im.point3D_ids.shape}")

ids = sorted(extr.keys())
print(f"\nimage_id range: {min(ids)} .. {max(ids)}  (count {len(ids)})")
print(f"camera_ids used: {sorted({im.camera_id for im in extr.values()})}")
