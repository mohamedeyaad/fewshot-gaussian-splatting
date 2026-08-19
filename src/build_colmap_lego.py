"""Turn NeRF-Synthetic lego into an ordinary COLMAP scene.

WHY THIS EXISTS. The Blender format cannot host the augmentation experiment,
for two independent reasons:

  - transforms_*.json stores ONE global camera_angle_x, so an outpainted view
    (same focal, larger canvas => wider FOV) cannot be expressed;
  - gen_guided.py anchors monocular depth to the sparse COLMAP points a view
    observed, and the Blender path has no sparse points at all - it starts
    from a random cloud.

Reconstructing the renders with COLMAP removes both at once, and costs no new
experiment code: the result has the same layout as truck and drjohnson, so
build_scene.py, gen_*.py and run_experiment.py all work unchanged.

WHAT IT DOES. Composites the RGBA frames over WHITE (the COLMAP loader reads
RGB straight off disk and does no compositing, so the background has to be
baked in), then runs feature extraction, exhaustive matching and mapping with
the TRUE intrinsics held fixed - NeRF-Synthetic is a single virtual camera, so
letting COLMAP guess them would only add error.

WHICH FRAMES. The 100 training frames plus every 6th test frame. Those 34 test
frames are exactly the held-out set the Blender-format run used, so if this
scene's full-data ceiling lands near the 33.77 dB already measured there - and
near the ~33 dB published - the pose recovery is verified end to end. Any
other choice of frames would throw that check away.

  python src/build_colmap_lego.py            # export + reconstruct
  python src/build_colmap_lego.py --export-only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(os.path.expanduser("~/fewshot_gs"))
SRC = ROOT / "data" / "blender" / "lego"
DST = ROOT / "data" / "legoc"
TEST_STRIDE = 6


def composite_white(src_png: Path, dst_png: Path):
    """RGBA over white. The alpha channel must not survive: kept alpha becomes
    a loss mask in train.py, which is the bug that cost 27 dB earlier."""
    im = np.asarray(Image.open(src_png).convert("RGBA"), dtype=np.float64) / 255.0
    rgb = im[:, :, :3] * im[:, :, 3:4] + 1.0 * (1.0 - im[:, :, 3:4])
    out = Image.fromarray(np.asarray(rgb * 255.0, dtype=np.uint8), "RGB")
    out.save(dst_png)
    return out.size


def export():
    img_dir = DST / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"train": [], "test": []}
    size = fovx = None

    for split, stride in (("train", 1), ("test", TEST_STRIDE)):
        meta = json.loads((SRC / ("transforms_%s.json" % split)).read_text())
        fovx = meta["camera_angle_x"]
        for i, frame in enumerate(meta["frames"]):
            if i % stride:
                continue
            rel = frame["file_path"].lstrip("./")            # e.g. train/r_0
            name = "%s_%s.png" % (split, Path(rel).name)     # train_r_0.png
            size = composite_white(SRC / (rel + ".png"), img_dir / name)
            manifest[split].append(name)

    W, H = size
    focal = 0.5 * W / math.tan(0.5 * fovx)
    n_tr, n_te = len(manifest["train"]), len(manifest["test"])
    (DST / "export.json").write_text(json.dumps(
        {"width": W, "height": H, "focal": focal, "camera_angle_x": fovx,
         "test_stride": TEST_STRIDE, **manifest}, indent=2))
    print("exported %d train + %d test = %d images at %dx%d"
          % (n_tr, n_te, n_tr + n_te, W, H))
    print("true intrinsics: PINHOLE fx=fy=%.4f cx=%.1f cy=%.1f" % (focal, W / 2, H / 2))
    return W, H, focal


def reconstruct(W, H, focal):
    db = DST / "database.db"
    if db.exists():
        db.unlink()
    sparse = DST / "sparse"
    sparse.mkdir(exist_ok=True)
    params = "%f,%f,%f,%f" % (focal, focal, W / 2.0, H / 2.0)

    steps = [
        ["colmap", "feature_extractor",
         "--database_path", str(db), "--image_path", str(DST / "images"),
         "--ImageReader.single_camera", "1",
         "--ImageReader.camera_model", "PINHOLE",
         "--ImageReader.camera_params", params,
         "--SiftExtraction.use_gpu", "0"],
        # Exhaustive, not sequential: the test frames are a separate orbit, so
        # neighbouring filenames are not neighbouring viewpoints.
        ["colmap", "exhaustive_matcher",
         "--database_path", str(db), "--SiftMatching.use_gpu", "0"],
        # Intrinsics are known exactly; refining them can only drift.
        ["colmap", "mapper",
         "--database_path", str(db), "--image_path", str(DST / "images"),
         "--output_path", str(sparse),
         "--Mapper.ba_refine_focal_length", "0",
         "--Mapper.ba_refine_principal_point", "0",
         "--Mapper.ba_refine_extra_params", "0"],
    ]
    for cmd in steps:
        print("\n$ %s" % cmd[1], flush=True)
        r = subprocess.run(cmd, cwd=str(DST))
        if r.returncode != 0:
            sys.exit("FAILED: %s (exit %d)" % (cmd[1], r.returncode))

    out = subprocess.run(["colmap", "model_analyzer", "--path", str(sparse / "0")],
                         capture_output=True, text=True)
    print(out.stdout, out.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-only", action="store_true")
    a = ap.parse_args()
    w, h, f = export()
    if not a.export_only:
        reconstruct(w, h, f)
