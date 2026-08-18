"""Verify depth supervision actually reaches the training views.

This exists because it did not, twice, without any error being raised:

  1. add_depths.py wrote each scene's depth_params.json THROUGH a symlink into
     the shared dataset, so scene 1's 37 entries replaced the dataset's 251 and
     every later scene read the truncated file.
  2. gaussian-splatting/utils/make_depth_scale.py mis-slices its remap output
     on OpenCV 5 and emits scale=inf for every image, which poisons the median
     in dataset_readers.py and fails every image at the gate in cameras.py.

Both produced training runs that completed normally and reproduced the
baseline exactly, which is the worst possible failure: a plausible null result
that is really a plumbing fault.

The only number that matters is how many TRAINING views survive to be
supervised - test views are loaded but never contribute to the depth loss, so
counting all images in a scene hides the problem.

  python src/check_depth_coverage.py            # exits 1 if any seed has none
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(os.path.expanduser("~/fewshot_gs"))


def main():
    source = ROOT / "data/tandt/truck"
    params_path = source / "sparse/0/depth_params.json"
    if not params_path.exists():
        print(f"MISSING {params_path}")
        return 1

    n_imgs = len(list((source / "images").iterdir()))
    n_deps = len(list((source / "depths").glob("*.png"))) if (source / "depths").exists() else 0
    params = json.loads(params_path.read_text())

    good = [v["scale"] for v in params.values()
            if isinstance(v.get("scale"), (int, float))
            and math.isfinite(v["scale"]) and v["scale"] > 0]
    print(f"images {n_imgs}   depth maps {n_deps}   params {len(params)}   "
          f"usable {len(good)}")
    if n_deps < n_imgs:
        print(f"  ! {n_imgs - n_deps} images have no depth map - training will "
              f"crash (cv2.imread returns None, camera_utils.py re-raises)")
    if not good:
        print("  ! no usable scales at all")
        return 1

    med = sorted(good)[len(good) // 2]
    print(f"median scale {med:.4f}")

    failed = 0
    for seed in (0, 1, 2):
        man = ROOT / f"subsets/truck_k5_seed{seed}_fps.json"
        if not man.exists():
            continue
        names = json.loads(man.read_text())["images"]
        ok = []
        for n in names:
            e = params.get(Path(n).stem)
            s = e.get("scale") if e else None
            # mirrors the reliability gate in scene/cameras.py:69
            if (s and math.isfinite(s) and s > 0
                    and 0.2 * med <= s <= 5 * med):
                ok.append(Path(n).stem)
        print(f"  seed{seed}: {len(ok)}/{len(names)} training views supervised")
        if not ok:
            failed += 1

    if failed:
        print("\nFAIL: depth regularisation would be a no-op for "
              f"{failed} seed(s)")
        return 1
    print("\nOK: every seed has supervised training views")
    return 0


if __name__ == "__main__":
    sys.exit(main())
