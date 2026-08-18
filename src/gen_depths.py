"""Monocular inverse-depth maps for depth-regularised training.

Depth regularisation attacks the few-shot problem from the opposite side to
augmentation. Augmentation adds coverage but drags multi-view inconsistency
along with it - that trade is the central finding of this study. A depth prior
adds a geometric CONSTRAINT with no extra view, so it contributes no
inconsistency at all.

Upstream 3DGS already implements the loss (train.py: L1 between rendered
inverse depth and the prior, weight decaying 1.0 -> 0.01). What it needs from
us is a depths/ folder and sparse/0/depth_params.json.

FORMAT, dictated by the consumers - both get this wrong silently if we do:
  * 16-bit single-channel PNG, read as `cv2.imread(...) / 2**16`
    (utils/camera_utils.py:28 and utils/make_depth_scale.py:38)
  * the values are INVERSE depth / disparity (bigger = nearer), which is what
    Depth Anything emits natively, so no inversion here
  * per-image min-max normalisation is fine: make_depth_scale.py fits a
    per-image scale and offset against the sparse COLMAP points afterwards,
    which is what puts every map into true scene units

  python src/gen_depths.py --source data/tandt/truck
  python src/gen_depths.py --images synthetic/foo/images --out synthetic/foo/depths
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def to_uint16_disparity(pred: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Normalise a raw disparity prediction to the full uint16 range.

    A constant map would divide by zero; those are degenerate anyway, so emit
    mid-grey and let make_depth_scale.py reject the image (it needs spread
    greater than 1e-3 to fit a scale, and assigns scale 0 otherwise).
    """
    d = pred.astype(np.float32)
    lo, hi = float(d.min()), float(d.max())
    d = np.full_like(d, 0.5) if hi - lo < 1e-8 else (d - lo) / (hi - lo)
    img = Image.fromarray((d * 65535.0).astype(np.uint16), mode="I;16")
    if img.size != size:
        img = img.resize(size, Image.BILINEAR)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None,
                    help="dataset root; uses <source>/images -> <source>/depths")
    ap.add_argument("--images", default=None, help="explicit image directory")
    ap.add_argument("--out", default=None, help="explicit output directory")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these filenames (default: all)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.source:
        img_dir = Path(args.source) / "images"
        out_dir = Path(args.out) if args.out else Path(args.source) / "depths"
    else:
        if not (args.images and args.out):
            ap.error("give --source, or both --images and --out")
        img_dir, out_dir = Path(args.images), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = sorted(p.name for p in img_dir.iterdir()
                   if p.suffix in EXTS and not p.name.startswith("."))
    if args.only:
        keep = set(args.only)
        names = [n for n in names if n in keep]
    if not names:
        print(f"no images in {img_dir}")
        return

    todo = [n for n in names
            if args.force or not (out_dir / f"{Path(n).stem}.png").exists()]
    print(f"{img_dir} -> {out_dir}")
    print(f"  {len(names)} images, {len(todo)} to generate")
    if not todo:
        print("  all present, nothing to do")
        return

    from transformers import pipeline as hf_pipeline
    dpipe = hf_pipeline("depth-estimation", model=DEPTH_MODEL, device=0)

    for i, n in enumerate(todo, 1):
        im = Image.open(img_dir / n).convert("RGB")
        pred = dpipe(im)["predicted_depth"]
        pred = pred.squeeze().cpu().numpy() if hasattr(pred, "cpu") else np.asarray(pred)
        to_uint16_disparity(pred, im.size).save(out_dir / f"{Path(n).stem}.png")
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}")
    print(f"  wrote {len(todo)} depth maps")


if __name__ == "__main__":
    main()
