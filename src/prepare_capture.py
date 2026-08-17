"""Normalise a folder of phone photos into COLMAP-ready input.

Three things matter here and each of them silently ruins a reconstruction if
skipped:

  1. EXIF orientation. Phones store the sensor image plus a rotation flag.
     Different apps apply it differently, so COLMAP can see a mix of portrait
     and landscape frames of the same scene and fail to match them.
     ImageOps.exif_transpose bakes the rotation into the pixels and clears the
     flag, making orientation unambiguous.
  2. Resolution. Exhaustive matching of 135 frames at 12 MP will not fit in
     7.6 GB of RAM. 1600 px on the long edge keeps plenty of features while
     staying well inside the budget.
  3. Consistent naming. Sequential zero-padded names so the LLFF hold-out rule
     (idx % 8) selects an evenly spread test set rather than a clustered one.

Usage:
    python src/prepare_capture.py --input ~/photos/mycar --name mycar
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image, ImageOps

# iPhones default to HEIC. Register the opener if the plugin is available.
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF = True
except ImportError:
    HEIF = False

ROOT = Path(os.path.expanduser("~/fewshot_gs"))
EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder of raw photos")
    ap.add_argument("--name", required=True, help="scene name, e.g. mycar")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-dim", type=int, default=1600)
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    src = Path(os.path.expanduser(args.input))
    if not src.is_dir():
        sys.exit(f"no such folder: {src}")

    # convert.py expects the raw frames in <scene>/input and writes the
    # undistorted PINHOLE model back to <scene>/images + <scene>/sparse/0.
    dst_root = Path(args.out) if args.out else ROOT / "data" / "custom" / args.name
    dst = dst_root / "input"
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        sys.exit(f"no images found in {src} (looked for {sorted(EXTS)})")

    heic = [p for p in files if p.suffix.lower() in {".heic", ".heif"}]
    if heic and not HEIF:
        sys.exit(f"{len(heic)} HEIC files but pillow-heif is not installed.\n"
                 f"  ./venv/bin/pip install pillow-heif")

    print(f"found {len(files)} images in {src}")
    sizes, kept = set(), 0
    for i, p in enumerate(files, start=1):
        try:
            img = Image.open(p)
            img = ImageOps.exif_transpose(img)      # bake rotation, clear flag
            img = img.convert("RGB")
        except Exception as e:
            print(f"  SKIP {p.name}: {type(e).__name__}: {e}")
            continue

        w, h = img.size
        scale = args.max_dim / max(w, h)
        if scale < 1.0:
            img = img.resize((round(w * scale), round(h * scale)),
                             Image.LANCZOS)
        sizes.add(img.size)
        img.save(dst / f"{i:06d}.jpg", "JPEG", quality=args.quality)
        kept += 1
        if kept % 25 == 0:
            print(f"  {kept}/{len(files)}")

    print(f"\nwrote {kept} images to {dst}")
    print(f"resolutions present: {sorted(sizes)}")

    if len(sizes) > 1:
        print("\n  WARNING: mixed resolutions. COLMAP is run with "
              "--ImageReader.single_camera 1, which assumes one shared set of\n"
              "  intrinsics. Mixed sizes usually mean the lens was switched "
              "mid-capture (0.5x / 2x), which breaks that assumption\n"
              "  and commonly causes a partial or failed reconstruction.")
    if kept < 100:
        print(f"\n  WARNING: only {kept} images. The project spec requires "
              f"N >= 100, and COLMAP will reject some during registration.")

    print(f"\nnext:  bash src/run_colmap.sh {args.name}")


if __name__ == "__main__":
    main()
