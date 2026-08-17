"""Duplication control: plain copies of the real views, no diffusion at all.

Outpainting preserves the real image region byte-for-byte and fabricates only
the border. So adding N outpainted images to a K-image training set does two
things at once:

  1. adds fabricated peripheral coverage
  2. re-weights the loss toward the real pixels, which now appear (1 + N/K)
     times instead of once

The k=5 outpainting gain (+0.426 dB at the 800% ratio) could be entirely (2).
This control isolates it: emit the same real images, at their exact original
poses, with no invented content whatsoever. Same image counts, same duplication
factor, zero synthetic pixels.

    K=5 FAKES="20 40" NGEN=40 STRATEGY=duplicate bash src/run_curve.sh

If duplication reproduces the outpainting gain, the finding is about loss
re-weighting rather than about diffusion. If it does not, the fabricated
borders are contributing something real.

No GPU, no model - this is a file copy with bookkeeping.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(os.path.expanduser("~/fewshot_gs"))
REPO = ROOT / "gaussian-splatting"
sys.path.insert(0, str(REPO))
from scene.colmap_loader import read_extrinsics_binary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--source", default=str(ROOT / "data/tandt/truck"))
    ap.add_argument("--out", default=str(ROOT / "synthetic"))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--label", default="duplicate")
    # Accepted and ignored, so run_curve.sh can drive this exactly like a real
    # generator without special-casing it.
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--negative", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--save-debug", action="store_true")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    src_root = Path(args.source)
    reals = list(manifest["images"])
    seed, k, method = manifest["seed"], manifest["k"], manifest["method"]

    tag = f"{manifest['scene']}_k{k}_seed{seed}_{method}_{args.label}"
    out_dir = Path(args.out) / tag
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    extr = read_extrinsics_binary(str(src_root / "sparse/0/images.bin"))
    by_name = {im.name: im for im in extr.values()}
    for r in reals:
        if r not in by_name:
            raise KeyError(f"{r} not in COLMAP model")

    print(f"[{tag}] copying {args.n} duplicates of {len(reals)} real views")
    print("  no diffusion, no invented pixels - byte-identical copies")

    records = []
    t0 = time.time()
    for i in range(args.n):
        # Round-robin in the same order the real generators use, so the nested
        # prefix property matches: avail[:n] spreads across source views first.
        src_name = reals[i % len(reals)]
        variant = i // len(reals)
        out_name = f"synth_{args.label}_{Path(src_name).stem}_v{variant:02d}.jpg"

        shutil.copyfile(src_root / "images" / src_name, img_dir / out_name)

        im = by_name[src_name]
        records.append({
            "name": out_name,
            "source_image": src_name,
            "strategy": args.label,
            "camera_id": int(im.camera_id),
            "qvec": [float(x) for x in im.qvec],
            "tvec": [float(x) for x in im.tvec],
        })

    elapsed = time.time() - t0
    meta = {
        "tag": tag, "strategy": args.label, "scene": manifest["scene"],
        "k": k, "seed": seed, "method": method,
        "source_manifest": str(args.manifest),
        "params": {"note": "exact duplicates of the real views; "
                           "no generative model involved"},
        "cost": {"total_seconds": round(elapsed, 1),
                 "seconds_per_image": round(elapsed / max(args.n, 1), 3)},
        "images": records,
    }
    (out_dir / "poses.json").write_text(json.dumps(meta, indent=2))
    print(f"\n  {args.n} copies in {elapsed:.1f}s")
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
