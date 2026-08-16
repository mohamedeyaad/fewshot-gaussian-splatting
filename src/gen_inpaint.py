"""Generate synthetic views by INPAINTING masked regions of real training views.

Strategy 1 of 3. The defining property: the camera pose is inherited from the
source photo for free, so there is zero pose error. Only the masked pixels are
synthetic - everything outside the mask is copied from the real photo bit for
bit, which isolates the effect of the hallucinated content.

Filenames encode provenance, e.g.
    synth_inpaint_000044_v02.jpg   <- derived from real view 000044.jpg

  python src/gen_inpaint.py --manifest subsets/truck_k10_seed0_fps.json --n 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(os.path.expanduser("~/fewshot_gs"))
REPO = ROOT / "gaussian-splatting"
sys.path.insert(0, str(REPO))
from scene.colmap_loader import read_extrinsics_binary  # noqa: E402

MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"
# SD1.5 wants multiples of 8 and degrades far from its 512x512 training size.
# 704x392 keeps the truck aspect ratio (1.7959 vs native 1.7930) almost exactly.
SD_W, SD_H = 704, 392


def make_mask(w: int, h: int, rng: np.random.Generator,
              area_lo=0.08, area_hi=0.22) -> Image.Image:
    """A random rectangle or ellipse covering a controlled fraction of frame.

    Kept to a modest area so the surrounding real context still constrains
    what the model paints - a mask covering most of the frame is closer to
    unconditional generation than to inpainting.
    """
    target = rng.uniform(area_lo, area_hi) * w * h
    aspect = rng.uniform(0.6, 1.7)
    bw = int(np.clip(np.sqrt(target * aspect), 32, w * 0.8))
    bh = int(np.clip(target / max(bw, 1), 32, h * 0.8))
    x0 = int(rng.integers(0, max(1, w - bw)))
    y0 = int(rng.integers(0, max(1, h - bh)))

    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    box = [x0, y0, x0 + bw, y0 + bh]
    if rng.random() < 0.5:
        d.rectangle(box, fill=255)
    else:
        d.ellipse(box, fill=255)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--source", default=str(ROOT / "data/tandt/truck"))
    ap.add_argument("--out", default=str(ROOT / "synthetic"))
    ap.add_argument("--n", type=int, required=True, help="how many to generate")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--prompt", default="a photo of a truck parked on a street, "
                                        "realistic, sharp focus, natural lighting")
    ap.add_argument("--negative", default="blurry, distorted, watermark, text, lowres")
    ap.add_argument("--save-debug", action="store_true",
                    help="also save the mask and pre-composite output")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    src_root = Path(args.source)
    reals = list(manifest["images"])
    seed, k, method = manifest["seed"], manifest["k"], manifest["method"]

    tag = f"{manifest['scene']}_k{k}_seed{seed}_{method}_inpaint"
    out_dir = Path(args.out) / tag
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Poses come straight from the source view - no estimation involved.
    extr = read_extrinsics_binary(str(src_root / "sparse/0/images.bin"))
    by_name = {im.name: im for im in extr.values()}
    for r in reals:
        if r not in by_name:
            raise KeyError(f"{r} not in COLMAP model")

    print(f"[{tag}] generating {args.n} inpainted views from {len(reals)} real views")

    from diffusers import StableDiffusionInpaintPipeline
    t0 = time.time()
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True, safety_checker=None, requires_safety_checker=False)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
    print(f"  pipeline ready in {time.time()-t0:.1f}s")

    rng = np.random.default_rng(seed * 1000 + 7)
    torch.cuda.reset_peak_memory_stats()
    records = []
    t_gen = time.time()

    for i in range(args.n):
        src_name = reals[i % len(reals)]          # round-robin over real views
        variant = i // len(reals)
        stem = Path(src_name).stem
        out_name = f"synth_inpaint_{stem}_v{variant:02d}.jpg"

        real = Image.open(src_root / "images" / src_name).convert("RGB")
        W, H = real.size

        small = real.resize((SD_W, SD_H), Image.LANCZOS)
        mask_small = make_mask(SD_W, SD_H, rng)

        gen = torch.Generator("cuda").manual_seed(int(rng.integers(0, 2**31 - 1)))
        painted = pipe(prompt=args.prompt, negative_prompt=args.negative,
                       image=small, mask_image=mask_small,
                       num_inference_steps=args.steps,
                       guidance_scale=args.guidance,
                       height=SD_H, width=SD_W, generator=gen).images[0]

        # Composite at NATIVE resolution: real pixels outside the mask are
        # preserved exactly, so only the masked region is synthetic.
        painted_full = painted.resize((W, H), Image.LANCZOS)
        mask_full = mask_small.resize((W, H), Image.NEAREST)
        out_img = Image.composite(painted_full, real, mask_full)
        out_img.save(img_dir / out_name, quality=95)

        if args.save_debug:
            dbg = out_dir / "debug"; dbg.mkdir(exist_ok=True)
            mask_full.save(dbg / f"{Path(out_name).stem}_mask.png")

        im = by_name[src_name]
        mask_frac = float(np.asarray(mask_small).mean() / 255.0)
        records.append({
            "name": out_name,
            "source_image": src_name,
            "strategy": "inpaint",
            "qvec": [float(x) for x in im.qvec],
            "tvec": [float(x) for x in im.tvec],
            "camera_id": int(im.camera_id),
            "pose_source": "copied from source view (exact)",
            "mask_fraction": round(mask_frac, 4),
        })
        print(f"  [{i+1}/{args.n}] {out_name}  from {src_name}  "
              f"mask {mask_frac*100:.1f}%")

    elapsed = time.time() - t_gen
    peak = torch.cuda.max_memory_allocated() / 1024**3

    (out_dir / "poses.json").write_text(json.dumps({
        "tag": tag, "strategy": "inpaint", "scene": manifest["scene"],
        "k": k, "seed": seed, "method": method,
        "source_manifest": str(args.manifest),
        "config": {"model": MODEL, "steps": args.steps,
                   "guidance": args.guidance, "sd_size": [SD_W, SD_H],
                   "prompt": args.prompt, "negative_prompt": args.negative},
        "cost": {"seconds": round(elapsed, 1),
                 "seconds_per_image": round(elapsed / max(args.n, 1), 2),
                 "peak_vram_gb": round(peak, 2)},
        "images": records,
    }, indent=2))

    print(f"\n  {args.n} images in {elapsed:.0f}s "
          f"({elapsed/max(args.n,1):.1f}s each), peak {peak:.2f} GB")
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
