"""Generate synthetic views by OUTPAINTING beyond the frame edges.

Strategy 2 of 3. The camera POSE is unchanged (same position, same
orientation) but the frame is widened, so the INTRINSICS change:

    focal length (px) : UNCHANGED - we did not change the lens or the pixel
                        pitch, we merely captured more of the image plane.
    principal point   : SHIFTS by the paste offset, so that the optical axis
                        still lands on the same physical spot.
    width / height    : grow by the expansion factor.
    => field of view GROWS, which is the entire point.

Getting this backwards (e.g. scaling the focal length with the frame) leaves
the pose looking right while every ray is subtly wrong - the classic silent
bug in this approach. We assert the FOV actually widened, and that the real
image region still maps to the original rays.

  python src/gen_outpaint.py --manifest subsets/truck_k10_seed0_fps.json --n 20
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(os.path.expanduser("~/fewshot_gs"))
REPO = ROOT / "gaussian-splatting"
sys.path.insert(0, str(REPO))
from scene.colmap_loader import (read_extrinsics_binary,  # noqa: E402
                                 read_intrinsics_binary)

MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"


def sd_size_for(aspect: float, target_w: int = 704):
    """Nearest multiple-of-8 size preserving aspect, near SD's sweet spot."""
    w = int(round(target_w / 8) * 8)
    h = int(round((w / aspect) / 8) * 8)
    return w, h


def fov_deg(size_px: float, focal_px: float) -> float:
    return math.degrees(2.0 * math.atan(size_px / (2.0 * focal_px)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--source", default=str(ROOT / "data/tandt/truck"))
    ap.add_argument("--out", default=str(ROOT / "synthetic"))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--expand", type=float, default=1.25,
                    help="linear frame expansion factor (1.25 = 25% wider)")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--prompt", default="a photo of a truck parked on a street, "
                                        "wide view, realistic, sharp focus")
    ap.add_argument("--negative", default="blurry, distorted, watermark, text, lowres")
    # Model-robustness ablation. Is the outpainting result a property of
    # augmentation, or of Stable Diffusion 1.5 specifically? --model swaps the
    # checkpoint; --label puts the output in its own directory so both versions
    # coexist and can be compared directly. NOTE build_scene.py names scenes
    # from the poses.json "strategy" field rather than the directory it is
    # given, so --label has to flow into that field too.
    ap.add_argument("--model", default=MODEL,
                    help="diffusers inpainting checkpoint")
    ap.add_argument("--label", default="outpaint",
                    help="name used in the output directory and filenames")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    src_root = Path(args.source)
    reals = list(manifest["images"])
    seed, k, method = manifest["seed"], manifest["k"], manifest["method"]
    s = args.expand

    tag = f"{manifest['scene']}_k{k}_seed{seed}_{method}_{args.label}"
    out_dir = Path(args.out) / tag
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    intr = read_intrinsics_binary(str(src_root / "sparse/0/cameras.bin"))
    extr = read_extrinsics_binary(str(src_root / "sparse/0/images.bin"))
    by_name = {im.name: im for im in extr.values()}

    cam = intr[1]
    Wc, Hc = int(cam.width), int(cam.height)
    fx, fy, cx, cy = (float(v) for v in cam.params)

    # ---- new intrinsics, computed in COLMAP pixel space ------------------
    Wc2, Hc2 = int(round(Wc * s)), int(round(Hc * s))
    ox_c, oy_c = (Wc2 - Wc) // 2, (Hc2 - Hc) // 2
    cx2, cy2 = cx + ox_c, cy + oy_c        # focal lengths deliberately unchanged

    print("=== intrinsics ===")
    print(f"  original : {Wc}x{Hc}  f=({fx:.2f},{fy:.2f})  c=({cx:.1f},{cy:.1f})"
          f"  FOV=({fov_deg(Wc,fx):.2f}, {fov_deg(Hc,fy):.2f}) deg")
    print(f"  expanded : {Wc2}x{Hc2}  f=({fx:.2f},{fy:.2f})  c=({cx2:.1f},{cy2:.1f})"
          f"  FOV=({fov_deg(Wc2,fx):.2f}, {fov_deg(Hc2,fy):.2f}) deg")

    assert fov_deg(Wc2, fx) > fov_deg(Wc, fx), "FOV must widen"
    # The original frame must still occupy the same rays: its left edge sits at
    # pixel ox_c in the new frame, so (0 - cx) and (ox_c - cx2) must agree.
    assert abs((0 - cx) - (ox_c - cx2)) < 1e-9, "principal point shift inconsistent"
    print("  checks: FOV widened OK, principal-point shift consistent OK")

    # ---- disk-space geometry (images on disk are ~half COLMAP size) ------
    probe = Image.open(src_root / "images" / reals[0])
    Wd, Hd = probe.size
    Wd2 = int(round(Wc2 * Wd / Wc))
    Hd2 = int(round(Hc2 * Hd / Hc))
    ox_d = int(round(ox_c * Wd / Wc))
    oy_d = int(round(oy_c * Hd / Hc))
    print(f"  disk: {Wd}x{Hd} -> {Wd2}x{Hd2}, paste offset ({ox_d},{oy_d})")

    sd_w, sd_h = sd_size_for(Wd2 / Hd2)
    print(f"  SD canvas: {sd_w}x{sd_h}")

    print(f"\n[{tag}] generating {args.n} outpainted views")

    from diffusers import StableDiffusionInpaintPipeline
    print(f"  model: {args.model}")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True, safety_checker=None, requires_safety_checker=False)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()

    rng = np.random.default_rng(seed * 1000 + 23)
    torch.cuda.reset_peak_memory_stats()
    records = []
    t0 = time.time()

    for i in range(args.n):
        src_name = reals[i % len(reals)]
        variant = i // len(reals)
        out_name = f"synth_{args.label}_{Path(src_name).stem}_v{variant:02d}.jpg"

        real = Image.open(src_root / "images" / src_name).convert("RGB")

        # Canvas at full disk resolution with the real frame centred.
        canvas = Image.new("RGB", (Wd2, Hd2), (0, 0, 0))
        canvas.paste(real, (ox_d, oy_d))
        keep = Image.new("L", (Wd2, Hd2), 0)
        ImageDraw.Draw(keep).rectangle(
            [ox_d, oy_d, ox_d + real.width - 1, oy_d + real.height - 1], fill=255)

        # SD works on the downscaled canvas. Feather the mask a few px INTO
        # the real region so the model has context to blend against.
        canvas_sd = canvas.resize((sd_w, sd_h), Image.LANCZOS)
        keep_sd = keep.resize((sd_w, sd_h), Image.NEAREST)
        mask_sd = Image.eval(keep_sd, lambda v: 255 - v)
        mask_sd = mask_sd.filter(ImageFilter.MaxFilter(5))   # grow into the seam

        gen = torch.Generator("cuda").manual_seed(int(rng.integers(0, 2**31 - 1)))
        painted = pipe(prompt=args.prompt, negative_prompt=args.negative,
                       image=canvas_sd, mask_image=mask_sd,
                       num_inference_steps=args.steps,
                       guidance_scale=args.guidance,
                       height=sd_h, width=sd_w, generator=gen).images[0]

        # Composite at full resolution: the real frame is preserved exactly,
        # only the new border is synthetic.
        painted_full = painted.resize((Wd2, Hd2), Image.LANCZOS)
        out_img = Image.composite(canvas, painted_full, keep)
        out_img.save(img_dir / out_name, quality=95)

        im = by_name[src_name]
        records.append({
            "name": out_name,
            "source_image": src_name,
            "strategy": args.label,
            "qvec": [float(x) for x in im.qvec],
            "tvec": [float(x) for x in im.tvec],
            "camera_id": 2,                      # the widened camera
            "pose_source": "copied from source view; intrinsics widened",
            "expand": s,
        })
        print(f"  [{i+1}/{args.n}] {out_name}  from {src_name}")

    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3

    (out_dir / "poses.json").write_text(json.dumps({
        "tag": tag, "strategy": args.label, "scene": manifest["scene"],
        "k": k, "seed": seed, "method": method,
        "source_manifest": str(args.manifest),
        "extra_cameras": {
            "2": {"model": "PINHOLE", "width": Wc2, "height": Hc2,
                  "params": [fx, fy, cx2, cy2],
                  "derived_from": 1, "expand": s,
                  "note": "focal unchanged; principal point shifted by paste offset"}
        },
        "config": {"model": MODEL, "steps": args.steps, "guidance": args.guidance,
                   "expand": s, "sd_size": [sd_w, sd_h],
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
