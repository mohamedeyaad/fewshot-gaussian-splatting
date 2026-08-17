"""Generate synthetic views at NEW camera poses via depth-warping + diffusion.

Strategy 3 of 3, and the only one that supplies genuinely new viewpoint
information - which is exactly what few-shot GS lacks, and exactly why it is
the riskiest.

Pipeline per image:
  1. take a real view and its known COLMAP pose
  2. predict a monocular depth map (Depth Anything V2)
  3. ANCHOR that depth to true scene scale using the sparse COLMAP 3D points
     this view observed. Monocular depth is relative - without this step every
     pixel lands at the wrong distance and the warp is meaningless.
  4. choose a new camera between this view and its nearest neighbour (slerp
     rotation, lerp centre) - we CHOOSE it, so the pose is known exactly
  5. back-project pixels to 3D, re-project into the new camera with a z-buffer
     -> a warped image with holes where hidden surfaces are revealed
  6. Stable Diffusion fills only the holes
  7. save with the chosen pose

Intrinsics are unchanged (same lens, camera moved), so these views reuse
camera_id 1.

  python src/gen_guided.py --manifest subsets/truck_k10_seed0_fps.json --n 20
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
from PIL import Image
from scipy.ndimage import binary_opening

# Structuring-element size for separating thin splatting gaps from genuine
# disocclusions. Hole structures narrower than this are interpolated away;
# anything that survives the opening goes to the diffusion model.
SPECKLE_OPEN = 5
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(os.path.expanduser("~/fewshot_gs"))
REPO = ROOT / "gaussian-splatting"
sys.path.insert(0, str(REPO))
from scene.colmap_loader import (read_extrinsics_binary,  # noqa: E402
                                 read_intrinsics_binary,
                                 read_next_bytes, qvec2rotmat)

SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


# ----------------------------------------------------------------- geometry
def qvec_to_scipy(q):
    """COLMAP stores [w,x,y,z]; scipy wants [x,y,z,w]."""
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


def scipy_to_qvec(q):
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)


def cam_centre(R, t):
    return -R.T @ t


def interp_pose(q1, t1, q2, t2, frac):
    """Interpolate between two COLMAP poses. Rotation by slerp, camera CENTRE
    by lerp (lerping tvec directly would swing the camera through an arc)."""
    rots = Rotation.from_quat(np.stack([qvec_to_scipy(q1), qvec_to_scipy(q2)]))
    slerp = Slerp([0.0, 1.0], rots)
    R_new = slerp([frac])[0].as_matrix()
    C1 = cam_centre(qvec2rotmat(q1), t1)
    C2 = cam_centre(qvec2rotmat(q2), t2)
    C_new = (1 - frac) * C1 + frac * C2
    t_new = -R_new @ C_new
    q_new = scipy_to_qvec(Rotation.from_matrix(R_new).as_quat())
    return q_new, t_new, R_new


def align_depth_to_colmap(pred_disp, xys, depths_true, W, H):
    """Fit pred_disp * a + b ~= 1/depth_true at observed keypoints.

    Monocular models emit disparity-like values (bigger = nearer) on an
    arbitrary scale. Aligning in DISPARITY space is the standard choice: it is
    linear there, and it weights near geometry more, which is what dominates
    a warp. Returns (a, b, stats).
    """
    u = np.round(xys[:, 0]).astype(int)
    v = np.round(xys[:, 1]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (depths_true > 1e-6)
    if ok.sum() < 20:
        return None, None, {"n": int(ok.sum()), "reason": "too few points"}
    p = pred_disp[v[ok], u[ok]].astype(np.float64)
    q = 1.0 / depths_true[ok].astype(np.float64)

    # Robust-ish: drop the extreme 5% of residuals after a first fit.
    A = np.stack([p, np.ones_like(p)], axis=1)
    sol, *_ = np.linalg.lstsq(A, q, rcond=None)
    resid = np.abs(A @ sol - q)
    keep = resid <= np.quantile(resid, 0.95)
    sol, *_ = np.linalg.lstsq(A[keep], q[keep], rcond=None)
    a, b = float(sol[0]), float(sol[1])

    fitted = a * p[keep] + b
    ss_res = float(np.sum((fitted - q[keep]) ** 2))
    ss_tot = float(np.sum((q[keep] - q[keep].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return a, b, {"n": int(keep.sum()), "r2": round(r2, 4),
                  "a": round(a, 6), "b": round(b, 6)}


def warp(img_rgb, depth, K, R1, t1, R2, t2):
    """Forward-warp with a z-buffer. Returns (warped, valid_mask)."""
    H, W = depth.shape
    fx, fy, cx, cy = K
    vs, us = np.mgrid[0:H, 0:W]
    z = depth.reshape(-1)
    u = us.reshape(-1).astype(np.float64)
    v = vs.reshape(-1).astype(np.float64)

    # pixel -> source camera coords -> world
    Xc = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=1)
    Pw = (Xc - t1) @ R1                       # R1.T @ (Xc - t1), vectorised

    # world -> target camera
    Xc2 = Pw @ R2.T + t2
    good = Xc2[:, 2] > 1e-6
    u2 = np.full(len(z), -1.0)
    v2 = np.full(len(z), -1.0)
    u2[good] = fx * Xc2[good, 0] / Xc2[good, 2] + cx
    v2[good] = fy * Xc2[good, 1] / Xc2[good, 2] + cy

    ui = np.round(u2).astype(int)
    vi = np.round(v2).astype(int)
    inb = good & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

    out = np.zeros((H, W, 3), dtype=np.uint8)
    zbuf = np.full((H, W), np.inf)
    src = img_rgb.reshape(-1, 3)

    # Painter's algorithm by depth: sort far->near so nearer writes last.
    order = np.argsort(-Xc2[:, 2])
    order = order[inb[order]]

    # Exact splat first - this alone must reproduce the source under an
    # identity warp, which test_warp.py asserts.
    flat0 = vi[order] * W + ui[order]
    out.reshape(-1, 3)[flat0] = src[order]
    zbuf.reshape(-1)[flat0] = Xc2[order, 2]
    exact = np.isfinite(zbuf)

    # Then fill ONLY the gaps using a 1px-offset splat. As the camera moves,
    # single-pixel splats spread apart and leave a regular lattice of
    # sub-pixel holes that reads as a chain-link mesh - a systematic artifact
    # that would confound training. Offsets never overwrite exact hits, so
    # identity stays pixel-perfect.
    if not exact.all():
        fill_rgb = np.zeros_like(out)
        fill_z = np.full((H, W), np.inf)
        for du, dv in ((1, 0), (0, 1), (1, 1)):
            uu = np.clip(ui[order] + du, 0, W - 1)
            vv = np.clip(vi[order] + dv, 0, H - 1)
            f = vv * W + uu
            fill_rgb.reshape(-1, 3)[f] = src[order]
            fill_z.reshape(-1)[f] = Xc2[order, 2]
        gap = (~exact) & np.isfinite(fill_z)
        out[gap] = fill_rgb[gap]
        zbuf[gap] = fill_z[gap]

    valid = np.isfinite(zbuf)
    # Classify holes by CONNECTED-COMPONENT AREA, not by morphology. The
    # splatting gaps form thin connected diagonal chains, which binary_closing
    # does not fill - leaving a chain-link artifact that also survives the
    # downscale to SD resolution (a 1px mask aliases away, so SD never repairs
    # it). Area-based classification catches them regardless of shape.
    holes = ~valid
    thin = np.zeros_like(holes)
    if holes.any():
        # Separate by THICKNESS, not area. The splatting gaps form a 1-2px
        # lattice that 8-connectivity merges into one enormous component, so
        # an area test never fires on it. A morphological opening erases thin
        # structures whatever their extent while leaving genuine disocclusions
        # (tens of px across) intact.
        thick = binary_opening(holes, structure=np.ones((SPECKLE_OPEN, SPECKLE_OPEN)))
        thin = holes & ~thick
    if thin.any():
        # Interpolate the lattice away. Left as a mask it would survive to the
        # SD stage, where downscaling aliases a 1px mask out of existence and
        # the black lattice is preserved into the final image.
        import cv2
        out = cv2.inpaint(out, thin.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    # Thin gaps are now real pixels; only genuine disocclusions stay holes.
    return out, valid | thin


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--source", default=str(ROOT / "data/tandt/truck"))
    ap.add_argument("--out", default=str(ROOT / "synthetic"))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--frac-lo", type=float, default=0.25)
    ap.add_argument("--frac-hi", type=float, default=0.50)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--prompt", default="a photo of a truck parked on a street, "
                                        "realistic, sharp focus, natural lighting")
    ap.add_argument("--negative", default="blurry, distorted, watermark, text, lowres")
    ap.add_argument("--save-debug", action="store_true")
    # ---- ablation control -------------------------------------------------
    # Pose-guided synthesis changes two things at once relative to a real view:
    # the camera pose, and the ~10% of pixels that diffusion invents to fill
    # disocclusions. Its measured damage cannot be attributed to either on its
    # own. This flag removes the second: warp to the new pose exactly as
    # before, but leave the holes as they come out of warp() - black, since
    # `out` is zero-initialised - and never load Stable Diffusion.
    #
    # Comparing the two isolates the diffusion step's contribution. Everything
    # upstream (depth estimation, COLMAP alignment, pose interpolation, the
    # z-buffered forward warp, thin-gap interpolation) is bit-identical,
    # including the RNG stream that picks source views and interpolation
    # fractions - so the two conditions see exactly the same poses.
    ap.add_argument("--no-diffusion", action="store_true",
                    help="warp only; leave disocclusion holes black")
    ap.add_argument("--label", default="guided",
                    help="name used in the output directory and filenames")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    src_root = Path(args.source)
    reals = list(manifest["images"])
    seed, k, method = manifest["seed"], manifest["k"], manifest["method"]

    tag = f"{manifest['scene']}_k{k}_seed{seed}_{method}_{args.label}"
    out_dir = Path(args.out) / tag
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    intr = read_intrinsics_binary(str(src_root / "sparse/0/cameras.bin"))
    extr = read_extrinsics_binary(str(src_root / "sparse/0/images.bin"))
    by_name = {im.name: im for im in extr.values()}

    cam = intr[1]
    Wc, Hc = int(cam.width), int(cam.height)
    fx_c, fy_c, cx_c, cy_c = (float(x) for x in cam.params)

    probe = Image.open(src_root / "images" / reals[0])
    Wd, Hd = probe.size
    sx, sy = Wd / Wc, Hd / Hc                 # COLMAP px -> disk px
    K_disk = (fx_c * sx, fy_c * sy, cx_c * sx, cy_c * sy)
    print(f"intrinsics (disk {Wd}x{Hd}): fx={K_disk[0]:.2f} fy={K_disk[1]:.2f} "
          f"cx={K_disk[2]:.2f} cy={K_disk[3]:.2f}")

    # id -> xyz lookup. The repo's read_points3D_binary returns bare arrays
    # without the point IDs, and we need IDs to match against image.point3D_ids,
    # so parse the file directly.
    id_to_xyz = {}
    with open(src_root / "sparse/0/points3D.bin", "rb") as fid:
        n_pts = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(n_pts):
            props = read_next_bytes(fid, 43, "QdddBBBd")
            pid = props[0]
            id_to_xyz[pid] = np.array(props[1:4], dtype=np.float64)
            tl = read_next_bytes(fid, 8, "Q")[0]
            read_next_bytes(fid, 8 * tl, "ii" * tl)
    print(f"loaded {len(id_to_xyz):,} sparse 3D points")

    # camera centres of the K real views, for nearest-neighbour selection
    centres = {}
    for n in reals:
        im = by_name[n]
        centres[n] = cam_centre(qvec2rotmat(im.qvec), im.tvec)

    # -------- phase 1: depth, aligned to COLMAP scale --------------------
    print(f"\n[{tag}] phase 1: depth estimation + scale alignment")
    from transformers import pipeline as hf_pipeline
    dpipe = hf_pipeline("depth-estimation", model=DEPTH_MODEL, device=0)

    depths, align_stats = {}, {}
    for n in reals:
        img = Image.open(src_root / "images" / n).convert("RGB")
        res = dpipe(img)
        pred = res["predicted_depth"]
        if hasattr(pred, "detach"):
            pred = pred.detach().float().cpu().numpy()
        pred = np.squeeze(pred)
        if pred.shape != (Hd, Wd):
            pred = np.array(Image.fromarray(pred).resize((Wd, Hd), Image.BILINEAR))

        im = by_name[n]
        ids = im.point3D_ids
        sel = ids > 0
        xys_c = im.xys[sel]
        R = qvec2rotmat(im.qvec)
        P = np.stack([id_to_xyz[i] for i in ids[sel] if i in id_to_xyz])
        keep = np.array([i in id_to_xyz for i in ids[sel]])
        xys_c = xys_c[keep]
        z_true = (P @ R.T + im.tvec)[:, 2]
        xys_d = xys_c * np.array([sx, sy])

        a, b, st = align_depth_to_colmap(pred, xys_d, z_true, Wd, Hd)
        if a is None:
            print(f"  {n}: alignment FAILED ({st}) - skipping this source")
            continue
        disp = a * pred + b
        disp = np.maximum(disp, 1e-4)
        depths[n] = 1.0 / disp
        align_stats[n] = st
        print(f"  {n}: {st['n']} pts, R2={st['r2']:.3f}, "
              f"depth range {depths[n].min():.2f}..{np.percentile(depths[n],99):.2f}")

    del dpipe
    torch.cuda.empty_cache()
    usable = [n for n in reals if n in depths]
    if not usable:
        print("no usable sources - aborting")
        sys.exit(1)
    mean_r2 = float(np.mean([align_stats[n]["r2"] for n in usable]))
    print(f"  mean alignment R2 = {mean_r2:.3f} over {len(usable)} views")

    # -------- phase 2: warp (+ inpaint, unless this is the control) -------
    pipe = None
    if args.no_diffusion:
        print(f"\nphase 2: warping to new poses, holes left BLACK (control)")
    else:
        print(f"\nphase 2: warping to new poses + filling holes")
        from diffusers import StableDiffusionInpaintPipeline
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            SD_MODEL, torch_dtype=torch.float16, variant="fp16",
            use_safetensors=True, safety_checker=None, requires_safety_checker=False)
        pipe = pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        pipe.enable_attention_slicing()
        pipe.vae.enable_slicing()

    # Keep SD near its 512x512 training resolution. Running at native 976x544
    # was tried and produces incoherent output - SD 1.5 degrades badly at ~2x
    # its training size. Mask aliasing is not a concern here because the
    # speckle is already interpolated away in warp(), so the only holes left
    # are large disocclusions that survive the downscale intact.
    sd_w = 704
    sd_h = int(round((sd_w / (Wd / Hd)) / 8) * 8)

    rng = np.random.default_rng(seed * 1000 + 41)
    torch.cuda.reset_peak_memory_stats()
    records = []
    t0 = time.time()

    for i in range(args.n):
        src_name = usable[i % len(usable)]
        variant = i // len(usable)
        out_name = f"synth_{args.label}_{Path(src_name).stem}_v{variant:02d}.jpg"

        im = by_name[src_name]
        R1, t1 = qvec2rotmat(im.qvec), np.asarray(im.tvec, dtype=float)

        # nearest OTHER real view to interpolate toward
        others = [n for n in usable if n != src_name]
        nb = min(others, key=lambda n: np.linalg.norm(centres[n] - centres[src_name]))
        im2 = by_name[nb]
        frac = float(rng.uniform(args.frac_lo, args.frac_hi))
        q_new, t_new, R2 = interp_pose(im.qvec, t1, im2.qvec, np.asarray(im2.tvec, float), frac)

        rgb = np.asarray(Image.open(src_root / "images" / src_name).convert("RGB"))
        warped, valid = warp(rgb, depths[src_name], K_disk, R1, t1, R2, t_new)
        hole_frac = float(1.0 - valid.mean())

        keep_mask = Image.fromarray((valid * 255).astype(np.uint8))
        # Draw from the RNG either way, so the pose sequence is identical
        # between the diffusion and warp-only conditions and the two are
        # comparable image-for-image.
        sd_seed = int(rng.integers(0, 2**31 - 1))

        if args.no_diffusion:
            # Holes stay as warp() left them: black. This is the control.
            out_img = Image.fromarray(warped)
        else:
            warped_img = Image.fromarray(warped).resize((sd_w, sd_h), Image.LANCZOS)
            hole = Image.fromarray(((~valid) * 255).astype(np.uint8)).resize(
                (sd_w, sd_h), Image.NEAREST)
            gen = torch.Generator("cuda").manual_seed(sd_seed)
            filled = pipe(prompt=args.prompt, negative_prompt=args.negative,
                          image=warped_img, mask_image=hole,
                          num_inference_steps=args.steps,
                          guidance_scale=args.guidance,
                          height=sd_h, width=sd_w, generator=gen).images[0]
            filled_full = filled.resize((Wd, Hd), Image.LANCZOS)
            out_img = Image.composite(Image.fromarray(warped), filled_full, keep_mask)
        out_img.save(img_dir / out_name, quality=95)

        if args.save_debug:
            dbg = out_dir / "debug"; dbg.mkdir(exist_ok=True)
            Image.fromarray(warped).save(dbg / f"{Path(out_name).stem}_warp.png")
            keep_mask.save(dbg / f"{Path(out_name).stem}_valid.png")

        records.append({
            "name": out_name,
            "source_image": src_name,
            "neighbour_image": nb,
            "strategy": args.label,
            "qvec": [float(x) for x in q_new],
            "tvec": [float(x) for x in t_new],
            "camera_id": 1,
            "pose_source": f"slerp/lerp {frac:.3f} from {src_name} toward {nb}",
            "interp_frac": round(frac, 4),
            "hole_fraction": round(hole_frac, 4),
            "depth_align_r2": align_stats[src_name]["r2"],
        })
        print(f"  [{i+1}/{args.n}] {out_name}  {src_name}->{nb} "
              f"frac={frac:.2f} holes={hole_frac*100:.1f}%")

    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3

    (out_dir / "poses.json").write_text(json.dumps({
        # build_scene.py names scenes from THIS field, not from the directory,
        # so a hardcoded value here silently mislabels any variant condition.
        "tag": tag, "strategy": args.label, "scene": manifest["scene"],
        "k": k, "seed": seed, "method": method,
        "source_manifest": str(args.manifest),
        "config": {"sd_model": SD_MODEL, "depth_model": DEPTH_MODEL,
                   "steps": args.steps, "guidance": args.guidance,
                   "frac_range": [args.frac_lo, args.frac_hi],
                   "sd_size": [sd_w, sd_h], "prompt": args.prompt,
                   "negative_prompt": args.negative},
        "depth_alignment": {"mean_r2": round(mean_r2, 4), "per_view": align_stats},
        "cost": {"seconds": round(elapsed, 1),
                 "seconds_per_image": round(elapsed / max(args.n, 1), 2),
                 "peak_vram_gb": round(peak, 2)},
        "images": records,
    }, indent=2))

    mean_holes = float(np.mean([r["hole_fraction"] for r in records]))
    print(f"\n  {args.n} images in {elapsed:.0f}s "
          f"({elapsed/max(args.n,1):.1f}s each), peak {peak:.2f} GB")
    print(f"  mean hole fraction {mean_holes*100:.1f}%, mean depth-align R2 {mean_r2:.3f}")
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
