"""Train + render + evaluate one condition, writing a single results.json.

Every number the report needs for one grid cell comes out of here:
PSNR / SSIM / LPIPS on the held-out views, wall-clock training time, peak
VRAM, and final Gaussian count.

  python src/run_experiment.py --scene scenes/truck_k10_seed0_fps_fake0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
ROOT = HOME / "fewshot_gs"
REPO = ROOT / "gaussian-splatting"
PY = str(ROOT / "venv/bin/python")


class VramMonitor(threading.Thread):
    """Poll nvidia-smi so we get a true peak, not a guess.

    NOTE: this is whole-GPU usage, so it includes the desktop compositor
    (~150 MiB here). Reported as-is and noted in the results file.
    """

    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        # NB: must not be called _stop - that shadows Thread._stop(), which
        # join() calls internally, and blows up with 'Event' not callable.
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    text=True, stderr=subprocess.DEVNULL, timeout=5)
                self.peak = max(self.peak, int(out.strip().split("\n")[0]))
            except Exception:
                pass
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=5)
        return self.peak


def sh(cmd, cwd=REPO, log=None):
    """Run a subprocess, tee output to a log file, return exit code."""
    with open(log, "ab") if log else open(os.devnull, "wb") as fh:
        p = subprocess.run(cmd, cwd=str(cwd), stdout=fh,
                           stderr=subprocess.STDOUT)
    return p.returncode


def evaluate(model_dir: Path, iteration: int, device="cuda"):
    """PSNR / SSIM / LPIPS over the rendered held-out views."""
    import torch
    from PIL import Image
    import numpy as np

    sys.path.insert(0, str(REPO))
    from utils.loss_utils import ssim as ssim_fn

    rdir = model_dir / "test" / f"ours_{iteration}" / "renders"
    gdir = model_dir / "test" / f"ours_{iteration}" / "gt"
    files = sorted(p.name for p in rdir.glob("*.png"))
    if not files:
        raise RuntimeError(f"no renders in {rdir}")

    import lpips
    lp = lpips.LPIPS(net="vgg").to(device)

    def load(p):
        a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)

    psnrs, ssims, lpipss = [], [], []
    with torch.no_grad():
        for f in files:
            r, g = load(rdir / f), load(gdir / f)
            mse = torch.mean((r - g) ** 2).item()
            psnrs.append(float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse))
            ssims.append(ssim_fn(r, g).item())
            lpipss.append(lp(r * 2 - 1, g * 2 - 1).item())

    f = lambda xs: {"mean": float(np.mean(xs)), "std": float(np.std(xs))}
    return {"n_views": len(files), "psnr": f(psnrs), "ssim": f(ssims),
            "lpips": f(lpipss),
            "per_view": {"files": files, "psnr": psnrs,
                         "ssim": ssims, "lpips": lpipss}}


def gaussian_count(model_dir: Path, iteration: int) -> int:
    from plyfile import PlyData
    p = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    return len(PlyData.read(str(p))["vertex"]) if p.exists() else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", default=None, help="model dir (default runs/<scene name>)")
    ap.add_argument("--iterations", type=int, default=7000)
    ap.add_argument("--resolution", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    # Depth regularisation. Passed straight through to train.py as -d, which
    # resolves it relative to the scene: <scene>/<depths>. Requires the scene
    # to have been through src/add_depths.py first.
    ap.add_argument("--depths", default="", help="depth dir inside the scene")
    args = ap.parse_args()

    scene = Path(args.scene).resolve()
    model = Path(args.out).resolve() if args.out else (ROOT / "runs" / scene.name)
    # The tag names the RUN, not the scene: one scene can be trained under
    # several configurations (with and without depth regularisation), and they
    # must not collide in runs/ or in the aggregated tables. Identical to the
    # previous behaviour whenever --out is omitted.
    tag = model.name
    results_path = model / "results.json"

    if results_path.exists() and not args.force:
        print(f"[skip] {tag} already has results.json (use --force)")
        return

    model.mkdir(parents=True, exist_ok=True)
    log = model / "train.log"
    if log.exists():
        log.unlink()

    split = json.loads((scene / "split.json").read_text())
    n_train, n_test = len(split["train"]), len(split["test"])
    print(f"[{tag}] {n_train} train / {n_test} test, "
          f"{args.iterations} iters, -r {args.resolution}")

    env_note = "peak_vram_mib includes desktop compositor (~150 MiB)"

    # ---- train --------------------------------------------------------
    # Reuse an existing checkpoint so a crash in the eval stage doesn't cost
    # another full training run. Timing/VRAM are then unknown, not faked.
    ckpt = model / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply"
    if ckpt.exists() and not args.force:
        print(f"  reusing checkpoint at iteration {args.iterations}")
        train_time, peak = None, None
    else:
        mon = VramMonitor(); mon.start()
        t0 = time.time()
        depth_arg = ["-d", args.depths] if args.depths else []
        rc = sh([PY, "train.py", "-s", str(scene), "-m", str(model), "--eval",
                 "-r", str(args.resolution), "--iterations", str(args.iterations),
                 "--test_iterations", str(args.iterations),
                 "--save_iterations", str(args.iterations),
                 *depth_arg,
                 "--disable_viewer", "--quiet"], log=log)
        train_time = time.time() - t0
        peak = mon.stop()
        if rc != 0:
            print(f"[FAIL] training exited {rc}; see {log}")
            sys.exit(rc)
        print(f"  trained in {train_time:.0f}s, peak {peak} MiB")

    # ---- render held-out views ----------------------------------------
    rc = sh([PY, "render.py", "-m", str(model), "--iteration",
             str(args.iterations), "--skip_train", "--quiet"], log=log)
    if rc != 0:
        print(f"[FAIL] render exited {rc}; see {log}")
        sys.exit(rc)

    # ---- metrics -------------------------------------------------------
    metrics = evaluate(model, args.iterations)
    ng = gaussian_count(model, args.iterations)

    prov = dict(split.get("provenance", {}))
    prov["depth_reg"] = bool(args.depths)

    rec = {
        "tag": tag,
        "scene": str(scene),
        "provenance": prov,
        "config": {"iterations": args.iterations, "resolution": args.resolution,
                   "n_train": n_train, "n_test": n_test,
                   "depths": args.depths or None},
        "cost": {"train_seconds": round(train_time, 1) if train_time else None,
                 "peak_vram_mib": peak, "note": env_note,
                 "reused_checkpoint": train_time is None,
                 "n_gaussians": ng},
        "metrics": metrics,
    }
    results_path.write_text(json.dumps(rec, indent=2))

    print(f"  PSNR {metrics['psnr']['mean']:.3f}  "
          f"SSIM {metrics['ssim']['mean']:.4f}  "
          f"LPIPS {metrics['lpips']['mean']:.4f}  "
          f"gaussians {ng:,}")
    print(f"  -> {results_path}")


if __name__ == "__main__":
    main()
