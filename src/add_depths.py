"""Attach depth supervision to an already-built scene.

Kept separate from build_scene.py on purpose: the 191 completed runs were all
produced by the current build_scene.py, and changing it would make the scenes
they were trained from unreproducible. This only ever ADDS to a scene.

Two things go in:

  depths/            one 16-bit PNG per image IN THE SCENE. Every image needs
                     one - camera_utils.py:28 calls cv2.imread on the path
                     unconditionally, and cv2 returns None for a missing file
                     rather than raising, so the .astype() on the next line
                     raises and kills the run. Test views included.

  sparse/0/depth_params.json
                     per-image {scale, offset} mapping normalised disparity
                     into scene units. Real views get theirs from
                     make_depth_scale.py, fitted against the sparse COLMAP
                     points each view observes.

SYNTHETIC VIEWS ARE DEPRIVED OF SUPERVISION, DELIBERATELY. They have no COLMAP
points, so no honest scale can be fitted for them; and supervising fabricated
geometry with a depth prior estimated FROM that fabrication would be circular.
They are written with scale 0, which cameras.py:69 compares against the median
scale, marks unreliable, and masks out of the loss - the same convention
make_depth_scale.py uses when it cannot fit an image. So depth constrains real
views only, and the "+both" condition stays interpretable.

  python src/add_depths.py --scene scenes/truck_k5_seed0_fps_fake0
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

ROOT = Path(os.path.expanduser("~/fewshot_gs"))


def link(target: Path, link_path: Path):
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(target.resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--source", default=str(ROOT / "data/tandt/truck"),
                    help="dataset the real images and depth_params come from")
    args = ap.parse_args()

    scene = Path(args.scene).resolve()
    source = Path(args.source).resolve()

    src_depths = source / "depths"
    src_params = source / "sparse/0/depth_params.json"
    for p in (src_depths, src_params):
        if not p.exists():
            raise SystemExit(
                f"missing {p}\nrun:  python src/gen_depths.py --source {source}\n"
                f"then: python gaussian-splatting/utils/make_depth_scale.py "
                f"--base_dir {source} --depths_dir {src_depths}")

    split = json.loads((scene / "split.json").read_text())
    prov = split.get("provenance", {})
    all_names = sorted(set(split["train"]) | set(split["test"]))
    real_params = json.loads(src_params.read_text())

    dep_dir = scene / "depths"
    dep_dir.mkdir(parents=True, exist_ok=True)

    params, n_real, n_synth, missing = {}, 0, 0, []
    synth_dir = prov.get("synthetic_dir")
    synth_depths = (ROOT / synth_dir / "depths") if synth_dir else None

    def usable(e):
        """A scale is only usable if it is finite and positive.

        make_depth_scale.py computes scale = s_colmap / s_mono with no guard
        against s_mono == 0, so degenerate images come out as inf. An inf in
        the file poisons the median that dataset_readers.py computes over all
        positive scales, and every image then fails the 0.2x/5x gate in
        cameras.py - one bad image silently disabling depth for the scene.
        """
        s, o = e.get("scale"), e.get("offset", 0)
        return (isinstance(s, (int, float)) and math.isfinite(s) and s > 0
                and isinstance(o, (int, float)) and math.isfinite(o))

    for name in all_names:
        stem = Path(name).stem
        png = f"{stem}.png"
        if (src_depths / png).exists():                    # a real view
            link(src_depths / png, dep_dir / png)
            if stem in real_params and usable(real_params[stem]):
                params[stem] = real_params[stem]
                n_real += 1
            else:
                # depth present but unfittable: mask it rather than supervise
                # with an unscaled map, which cameras.py would otherwise do
                params[stem] = {"scale": 0, "offset": 0}
                missing.append(stem)
        elif synth_depths and (synth_depths / png).exists():  # a synthetic view
            link(synth_depths / png, dep_dir / png)
            params[stem] = {"scale": 0, "offset": 0}
            n_synth += 1
        else:
            raise SystemExit(
                f"no depth map for {name}\n"
                + (f"generate them with:\n  python src/gen_depths.py "
                   f"--images {ROOT / synth_dir}/images --out {synth_depths}"
                   if synth_dir else ""))

    # The scene's sparse/ may be a symlink to the shared dataset. Replace it
    # with a real directory holding links to the originals, so that writing
    # depth_params.json here cannot reach the dataset.
    sp = scene / "sparse"
    if sp.is_symlink():
        real_sp = sp.resolve()
        sp.unlink()
        (sp / "0").mkdir(parents=True)
        for f in (real_sp / "0").iterdir():
            link(f, sp / "0" / f.name)

    # CRITICAL: depth_params.json must be a real file, never a link. If the
    # dataset already has one, the loop above just linked to it, and writing
    # through that link would overwrite the DATASET's copy - which is exactly
    # the bug that silently disabled depth supervision on every training view
    # (scene 1's 37 entries replaced the dataset's 251, and every later scene
    # then read the truncated file).
    dst = sp / "0" / "depth_params.json"
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.write_text(json.dumps(params, indent=2))

    # The depth loss only ever touches TRAINING views, so the count that
    # decides whether this experiment means anything is how many of THOSE are
    # supervised - not how many images in the scene have an entry. Reporting
    # the latter is what hid the symlink bug: it read "37 real supervised"
    # while every one of the five training views was masked.
    train_ok = sum(1 for n in split["train"]
                   if params.get(Path(n).stem, {}).get("scale", 0) > 0)
    n_train = len(split["train"])
    print(f"  {scene.name}: {train_ok}/{n_train} TRAINING views supervised "
          f"({n_synth} synthetic masked"
          + (f", {len(missing)} unfittable" if missing else "") + ")")
    if missing:
        print(f"    unfittable: {', '.join(missing[:6])}"
              + (" ..." if len(missing) > 6 else ""))
    if train_ok == 0:
        raise SystemExit(
            f"  ABORT: no training view in {scene.name} has a usable depth "
            f"scale, so depth regularisation would be a no-op and the run "
            f"would silently reproduce the baseline.")


if __name__ == "__main__":
    main()
