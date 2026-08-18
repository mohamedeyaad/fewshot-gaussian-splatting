"""Build a scene directory for one experimental condition.

Two modes:

REAL-ONLY (n_fake = 0)
    images/ and sparse/ are symlinks to the original dataset; the only real
    content is split.json. Costs a few KB.

AUGMENTED (n_fake > 0)
    We must add camera entries for the synthetic views, so we emit COLMAP
    *text* format (images.txt / cameras.txt). readColmapSceneInfo tries the
    binary readers first and falls back to text, so writing only .txt is
    enough - and text is far easier to author correctly than .bin.
    images/ becomes a directory of per-file symlinks (real) plus symlinks to
    the generated files (synthetic).

Synthetic views are taken as a NESTED prefix of the generated set: the first
N of them. So fake2 is a strict subset of fake5, of fake10, of fake20 - which
means moving along the ratio axis only ever ADDS images and never swaps them.

  python src/build_scene.py --manifest subsets/truck_k10_seed0_fps.json
  python src/build_scene.py --manifest subsets/truck_k10_seed0_fps.json \
      --synthetic synthetic/truck_k10_seed0_fps_inpaint --n-fake 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
ROOT = HOME / "fewshot_gs"
REPO = ROOT / "gaussian-splatting"
sys.path.insert(0, str(REPO))
from scene.colmap_loader import (read_extrinsics_binary,  # noqa: E402
                                 read_intrinsics_binary)

DEFAULT_SRC = ROOT / "data/tandt/truck"


def write_cameras_txt(path: Path, cameras: dict):
    lines = ["# Camera list with one line of data per camera:",
             "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
             f"# Number of cameras: {len(cameras)}"]
    for cid, c in sorted(cameras.items()):
        params = " ".join(repr(float(p)) for p in c.params)
        lines.append(f"{cid} {c.model} {c.width} {c.height} {params}")
    path.write_text("\n".join(lines) + "\n")


def write_images_txt(path: Path, entries: list):
    """entries: list of dicts with image_id, qvec, tvec, camera_id, name.

    Each image needs a second line of POINTS2D. The 3DGS loader parses it but
    never uses it, so one dummy point with point3D_id = -1 is sufficient and
    keeps the parser happy (a truly blank line is riskier).
    """
    lines = ["# Image list with two lines of data per image:",
             "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
             "#   POINTS2D[] as (X, Y, POINT3D_ID)",
             f"# Number of images: {len(entries)}"]
    for e in entries:
        q = " ".join(repr(float(x)) for x in e["qvec"])
        t = " ".join(repr(float(x)) for x in e["tvec"])
        lines.append(f'{e["image_id"]} {q} {t} {e["camera_id"]} {e["name"]}')
        lines.append("0.0 0.0 -1")
    path.write_text("\n".join(lines) + "\n")


def link(target: Path, link_path: Path):
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(target.resolve())


def build(manifest_path: Path, source: Path, out_root: Path,
          synthetic_dir: Path | None = None, n_fake: int = 0,
          force: bool = False) -> Path:
    manifest = json.loads(manifest_path.read_text())
    # Prefer the scene-qualified split; fall back to the unqualified name for
    # the truck manifests, which were written before scenes were separated.
    split_path = manifest_path.parent / f"{manifest['scene']}_test_split.json"
    if not split_path.exists():
        split_path = manifest_path.parent / "test_split.json"
    split_all = json.loads(split_path.read_text())
    if split_all["scene"] != manifest["scene"]:
        raise ValueError(f"{split_path.name} is for scene "
                         f"'{split_all['scene']}' but the manifest is for "
                         f"'{manifest['scene']}' - refusing to mix test splits")

    test_images = list(split_all["test_images"])
    real_train = list(manifest["images"])

    strategy = "none"
    synth_records = []
    extra_cameras = {}
    if n_fake > 0:
        if synthetic_dir is None:
            raise ValueError("--n-fake > 0 requires --synthetic")
        poses = json.loads((synthetic_dir / "poses.json").read_text())
        strategy = poses["strategy"]
        avail = poses["images"]
        if n_fake > len(avail):
            raise ValueError(f"asked for {n_fake} synthetic views but only "
                             f"{len(avail)} generated in {synthetic_dir}")
        synth_records = avail[:n_fake]          # nested prefix
        extra_cameras = poses.get("extra_cameras", {})

    tag = (f"{manifest['scene']}_k{manifest['k']}_seed{manifest['seed']}"
           f"_{manifest['method']}"
           + (f"_{strategy}" if n_fake > 0 else "")
           + f"_fake{n_fake}")
    scene = out_root / tag

    if scene.exists() and not force:
        print(f"  exists, skipping: {tag}  (use --force)")
        return scene
    if scene.exists():
        import shutil
        shutil.rmtree(scene)
    scene.mkdir(parents=True, exist_ok=True)

    if n_fake == 0:
        # Nothing to add - symlink the originals wholesale.
        for sub in ("images", "sparse"):
            link(source / sub, scene / sub)
    else:
        intr = read_intrinsics_binary(str(source / "sparse/0/cameras.bin"))
        extr = read_extrinsics_binary(str(source / "sparse/0/images.bin"))
        by_name = {im.name: im for im in extr.values()}

        img_dir = scene / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        sp_dir = scene / "sparse" / "0"
        sp_dir.mkdir(parents=True, exist_ok=True)

        entries = []
        # Real views (train subset + all test views) keep their original ids.
        for name in sorted(set(real_train) | set(test_images)):
            im = by_name[name]
            entries.append({"image_id": int(im.id), "qvec": im.qvec,
                            "tvec": im.tvec, "camera_id": int(im.camera_id),
                            "name": name})
            link(source / "images" / name, img_dir / name)

        # Synthetic views get fresh ids above the originals.
        next_id = max(int(im.id) for im in extr.values()) + 1
        for j, rec in enumerate(synth_records):
            entries.append({"image_id": next_id + j, "qvec": rec["qvec"],
                            "tvec": rec["tvec"],
                            "camera_id": int(rec["camera_id"]),
                            "name": rec["name"]})
            link(synthetic_dir / "images" / rec["name"], img_dir / rec["name"])

        # Strategies that change the frame (outpainting) introduce additional
        # cameras; merge them in alongside the originals.
        from types import SimpleNamespace
        cams = dict(intr)
        for cid, spec in extra_cameras.items():
            cams[int(cid)] = SimpleNamespace(
                id=int(cid), model=spec["model"],
                width=int(spec["width"]), height=int(spec["height"]),
                params=list(spec["params"]))
        used = {e["camera_id"] for e in entries}
        missing = used - set(cams)
        if missing:
            raise ValueError(f"images reference camera ids {sorted(missing)} "
                             f"that are not defined")
        write_cameras_txt(sp_dir / "cameras.txt", cams)
        write_images_txt(sp_dir / "images.txt", entries)

        # Point cloud: reuse the already-converted ply if present, else the bin.
        src_ply = source / "sparse/0/points3D.ply"
        src_bin = source / "sparse/0/points3D.bin"
        if src_ply.exists():
            link(src_ply, sp_dir / "points3D.ply")
        elif src_bin.exists():
            link(src_bin, sp_dir / "points3D.bin")
        else:
            raise FileNotFoundError("no points3D.ply or .bin in source")

    train_images = real_train + [r["name"] for r in synth_records]
    (scene / "split.json").write_text(json.dumps({
        "train": train_images,
        "test": test_images,
        "provenance": {
            "manifest": str(manifest_path),
            "k": manifest["k"], "seed": manifest["seed"],
            "method": manifest["method"],
            "strategy": strategy,
            "n_real": len(real_train), "n_synthetic": len(synth_records),
            "ratio_pct": round(100.0 * len(synth_records) / max(len(real_train), 1)),
            "synthetic_dir": str(synthetic_dir) if synthetic_dir else None,
            "source": str(source),
        },
    }, indent=2))

    print(f"  built {tag}: {len(real_train)} real + {len(synth_records)} synth "
          f"= {len(train_images)} train / {len(test_images)} test")
    return scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--source", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default=str(ROOT / "scenes"))
    ap.add_argument("--synthetic", default=None)
    ap.add_argument("--n-fake", type=int, nargs="+", default=[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    syn = Path(args.synthetic) if args.synthetic else None
    for m in args.manifest:
        for nf in args.n_fake:
            build(Path(m), Path(args.source), out_root, syn, nf, args.force)


if __name__ == "__main__":
    main()
