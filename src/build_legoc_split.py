"""Freeze the train/test split for the COLMAP-reconstructed lego scene.

The scene is called `legoc` to keep it distinct from the Blender-format `lego`
runs, which stay in the report as the published-anchored scaling curve.

The split is NOT the llffhold rule here. build_colmap_lego.py exported the 100
NeRF-Synthetic training frames as train_*.png and every 6th test frame as
test_*.png, so the intended split is already encoded in the filenames. Holding
out every 8th image instead would mix the two orbits and destroy the one
external check this scene has: its full-data ceiling should reproduce the
33.77 dB measured on exactly these 34 held-out frames in the Blender format,
which in turn matches the ~33 dB published for lego.

Images COLMAP failed to register are dropped, and reported. A test frame that
did not register cannot be scored; a training frame that did not register just
shrinks the pool. Both change what the ceiling means, so neither is silent.

  python src/build_legoc_split.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.path.expanduser("~/fewshot_gs"))
DST = ROOT / "data" / "legoc"
SUB = ROOT / "subsets"
SCENE = "legoc"

sys.path.insert(0, str(ROOT / "gaussian-splatting"))
from scene.colmap_loader import read_extrinsics_binary  # noqa: E402


def main():
    export = json.loads((DST / "export.json").read_text())
    extr = read_extrinsics_binary(str(DST / "sparse" / "0" / "images.bin"))
    registered = {im.name for im in extr.values()}

    train = [n for n in export["train"] if n in registered]
    test = [n for n in export["test"] if n in registered]
    lost_tr = [n for n in export["train"] if n not in registered]
    lost_te = [n for n in export["test"] if n not in registered]

    print("registered %d / %d images" % (len(registered), len(export["train"]) + len(export["test"])))
    print("  train pool : %d / %d" % (len(train), len(export["train"])))
    print("  held out   : %d / %d" % (len(test), len(export["test"])))
    if lost_tr:
        print("  DROPPED from train: %s%s"
              % (", ".join(lost_tr[:5]), " ..." if len(lost_tr) > 5 else ""))
    if lost_te:
        print("  DROPPED from test : %s%s" % (", ".join(lost_te[:5]),
                                              " ..." if len(lost_te) > 5 else ""))
        print("  NOTE: the held-out set no longer matches the Blender run's 34 "
              "frames, so the 33.77 dB comparison is only approximate.")

    if len(test) < 20 or len(train) < 50:
        sys.exit("too few registered images - reconstruction is not usable")

    out = SUB / ("%s_test_split.json" % SCENE)
    out.write_text(json.dumps({
        "scene": SCENE,
        "llffhold": None,
        "note": "explicit split from NeRF-Synthetic orbits, not the llffhold rule",
        "test_images": sorted(test),
        "train_pool": sorted(train),
    }, indent=2))
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
