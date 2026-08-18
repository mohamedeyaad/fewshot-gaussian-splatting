"""Emit a manifest using the ENTIRE training pool - the upper-bound baseline.

Same format as the few-shot manifests so build_scene.py handles it unchanged.
"""
import argparse, json, os
from pathlib import Path

ap = argparse.ArgumentParser()
# One test_split.json per scene, so a second scene does not overwrite the
# first. Defaults to the unsuffixed file for backwards compatibility with the
# truck runs, which were produced before this was parameterised.
ap.add_argument("--split", default=None,
                help="path to test_split.json (default: subsets/test_split.json)")
args = ap.parse_args()

SUB = Path(os.path.expanduser("~/fewshot_gs/subsets"))
split = json.loads(Path(args.split).read_text() if args.split
                   else (SUB / "test_split.json").read_text())
pool = split["train_pool"]

rec = {
    "scene": split["scene"],
    "method": "full",
    "k": len(pool),
    "seed": 0,
    "llffhold": split["llffhold"],
    "images": sorted(pool),
    "coverage": {"note": "entire training pool - upper bound"},
}
out = SUB / f"{split['scene']}_k{len(pool)}_seed0_full.json"
out.write_text(json.dumps(rec, indent=2))
print(f"wrote {out} with {len(pool)} training images")
