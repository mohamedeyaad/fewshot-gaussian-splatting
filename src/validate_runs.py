"""Independent checks on runs that have no published number to compare against.

The lego CEILING could be validated because 3DGS reports ~33 dB for it. No such
anchor exists for a 5-view run, or for any condition in this study - so the
numbers have to be checked by internal consistency instead.

Each check below targets a specific failure that actually occurred in this
project, or that would silently invalidate a comparison:

  1. GROUND TRUTH IDENTITY. Every run of a scene must be scored against
     byte-identical held-out images. This is the strongest check available: if
     two conditions disagree about what the answer sheet is, their PSNRs are
     not comparable and no amount of correct training will fix it. The white/
     black background bug that cost 27 dB would have been caught instantly by
     this - the ceiling's GT differed from nothing, because it was the only
     run, but across conditions it shows up immediately.

  2. TRAIN/TEST DISJOINTNESS. A training image appearing in the held-out set
     turns the metric into a memorisation score. Cheap to check, fatal to miss.

  3. SUBSET SIZE. The scene must actually train on k images. A manifest saying
     k=5 while the loader reads 100 would produce a "few-shot" number that is
     really a full-data one.

  4. MONOTONICITY. More real views must not score worse. This cannot prove the
     numbers are right, but a violation proves something is wrong - and it is
     the only check here that tests the training rather than the plumbing.

  python src/validate_runs.py              # all scenes
  python src/validate_runs.py lego
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.path.expanduser("~/fewshot_gs"))
ITER = 7000


def digest(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()[:12]


def gt_fingerprint(run: Path):
    """One hash covering every held-out image of a run, in order."""
    d = run / "test" / f"ours_{ITER}" / "gt"
    if not d.is_dir():
        return None, 0
    files = sorted(d.glob("*.png"))
    if not files:
        return None, 0
    h = hashlib.sha256()
    for f in files:
        h.update(f.read_bytes())
    return h.hexdigest()[:12], len(files)


def main():
    want = sys.argv[1] if len(sys.argv) > 1 else None
    runs = sorted(p.parent for p in (ROOT / "runs").glob("*/results.json"))

    by_scene = defaultdict(list)
    for r in runs:
        scene = r.name.split("_k")[0]
        if want and scene != want:
            continue
        by_scene[scene].append(r)

    overall_ok = True
    for scene, rs in sorted(by_scene.items()):
        print(f"\n{'='*66}\n{scene}  ({len(rs)} runs)\n{'='*66}")
        ok = True

        # ---- 1. ground truth identity ---------------------------------
        groups = defaultdict(list)
        for r in rs:
            fp, n = gt_fingerprint(r)
            groups[(fp, n)].append(r.name)
        if len(groups) == 1:
            (fp, n), names = next(iter(groups.items()))
            print(f"  [1] ground truth  OK    all {len(names)} runs share "
                  f"{n} identical held-out images (sha {fp})")
        else:
            ok = False
            print(f"  [1] ground truth  FAIL  {len(groups)} DIFFERENT held-out sets:")
            for (fp, n), names in sorted(groups.items(), key=lambda x: -len(x[1])):
                print(f"        sha {fp}  n={n:<4} {len(names)} runs, e.g. {names[0]}")
                if len(names) <= 3:
                    for nm in names:
                        print(f"           {nm}")

        # ---- 2/3. split sanity ----------------------------------------
        bad_overlap, bad_count = [], []
        for r in rs:
            sp = ROOT / "scenes" / r.name
            f = sp / "split.json"
            if not f.exists():
                continue
            s = json.loads(f.read_text())
            tr, te = set(s["train"]), set(s["test"])
            if tr & te:
                bad_overlap.append((r.name, sorted(tr & te)[:3]))
            k = (s.get("provenance") or {}).get("k")
            n_syn = (s.get("provenance") or {}).get("n_synthetic", 0)
            if k is not None and len(s["train"]) != k + n_syn:
                bad_count.append((r.name, k, n_syn, len(s["train"])))
        if bad_overlap:
            ok = False
            print(f"  [2] train/test    FAIL  {len(bad_overlap)} runs leak test images")
            for nm, ex in bad_overlap[:3]:
                print(f"        {nm}: {ex}")
        else:
            print(f"  [2] train/test    OK    disjoint in every run")
        if bad_count:
            ok = False
            print(f"  [3] subset size   FAIL  {len(bad_count)} runs train on the wrong count")
            for nm, k, ns, got in bad_count[:3]:
                print(f"        {nm}: expected {k}+{ns}, got {got}")
        else:
            print(f"  [3] subset size   OK    train count matches k + n_synthetic")

        # ---- 4. monotonicity of the baselines -------------------------
        floors = defaultdict(list)
        for r in rs:
            rec = json.loads((r / "results.json").read_text())
            p = rec["provenance"]
            if p.get("n_synthetic", 0) or p.get("depth_reg") or r.name.endswith("_depth"):
                continue
            floors[p.get("k")].append(rec["metrics"]["psnr"]["mean"])
        ks = sorted(floors)
        if len(ks) >= 2:
            means = [(k, sum(v) / len(v)) for k, v in sorted(floors.items())]
            seq = "  ".join(f"k={k}:{m:.2f}" for k, m in means)
            bad = [(a, b) for (ka, a), (kb, b) in zip(means, means[1:]) if b < a]
            if bad:
                ok = False
                print(f"  [4] monotonic     FAIL  {seq}")
            else:
                print(f"  [4] monotonic     OK    {seq}")
        else:
            print(f"  [4] monotonic     --      only {len(ks)} subset size(s) present")

        print(f"\n  => {scene}: {'ALL CHECKS PASS' if ok else 'PROBLEMS ABOVE'}")
        overall_ok &= ok

    print()
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
