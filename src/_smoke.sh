#!/usr/bin/env bash
set -u
cd "$HOME/fewshot_gs" || exit 1
VPY="$HOME/fewshot_gs/venv/bin/python"

echo "--- scenes required by the sweep ---"
for s in 0 1 2; do
  for d in "scenes/truck_k5_seed${s}_fps_fake0" \
           "scenes/truck_k5_seed${s}_fps_outpaint_fake10" \
           "synthetic/truck_k5_seed${s}_fps_outpaint/images"; do
    [ -e "$d" ] && echo "  OK      $d" || echo "  MISSING $d"
  done
done

echo
echo "--- generating 2 depth maps as a format check ---"
FIRST=$(ls -1 data/tandt/truck/images | head -2 | tr '\n' ' ')
"$VPY" src/gen_depths.py --source data/tandt/truck \
    --out /tmp/depthtest --only $FIRST --force || exit 1

echo
echo "--- verifying the written format ---"
"$VPY" - <<'PY'
import glob, numpy as np, cv2
from PIL import Image
fs = sorted(glob.glob("/tmp/depthtest/*.png"))
if not fs:
    raise SystemExit("NO FILES WRITTEN")
for f in fs:
    im = Image.open(f)
    a = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    print(f"  {f.split('/')[-1]}")
    print(f"    PIL mode {im.mode}  size {im.size}")
    print(f"    cv2 dtype {a.dtype}  shape {a.shape}")
    print(f"    raw range {a.min()} .. {a.max()}")
    n = a.astype(np.float32) / 2**16
    print(f"    as consumed (/2**16): {n.min():.4f} .. {n.max():.4f}")
    ok = (a.dtype == np.uint16 and a.ndim == 2 and a.max() > 60000)
    print(f"    -> {'OK' if ok else 'WRONG FORMAT'}")
PY
echo
echo "--- source image size, for comparison ---"
"$VPY" - <<'PY'
import glob
from PIL import Image
f = sorted(glob.glob("/home/mooeyad/fewshot_gs/data/tandt/truck/images/*"))[0]
print(f"  {f.split('/')[-1]}: {Image.open(f).size}")
PY
rm -rf /tmp/depthtest
