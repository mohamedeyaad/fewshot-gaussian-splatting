#!/usr/bin/env bash
# The scaling curve on an isolated object: k = 5, 10, 20, plus the 100-view
# ceiling (already trained, 33.77 dB against ~33 published).
#
#   bash src/run_lego_floors.sh
#
# A third capture regime after an outdoor object (truck) and an indoor room
# (drjohnson). The interesting comparison is not the absolute PSNR - lego is a
# far easier scene - but the SHAPE of the curve: how much of the full-data
# quality survives when you cut to five views.
#
# Resolution 1 (native 800x800) and --white-background throughout, matching the
# ceiling so the curve is internally consistent. -w matters only now that
# patch 03 makes it reach the pixels; before that it silently mismatched the
# ground truth and cost 27 dB.
set -u
cd "$HOME/fewshot_gs" || exit 1
VPY="$HOME/fewshot_gs/venv/bin/python"
SCENE=lego
SEEDS="${SEEDS:-0 1 2}"
KS="${KS:-5 10 20}"

t0=$(date +%s)
for k in $KS; do
  for s in $SEEDS; do
    SC="scenes/${SCENE}_k${k}_seed${s}_fps_fake0"
    [ -d "$SC" ] || { echo "missing $SC"; continue; }
    echo "########## $(basename "$SC") ##########"
    "$VPY" -u src/run_experiment.py --scene "$SC" \
        --out "runs/$(basename "$SC")" \
        --iterations 7000 --resolution 1 --white-background 2>&1 \
      | grep -viE 'warn|deprecat|%\|' | grep -vE '^\s*$'
    echo
  done
done

echo "############## SCALING CURVE ##############"
"$VPY" - <<'PY'
import json, os, glob
from statistics import mean, stdev
os.chdir(os.path.expanduser("~/fewshot_gs"))
def m(tag, key="psnr"):
    p = f"runs/{tag}/results.json"
    return json.load(open(p))["metrics"][key]["mean"] if os.path.exists(p) else None

print(f"{'k':>5} {'PSNR':>16} {'SSIM':>8} {'LPIPS':>8} {'gauss':>10}")
print("-" * 52)
rows = []
for k in (5, 10, 20):
    vals = [m(f"lego_k{k}_seed{s}_fps_fake0") for s in (0, 1, 2)]
    vals = [v for v in vals if v is not None]
    if not vals:
        continue
    sd = stdev(vals) if len(vals) > 1 else 0.0
    ss = mean([m(f"lego_k{k}_seed{s}_fps_fake0", "ssim") for s in (0,1,2)
               if m(f"lego_k{k}_seed{s}_fps_fake0", "ssim") is not None])
    lp = mean([m(f"lego_k{k}_seed{s}_fps_fake0", "lpips") for s in (0,1,2)
               if m(f"lego_k{k}_seed{s}_fps_fake0", "lpips") is not None])
    g = json.load(open(f"runs/lego_k{k}_seed0_fps_fake0/results.json"))["cost"]["n_gaussians"]
    rows.append((k, mean(vals)))
    print(f"{k:>5} {mean(vals):9.2f} ± {sd:4.2f} {ss:8.4f} {lp:8.4f} {g:10,}")
c = m("lego_k100_seed0_full_fake0")
if c:
    cs = m("lego_k100_seed0_full_fake0", "ssim")
    cl = m("lego_k100_seed0_full_fake0", "lpips")
    g = json.load(open("runs/lego_k100_seed0_full_fake0/results.json"))["cost"]["n_gaussians"]
    print(f"{100:>5} {c:9.2f}        {cs:8.4f} {cl:8.4f} {g:10,}")
    if rows:
        print(f"\n  gap at k=5: {c - rows[0][1]:.2f} dB")
        print(f"  fraction of ceiling reached at k=5: {rows[0][1]/c:.1%}")
PY
t1=$(date +%s)
printf '\nLEGO FLOORS COMPLETE in %dh %dm\n' $(( (t1-t0)/3600 )) $(( ((t1-t0)%3600)/60 ))
