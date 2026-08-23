#!/usr/bin/env bash
# Does the DEPTH PRIOR survive convergence, when outpainting does not?
#
#   bash src/run_30k_depth.sh
#
# The 30,000-iteration check killed the study's only positive result:
# outpainting at k=5 falls from +0.285 to -0.078, indistinguishable from zero.
# It also showed why - both unaugmented baselines get WORSE with longer
# training (15.20 -> 15.04, 19.74 -> 19.49), so few-shot 3DGS at 30k is
# overfitting, not converging.
#
# That makes this the sharpest test the mechanism has faced. The account says a
# synthetic view supplies coverage AND inconsistency, while the depth prior
# supplies constraint and no inconsistency at all. Under overfitting those two
# should behave in opposite directions:
#
#   - outpainting gives the optimiser MORE contradictory data to memorise, so
#     its benefit should die first - it does;
#   - depth regularisation is a REGULARISER, so it should survive, and might
#     help MORE at 30k than at 7k, because overfitting is exactly what it
#     opposes.
#
# If depth also collapses, then nothing in this study survives convergence and
# the whole result is a property of early stopping. If it survives, the
# distinction the report draws between constraint and invention is real and
# holds at both operating points - which is a considerably stronger claim than
# the one currently written.
#
# 12 runs completing the 2x2 at both ends: baseline and outpainting already
# exist at 30k, so this adds +depth and +both at k=5 and k=20. Roughly 5 hours.
# Nothing is generated - the depth maps and scale parameters are already
# attached to these scenes.
set -u
cd "$HOME/fewshot_gs" || exit 1
VPY="$HOME/fewshot_gs/venv/bin/python"

ITERS=30000
OUTDIR=runs_30k
mkdir -p "$OUTDIR"

t0=$(date +%s)

for K in 5 20; do
  case "$K" in
    5)  NF=10 ;;
    20) NF=40 ;;
  esac
  for s in 0 1 2; do
    for SC in "truck_k${K}_seed${s}_fps_fake0" \
              "truck_k${K}_seed${s}_fps_outpaint_fake${NF}"; do
      TAG="${SC}_depth"
      [ -d "scenes/$SC" ] || { echo "MISSING scenes/$SC"; continue; }
      if [ ! -f "scenes/$SC/sparse/0/depth_params.json" ]; then
        echo "MISSING depth_params.json for $SC - run src/run_depth_reg.sh first"
        continue
      fi
      if [ -f "$OUTDIR/$TAG/results.json" ]; then
        echo "--- $TAG already done, skipping ---"
        continue
      fi
      echo "########## $TAG  @ ${ITERS} ##########"
      "$VPY" -u src/run_experiment.py --scene "scenes/$SC" \
          --out "$OUTDIR/$TAG" --iterations "$ITERS" --resolution 2 \
          --depths depths 2>&1 \
        | grep -viE 'warn|deprecat|%\|' | grep -vE '^\s*$'
      echo
    done
  done
  rm -rf "$OUTDIR"/*/point_cloud "$OUTDIR"/*/input.ply
done

echo
echo "############## THE 2x2, AT BOTH TRAINING LENGTHS ##############"
"$VPY" - <<'PY'
import json, os
from statistics import mean, stdev
os.chdir(os.path.expanduser("~/fewshot_gs"))

NF = {5: 10, 20: 40}

def psnr(d, tag):
    p = f"{d}/{tag}/results.json"
    return json.load(open(p))["metrics"]["psnr"]["mean"] if os.path.exists(p) else None

def delta(d, k, suffix):
    """Paired against the same seed's own plain baseline IN THE SAME DIRECTORY -
    a 30k run must never be compared to a 7k baseline."""
    ds = []
    for s in (0, 1, 2):
        b = psnr(d, f"truck_k{k}_seed{s}_fps_fake0")
        a = psnr(d, f"truck_k{k}_seed{s}_fps_{suffix}".format(s=s))
        if b is not None and a is not None:
            ds.append(a - b)
    if not ds:
        return None
    sd = stdev(ds) if len(ds) > 1 else 0.0
    return mean(ds), sd, len(ds)

def cell(v):
    if v is None:
        return "        --        "
    m, sd, n = v
    return f"{m:+8.3f} ± {sd:5.3f}{'*' if abs(m) > sd > 0 else ' '}"

for k in (5, 20):
    print(f"--- k={k} ---")
    print(f"{'':>16} {'7,000':>19} {'30,000':>19}")
    for label, suffix in (("+ depth", "fake0_depth"),
                          ("+ outpainting", f"outpaint_fake{NF[k]}"),
                          ("+ both", f"outpaint_fake{NF[k]}_depth")):
        a = delta("runs", k, suffix)
        b = delta("runs_30k", k, suffix)
        print(f"{label:>16} {cell(a)} {cell(b)}")
    b7 = [psnr("runs", f"truck_k{k}_seed{s}_fps_fake0") for s in (0, 1, 2)]
    b30 = [psnr("runs_30k", f"truck_k{k}_seed{s}_fps_fake0") for s in (0, 1, 2)]
    b7 = [x for x in b7 if x is not None]
    b30 = [x for x in b30 if x is not None]
    if b7 and b30:
        print(f"{'baseline PSNR':>16} {mean(b7):19.2f} {mean(b30):19.2f}")
    print()
print("The mechanism predicts the depth column survives at 30,000 while the")
print("outpainting column does not: one adds constraint, the other adds")
print("contradictions for a longer run to memorise.")
PY
t1=$(date +%s)
printf '\n30K DEPTH SWEEP COMPLETE in %dh %dm\n' $(( (t1-t0)/3600 )) $(( ((t1-t0)%3600)/60 ))
