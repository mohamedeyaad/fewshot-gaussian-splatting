#!/usr/bin/env bash
# Does the crossover survive convergence?
#
#   bash src/run_30k.sh
#
# Every one of the 236 runs in this study trains for 7,000 iterations. Upstream
# 3DGS schedules densification to 15,000 and trains to 30,000, so 7,000 stops
# while the model is still gaining primitives. That is applied identically to
# every condition, so the comparisons hold - but it leaves one question
# unanswered: is the sign change a property of few-shot reconstruction, or of
# UNDER-TRAINED few-shot reconstruction?
#
# There is a real reason it might not survive. A synthetic view's contribution
# is coverage plus inconsistency. Extra iterations give the optimiser more
# opportunity to fit the inconsistent views - so the harm could grow - but also
# more opportunity to converge the real ones, so the baseline rises too. Which
# moves faster is not predictable from the 7k data.
#
# 12 runs: k=5 and k=20, baseline and outpainting at the 200% ratio, three
# seeds. Roughly 5 hours. Nothing is generated - the scenes and their synthetic
# views already exist.
#
# WHY A SEPARATE DIRECTORY. runs/truck_k5_seed0_fps_fake0 already exists at
# 7,000 iterations. A 30,000-iteration run of the same condition has the same
# scene, k, seed, selection method and strategy, so it would collide in every
# baseline key in the analysis - the identical failure that fps-vs-random just
# produced, where one silently overwrote the other and flipped a reported sign.
# runs_noise/ is isolated for the same reason. The comparison below reads both
# directories explicitly.
set -u
cd "$HOME/fewshot_gs" || exit 1
VPY="$HOME/fewshot_gs/venv/bin/python"

# Parameterised so the same driver can fill in a third point on the
# training-length axis:
#
#   ITERS=15000 OUTDIR=runs_15k bash src/run_30k.sh
#
# 15,000 is where densification stops, and it is the point that decides whether
# 7,000 is genuinely near this regime's optimum or merely the setting that
# happened to be used. Two points cannot distinguish those.
ITERS="${ITERS:-30000}"
OUTDIR="${OUTDIR:-runs_30k}"
mkdir -p "$OUTDIR"

t0=$(date +%s)

for K in 5 20; do
  case "$K" in
    5)  NF=10 ;;    # 200%
    20) NF=40 ;;    # 200%
  esac
  for s in 0 1 2; do
    for SC in "truck_k${K}_seed${s}_fps_fake0" \
              "truck_k${K}_seed${s}_fps_outpaint_fake${NF}"; do
      [ -d "scenes/$SC" ] || { echo "MISSING scenes/$SC"; continue; }
      if [ -f "$OUTDIR/$SC/results.json" ]; then
        echo "--- $SC already done, skipping ---"
        continue
      fi
      echo "########## $SC  @ ${ITERS} ##########"
      "$VPY" -u src/run_experiment.py --scene "scenes/$SC" \
          --out "$OUTDIR/$SC" --iterations "$ITERS" --resolution 2 2>&1 \
        | grep -viE 'warn|deprecat|%\|' | grep -vE '^\s*$'
      echo
    done
  done
  rm -rf "$OUTDIR"/*/point_cloud "$OUTDIR"/*/input.ply
done

echo
echo "############## TRAINING LENGTH vs THE EFFECT ##############"
"$VPY" - <<'PY'
import json, os
from statistics import mean, stdev
os.chdir(os.path.expanduser("~/fewshot_gs"))

NF = {5: 10, 20: 40}
# Every training length present, so a third point slots in without edits.
DIRS = [("runs", "7,000"), ("runs_15k", "15,000"), ("runs_30k", "30,000")]

def psnr(d, tag):
    p = f"{d}/{tag}/results.json"
    return json.load(open(p))["metrics"]["psnr"]["mean"] if os.path.exists(p) else None

print(f"{'':>6} {'iters':>9} {'baseline':>9} {'+200%':>9} {'delta':>17}")
print("-" * 56)
for k in (5, 20):
    for d, lab in DIRS:
        ds, bs = [], []
        for s in (0, 1, 2):
            b = psnr(d, f"truck_k{k}_seed{s}_fps_fake0")
            a = psnr(d, f"truck_k{k}_seed{s}_fps_outpaint_fake{NF[k]}")
            if b is not None and a is not None:
                ds.append(a - b); bs.append(b)
        if not ds:
            continue
        sd = stdev(ds) if len(ds) > 1 else 0.0
        star = "*" if abs(mean(ds)) > sd > 0 else " "
        print(f"k={k:<3} {lab:>9} {mean(bs):9.2f} {mean(bs)+mean(ds):9.2f} "
              f"{mean(ds):+9.3f} ± {sd:5.3f}{star}")
    print()
print("Two things to read off the BASELINE column. If it peaks at or near")
print("7,000, early stopping is this regime's optimum and the reported gain")
print("sits at the right operating point. If it is still rising at 15,000,")
print("then 7,000 was simply the setting used, and the report has to say so.")
PY
t1=$(date +%s)
printf '\n30K SWEEP COMPLETE in %dh %dm\n' $(( (t1-t0)/3600 )) $(( ((t1-t0)%3600)/60 ))
