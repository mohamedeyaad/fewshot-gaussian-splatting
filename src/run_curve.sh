#!/usr/bin/env bash
# Generic driver: generate synthetic views for one strategy across all seeds,
# build the nested ratio scenes, train and evaluate each.
#
#   STRATEGY=outpaint bash src/run_curve.sh
#
# Idempotent: run_experiment.py skips anything that already has results.json.
set -u
cd "$HOME/fewshot_gs" || exit 1
VPY="$HOME/fewshot_gs/venv/bin/python"
STRATEGY="${STRATEGY:?set STRATEGY=inpaint|outpaint|guided}"
SEEDS="${SEEDS:-0 1 2}"
FAKES="${FAKES:-2 5 10 20}"
NGEN="${NGEN:-20}"
ITERS="${ITERS:-7000}"
GEN="src/gen_${STRATEGY}.py"

[ -f "$GEN" ] || { echo "no generator $GEN"; exit 1; }

echo "############## GENERATE ($STRATEGY) ##############"
for s in $SEEDS; do
  D="synthetic/truck_k10_seed${s}_fps_${STRATEGY}"
  if [ -f "$D/poses.json" ] && [ "$(ls "$D/images" 2>/dev/null | wc -l)" -ge "$NGEN" ]; then
    echo "--- seed $s: already have >= $NGEN images, skipping ---"
    continue
  fi
  echo "--- seed $s ---"
  "$VPY" -u "$GEN" --manifest "subsets/truck_k10_seed${s}_fps.json" --n "$NGEN" 2>&1 \
    | grep -viE 'warn|deprecat|it/s\]$' | grep -vE '^\s*$'
done

echo
echo "############## BUILD SCENES ##############"
for s in $SEEDS; do
  "$VPY" src/build_scene.py \
    --manifest "subsets/truck_k10_seed${s}_fps.json" \
    --synthetic "synthetic/truck_k10_seed${s}_fps_${STRATEGY}" \
    --n-fake $FAKES --force
done

echo
echo "############## TRAIN ##############"
for s in $SEEDS; do
  for f in $FAKES; do
    SC="scenes/truck_k10_seed${s}_fps_${STRATEGY}_fake${f}"
    [ -d "$SC" ] || { echo "missing $SC"; continue; }
    echo "########## $(basename "$SC") ##########"
    "$VPY" -u src/run_experiment.py --scene "$SC" --iterations "$ITERS" 2>&1 \
      | grep -viE 'warn|deprecat|%\|' | grep -vE '^\s*$'
    echo
  done
done

echo "############## COLLECT ##############"
"$VPY" src/collect_results.py
"$VPY" src/plot_curves.py
echo "${STRATEGY} CURVE COMPLETE"
