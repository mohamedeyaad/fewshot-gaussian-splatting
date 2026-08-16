#!/usr/bin/env bash
# Full inpainting ablation: generate 20 synthetic views per seed, build the
# nested 2/5/10/20 scenes, train and evaluate each.
# Safe to re-run: generation overwrites deterministically, scenes rebuild,
# and run_experiment.py skips anything that already has results.json.
set -u
cd "$HOME/fewshot_gs" || exit 1
VPY="$HOME/fewshot_gs/venv/bin/python"
SEEDS="${SEEDS:-0 1 2}"
FAKES="${FAKES:-2 5 10 20}"
NGEN="${NGEN:-20}"
ITERS="${ITERS:-7000}"

echo "############## STAGE 1: GENERATE ##############"
for s in $SEEDS; do
  M="subsets/truck_k10_seed${s}_fps.json"
  echo "--- seed $s ---"
  "$VPY" -u src/gen_inpaint.py --manifest "$M" --n "$NGEN" 2>&1 \
    | grep -viE 'warn|deprecat|it/s\]$' | grep -vE '^\s*$'
done

echo
echo "############## STAGE 2: BUILD SCENES ##############"
for s in $SEEDS; do
  "$VPY" src/build_scene.py \
    --manifest "subsets/truck_k10_seed${s}_fps.json" \
    --synthetic "synthetic/truck_k10_seed${s}_fps_inpaint" \
    --n-fake $FAKES --force
done

echo
echo "############## STAGE 3: TRAIN ##############"
for s in $SEEDS; do
  for f in $FAKES; do
    SC="scenes/truck_k10_seed${s}_fps_inpaint_fake${f}"
    [ -d "$SC" ] || { echo "missing $SC"; continue; }
    echo "########## $(basename "$SC") ##########"
    "$VPY" -u src/run_experiment.py --scene "$SC" --iterations "$ITERS" 2>&1 \
      | grep -viE 'warn|deprecat|%\|' | grep -vE '^\s*$'
    echo
  done
done

echo "############## COLLECTING ##############"
"$VPY" src/collect_results.py
echo "INPAINT CURVE COMPLETE"
