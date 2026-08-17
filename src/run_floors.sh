#!/usr/bin/env bash
# Train the real-only (0 synthetic) baselines at each subset size.
#
#   bash src/run_floors.sh            # k = 5 and 20
#   KS="5 20" bash src/run_floors.sh
#
# k=10 and the k=219 full-data ceiling already exist; this fills in the sizes
# the spec asks for (5/10/20) that were selected but never trained.
#
# Idempotent: run_experiment.py skips anything with an existing results.json.
set -u
cd "$HOME/fewshot_gs" || exit 1
VPY="$HOME/fewshot_gs/venv/bin/python"
KS="${KS:-5 20}"
SEEDS="${SEEDS:-0 1 2}"
ITERS="${ITERS:-7000}"

echo "############## BUILD SCENES ##############"
for k in $KS; do
  for s in $SEEDS; do
    M="subsets/truck_k${k}_seed${s}_fps.json"
    [ -f "$M" ] || { echo "MISSING MANIFEST $M"; exit 1; }
    "$VPY" src/build_scene.py --manifest "$M" --force
  done
done

echo
echo "############## TRAIN ##############"
for k in $KS; do
  for s in $SEEDS; do
    SC="scenes/truck_k${k}_seed${s}_fps_fake0"
    [ -d "$SC" ] || { echo "missing $SC"; continue; }
    echo "########## $(basename "$SC") ##########"
    "$VPY" -u src/run_experiment.py --scene "$SC" --iterations "$ITERS" 2>&1 \
      | grep -viE 'warn|deprecat|%\|' | grep -vE '^\s*$'
    echo
  done
done

echo "############## SUMMARY ##############"
"$VPY" src/collect_results.py
echo "FLOORS COMPLETE"
