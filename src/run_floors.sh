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
SCENE="${SCENE:-truck}"
SOURCE="${SOURCE:-data/tandt/truck}"
KS="${KS:-5 20}"
SEEDS="${SEEDS:-0 1 2}"
ITERS="${ITERS:-7000}"
# Resolution divisor passed to run_experiment.py. Must be identical across
# every condition of a scene: mixing -r 2 and -r 4 within one scene makes the
# scaling curve meaningless. drjohnson needs -r 4 because at -r 2 a 230-view
# run exceeds 4 GB and thrashes (5 s/iter instead of 18 it/s).
RES="${RES:-2}"
# Subset selection method. `fps` spreads the K views over the camera
# trajectory; `random` draws them uniformly, which covers the scene noticeably
# worse (truck k=5: max_nn_dist 5.15 against fps's 3.54) and is the more
# realistic "someone took five casual photos" case. Manifests for both are
# written by select_subsets.py --methods.
METHOD="${METHOD:-fps}"

echo "############## BUILD SCENES ##############"
for k in $KS; do
  for s in $SEEDS; do
    M="subsets/${SCENE}_k${k}_seed${s}_${METHOD}.json"
    [ -f "$M" ] || { echo "MISSING MANIFEST $M"; exit 1; }
    "$VPY" src/build_scene.py --manifest "$M" --source "$SOURCE" --force
  done
done

echo
echo "############## TRAIN ##############"
for k in $KS; do
  for s in $SEEDS; do
    SC="scenes/${SCENE}_k${k}_seed${s}_${METHOD}_fake0"
    [ -d "$SC" ] || { echo "missing $SC"; continue; }
    echo "########## $(basename "$SC") ##########"
    "$VPY" -u src/run_experiment.py --scene "$SC" --iterations "$ITERS" --resolution "$RES" 2>&1 \
      | grep -viE 'warn|deprecat|%\|' | grep -vE '^\s*$'
    echo
  done
done

echo "############## SUMMARY ##############"
"$VPY" src/collect_results.py
echo "FLOORS COMPLETE"
