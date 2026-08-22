#!/usr/bin/env bash
# Generic driver: generate synthetic views for one strategy across all seeds,
# build the nested ratio scenes, train and evaluate each.
#
#   STRATEGY=outpaint bash src/run_curve.sh
#   K=20 FAKES="5 10 20 40" NGEN=40 STRATEGY=outpaint bash src/run_curve.sh
#
# K is the number of REAL training views. FAKES are absolute synthetic counts,
# so the ratio each one represents depends on K:
#
#   K=5   FAKES="1 2 5 10"     ->  20 /  40 / 100 / 200 %
#   K=10  FAKES="2 5 10 20"    ->  20 /  50 / 100 / 200 %
#   K=20  FAKES="5 10 20 40"   ->  25 /  50 / 100 / 200 %
#
# The spec asks for 25/50/100/200%. Only K=20 divides cleanly; at K=5 and K=10
# the 25% point is fractional (1.25 and 2.5 images) and is rounded down.
#
# Idempotent: run_experiment.py skips anything that already has results.json.
set -u
cd "$HOME/fewshot_gs" || exit 1
VPY="$HOME/fewshot_gs/venv/bin/python"
STRATEGY="${STRATEGY:?set STRATEGY=inpaint|outpaint|guided}"
K="${K:-10}"
SEEDS="${SEEDS:-0 1 2}"
FAKES="${FAKES:-2 5 10 20}"
NGEN="${NGEN:-20}"
ITERS="${ITERS:-7000}"
# Resolution divisor passed to run_experiment.py. Must be identical across
# every condition of a scene: mixing -r 2 and -r 4 within one scene makes the
# scaling curve meaningless. drjohnson needs -r 4 because at -r 2 a 230-view
# run exceeds 4 GB and thrashes (5 s/iter instead of 18 it/s).
RES="${RES:-2}"
# Rasteriser background. legoc has a white background BAKED INTO its images -
# the COLMAP loader does no compositing - so the rasteriser has to match, or
# every one of the 70% background pixels is scored against the wrong colour.
# That mismatch is exactly what cost 27 dB on the Blender path.
WB="${WB:-0}"
# Subset selection method, matching run_floors.sh. Must agree with the floors
# the curve is paired against: a `random` augmented run compared to an `fps`
# baseline would attribute the whole coverage difference to augmentation.
METHOD="${METHOD:-fps}"
# Scene selection. SCENE must match the directory name under data/, because
# select_subsets.py derives manifest names from it.
SCENE="${SCENE:-truck}"
SOURCE="${SOURCE:-data/tandt/truck}"
# The prompt has to describe the actual scene; the truck default produces
# nonsense indoors.
PROMPT="${PROMPT:-}"
GEN="src/gen_${STRATEGY}.py"
TAG="${SCENE}_k${K}"

PROMPT_ARG=()
[ -n "$PROMPT" ] && PROMPT_ARG=(--prompt "$PROMPT")
WB_ARG=()
[ "$WB" = "1" ] && WB_ARG=(--white-background)

[ -f "$GEN" ] || { echo "no generator $GEN"; exit 1; }

echo "############## GENERATE ($STRATEGY, k=$K) ##############"
for s in $SEEDS; do
  D="synthetic/${TAG}_seed${s}_${METHOD}_${STRATEGY}"
  if [ -f "$D/poses.json" ] && [ "$(ls "$D/images" 2>/dev/null | wc -l)" -ge "$NGEN" ]; then
    echo "--- seed $s: already have >= $NGEN images, skipping ---"
    continue
  fi
  echo "--- seed $s ---"
  "$VPY" -u "$GEN" --manifest "subsets/${TAG}_seed${s}_${METHOD}.json" --n "$NGEN" \
    --source "$SOURCE" "${PROMPT_ARG[@]}" 2>&1 \
    | grep -viE 'warn|deprecat|it/s\]$' | grep -vE '^\s*$'
done

echo
echo "############## BUILD SCENES ##############"
for s in $SEEDS; do
  "$VPY" src/build_scene.py \
    --manifest "subsets/${TAG}_seed${s}_${METHOD}.json" \
    --synthetic "synthetic/${TAG}_seed${s}_${METHOD}_${STRATEGY}" \
    --source "$SOURCE" --n-fake $FAKES --force
done

echo
echo "############## TRAIN ##############"
for s in $SEEDS; do
  for f in $FAKES; do
    SC="scenes/${TAG}_seed${s}_${METHOD}_${STRATEGY}_fake${f}"
    [ -d "$SC" ] || { echo "missing $SC"; continue; }
    echo "########## $(basename "$SC") ##########"
    "$VPY" -u src/run_experiment.py --scene "$SC" --iterations "$ITERS" \
        --resolution "$RES" "${WB_ARG[@]}" 2>&1 \
      | grep -viE 'warn|deprecat|%\|' | grep -vE '^\s*$'
    echo
  done
done

echo "############## COLLECT ##############"
"$VPY" src/collect_results.py
"$VPY" src/plot_curves.py
echo "${STRATEGY} CURVE COMPLETE"
