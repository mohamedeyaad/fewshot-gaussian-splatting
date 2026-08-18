#!/usr/bin/env bash
# Depth regularisation at k=5, the 2x2 that makes it a result.
#
#   bash src/run_depth_reg.sh
#
#                       no depth        + depth prior
#   real only           (have it)       run
#   + outpainting 200%  (have it)       run
#
# The two existing cells are already trained, so this adds 6 runs (2 conditions
# x 3 seeds), roughly 40 minutes including depth generation.
#
# WHY THE 2x2 AND NOT JUST "+depth": the interesting question is not whether a
# depth prior helps - it is whether it helps for the SAME reason augmentation
# does. This study's finding is that a synthetic view supplies coverage and
# inconsistency together. A depth prior supplies constraint with no view at
# all. If the two attack the same deficiency their gains will not add; if they
# attack different ones, they will. Either outcome is a result, and neither is
# available from the "+depth" column alone.
set -u
cd "$HOME/fewshot_gs" || exit 1
VPY="$HOME/fewshot_gs/venv/bin/python"

SCENE=truck
SOURCE=data/tandt/truck
K=5
SEEDS="${SEEDS:-0 1 2}"
ITERS=7000
RES=2
NF=10          # 200% ratio at k=5 - the best-performing outpainting condition

t0=$(date +%s)

echo "############## 1. DEPTH MAPS FOR THE REAL DATASET ##############"
# Every image in the dataset, not just the training subsets: the test views are
# loaded too, and camera_utils.py crashes on a missing depth file.
"$VPY" src/gen_depths.py --source "$SOURCE" || exit 1

echo
echo "############## 2. SCALE + OFFSET AGAINST COLMAP POINTS ##############"
# NOT gaussian-splatting/utils/make_depth_scale.py: its keypoint sampling is
# broken on OpenCV 5 (it collapses every image to one sample and emits
# scale=inf, which disables depth for the whole scene without any error).
# src/make_depth_params.py does the same job with correct sampling and reports
# the fit quality. See its docstring.
"$VPY" src/make_depth_params.py --source "$SOURCE" || exit 1

echo
echo "############## 3. DEPTH MAPS FOR THE SYNTHETIC VIEWS ##############"
# Needed only so the loader finds a file; add_depths.py masks them out of the
# loss. Generating them is cheaper than special-casing the reader.
for s in $SEEDS; do
  SD="synthetic/${SCENE}_k${K}_seed${s}_fps_outpaint"
  [ -d "$SD/images" ] || { echo "  missing $SD/images"; exit 1; }
  "$VPY" src/gen_depths.py --images "$SD/images" --out "$SD/depths" || exit 1
done

echo
echo "############## 4. ATTACH DEPTHS TO THE SCENES ##############"
for s in $SEEDS; do
  for SC in "scenes/${SCENE}_k${K}_seed${s}_fps_fake0" \
            "scenes/${SCENE}_k${K}_seed${s}_fps_outpaint_fake${NF}"; do
    [ -d "$SC" ] || { echo "  MISSING $SC - rebuild with src/build_scene.py"; exit 1; }
    "$VPY" src/add_depths.py --scene "$SC" --source "$SOURCE" || exit 1
  done
done

echo
echo "############## 4b. GATE: is anything actually supervised? ##############"
# Refuse to spend GPU hours on a run that would reproduce the baseline because
# of a plumbing fault. Both earlier failures looked exactly like a null result.
"$VPY" src/check_depth_coverage.py || {
  echo "ABORTING: depth would be a no-op - fix the above before training."
  exit 1
}

echo
echo "############## 5. TRAIN ##############"
for s in $SEEDS; do
  for SC in "scenes/${SCENE}_k${K}_seed${s}_fps_fake0" \
            "scenes/${SCENE}_k${K}_seed${s}_fps_outpaint_fake${NF}"; do
    TAG="$(basename "$SC")_depth"
    echo "########## $TAG ##########"
    "$VPY" -u src/run_experiment.py --scene "$SC" --out "runs/$TAG" \
        --iterations "$ITERS" --resolution "$RES" --depths depths 2>&1 \
      | grep -viE 'warn|deprecat|%\|' | grep -vE '^\s*$'
    echo
  done
done

echo "############## 6. RESULT ##############"
"$VPY" src/depth_compare.py
t1=$(date +%s)
printf '\nDEPTH REGULARISATION COMPLETE in %dh %dm\n' \
  $(( (t1-t0)/3600 )) $(( ((t1-t0)%3600)/60 ))
