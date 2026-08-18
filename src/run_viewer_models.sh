#!/usr/bin/env bash
# Regenerate viewable .ply point clouds for the conditions worth showing.
#
#   bash src/run_viewer_models.sh
#
# WHY THIS IS NEEDED: every run in runs/ kept only metrics and rendered test
# images. The point clouds were deleted - by reclaim() in the sweep drivers and
# by the manual cleanup during the disk crisis - so there is nothing left to
# open in a viewer. Retraining is the only way back.
#
# WHY IT WRITES TO viewer/ AND NOT runs/: training is not bit-deterministic
# (measured noise floor sigma = 0.039 dB). Retraining into runs/ would
# overwrite results.json and shift every published number slightly, desyncing
# the report and README from the committed record. viewer/ is disposable and
# gitignored; runs/ stays frozen as the experimental record.
#
# Seven conditions, ~40 min total, ~1.3 GB of .ply. Idempotent: skips any
# model whose .ply already exists.
set -u
cd "$HOME/fewshot_gs" || exit 1
VPY="$HOME/fewshot_gs/venv/bin/python"
ITERS=7000
RES=2

mkdir -p viewer

# scene dir : label for the presentation
CONDS=(
  "truck_k5_seed0_fps_fake0|k05-floor|5 real views, no augmentation (15.28 dB)"
  "truck_k5_seed0_fps_outpaint_fake10|k05-outpaint-200|5 real + 10 synthetic, outpainting (15.63 dB)"
  "truck_k5_seed0_fps_outpaint_fake40|k05-outpaint-800|5 real + 40 synthetic, best in study (15.79 dB)"
  "truck_k20_seed0_fps_fake0|k20-floor|20 real views, no augmentation (19.58 dB)"
  "truck_k20_seed0_fps_outpaint_fake20|k20-outpaint-100|20 real + 20 synthetic, augmentation HURTS (18.70 dB)"
  "truck_k10_seed0_fps_guided_fake20|k10-guided-200|10 real + 20 pose-guided, worst failure (16.03 dB)"
  "truck_k219_seed0_full_fake0|k219-ceiling|all 219 real views (25.23 dB)"
)

t0=$(date +%s)
n=0
for entry in "${CONDS[@]}"; do
  IFS='|' read -r scene label desc <<< "$entry"
  n=$((n + 1))
  out="viewer/${label}"
  ply="${out}/point_cloud/iteration_${ITERS}/point_cloud.ply"

  echo
  echo "################################################################"
  echo "###  [${n}/${#CONDS[@]}] ${label}"
  echo "###  ${desc}"
  echo "################################################################"

  if [ -f "$ply" ]; then
    echo "  already built: $ply"
    continue
  fi
  if [ ! -d "scenes/${scene}" ]; then
    echo "  !! missing scenes/${scene} - rebuild with src/build_scene.py"
    continue
  fi

  "$VPY" -u src/run_experiment.py --scene "scenes/${scene}" --out "$out" \
      --iterations "$ITERS" --resolution "$RES" 2>&1 \
    | grep -viE 'warn|deprecat|%\|' | grep -vE '^\s*$'
done

t1=$(date +%s)
echo
echo "############## VIEWER MODELS ##############"
for entry in "${CONDS[@]}"; do
  IFS='|' read -r scene label desc <<< "$entry"
  ply="viewer/${label}/point_cloud/iteration_${ITERS}/point_cloud.ply"
  if [ -f "$ply" ]; then
    printf '  %-22s %6s  %s\n' "$label" "$(du -h "$ply" | cut -f1)" "$desc"
  else
    printf '  %-22s %6s  %s\n' "$label" "MISSING" "$desc"
  fi
done
printf '\nDONE in %dh %dm.  Files are under ~/fewshot_gs/viewer/\n' \
  $(( (t1-t0)/3600 )) $(( ((t1-t0)%3600)/60 ))
echo "From Windows: \\\\wsl.localhost\\Ubuntu-24.04\\home\\mooeyad\\fewshot_gs\\viewer\\"
echo "Drag a point_cloud.ply onto https://superspl.at/editor"
