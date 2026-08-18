#!/usr/bin/env bash
# Stage 2 of the second scene: does the outpainting crossover reproduce indoors?
#
#   bash src/run_drjohnson_sweep.sh
#
# Outpainting only, at k=5 and k=20 - the two subset sizes where truck showed
# opposite signs (+0.182 and -0.983 at the 100% ratio). Those are the two ends
# of the crossover, so they are what a generalisation test needs. Running all
# three strategies at both sizes would be 72 runs to re-confirm two things
# already known: inpainting is flat, pose-guided is harmful everywhere.
#
# 24 training runs, roughly 3.5 hours. Fully idempotent.
#
# DISK: drjohnson images are 2.2x truck's pixel count at -r 2, so checkpoints
# are correspondingly larger. Every metric is extracted into results.json
# during the run, so point_cloud/ is deleted after each block to keep the
# footprint flat - a full host disk killed an earlier sweep mid-flight and
# left WSL unable to start.
set -u
cd "$HOME/fewshot_gs" || exit 1
VPY="$HOME/fewshot_gs/venv/bin/python"

SCENE=drjohnson
export RES=4   # see run_floors.sh: -r 2 exceeds 4 GB on this scene
SOURCE=data/db/drjohnson
# The truck prompt produces nonsense indoors. Kept deliberately generic: a
# prompt that described the room too specifically would be doing the model's
# job for it and would not transfer to another scene.
PROMPT="an indoor room interior, furniture and walls, realistic photograph, sharp focus"

t0=$(date +%s)

reclaim() {
  local before after
  before=$(du -sh runs 2>/dev/null | cut -f1)
  rm -rf runs/*/point_cloud runs/*/input.ply
  after=$(du -sh runs 2>/dev/null | cut -f1)
  echo "  [disk] runs/ ${before} -> ${after}; host free: $(df -h / | awk 'NR==2{print $4}')"
}

for K in 5 20; do
  case "$K" in
    5)  FAKES="1 2 5 10";  NGEN=10 ;;   #  20 /  40 / 100 / 200 %
    20) FAKES="5 10 20 40"; NGEN=40 ;;  #  25 /  50 / 100 / 200 %
  esac
  echo
  echo "################################################################"
  echo "###  drjohnson  k=${K}  outpaint   fakes: ${FAKES}"
  echo "################################################################"
  SCENE="$SCENE" SOURCE="$SOURCE" PROMPT="$PROMPT" \
    K="$K" FAKES="$FAKES" NGEN="$NGEN" STRATEGY=outpaint RES="$RES" \
    bash src/run_curve.sh
  reclaim
done

echo
echo "############## COLLECT ##############"
"$VPY" src/collect_results.py
t1=$(date +%s)
printf '\nDRJOHNSON SWEEP COMPLETE in %dh %dm\n' $(( (t1-t0)/3600 )) $(( ((t1-t0)%3600)/60 ))
