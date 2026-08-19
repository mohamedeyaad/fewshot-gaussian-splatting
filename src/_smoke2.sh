#!/usr/bin/env bash
set -u
cd "$HOME/fewshot_gs" || exit 1
VPY=./venv/bin/python

echo "############## BUILD SCENES ##############"
$VPY src/build_blender_scene.py --k 5 10 20 --seeds 0 1 2 --full || exit 1

echo
echo "############## SANITY: which loader will fire? ##############"
SC=scenes/lego_k5_seed0_fps_fake0
ls -1 "$SC"
echo -n "  sparse/ present (would force COLMAP): "
[ -e "$SC/sparse" ] && echo "YES - PROBLEM" || echo "no (Blender loader will be used)"

echo
echo "############## SMOKE: one k=5 run ##############"
$VPY -u src/run_experiment.py --scene "$SC" --out runs/lego_k5_seed0_fps_fake0 \
    --iterations 7000 --resolution 1 --white-background 2>&1 \
  | grep -viE 'warn|deprecat|%\|' | grep -vE '^\s*$'
