#!/usr/bin/env bash
# The control that decides what the isolated-object result actually shows.
#
#   bash src/run_legoc_control.sh
#
# Outpainting cost legoc 9.3 dB at k=5 and 15.6 dB at k=20. Two explanations
# survive those numbers and the sweep cannot tell them apart:
#
#   A. the FABRICATED CONTENT is destructive - the diffusion model paints
#      confetti where the truth is empty white, and the optimiser cannot
#      reconcile it across views;
#   B. the WIDENED FRUSTUM alone is destructive - a synthetic camera with a
#      48.5 deg FOV where every real camera has 39.6 deg, whatever fills it.
#
# --fill white isolates A: identical canvas, identical focal, identical
# widened camera id 2, identical paste offset, identical poses - verified
# equal in poses.json - with plain white in the new border instead of
# diffusion output. On an isolated object plain white is the TRUE content out
# there, so under explanation A this returns to baseline and under B it does
# not.
#
# 6 training runs at the 100% ratio, roughly 40 minutes. Generation is
# instant: no model is loaded.
set -u
cd "$HOME/fewshot_gs" || exit 1
VPY="$HOME/fewshot_gs/venv/bin/python"

SCENE=legoc
SOURCE=data/legoc
SEEDS="0 1 2"
LABEL=outwhite

t0=$(date +%s)

echo "############## GENERATE (white fill) ##############"
for K in 5 20; do
  for s in $SEEDS; do
    "$VPY" -u src/gen_outpaint.py \
        --manifest "subsets/${SCENE}_k${K}_seed${s}_fps.json" \
        --n "$K" --source "$SOURCE" --fill white --label "$LABEL" 2>&1 \
      | grep -viE 'warn|deprecat' | tail -3
  done
done

echo
echo "############## BUILD SCENES ##############"
for K in 5 20; do
  for s in $SEEDS; do
    "$VPY" src/build_scene.py \
        --manifest "subsets/${SCENE}_k${K}_seed${s}_fps.json" \
        --synthetic "synthetic/${SCENE}_k${K}_seed${s}_fps_${LABEL}" \
        --source "$SOURCE" --n-fake "$K" --force
  done
done

echo
echo "############## TRAIN ##############"
for K in 5 20; do
  for s in $SEEDS; do
    SC="scenes/${SCENE}_k${K}_seed${s}_fps_${LABEL}_fake${K}"
    [ -d "$SC" ] || { echo "missing $SC"; continue; }
    echo "########## $(basename "$SC") ##########"
    "$VPY" -u src/run_experiment.py --scene "$SC" --out "runs/$(basename "$SC")" \
        --iterations 7000 --resolution 1 --white-background 2>&1 \
      | grep -viE 'warn|deprecat|%\|' | grep -vE '^\s*$'
  done
done

rm -rf runs/*/point_cloud runs/*/input.ply

echo
echo "############## VERDICT ##############"
"$VPY" - <<'PY'
import json, os
from statistics import mean
os.chdir(os.path.expanduser("~/fewshot_gs"))

def psnr(tag):
    p = f"runs/{tag}/results.json"
    return json.load(open(p))["metrics"]["psnr"]["mean"] if os.path.exists(p) else None

print(f"{'':>6} {'baseline':>9} {'+white':>9} {'delta':>8}   {'+diffusion':>10} {'delta':>8}")
print("-" * 60)
for k in (5, 20):
    rows = []
    for s in (0, 1, 2):
        b = psnr(f"legoc_k{k}_seed{s}_fps_fake0")
        w = psnr(f"legoc_k{k}_seed{s}_fps_outwhite_fake{k}")
        d = psnr(f"legoc_k{k}_seed{s}_fps_outpaint_fake{k}")
        if None not in (b, w, d):
            rows.append((b, w, d))
    if not rows:
        continue
    b, w, d = (mean(x) for x in zip(*rows))
    print(f"k={k:<4} {b:9.2f} {w:9.2f} {w-b:+8.3f}   {d:10.2f} {d-b:+8.3f}")
print()
print("If '+white' sits near baseline while '+diffusion' collapses, the damage")
print("is the fabricated content, not the widened frustum.")
PY
t1=$(date +%s)
printf '\nCONTROL COMPLETE in %dh %dm\n' $(( (t1-t0)/3600 )) $(( ((t1-t0)%3600)/60 ))
