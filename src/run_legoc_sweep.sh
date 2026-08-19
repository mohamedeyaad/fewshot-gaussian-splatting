#!/usr/bin/env bash
# The augmentation experiment on an isolated object.
#
#   bash src/run_legoc_sweep.sh
#
# THE PREDICTION THIS TESTS. On truck and drjohnson, outpainting helps at k=5
# and harms at k=20, and the explanation offered is coverage: synthetic views
# pay off exactly as far as they supply scene content no real camera saw.
#
# An isolated object on a clean background splits that claim in two, because
# its two kinds of missing information come apart:
#
#   - FRAMING deficit: what lies outside the frame. Here that is white
#     background. Outpainting can only invent it, so it should be worth
#     nothing - or less than nothing, if diffusion hallucinates scenery into
#     what should be empty.
#   - ANGULAR deficit: the unseen sides of the digger. Large at k=5, and the
#     only thing pose-guided synthesis addresses.
#
# So the coverage story predicts outpainting flat-to-negative at both subset
# sizes here, while pose-guided is the strategy that can help. On truck the two
# were confounded - outpainting supplied both kinds at once. If outpainting
# helps at k=5 anyway, coverage is not the mechanism and the report has to say
# so.
#
# DESIGN. k=5 and k=20 only: they are the two ends of the crossover, where
# truck showed opposite signs. Ratios 100% and 200%, three seeds, both
# strategies, plus the fake0 baselines and the full-data ceiling. 31 training
# runs and ~300 diffusion images, roughly 3-4 hours.
#
# Everything is idempotent - run_experiment.py skips anything that already has
# results.json, and run_curve.sh skips generation when the images exist.
set -u
cd "$HOME/fewshot_gs" || exit 1
VPY="$HOME/fewshot_gs/venv/bin/python"

SCENE=legoc
SOURCE=data/legoc
SPLIT=subsets/legoc_test_split.json
RES=1                      # native 800x800, matching the Blender lego runs
WB=1                       # white baked into the images; rasteriser must agree
SEEDS="0 1 2"
# Deliberately generic, like the drjohnson prompt: naming the digger would be
# doing the model's job and would not transfer to another object.
PROMPT="a toy construction vehicle on a plain white background, product photograph, sharp focus"

# What the Blender-format ceiling scored on these same 34 held-out frames.
# Reproducing it is the only external check this scene has.
REFERENCE=33.774
FLOOR=31.5

t0=$(date +%s)

reclaim() {
  rm -rf runs/*/point_cloud runs/*/input.ply
  echo "  [disk] host free: $(df -h / | awk 'NR==2{print $4}')"
}

# ---------------------------------------------------------------------------
# GATE. Train the full-data ceiling first and refuse to spend three hours on
# few-shot conditions if the COLMAP poses did not survive the rebuild. A broken
# reconstruction would still produce a plausible-looking set of numbers - every
# condition equally wrong - which is precisely the failure this project keeps
# hitting.
# ---------------------------------------------------------------------------
echo "############## GATE: full-data ceiling ##############"
SC=scenes/${SCENE}_k100_seed0_full_fake0
[ -d "$SC" ] || { echo "missing $SC - run build_scene.py first"; exit 1; }
"$VPY" -u src/run_experiment.py --scene "$SC" --out "runs/$(basename "$SC")" \
    --iterations 7000 --resolution "$RES" --white-background 2>&1 \
  | grep -viE 'warn|deprecat|%\|' | grep -vE '^\s*$'

CEIL=$("$VPY" -c "import json; print(json.load(open('runs/${SCENE}_k100_seed0_full_fake0/results.json'))['metrics']['psnr']['mean'])")
echo
echo "  ceiling      : ${CEIL} dB"
echo "  Blender ref  : ${REFERENCE} dB (same 34 held-out frames)"
awk -v c="$CEIL" -v f="$FLOOR" 'BEGIN { exit !(c < f) }' && {
  echo
  echo "  ABORTING: ceiling is below ${FLOOR} dB, so the reconstruction or the"
  echo "  split is wrong. Fix that before running any few-shot condition -"
  echo "  masking the background during feature extraction is the first thing"
  echo "  to try (build_colmap_lego.py, --ImageReader.mask_path)."
  exit 1
}
echo "  GATE PASSED - poses reproduce the reference, sweep is safe to run"
reclaim

# ---------------------------------------------------------------------------
# SUBSETS. --split bypasses the llffhold rule: the train and test frames are
# two separate NeRF-Synthetic orbits and interleaving them would silently throw
# away the comparison the gate above depends on.
# ---------------------------------------------------------------------------
echo
echo "############## SUBSETS ##############"
"$VPY" src/select_subsets.py --source "$SOURCE" --split "$SPLIT" \
    --k 5 20 --seeds $SEEDS --methods fps --plot

# ---------------------------------------------------------------------------
# BASELINES. The fake0 runs every augmented condition is measured against.
# Paired within seed, so these must exist before anything else is interpreted.
# ---------------------------------------------------------------------------
echo
echo "############## BASELINES (fake0) ##############"
for K in 5 20; do
  for s in $SEEDS; do
    "$VPY" src/build_scene.py --manifest "subsets/${SCENE}_k${K}_seed${s}_fps.json" \
        --source "$SOURCE" --n-fake 0
  done
done
for K in 5 20; do
  for s in $SEEDS; do
    SC="scenes/${SCENE}_k${K}_seed${s}_fps_fake0"
    [ -d "$SC" ] || { echo "missing $SC"; continue; }
    echo "########## $(basename "$SC") ##########"
    "$VPY" -u src/run_experiment.py --scene "$SC" --out "runs/$(basename "$SC")" \
        --iterations 7000 --resolution "$RES" --white-background 2>&1 \
      | grep -viE 'warn|deprecat|%\|' | grep -vE '^\s*$'
  done
done
reclaim

# ---------------------------------------------------------------------------
# THE TWO STRATEGIES. Ratios 100% and 200% only - the 25% and 50% points on
# truck sat inside the noise floor at both ends, so they would cost an hour to
# re-confirm nothing.
# ---------------------------------------------------------------------------
for STRAT in outpaint guided; do
  for K in 5 20; do
    case "$K" in
      5)  FAKES="5 10";   NGEN=10 ;;   # 100 / 200 %
      20) FAKES="20 40";  NGEN=40 ;;   # 100 / 200 %
    esac
    echo
    echo "################################################################"
    echo "###  ${SCENE}  k=${K}  ${STRAT}   fakes: ${FAKES}"
    echo "################################################################"
    SCENE="$SCENE" SOURCE="$SOURCE" PROMPT="$PROMPT" \
      K="$K" FAKES="$FAKES" NGEN="$NGEN" STRATEGY="$STRAT" \
      RES="$RES" WB="$WB" SEEDS="$SEEDS" \
      bash src/run_curve.sh
    reclaim
  done
done

echo
echo "############## COLLECT ##############"
"$VPY" src/collect_results.py
"$VPY" src/validate_runs.py "$SCENE"
t1=$(date +%s)
printf '\nLEGOC SWEEP COMPLETE in %dh %dm\n' $(( (t1-t0)/3600 )) $(( ((t1-t0)%3600)/60 ))
