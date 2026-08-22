#!/usr/bin/env bash
# Does the crossover depend on having chosen the five views well?
#
#   bash src/run_random_sweep.sh
#
# THE GAP THIS FILLS. Every result in the study uses farthest-point sampling,
# which spreads the K views over the camera trajectory. That is the best case,
# and an obvious objection: nobody taking five casual photographs gets an
# optimal spread. select_subsets.py has always written `random` manifests
# alongside the `fps` ones - they were never trained on.
#
# THE PREDICTION. Random draws cover the scene measurably worse (truck k=5:
# max_nn_dist 5.15 against fps's 3.54 - some pool cameras sit far from anything
# chosen). If augmentation pays off in proportion to the coverage it supplies,
# then on random subsets it should:
#
#   - help MORE at k=5, because there is more missing coverage to fill;
#   - cross over LATER, i.e. still be positive at a k where fps has already
#     turned negative.
#
# k=10 is the sharp test. On fps, outpainting at 200% is worth -0.003 dB there
# - the crossover point itself, indistinguishable from zero. If the coverage
# account is right, random at k=10 should be clearly positive. If augmentation
# helps by the same amount however well the real views already cover the scene,
# the account is wrong and the report has to say so.
#
# 18 training runs (9 baselines + 9 augmented) and ~210 generated images,
# roughly 3 hours. Idempotent.
set -u
cd "$HOME/fewshot_gs" || exit 1
VPY="$HOME/fewshot_gs/venv/bin/python"

SCENE=truck
SOURCE=data/tandt/truck
RES=2
SEEDS="0 1 2"
export METHOD=random

t0=$(date +%s)

# ---------------------------------------------------------------------------
# BASELINES. Nine of them, and they are not optional: every delta below is
# paired against the same seed's own random baseline. Pairing a random
# augmented run against an fps baseline would book the entire coverage
# difference as an augmentation effect.
# ---------------------------------------------------------------------------
echo "############## RANDOM BASELINES ##############"
SCENE="$SCENE" SOURCE="$SOURCE" KS="5 10 20" SEEDS="$SEEDS" RES="$RES" \
  METHOD=random bash src/run_floors.sh

# ---------------------------------------------------------------------------
# OUTPAINTING at the 200% ratio only - the one ratio present at every subset
# size on the fps side, so the two methods are compared like for like.
# ---------------------------------------------------------------------------
for K in 5 10 20; do
  case "$K" in
    5)  FAKES="10"; NGEN=10 ;;
    10) FAKES="20"; NGEN=20 ;;
    20) FAKES="40"; NGEN=40 ;;
  esac
  echo
  echo "################################################################"
  echo "###  ${SCENE}  k=${K}  random  outpaint  200%"
  echo "################################################################"
  SCENE="$SCENE" SOURCE="$SOURCE" K="$K" FAKES="$FAKES" NGEN="$NGEN" \
    STRATEGY=outpaint RES="$RES" SEEDS="$SEEDS" METHOD=random \
    bash src/run_curve.sh
  rm -rf runs/*/point_cloud runs/*/input.ply
done

echo
echo "############## FPS vs RANDOM ##############"
"$VPY" - <<'PY'
import json, os
from statistics import mean, stdev
os.chdir(os.path.expanduser("~/fewshot_gs"))

NF = {5: 10, 10: 20, 20: 40}          # the 200% ratio at each subset size

def psnr(tag):
    p = f"runs/{tag}/results.json"
    return json.load(open(p))["metrics"]["psnr"]["mean"] if os.path.exists(p) else None

def cov(k, m):
    p = f"subsets/truck_k{k}_seed0_{m}.json"
    return json.load(open(p))["coverage"]["mean_nn_dist"] if os.path.exists(p) else None

print(f"{'':>5} {'method':>7} {'baseline':>9} {'+200%':>9} {'delta':>17} {'coverage':>9}")
print("-" * 62)
for k in (5, 10, 20):
    for m in ("fps", "random"):
        ds, bs = [], []
        for s in (0, 1, 2):
            b = psnr(f"truck_k{k}_seed{s}_{m}_fake0")
            a = psnr(f"truck_k{k}_seed{s}_{m}_outpaint_fake{NF[k]}")
            if b is not None and a is not None:
                ds.append(a - b); bs.append(b)
        if not ds:
            continue
        sd = stdev(ds) if len(ds) > 1 else 0.0
        star = "*" if abs(mean(ds)) > sd > 0 else " "
        c = cov(k, m)
        print(f"k={k:<3} {m:>7} {mean(bs):9.2f} {mean(bs)+mean(ds):9.2f} "
              f"{mean(ds):+9.3f} ± {sd:4.3f}{star} {c if c is None else round(c,3):>9}")
    print()
print("Prediction: random should be MORE positive at every k, and still")
print("positive at k=10 where fps sits at -0.003 (the crossover point).")
PY
t1=$(date +%s)
printf '\nRANDOM SWEEP COMPLETE in %dh %dm\n' $(( (t1-t0)/3600 )) $(( ((t1-t0)%3600)/60 ))
