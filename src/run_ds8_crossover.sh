#!/usr/bin/env bash
# Finish the model ablation: does the CROSSOVER reproduce, or only the gain?
#
#   bash src/run_ds8_crossover.sh
#
# Dreamshaper-8 was already swept at k=5, where it reproduces the positive
# effect at every ratio (+0.074 / +0.075 / +0.159 / +0.193 against SD 1.5's
# +0.140 / +0.140 / +0.182 / +0.285). Smaller, same sign - so the GAIN is not
# an artefact of one checkpoint.
#
# But the finding of this study is not the gain, it is the SIGN CHANGE, and
# k=10 and k=20 were never run with the second model. As it stands the
# ablation supports the half of the claim that was least in doubt. If
# outpainting at k=20 comes out positive with Dreamshaper, the crossover IS a
# property of SD 1.5 and the central result is much weaker than reported.
#
# k=10 is included because it is where SD 1.5 sits at -0.003, the crossover
# point itself: a model that shifts the whole curve up or down moves that
# point, and only running the two ends would hide it.
#
# 9 training runs and ~180 generated images, roughly 1.5 hours. Idempotent.
set -u
cd "$HOME/fewshot_gs" || exit 1
VPY="$HOME/fewshot_gs/venv/bin/python"

t0=$(date +%s)

for K in 10 20; do
  case "$K" in
    10) FAKES="20";    NGEN=20 ;;   # 200%
    20) FAKES="20 40"; NGEN=40 ;;   # 100%, 200%
  esac
  echo
  echo "################################################################"
  echo "###  truck  k=${K}  outpaint_ds8   fakes: ${FAKES}"
  echo "################################################################"
  SCENE=truck SOURCE=data/tandt/truck K="$K" FAKES="$FAKES" NGEN="$NGEN" \
    STRATEGY=outpaint_ds8 RES=2 SEEDS="0 1 2" METHOD=fps \
    bash src/run_curve.sh
  rm -rf runs/*/point_cloud runs/*/input.ply
done

echo
echo "############## SD 1.5 vs DREAMSHAPER-8 ##############"
"$VPY" - <<'PY'
import json, os
from statistics import mean, stdev
os.chdir(os.path.expanduser("~/fewshot_gs"))

NF = {5: 10, 10: 20, 20: 40}          # the 200% ratio at each subset size

def psnr(tag):
    p = f"runs/{tag}/results.json"
    return json.load(open(p))["metrics"]["psnr"]["mean"] if os.path.exists(p) else None

print(f"{'':>6} {'SD 1.5':>18} {'Dreamshaper-8':>18}")
print("-" * 46)
for k in (5, 10, 20):
    cells = []
    for strat in ("outpaint", "outpaint_ds8"):
        ds = []
        for s in (0, 1, 2):
            b = psnr(f"truck_k{k}_seed{s}_fps_fake0")
            a = psnr(f"truck_k{k}_seed{s}_fps_{strat}_fake{NF[k]}")
            if b is not None and a is not None:
                ds.append(a - b)
        if not ds:
            cells.append("        --       ")
            continue
        sd = stdev(ds) if len(ds) > 1 else 0.0
        star = "*" if abs(mean(ds)) > sd > 0 else " "
        cells.append(f"{mean(ds):+9.3f} ± {sd:5.3f}{star}")
    print(f"k={k:<4} {cells[0]} {cells[1]}")
print()
print("At the 200% ratio, paired within seed. The claim survives only if the")
print("Dreamshaper column changes sign between k=5 and k=20 as SD 1.5 does.")
PY
t1=$(date +%s)
printf '\nDS8 CROSSOVER COMPLETE in %dh %dm\n' $(( (t1-t0)/3600 )) $(( ((t1-t0)%3600)/60 ))
