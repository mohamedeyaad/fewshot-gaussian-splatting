#!/usr/bin/env bash
# Full ratio sweep at the subset sizes the spec asks for but that were never
# trained: k=5 and k=20. (k=10 is already complete.)
#
#   bash src/run_all_scales.sh
#
# Absolute synthetic counts per size, and the ratio each represents:
#
#   k=5   1  2  5 10   ->  20 /  40 / 100 / 200 %
#   k=20  5 10 20 40   ->  25 /  50 / 100 / 200 %
#
# Only k=20 divides cleanly into the spec's 25/50/100/200%; at k=5 the 25% and
# 50% points are fractional (1.25 and 2.5 images) and are rounded down.
#
# 72 training runs plus generation. Expect roughly 8 hours. Fully idempotent -
# every stage skips work that already has an output, so this survives being
# interrupted and restarted.
#
# GPU serialisation matters: diffusion peaks at 2.65 GB and training at 3.0 GB
# on a 4 GB card, so they must never overlap. run_curve.sh finishes all
# generation before it starts training, and the loop below is sequential.
set -u
cd "$HOME/fewshot_gs" || exit 1
VPY="$HOME/fewshot_gs/venv/bin/python"
ITERS="${ITERS:-7000}"
STRATEGIES="${STRATEGIES:-inpaint outpaint guided}"

t0=$(date +%s)

run_scale() {
  local k="$1" fakes="$2" ngen="$3"
  for strat in $STRATEGIES; do
    echo
    echo "################################################################"
    echo "###  k=${k}  ${strat}   fakes: ${fakes}"
    echo "################################################################"
    K="$k" FAKES="$fakes" NGEN="$ngen" STRATEGY="$strat" ITERS="$ITERS" \
      bash src/run_curve.sh
  done
}

echo "=============== k=5 (36 runs) ==============="
run_scale 5 "1 2 5 10" 10

echo
echo "=============== k=20 (36 runs) ==============="
run_scale 20 "5 10 20 40" 40

echo
echo "############## COLLECT ##############"
"$VPY" src/collect_results.py
"$VPY" src/plot_curves.py

t1=$(date +%s)
printf '\nALL SCALES COMPLETE in %dh %dm\n' $(( (t1-t0)/3600 )) $(( ((t1-t0)%3600)/60 ))
