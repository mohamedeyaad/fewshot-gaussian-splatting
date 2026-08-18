#!/usr/bin/env bash
# Depth regularisation across the crossover: k=10 and k=20.
#
# The prediction under test: a depth prior supplies constraint WITHOUT
# multi-view inconsistency, so unlike outpainting it should keep helping at
# k=20, where outpainting turns harmful (-0.618 at the 200% ratio). If it
# does, the crossover is demonstrably caused by inconsistency specifically
# rather than by augmentation in general.
set -u
cd "$HOME/fewshot_gs" || exit 1
t0=$(date +%s)
for cfg in "10 20" "20 40"; do
  set -- $cfg
  echo
  echo "################################################################"
  echo "###  DEPTH REGULARISATION AT k=$1   (200% ratio = $2 synthetic)"
  echo "################################################################"
  K="$1" NF="$2" bash src/run_depth_reg.sh || exit 1
done
t1=$(date +%s)
printf '\nALL DEPTH SWEEPS COMPLETE in %dh %dm\n' $(( (t1-t0)/3600 )) $(( ((t1-t0)%3600)/60 ))
