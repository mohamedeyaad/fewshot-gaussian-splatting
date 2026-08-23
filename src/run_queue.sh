#!/usr/bin/env bash
# Run the last two sweeps back to back, once the GPU is free.
#
#   bash src/run_queue.sh
#
#   1. 15,000 iterations, the two headline cells        12 runs, ~2.5 h
#   2. depth regularisation on drjohnson, the 2x2 at k=5 6 runs, ~3 h
#
# WHY 15,000. The convergence check has only two points, 7,000 and 30,000, and
# the report now argues from them that 7,000 is near this regime's optimum -
# the baselines fall by 30,000, so training longer overfits. Two points cannot
# actually establish that. 15,000 is where densification stops and is the point
# that decides it: if the baseline is still rising there, 7,000 was merely the
# setting used and the argument in the report is too strong.
#
# WHY DRJOHNSON DEPTH. The depth prior is the best result in the study (+0.714
# combined at k=5) and the only major one confirmed on a single scene. The
# crossover is two-scene; this is not.
#
# The card holds 4 GB and one training run saturates it, so this waits rather
# than competing with whatever is already running.
set -u
cd "$HOME/fewshot_gs" || exit 1

echo "waiting for the GPU..."
while pgrep -f 'run_30k_depth.sh|run_30k.sh|run_curve.sh|train\.py' >/dev/null 2>&1; do
  sleep 60
done
echo "GPU free at $(date '+%H:%M'), starting."

echo
echo "################################################################"
echo "###  1/2  15,000 iterations"
echo "################################################################"
ITERS=15000 OUTDIR=runs_15k bash src/run_30k.sh

echo
echo "################################################################"
echo "###  2/2  depth regularisation on drjohnson"
echo "################################################################"
# RES=4 matches every other drjohnson run: at -r 2 a 230-view run exceeds the
# card. NF=10 is the 200% ratio at k=5, the same cell the truck 2x2 used.
SCENE=drjohnson SOURCE=data/db/drjohnson RES=4 K=5 NF=10 \
  bash src/run_depth_reg.sh

echo
echo "############## ALL QUEUED SWEEPS COMPLETE ##############"
