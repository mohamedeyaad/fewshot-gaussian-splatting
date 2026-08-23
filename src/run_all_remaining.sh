#!/usr/bin/env bash
# Everything still outstanding, in one chain, detached from any session.
#
#   bash src/launch_detached.sh          # start it and walk away
#   tail -f logs/remaining.log           # watch it
#
#   1. depth regularisation at 30,000        11 runs left, ~5 h
#   2. 15,000 iterations, the headline cells 12 runs,      ~2.5 h
#   3. depth regularisation on drjohnson      6 runs,      ~3 h
#
# WHY THIS EXISTS. These were launched as editor background tasks and died
# when the session ended - one run into a twelve-run sweep. Anything measured
# in hours has to outlive the thing that started it, so launch_detached.sh puts
# this under setsid with its own log.
#
# Every stage is idempotent: run_experiment.py skips any tag that already has
# results.json, so re-running resumes rather than repeating. The one run that
# did finish is kept.
set -u
cd "$HOME/fewshot_gs" || exit 1
mkdir -p logs

echo "=== started $(date '+%F %H:%M') ==="

echo
echo "################################################################"
echo "###  1/3  depth regularisation at 30,000"
echo "################################################################"
bash src/run_30k_depth.sh

echo
echo "################################################################"
echo "###  2/3  15,000 iterations"
echo "################################################################"
ITERS=15000 OUTDIR=runs_15k bash src/run_30k.sh

echo
echo "################################################################"
echo "###  3/3  depth regularisation on drjohnson"
echo "################################################################"
SCENE=drjohnson SOURCE=data/db/drjohnson RES=4 K=5 NF=10 \
  bash src/run_depth_reg.sh

echo
echo "=== all remaining sweeps complete $(date '+%F %H:%M') ==="
