#!/usr/bin/env bash
# Run a list of scenes sequentially. Each writes its own results.json and is
# skipped if that already exists, so this is safe to re-run after a crash.
cd "$HOME/fewshot_gs" || exit 1
VPY="$HOME/fewshot_gs/venv/bin/python"
ITERS="${ITERS:-7000}"

for scene in "$@"; do
  echo "############ $(basename "$scene") ############"
  "$VPY" -u src/run_experiment.py --scene "$scene" --iterations "$ITERS" 2>&1 \
    | grep -viE 'warn|deprecat|^ *[0-9]+%\|' \
    | grep -vE '^\s*$'
  echo
done
echo "ALL DONE"
