#!/usr/bin/env bash
# Start run_all_remaining.sh so it survives the session that started it.
#
#   bash src/launch_detached.sh
#
# setsid puts the job in its own session and process group, so it is not
# signalled when the terminal, editor or agent that launched it goes away.
# Output goes to logs/remaining.log rather than a pipe, for the same reason:
# a pipe dies with its reader.
#
# Refuses to start a second copy. Two training runs on a 4 GB card thrash
# rather than share, and both would be slower than either alone.
set -u
cd "$HOME/fewshot_gs" || exit 1
mkdir -p logs

if pgrep -f 'run_all_remaining.sh' >/dev/null 2>&1; then
  echo "already running (pid $(pgrep -f 'run_all_remaining.sh' | head -1))"
  echo "watch it with:  tail -f logs/remaining.log"
  exit 0
fi

LOG="logs/remaining.log"
setsid nohup bash src/run_all_remaining.sh >"$LOG" 2>&1 < /dev/null &
disown 2>/dev/null || true
sleep 2

PID=$(pgrep -f 'run_all_remaining.sh' | head -1)
if [ -n "$PID" ]; then
  echo "launched, pid $PID, detached"
  echo "log:    $HOME/fewshot_gs/$LOG"
  echo "watch:  tail -f $LOG"
  echo "stop:   pkill -f run_all_remaining.sh"
else
  echo "FAILED to launch - check $LOG"
  exit 1
fi
