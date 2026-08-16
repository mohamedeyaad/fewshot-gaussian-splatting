#!/usr/bin/env bash
cd "$HOME/fewshot_gs/gaussian-splatting"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
VPY="$HOME/fewshot_gs/venv/bin/python"

SRC="$HOME/fewshot_gs/data/tandt/truck"
OUT="$HOME/fewshot_gs/runs/smoke_test"
rm -rf "$OUT"; mkdir -p "$OUT"

# Poll VRAM in the background so we get a real peak, not a guess.
( while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> /tmp/vram_poll.txt 2>/dev/null
    sleep 1
  done ) &
POLLER=$!
rm -f /tmp/vram_poll.txt

echo "=== training: truck, -r 2, 1000 iterations ==="
START=$(date +%s)
"$VPY" train.py \
  -s "$SRC" \
  -m "$OUT" \
  --eval \
  -r 2 \
  --iterations 1000 \
  --test_iterations 1000 \
  --save_iterations 1000 \
  --disable_viewer \
  2>&1 | tail -25
RC=${PIPESTATUS[0]}
END=$(date +%s)

kill $POLLER 2>/dev/null

echo
echo "=== result ==="
echo "exit code: $RC"
echo "wall time: $((END-START)) s"
if [ -f /tmp/vram_poll.txt ]; then
  echo "peak VRAM (MiB, incl. desktop): $(sort -n /tmp/vram_poll.txt | tail -1)"
fi

echo
echo "=== outputs ==="
find "$OUT" -name "*.ply" -exec ls -lh {} \; 2>/dev/null | awk '{print $5, $9}'
if [ -f "$OUT/point_cloud/iteration_1000/point_cloud.ply" ]; then
  "$VPY" - <<EOF
from plyfile import PlyData
p = PlyData.read("$OUT/point_cloud/iteration_1000/point_cloud.ply")
print("gaussians:", len(p['vertex']))
EOF
fi
echo
echo "=== train/test split detected ==="
"$VPY" - <<EOF
import json, os
c = os.path.join("$OUT", "cameras.json")
if os.path.exists(c):
    print("cameras.json entries:", len(json.load(open(c))))
EOF
rm -f /tmp/vram_poll.txt
