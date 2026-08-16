#!/usr/bin/env bash
# Quantify run-to-run training noise: same scene, same config, N repeats.
# Python RNG is seeded by the repo, so anything left is CUDA nondeterminism
# (atomicAdd ordering in the rasterizer backward pass) amplified by
# densification. This sets the floor below which no effect is meaningful.
set -u
cd "$HOME/fewshot_gs" || exit 1
VPY="$HOME/fewshot_gs/venv/bin/python"
SCENE="${SCENE:-scenes/truck_k10_seed0_fps_fake0}"
REPEATS="${REPEATS:-3}"
ITERS="${ITERS:-7000}"

for i in $(seq 1 "$REPEATS"); do
  OUT="runs_noise/repeat${i}"
  echo "########## repeat $i ##########"
  "$VPY" -u src/run_experiment.py --scene "$SCENE" --out "$OUT" \
      --iterations "$ITERS" --force 2>&1 \
    | grep -viE 'warn|deprecat|%\|' | grep -vE '^\s*$'
done

echo
echo "########## SUMMARY ##########"
"$VPY" - <<'EOF'
import json, glob, statistics as st
rows = []
for p in sorted(glob.glob("runs_noise/repeat*/results.json")):
    r = json.load(open(p))
    m = r["metrics"]
    rows.append((p.split("/")[1], m["psnr"]["mean"], m["ssim"]["mean"],
                 m["lpips"]["mean"], r["cost"]["n_gaussians"],
                 r["cost"]["train_seconds"]))
print(f"{'run':10s} {'PSNR':>8s} {'SSIM':>8s} {'LPIPS':>8s} {'gauss':>10s} {'sec':>7s}")
for n, p, s, l, g, t in rows:
    print(f"{n:10s} {p:8.4f} {s:8.4f} {l:8.4f} {g:10,} {t or 0:7.0f}")
if len(rows) > 1:
    for i, name in enumerate(["PSNR", "SSIM", "LPIPS"], start=1):
        vals = [r[i] for r in rows]
        print(f"{name}: mean {st.mean(vals):.4f}  std {st.stdev(vals):.4f}  "
              f"range {max(vals)-min(vals):.4f}")
EOF
echo "NOISE MEASUREMENT COMPLETE"
