#!/usr/bin/env bash
set -e
cd "$HOME/fewshot_gs"

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
# Build only for this GPU's architecture (RTX 3050 Ti = Ampere, sm_86).
# Massively cuts compile time vs building every arch.
export TORCH_CUDA_ARCH_LIST="8.6"
# Parallel compile, but capped - each nvcc job is memory hungry and RAM is 7.6GB
export MAX_JOBS=4

VPIP="$HOME/fewshot_gs/venv/bin/pip"
VPY="$HOME/fewshot_gs/venv/bin/python"

echo "=== toolchain ==="
nvcc --version | tail -2
gcc --version | head -1
"$VPY" -c "import torch; print('torch', torch.__version__, '| torch.version.cuda', torch.version.cuda)"

echo
echo "=== supporting python deps ==="
"$VPIP" install -q plyfile tqdm opencv-python-headless joblib
echo "ok"

echo
echo "=== diagnosing: run setup.py directly to see the real error ==="
( cd ./gaussian-splatting/submodules/diff-gaussian-rasterization && "$VPY" setup.py --version 2>&1 | tail -20 ) || true

# --no-build-isolation: these setup.py files import torch at module level,
# which is unavailable inside pip's isolated PEP 517 build environment.
echo
echo "=== building diff-gaussian-rasterization (the long one) ==="
"$VPIP" install --no-build-isolation ./gaussian-splatting/submodules/diff-gaussian-rasterization 2>&1 | tail -20

echo
echo "=== building simple-knn ==="
"$VPIP" install --no-build-isolation ./gaussian-splatting/submodules/simple-knn 2>&1 | tail -15

echo
echo "=== building fused-ssim ==="
"$VPIP" install --no-build-isolation ./gaussian-splatting/submodules/fused-ssim 2>&1 | tail -15

echo
echo "=== import test ==="
"$VPY" - <<'EOF'
import torch
ok = True
for mod in ["diff_gaussian_rasterization", "simple_knn._C", "fused_ssim"]:
    try:
        __import__(mod)
        print(f"  {mod}: OK")
    except Exception as e:
        ok = False
        print(f"  {mod}: FAILED -> {type(e).__name__}: {e}")
print("ALL IMPORTS OK" if ok else "SOME IMPORTS FAILED")
EOF
