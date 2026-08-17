#!/usr/bin/env bash
# Install COLMAP for custom-scene reconstruction.
# Reads the sudo password from stdin so it never appears in a command line,
# a process listing, or shell history.
set -u

read -r PW
PW="${PW%$'\r'}"          # PowerShell appends CR when piping

s() { echo "$PW" | sudo -S -p '' "$@" 2>&1; }

echo "=== sudo check ==="
if ! s true >/dev/null; then
  echo "SUDO FAILED - wrong password?"
  exit 1
fi
echo "ok"

echo
echo "=== apt update ==="
s apt-get update -qq | tail -3

echo
echo "=== installing colmap ==="
s env DEBIAN_FRONTEND=noninteractive apt-get install -y colmap | tail -8

echo
echo "=== verify ==="
if command -v colmap >/dev/null; then
  colmap --help 2>&1 | head -3
  echo
  # COLMAP's own banner is authoritative. ldd is not: the Ubuntu package links
  # a CUDA library transitively even in a CPU-only build, so checking ldd
  # reports GPU support that does not exist.
  echo "CUDA support:"
  if colmap --help 2>&1 | head -3 | grep -qi "without CUDA"; then
    echo "  CPU-only build -> pass --SiftExtraction.use_gpu 0"
    echo "                    and  --SiftMatching.use_gpu 0"
  else
    echo "  CUDA build (GPU SIFT available)"
  fi
  echo "COLMAP INSTALL OK"
else
  echo "COLMAP NOT FOUND AFTER INSTALL"
  exit 1
fi
