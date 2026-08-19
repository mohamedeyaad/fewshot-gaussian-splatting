#!/usr/bin/env bash
# Fetch ONE scene of the NeRF-Synthetic (Blender) dataset.
#
#   bash src/get_blender_scene.sh [scene]        # default: lego
#
# Why one scene and not the 700 MB archive: the host disk is the binding
# constraint here (the WSL vhdx grows on C:, which has ~19 GB free), and a
# single scene is all the experiment needs. This mirror stores files
# individually, so a per-scene fetch is possible.
#
# Only the RGB frames referenced by transforms_{train,test}.json are pulled -
# the mirror also carries *_depth_* and *_normal_* auxiliaries that would
# triple the download for no benefit here.
set -u
cd "$HOME/fewshot_gs" || exit 1
SCENE="${1:-lego}"
REPO="pablovela5620/nerf-synthetic-mirror"
BASE="https://huggingface.co/datasets/$REPO/resolve/main"
DEST="data/blender/$SCENE"

mkdir -p "$DEST"
echo "=== $SCENE -> $DEST ==="

for f in transforms_train.json transforms_test.json; do
  if [ ! -s "$DEST/$f" ]; then
    curl -sfL -m 60 "$BASE/$SCENE/$f" -o "$DEST/$f" \
      || { echo "  FAILED to fetch $f"; exit 1; }
  fi
  echo "  $f  $(wc -c < "$DEST/$f") bytes"
done

# Collect the frame paths the transforms actually reference.
./venv/bin/python - "$DEST" <<'PY' > /tmp/blender_files.txt
import json, sys, os
d = sys.argv[1]
out = []
for split in ("train", "test"):
    j = json.load(open(f"{d}/transforms_{split}.json"))
    for fr in j["frames"]:
        # file_path is like "./train/r_0" - no extension in this dataset
        rel = fr["file_path"].lstrip("./")
        out.append(rel + ".png")
print("\n".join(out))
PY

n=$(wc -l < /tmp/blender_files.txt)
echo "  $n frames referenced"

missing=$(while read -r rel; do
  [ -s "$DEST/$rel" ] || echo "$rel"
done < /tmp/blender_files.txt | wc -l)
echo "  $missing to download"

if [ "$missing" -gt 0 ]; then
  export BASE SCENE DEST
  while read -r rel; do
    [ -s "$DEST/$rel" ] && continue
    echo "$rel"
  done < /tmp/blender_files.txt \
  | xargs -P 8 -I{} bash -c '
      mkdir -p "$(dirname "$DEST/{}")"
      curl -sfL -m 120 "$BASE/$SCENE/{}" -o "$DEST/{}" || echo "  FAIL {}"
    '
fi

echo
echo "=== result ==="
echo "  train png: $(ls -1 "$DEST/train"/*.png 2>/dev/null | wc -l)"
echo "  test  png: $(ls -1 "$DEST/test"/*.png 2>/dev/null | wc -l)"
echo "  size     : $(du -sh "$DEST" | cut -f1)"
./venv/bin/python - "$DEST" <<'PY'
import json, sys
d = sys.argv[1]
j = json.load(open(f"{d}/transforms_train.json"))
print(f"  camera_angle_x: {j['camera_angle_x']:.6f}")
print(f"  train frames  : {len(j['frames'])}")
print(f"  test frames   : {len(json.load(open(f'{d}/transforms_test.json'))['frames'])}")
PY
