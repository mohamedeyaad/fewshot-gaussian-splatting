#!/usr/bin/env bash
cd "$HOME/fewshot_gs" || exit 1
for repo in pleasure97/NeRF_synthetic_SfM_Points pablovela5620/nerf-synthetic-mirror XayahHina/nerf_synthetic; do
  echo "=== $repo ==="
  curl -s -m 30 "https://huggingface.co/api/datasets/$repo" \
    | ./venv/bin/python -c "
import json,sys
d=json.load(sys.stdin)
fs=[f['rfilename'] for f in d.get('siblings',[])]
print('   files:', len(fs))
for f in fs[:14]:
    print('     ', f)
if len(fs)>14: print('      ...')
" 2>/dev/null || echo "   query failed"
  echo
done
