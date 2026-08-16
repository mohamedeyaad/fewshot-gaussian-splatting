#!/usr/bin/env bash
# Which alternative diffusion models exist, and how big are their fp16 weights?
check() {
  local repo="$1" label="$2"
  code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 25 \
        "https://huggingface.co/api/models/$repo")
  sz=$(curl -sS --max-time 30 "https://huggingface.co/api/models/$repo?blobs=true" 2>/dev/null \
     | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
except Exception:
    print(''); raise SystemExit
tot16=tot32=0
for f in d.get('siblings',[]):
    n=f.get('rfilename',''); s=f.get('size') or 0
    if not n.endswith(('.safetensors','.bin')): continue
    if 'fp16' in n: tot16+=s
    elif n.endswith('.safetensors'): tot32+=s
best = tot16 or tot32
print(f'{best/1073741824:.1f} GB' + (' (fp16)' if tot16 else ' (fp32 only)') if best else '?')
" 2>/dev/null)
  printf '  %-4s %-46s %-14s %s\n' "$code" "$repo" "${sz:-?}" "$label"
}

echo "=== INPAINTING models that could fit 4GB ==="
check "stable-diffusion-v1-5/stable-diffusion-inpainting" "SD1.5 (current)"
check "stabilityai/stable-diffusion-2-inpainting"         "SD2.0 - newer, 512 native"
check "Lykon/dreamshaper-8-inpainting"                    "SD1.5 finetune, photoreal"
check "kandinsky-community/kandinsky-2-2-decoder-inpaint" "Kandinsky 2.2"

echo
echo "=== TOO BIG for 4GB (listed for the report) ==="
check "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"  "SDXL - needs ~8GB"
check "black-forest-labs/FLUX.1-Fill-dev"                 "FLUX - 12B params"

echo
echo "=== MULTI-VIEW CONSISTENT generators (the real fix) ==="
check "sudo-ai/zero123plus-v1.2"                          "Zero123++ pose-conditioned"
check "stabilityai/sv3d"                                  "SV3D orbit video"
check "ashawkey/imagedream-ipmv-diffusers"                "ImageDream multi-view"
