"""Smoke test: can this 4GB GPU run SD1.5 inpainting on a real truck photo?

Loads the inpainting pipeline in fp16, masks a region of an actual dataset
image, regenerates it, and reports wall time + peak VRAM.
"""
import os, time, glob
import torch
from PIL import Image, ImageDraw

HOME = os.path.expanduser("~")
SRC = f"{HOME}/fewshot_gs/data/tandt/truck/images"
OUT = f"{HOME}/fewshot_gs/runs/diffusion_smoke"
os.makedirs(OUT, exist_ok=True)

MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"

print("=== torch / gpu ===")
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
props = torch.cuda.get_device_properties(0)
print(f"gpu {props.name} | {props.total_memory/1024**3:.2f} GB")

# --- pick a real training image -------------------------------------------
imgs = sorted(glob.glob(os.path.join(SRC, "*")))
print(f"\nfound {len(imgs)} source images; using {os.path.basename(imgs[0])}")
src = Image.open(imgs[0]).convert("RGB")
print("native size:", src.size)

# SD1.5 works at 512x512; keep it simple and square for the smoke test.
W = H = 512
img = src.resize((W, H), Image.LANCZOS)

# Mask a centre rectangle - this is the "hole" SD must invent content for.
mask = Image.new("L", (W, H), 0)
ImageDraw.Draw(mask).rectangle([160, 160, 352, 352], fill=255)
img.save(f"{OUT}/00_input.png")
mask.save(f"{OUT}/01_mask.png")

# --- load pipeline --------------------------------------------------------
from diffusers import StableDiffusionInpaintPipeline

print("\n=== loading pipeline (fp16) ===")
t0 = time.time()
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    variant="fp16",               # halves download+disk (~2GB not ~4GB)
    use_safetensors=True,
    safety_checker=None,          # saves VRAM, load time and a 1.1GB download
    requires_safety_checker=False,
)
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)
# Memory savers that matter a lot on 4GB:
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
print(f"loaded in {time.time()-t0:.1f}s")

torch.cuda.reset_peak_memory_stats()

# --- run inpainting -------------------------------------------------------
print("\n=== inpainting ===")
t0 = time.time()
result = pipe(
    prompt="a photo of a truck parked outdoors, sharp focus, realistic",
    image=img,
    mask_image=mask,
    num_inference_steps=25,
    guidance_scale=7.5,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]
elapsed = time.time() - t0
result.save(f"{OUT}/02_inpainted.png")

peak = torch.cuda.max_memory_allocated() / 1024**3
reserved = torch.cuda.max_memory_reserved() / 1024**3

print(f"\n=== results ===")
print(f"time for 25 steps : {elapsed:.1f} s")
print(f"peak VRAM alloc   : {peak:.2f} GB")
print(f"peak VRAM reserved: {reserved:.2f} GB")
print(f"of total          : {props.total_memory/1024**3:.2f} GB")
print(f"outputs written to: {OUT}")
for f in sorted(os.listdir(OUT)):
    print("   ", f)
