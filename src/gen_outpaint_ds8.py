"""Outpainting with Dreamshaper-8 instead of Stable Diffusion 1.5.

Model-robustness ablation. Outpainting at k=5 is the only condition in this
study that improves reconstruction quality, so the obvious challenge to it is
that the effect belongs to one particular checkpoint rather than to the
augmentation strategy. Re-running the same sweep with a different model tests
that directly.

Dreamshaper-8-inpainting is an SD 1.5 finetune: same architecture, same VRAM
budget (2.6 GB fp16), materially better photorealism. That makes it the right
comparison - it changes image quality while holding everything structural
fixed. A larger model like SDXL would confound quality with capacity, and does
not fit in 4 GB anyway.

    STRATEGY=outpaint_ds8 K=5 FAKES="1 2 5 10" NGEN=10 bash src/run_curve.sh

Everything else - poses, intrinsics, expansion factor, prompts, seeds, the RNG
stream - is identical to gen_outpaint.py.
"""
import sys
from pathlib import Path

MODEL = "Lykon/dreamshaper-8-inpainting"

sys.argv = [sys.argv[0]] + sys.argv[1:] + ["--model", MODEL,
                                           "--label", "outpaint_ds8"]
sys.path.insert(0, str(Path(__file__).parent))

from gen_outpaint import main  # noqa: E402

if __name__ == "__main__":
    main()
