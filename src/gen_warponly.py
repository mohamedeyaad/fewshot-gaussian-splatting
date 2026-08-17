"""Warp-only control: pose-guided synthesis with the diffusion step removed.

Thin wrapper so run_curve.sh can drive it as STRATEGY=warponly, exactly like
the other three. Everything is gen_guided.py; the only differences are that
disocclusion holes are left black and Stable Diffusion is never loaded.

    STRATEGY=warponly K=10 bash src/run_curve.sh

Why this exists: the pose-guided condition changes two things at once relative
to a real training view - the camera pose, and the ~10% of pixels diffusion
invents to cover disocclusions. Its measured damage cannot be attributed to
either alone. Holding the pose sequence fixed and deleting only the diffusion
step isolates that second term.

The comparison is not perfectly clean - black holes are themselves an artifact,
just a different one - but it answers the question that was actually open:
whether the diffusion step is contributing anything, positive or negative.
"""
import sys
from pathlib import Path

sys.argv = [sys.argv[0]] + sys.argv[1:] + ["--no-diffusion", "--label", "warponly"]
sys.path.insert(0, str(Path(__file__).parent))

from gen_guided import main  # noqa: E402

if __name__ == "__main__":
    main()
