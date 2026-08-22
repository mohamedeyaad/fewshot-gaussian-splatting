"""Which scene a run belongs to.

results.json has no explicit `scene` field - it was written when truck was the
only scene, so (k, seed) was a unique key. It stopped being unique the moment
drjohnson was trained: drjohnson k=5 seed0 and truck k=5 seed0 collide, and an
augmented run of one scene silently gets paired against the other scene's
baseline. The symptom is a delta table with n=6 seeds per row and standard
deviations an order of magnitude too large.

Rather than rewrite 191 results.json files, derive the scene from provenance.
`source` is the dataset directory and is the most direct signal, but it is
absolute for the truck runs and relative for drjohnson, so take the basename.
Fall back to the manifest name and then the tag, both of which are formatted
`{scene}_k{k}_seed{s}_...`.
"""
from __future__ import annotations

from pathlib import PurePosixPath


def scene_of(rec: dict) -> str:
    prov = rec.get("provenance", {}) or {}
    src = prov.get("source")
    if src:
        return PurePosixPath(str(src)).name
    man = prov.get("manifest")
    if man:
        return PurePosixPath(str(man)).stem.split("_k")[0]
    return str(rec.get("tag", "")).split("_k")[0]


def select_of(rec: dict) -> str:
    """How the K real views were CHOSEN: 'fps', 'random', or 'full'.

    The third collision axis, and the one that stayed hidden longest. Every
    result in the study used farthest-point sampling, so (scene, k, seed) was
    unique - until the `random` manifests, written by select_subsets.py at the
    very start, were finally trained on. Then truck_k20_seed0_fps_fake0 and
    truck_k20_seed0_random_fake0 shared a baseline key, one silently
    overwrote the other, and EVERY fps delta in the study was recomputed
    against a random baseline that is ~0.7 dB lower. Inpainting at k=20 moved
    from -0.161 to +0.515 - a sign flip, from a run that had nothing to do
    with inpainting.

    This is the same failure as scene_of() one axis over: a key that was
    unique only because an experiment had not been run yet.
    """
    prov = rec.get("provenance", {}) or {}
    m = prov.get("method")
    if m:
        return str(m)
    man = prov.get("manifest")
    if man:                       # subsets/truck_k20_seed0_random.json
        return PurePosixPath(str(man)).stem.rsplit("_", 1)[-1]
    return "fps"                  # pre-dates the random sweep


def is_depth(rec: dict) -> bool:
    """Was this run trained with depth regularisation?

    Depth runs share a scene with their non-depth twin, so they carry an
    IDENTICAL provenance: same k, same seed, same strategy, same n_synthetic.
    That makes them indistinguishable to any table keyed on those fields, and
    they silently contaminate it in two ways - a `..._fake0_depth` run counts
    as a baseline and overwrites the real one, and a `..._outpaint_fakeN_depth`
    run joins the outpainting bucket and doubles its seed count.

    They belong in the depth analysis (src/depth_compare.py), not in the
    strategy tables, so every general loader filters on this.

    Runs predating depth regularisation have no such key and are never depth.
    """
    prov = rec.get("provenance", {}) or {}
    if prov.get("depth_reg"):
        return True
    # Belt and braces: the tag suffix is set by run_experiment.py's --out.
    return str(rec.get("tag", "")).endswith("_depth")
