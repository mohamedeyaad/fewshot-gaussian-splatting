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
