"""Choose the K real training photos for each seed, reproducibly.

The split MUST match what gaussian-splatting's train.py does internally:
  test  = every `llffhold`-th image, by SORTED FILENAME, starting at index 0
  train = everything else
(see gaussian-splatting/scene/dataset_readers.py:readColmapSceneInfo)

Two selection methods:
  fps    - farthest-point sampling over camera POSITIONS. Greedily picks the
           camera furthest from everything already chosen, so the K views are
           spread around the scene. Low variance between seeds.
  random - uniform random draw. Higher variance; closer to "someone just took
           K casual photos". Useful as a robustness check.

Writes one JSON manifest per (k, seed, method) plus a top-down camera map.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Reuse the repo's own COLMAP reader so we parse the binaries identically.
REPO = Path(__file__).resolve().parents[1] / "gaussian-splatting"
sys.path.insert(0, str(REPO))
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat  # noqa: E402


def camera_centres(sparse_dir: Path):
    """Return (names, centres) sorted by filename.

    COLMAP stores world->camera as (R, t). The camera's position in world
    space is -R^T @ t, which is what we want to measure distances between.
    """
    extr = read_extrinsics_binary(str(sparse_dir / "images.bin"))
    rows = []
    for img in extr.values():
        R = qvec2rotmat(img.qvec)
        centre = -R.T @ img.tvec
        rows.append((img.name, centre))
    rows.sort(key=lambda r: r[0])
    names = [r[0] for r in rows]
    centres = np.stack([r[1] for r in rows])
    return names, centres


def split_train_test(names, llffhold: int = 8):
    """Mirror readColmapSceneInfo: index % llffhold == 0 -> test."""
    test_idx = [i for i in range(len(names)) if i % llffhold == 0]
    train_idx = [i for i in range(len(names)) if i % llffhold != 0]
    return train_idx, test_idx


def farthest_point_sample(centres: np.ndarray, k: int, seed: int):
    """Greedy FPS. The seed only picks the STARTING camera; the rest is
    deterministic, which is what keeps coverage good across seeds."""
    n = len(centres)
    if k > n:
        raise ValueError(f"asked for {k} views but pool only has {n}")
    rng = np.random.default_rng(seed)
    first = int(rng.integers(n))
    chosen = [first]
    # dist[i] = distance from camera i to the nearest already-chosen camera
    dist = np.linalg.norm(centres - centres[first], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(centres - centres[nxt], axis=1))
    return chosen


def random_sample(centres: np.ndarray, k: int, seed: int):
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(len(centres), size=k, replace=False).tolist())


def coverage_stats(centres: np.ndarray, sel_local, pool_centres):
    """How well do the chosen cameras cover the pool?

    mean_nn_dist: for every pool camera, distance to the nearest CHOSEN
    camera, averaged. Lower = better coverage. This is the number that
    separates a good draw from a bad one.
    """
    chosen = pool_centres[sel_local]
    d = np.linalg.norm(pool_centres[:, None, :] - chosen[None, :, :], axis=2)
    nn = d.min(axis=1)
    pair = np.linalg.norm(chosen[:, None, :] - chosen[None, :, :], axis=2)
    iu = np.triu_indices(len(chosen), k=1)
    return {
        "mean_nn_dist": float(nn.mean()),
        "max_nn_dist": float(nn.max()),
        "min_pairwise_dist": float(pair[iu].min()) if len(chosen) > 1 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=os.path.expanduser("~/fewshot_gs/data/tandt/truck"))
    ap.add_argument("--out", default=os.path.expanduser("~/fewshot_gs/subsets"))
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--methods", nargs="+", default=["fps", "random"])
    ap.add_argument("--llffhold", type=int, default=8)
    ap.add_argument("--plot", action="store_true", help="render camera maps")
    args = ap.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    names, centres = camera_centres(source / "sparse" / "0")
    train_idx, test_idx = split_train_test(names, args.llffhold)

    print(f"scene      : {source.name}")
    print(f"images     : {len(names)}")
    print(f"test  (held out, every {args.llffhold}th): {len(test_idx)}")
    print(f"train pool                              : {len(train_idx)}")

    # Persist the test split once - it is identical for EVERY condition.
    (out / "test_split.json").write_text(json.dumps({
        "scene": source.name, "llffhold": args.llffhold,
        "test_images": [names[i] for i in test_idx],
        "train_pool": [names[i] for i in train_idx],
    }, indent=2))

    pool_centres = centres[train_idx]
    manifests = []

    for method in args.methods:
        for k in args.k:
            for seed in args.seeds:
                if method == "fps":
                    local = farthest_point_sample(pool_centres, k, seed)
                else:
                    local = random_sample(pool_centres, k, seed)
                gidx = [train_idx[i] for i in local]
                sel = [names[i] for i in gidx]
                stats = coverage_stats(centres, local, pool_centres)

                rec = {
                    "scene": source.name, "method": method, "k": k, "seed": seed,
                    "llffhold": args.llffhold,
                    "images": sorted(sel),
                    "coverage": stats,
                }
                name = f"{source.name}_k{k}_seed{seed}_{method}.json"
                (out / name).write_text(json.dumps(rec, indent=2))
                manifests.append(rec)
                print(f"  {method:6s} k={k:2d} seed={seed}  "
                      f"mean_nn={stats['mean_nn_dist']:.3f}  "
                      f"max_nn={stats['max_nn_dist']:.3f}")

    print(f"\nwrote {len(manifests)} manifests to {out}")

    if args.plot:
        make_plots(names, centres, train_idx, test_idx, manifests, out, args)


def make_plots(names, centres, train_idx, test_idx, manifests, out, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Project onto the two axes with the most spread -> a sensible "top-down".
    spread = centres.std(axis=0)
    a, b = np.argsort(spread)[::-1][:2]
    lbl = {0: "X", 1: "Y", 2: "Z"}

    for k in args.k:
        for method in args.methods:
            recs = [m for m in manifests if m["k"] == k and m["method"] == method]
            if not recs:
                continue
            fig, axes = plt.subplots(1, len(recs), figsize=(4.2 * len(recs), 4.4),
                                     squeeze=False)
            for ax, rec in zip(axes[0], recs):
                ax.scatter(centres[train_idx, a], centres[train_idx, b],
                           s=14, c="#d8d8d8", label=f"pool ({len(train_idx)})",
                           zorder=1)
                ax.scatter(centres[test_idx, a], centres[test_idx, b],
                           s=16, c="#7fb3ff", marker="s",
                           label=f"test ({len(test_idx)})", zorder=2)
                sel = [names.index(n) for n in rec["images"]]
                ax.scatter(centres[sel, a], centres[sel, b],
                           s=95, c="#e8443a", marker="*",
                           edgecolors="black", linewidths=0.5,
                           label=f"chosen ({k})", zorder=3)
                ax.set_title(f"{method}  k={k}  seed={rec['seed']}\n"
                             f"mean_nn={rec['coverage']['mean_nn_dist']:.3f}",
                             fontsize=10)
                ax.set_xlabel(lbl[a]); ax.set_ylabel(lbl[b])
                ax.set_aspect("equal", adjustable="datalim")
                ax.grid(alpha=0.25, linewidth=0.5)
            axes[0][0].legend(fontsize=7, loc="best")
            fig.suptitle(f"Camera positions - {method.upper()} selection, k={k}",
                         fontsize=12)
            fig.tight_layout()
            p = out / f"cameras_k{k}_{method}.png"
            fig.savefig(p, dpi=130)
            plt.close(fig)
            print(f"  plot -> {p}")


if __name__ == "__main__":
    main()
