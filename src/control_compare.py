"""Warp-only control vs pose-guided: what did the diffusion step contribute?

Both conditions warp to bit-identical poses with bit-identical hole masks. The
only difference is whether disoccluded pixels receive diffusion content or stay
black. The column `warponly - guided` is therefore the diffusion step's
contribution, isolated.
"""
import glob
import json
import statistics as st
from collections import defaultdict

base, runs = {}, defaultdict(list)
for f in glob.glob("runs/truck_k10_*/results.json"):
    r = json.load(open(f))
    p = r["provenance"]
    if p.get("method") == "full":
        continue
    s, nf = p.get("seed"), p.get("n_synthetic", 0)
    if nf == 0:
        base[s] = r["metrics"]
    else:
        runs[(p.get("strategy"), nf)].append((s, r["metrics"]))


def delta(strat, nf, key="psnr"):
    ds = [m[key]["mean"] - base[s][key]["mean"]
          for s, m in runs.get((strat, nf), []) if s in base]
    if not ds:
        return None
    return st.mean(ds), (st.stdev(ds) if len(ds) > 1 else 0.0), len(ds)


print(f"k=10 baseline {st.mean([m['psnr']['mean'] for m in base.values()]):.2f} dB "
      f"({len(base)} seeds)\n")
print("           pose-guided        warp-only          diffusion")
print("ratio      (warp + SD)        (holes black)      contribution")
print("-" * 66)
for nf, lab in ((2, "20%"), (5, "50%"), (10, "100%"), (20, "200%")):
    g, w = delta("guided", nf), delta("warponly", nf)
    if not (g and w):
        continue
    # Diffusion's contribution is (with SD) - (without SD). Writing it the
    # other way round inverts the sign against the legend below.
    diff = g[0] - w[0]
    sig = "*" if abs(g[0]) > g[1] > 0 else " "
    sigw = "*" if abs(w[0]) > w[1] > 0 else " "
    print(f"{lab:>5s}    {g[0]:+7.3f} ± {g[1]:.3f}{sig}   "
          f"{w[0]:+7.3f} ± {w[1]:.3f}{sigw}   {diff:+7.3f}")

print("\nPositive 'diffusion contribution' = filling holes with SD was BETTER")
print("than leaving them black. Negative = the diffusion content actively hurt.")

print("\n--- SSIM / LPIPS agreement ---")
for key, better in (("ssim", "higher"), ("lpips", "lower")):
    print(f"\n{key} ({better} is better)")
    for nf, lab in ((2, "20%"), (5, "50%"), (10, "100%"), (20, "200%")):
        g, w = delta("guided", nf, key), delta("warponly", nf, key)
        if g and w:
            print(f"  {lab:>5s}  guided {g[0]:+.4f}   warponly {w[0]:+.4f}   "
                  f"diff {w[0]-g[0]:+.4f}")
