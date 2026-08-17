"""The headline table: does augmentation help, and does that depend on how many
real views you already have?

Rows are synthetic ratio, columns are subset size. Reading a row left to right
shows how the value of a synthetic image changes as real data accumulates.
"""
import glob
import json
import statistics as st
from collections import defaultdict

RATIO_LABEL = {5: {1: 20, 2: 40, 5: 100, 10: 200},
               10: {2: 20, 5: 50, 10: 100, 20: 200},
               20: {5: 25, 10: 50, 20: 100, 40: 200}}
ORDER = [20, 50, 100, 200]          # nominal ratio buckets

base, runs = {}, defaultdict(list)
for f in glob.glob("runs/truck_k*/results.json"):
    r = json.load(open(f))
    p = r["provenance"]
    if p.get("method") == "full":
        continue
    k, s, nf = p.get("k"), p.get("seed"), p.get("n_synthetic", 0)
    if nf == 0:
        base[(k, s)] = r["metrics"]["psnr"]["mean"]
    else:
        runs[(k, p.get("strategy"), nf)].append((s, r["metrics"]["psnr"]["mean"]))


def cell(k, strat, ratio):
    for nf, lab in RATIO_LABEL[k].items():
        # k=5's second point is 40%, not 50% - 1.25 and 2.5 images are not
        # available, so the spec's ratios round to the nearest whole image.
        if abs(lab - ratio) <= 10 and (k, strat, nf) in runs:
            ds = [v - base[(k, s)] for s, v in runs[(k, strat, nf)] if (k, s) in base]
            if not ds:
                return "      -   "
            m = st.mean(ds)
            sd = st.stdev(ds) if len(ds) > 1 else 0.0
            return f"{m:+7.3f}{'*' if sd > 0 and abs(m) > sd else ' '} "
    return "      -   "


for strat in ("outpaint", "inpaint", "guided"):
    print(f"\n=== {strat} ===")
    print(f"{'ratio':>6s}   {'k=5':>10s} {'k=10':>10s} {'k=20':>10s}")
    for ratio in ORDER:
        print(f"{ratio:5d}%  " + "".join(cell(k, strat, ratio) for k in (5, 10, 20)))

print("\n\n=== baselines (real views only) ===")
for k in (5, 10, 20):
    vs = [v for (kk, s), v in base.items() if kk == k]
    print(f"  k={k:3d}  {st.mean(vs):6.2f} +- {st.stdev(vs) if len(vs) > 1 else 0:.2f} dB")
full = [json.load(open(f)) for f in glob.glob("runs/truck_k219*/results.json")]
if full:
    print(f"  k=219  {full[0]['metrics']['psnr']['mean']:6.2f} dB  (ceiling)")
print("\n* = |mean| exceeds the between-seed standard deviation")
