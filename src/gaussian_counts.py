"""Gaussian count per condition: is the optimiser spending primitives on artifacts?"""
import json, glob, statistics as st
from collections import defaultdict

g, p = defaultdict(list), defaultdict(list)
for f in glob.glob("runs/truck_k10_*/results.json"):
    r = json.load(open(f))
    name = f.split("/")[1]
    strat = "baseline" if "fake0" in name else name.split("_")[4]
    fake = int(name.split("fake")[1])
    g[(strat, fake)].append(r["cost"]["n_gaussians"])
    p[(strat, fake)].append(r["metrics"]["psnr"]["mean"])

base = st.mean(g[("baseline", 0)])
print(f"{'condition':22s} {'Gaussians':>10s} {'vs base':>9s} {'PSNR':>7s}")
for k in sorted(g):
    m = st.mean(g[k])
    print(f"{k[0] + ' fake' + str(k[1]):22s} {m:10,.0f} "
          f"{100 * (m - base) / base:+8.0f}% {st.mean(p[k]):7.2f}")
