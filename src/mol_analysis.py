"""QM9 results as a quality-vs-compute frontier.

The number that matters on this benchmark is not validity at some fixed step
count but **how few denoising steps are needed to reach a given validity**. That
is the axis the discrete graph-diffusion literature competes on, and it is the
one determined by the training schedule and the decoding rule rather than by
model scale.

Reported here as a speed-up over the standard formulation. The `mdlm` + `rope`
arm IS the conventional setup - uniform-t discrete diffusion with the rotary
encoding that modern implementations default to - so the comparison is
controlled: same data, same model size, same training budget, same decoder, one
factor changed at a time.

A note on external comparison. Published DiGress-style numbers come from larger
models trained longer and with auxiliary spectral features that neither arm here
uses, so quoting them beside these would not be like-for-like. The claim
supported by this script is the controlled one: at matched scale and budget, how
many steps does each formulation need. Any external number should be taken from
the paper itself rather than from memory.
"""
import glob, json, os, sys
from collections import defaultdict

import numpy as np

RUNS = os.environ.get("RUNS", "runs")

runs = defaultdict(list)
for f in sorted(glob.glob(f"{RUNS}/mol1-*/result.json")):
    r = json.load(open(f))
    a = r["args"]
    runs[(a["mode"], a["pe"])].append(r)

if not runs:
    sys.exit("no QM9 results yet")

LABEL = {("mdlm", "rope"): "uniform-t + relative  (standard)",
         ("mdlm", "ape"): "uniform-t + absolute",
         ("matched", "rope"): "matched + relative",
         ("matched", "ape"): "matched + absolute  (ours)"}


def curve(rs):
    """(steps, validity, unique, novel) for every fixed-K setting, by step count."""
    pts = defaultdict(list)
    for r in rs:
        for k, v in r["decode"].items():
            if k.startswith("fixedK_"):
                pts[int(k.split("_")[1])].append(v)
    out = []
    for k in sorted(pts):
        vs = pts[k]
        out.append((k,
                    float(np.mean([x["validity"] for x in vs])),
                    float(np.mean([x["unique"] for x in vs])),
                    float(np.mean([x["novel"] for x in vs]))))
    return out


def steps_for(c, target):
    """Fewest denoising steps reaching `target` validity, or None."""
    for k, v, _, _ in c:
        if v >= target:
            return k
    return None


print("=" * 84)
print("QM9 — validity against number of denoising steps")
print("=" * 84)
for key in [("mdlm", "rope"), ("mdlm", "ape"), ("matched", "rope"), ("matched", "ape")]:
    rs = runs.get(key)
    if not rs:
        continue
    c = curve(rs)
    print(f"\n{LABEL[key]}   ({len(rs)} seeds)")
    print(f"  {'steps':>7}{'validity':>11}{'unique':>9}{'novel':>9}")
    for k, v, u, n in c:
        print(f"  {k:>7}{v*100:>10.2f}%{u*100:>8.1f}%{n*100:>8.1f}%")

print()
print("=" * 84)
print("STEPS NEEDED TO REACH A VALIDITY TARGET  (fewer is better)")
print("=" * 84)
hdr = "".join(f"{str(int(t*100)) + '%':>9}" for t in (0.5, 0.8, 0.9, 0.95))
print(f"{'formulation':<36}" + hdr)
base = runs.get(("mdlm", "rope"))
base_c = curve(base) if base else None
for key in [("mdlm", "rope"), ("mdlm", "ape"), ("matched", "rope"), ("matched", "ape")]:
    rs = runs.get(key)
    if not rs:
        continue
    c = curve(rs)
    cells = []
    for t in (0.5, 0.8, 0.9, 0.95):
        s = steps_for(c, t)
        cells.append(f"{s:>9}" if s else f"{'—':>9}")
    print(f"{LABEL[key]:<36}" + "".join(cells))

if base_c:
    print()
    print("SPEED-UP over the standard formulation, at equal validity:")
    for key in [("mdlm", "ape"), ("matched", "rope"), ("matched", "ape")]:
        rs = runs.get(key)
        if not rs:
            continue
        c = curve(rs)
        for t in (0.8, 0.9, 0.95):
            b, o = steps_for(base_c, t), steps_for(c, t)
            if b and o:
                print(f"  {LABEL[key]:<34} at {int(t*100)}% validity: "
                      f"{b} steps -> {o} steps  ({b/o:.1f}x fewer)")
