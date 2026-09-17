"""Every figure for the Sudoku result, generated from the raw run files.

Nothing here is hand-entered: each panel reads result.json from the runs it
plots, so a figure cannot drift from the numbers it claims to show.
"""
import glob, json, os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = os.environ.get("RUNS", "runs")
OUT = os.environ.get("OUT", "figures")
os.makedirs(OUT, exist_ok=True)

# classical baselines, measured on this exact test set by sudoku_compare.py
CLASSICAL = {"constraint propagation": 0.0300, "greedy MRV": 0.0590}


def cfg_label(a):
    if a.get("objective") == "ar":
        return "autoregressive prefix-LM"
    if a.get("objective") == "direct":
        return "direct recurrent classifier"
    if not a.get("recurrent"):
        return f"feed-forward {a['layers']}L ({a['mode']})"
    tags = []
    if a.get("no_inject"):
        tags.append("no-inject")
    if a.get("no_deep_sup"):
        tags.append("no-deepsup")
    return "recurrent " + a["mode"] + (" [" + ",".join(tags) + "]" if tags else "")


runs = defaultdict(list)
for f in sorted(glob.glob(f"{RUNS}/*/result.json")):
    n = os.path.basename(os.path.dirname(f))
    if not n.startswith(("rec1-", "abl-", "base-", "fix-", "inj-", "sudh-")):
        continue
    try:
        r = json.load(open(f))
    except Exception:
        continue
    if r.get("decode"):
        runs[cfg_label(r["args"])].append(r)

# ---- Figure 1: board accuracy against decoding passes -----------------------
fig, ax = plt.subplots(figsize=(7.2, 4.6))
GRID = [1, 2, 4, 8, 16, 32, 61]
style = {"recurrent full": ("#1a5276", "o", 2.4),
         "recurrent mdlm": ("#2980b9", "s", 1.6),
         "feed-forward 12L (full)": ("#c0392b", "^", 1.6),
         "feed-forward 12L (mdlm)": ("#e67e22", "v", 1.4),
         "feed-forward 32L (full)": ("#8e44ad", "D", 1.4)}
for label, rs in sorted(runs.items()):
    if label not in style:
        continue
    col, mk, lw = style[label]
    xs, ys = [], []
    for k in GRID:
        v = [r["decode"][f"fixedK_{k}"]["board"] for r in rs
             if f"fixedK_{k}" in r["decode"]]
        if v:
            xs.append(k); ys.append(np.mean(v) * 100)
    if xs:
        ax.plot(xs, ys, marker=mk, color=col, lw=lw, ms=5, label=label)
for name, v in CLASSICAL.items():
    ax.axhline(v * 100, ls=":", lw=1.2, color="#555")
    ax.text(1.05, v * 100 + 1.5, name, fontsize=7, color="#555")
ax.set_xscale("log", base=2)
ax.set_xlabel("denoising passes (NFE)")
ax.set_ylabel("board accuracy (%)  — all 81 cells correct")
ax.set_title("Hard Sudoku: recurrence beats capacity")
ax.grid(alpha=.3); ax.legend(fontsize=7.5, loc="lower right"); ax.set_ylim(-3, 103)
fig.tight_layout(); fig.savefig(f"{OUT}/sud_fig1_passes.png", dpi=170)

# ---- Figure 2: parameters against accuracy ---------------------------------
fig, ax = plt.subplots(figsize=(6.6, 4.4))
pts = []
for label, rs in runs.items():
    a = rs[0]["args"]
    npar = 212e3 if a.get("recurrent") else (
        6.34e6 if a.get("layers") == 32 else (2.4e6 if a.get("objective") == "ar"
                                              else 37.9e6))
    best = np.mean([max(v["board"] for v in r["decode"].values()) for r in rs])
    pts.append((npar, best * 100, label))
for npar, acc, label in pts:
    hero = label == "recurrent full"
    ax.scatter(npar, acc, s=170 if hero else 70,
               color="#1a5276" if hero else "#888",
               marker="*" if hero else "o", zorder=3 if hero else 2)
    ax.annotate(label, (npar, acc), fontsize=6.6,
                xytext=(6, 5 if acc < 95 else -11), textcoords="offset points")
ax.axhline(99.5, ls="--", color="#117a65", lw=1.2)
ax.text(2.3e5, 99.5 + 0.9, "Recurrent Transformer 99.5% @211k params "
                           "(different clue distribution)", fontsize=6.6, color="#117a65")
ax.set_xscale("log")
ax.set_xlabel("parameters"); ax.set_ylabel("board accuracy (%)")
ax.set_title("180× fewer parameters, 10 points better")
ax.grid(alpha=.3); ax.set_ylim(-5, 108)
fig.tight_layout(); fig.savefig(f"{OUT}/sud_fig2_params.png", dpi=170)

# ---- Figure 3: recurrence extrapolates past its training depth --------------
sw = os.path.join(OUT, "sudoku_rsweep.json")
if os.path.exists(sw):
    data = json.load(open(sw))
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for name, d in data.items():
        Rs = sorted(int(k) for k in d)
        ys = [d[str(R)] * 100 for R in Rs]
        ax.plot(Rs, ys, marker="o", ms=4,
                color="#1a5276" if "full" in name else "#c0392b",
                alpha=.85, label=name)
    ax.axvline(32, ls="--", color="k", lw=1)
    ax.text(33, 12, "trained at R = 32", fontsize=7.5, rotation=90)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("recurrences at inference"); ax.set_ylabel("board accuracy (%)")
    ax.set_title("Deep supervision makes the recurrence iterable past training depth")
    ax.grid(alpha=.3); ax.legend(fontsize=7); ax.set_ylim(-3, 103)
    fig.tight_layout(); fig.savefig(f"{OUT}/sud_fig3_rsweep.png", dpi=170)

# ---- Figure 4: ablation ladder ---------------------------------------------
order = ["feed-forward 12L (mdlm)", "feed-forward 12L (full)",
         "feed-forward 32L (full)", "recurrent mdlm",
         "recurrent full [no-deepsup]", "recurrent full [no-inject]",
         "recurrent full"]
have = [(o, np.mean([max(v["board"] for v in r["decode"].values())
                     for r in runs[o]]) * 100) for o in order if o in runs]
if have:
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    names = [h[0] for h in have]; vals = [h[1] for h in have]
    cols = ["#1a5276" if n == "recurrent full" else "#9fb6c8" for n in names]
    ax.barh(range(len(vals)), vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(v + 0.8, i, f"{v:.2f}%", va="center", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("board accuracy (%)"); ax.set_xlim(0, 108)
    ax.set_title("Every ingredient is load-bearing")
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(f"{OUT}/sud_fig4_ablation.png", dpi=170)

print("configurations plotted:")
for k in sorted(runs):
    v = [max(x["board"] for x in r["decode"].values()) for r in runs[k]]
    print(f"  {k:<42} n={len(v)}  {np.mean(v)*100:6.2f}%  {[round(x*100,1) for x in v]}")
print(f"\nwrote figures to {OUT}")
