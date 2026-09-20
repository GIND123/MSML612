"""The matched R-MDM comparison: what input injection is worth, and where.

Reads figures/rmdm_compare.json, written by eval_ckpt.py from the step-80,000
checkpoints of three runs that differ in exactly one component. Nothing here is
hand-entered.
"""
import json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.environ.get("OUT", "figures")
src = os.path.join(OUT, "rmdm_compare.json")
if not os.path.exists(src):
    sys.exit(f"missing {src}")
res = json.load(open(src))

Ks = [1, 4, 8, 16, 32, 61]
off = [r for r in res.values() if not r["inject"]]
on = [r for r in res.values() if r["inject"]]


def curve(rs):
    """Mean board accuracy at each budget, plus the per-seed spread."""
    m = [np.mean([r["decode"][str(k)]["board"] for r in rs]) * 100 for k in Ks]
    lo = [np.min([r["decode"][str(k)]["board"] for r in rs]) * 100 for k in Ks]
    hi = [np.max([r["decode"][str(k)]["board"] for r in rs]) * 100 for k in Ks]
    return np.array(m), np.array(lo), np.array(hi)


fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.3))

for rs, lab, col in ((off, f"R-MDM faithful, injection OFF (n={len(off)})", "#c0392b"),
                     (on, f"+ input injection (n={len(on)})", "#1a5276")):
    if not rs:
        continue
    m, lo, hi = curve(rs)
    ax.plot(Ks, m, "o-", color=col, lw=2, ms=5.5, label=lab)
    if len(rs) > 1:
        ax.fill_between(Ks, lo, hi, color=col, alpha=.15, lw=0)
ax.set_xscale("log", base=2)
ax.set_xlabel("denoising passes (NFE)")
ax.set_ylabel("board accuracy (%)")
ax.set_title("Matched comparison: identical architecture,\n"
             "step embedding on both sides, 80k steps")
ax.text(1.03, 12, "shaded = 2 seeds of the same\nconfiguration, 67 points apart",
        fontsize=6.6, color="#c0392b")
ax.grid(alpha=.3); ax.legend(fontsize=7.5, loc="lower right")

# --- is the injection-ON seed even outside the injection-OFF seed range? -----
# With one seed on the ON side, the only defensible question is whether it lands
# outside the range spanned by the two OFF seeds. At five of six budgets it does
# not, so there is nothing to claim there.
if off and on:
    mo, lo, hi = curve(off)
    mn, _, _ = curve(on)
    x = np.arange(len(Ks))
    ax2.vlines(x, lo, hi, color="#c0392b", lw=9, alpha=.32,
               label="injection OFF: range of 2 seeds")
    ax2.plot(x, mo, "_", color="#c0392b", ms=17, mew=2.4, label="injection OFF: mean")
    inside = (mn >= lo) & (mn <= hi)
    ax2.scatter(x[inside], mn[inside], s=64, color="#7f8c8d", zorder=4,
                marker="o", label="injection ON (n=1): inside the OFF range")
    ax2.scatter(x[~inside], mn[~inside], s=110, color="#1a5276", zorder=4,
                marker="*", label="injection ON (n=1): outside")
    for i in range(len(Ks)):
        ax2.annotate("inside" if inside[i] else "outside", (x[i], mn[i]),
                     fontsize=6.4, xytext=(0, 9 if not inside[i] else -14),
                     textcoords="offset points", ha="center",
                     color="#1a5276" if not inside[i] else "#666")
    ax2.set_xticks(x); ax2.set_xticklabels(Ks)
    ax2.set_xlabel("denoising passes (NFE)")
    ax2.set_ylabel("board accuracy (%)")
    ax2.set_title("At 5 of 6 budgets the injection-ON run lands\n"
                  "inside the spread of two injection-OFF seeds", fontsize=10)
    ax2.grid(alpha=.3, axis="y"); ax2.legend(fontsize=6.3, loc="lower right")
    ax2.set_ylim(0, 104)

fig.tight_layout()
fig.savefig(f"{OUT}/sud_fig9_rmdm.png", dpi=170)
print(f"wrote {OUT}/sud_fig9_rmdm.png")

print("\nmatched R-MDM comparison (step 80,000):")
print(f"  {'NFE':>5}{'inj OFF':>10}{'inj ON':>10}{'delta':>9}")
for i, k in enumerate(Ks):
    mo = np.mean([r["decode"][str(k)]["board"] for r in off]) * 100
    mn = np.mean([r["decode"][str(k)]["board"] for r in on]) * 100
    print(f"  {k:>5}{mo:>9.2f}%{mn:>9.2f}%{mn-mo:>+8.2f}")
bo = np.mean([r["best"] for r in off]) * 100
bn = np.mean([r["best"] for r in on]) * 100
print(f"  {'best':>5}{bo:>9.2f}%{bn:>9.2f}%{bn-bo:>+8.2f}")
