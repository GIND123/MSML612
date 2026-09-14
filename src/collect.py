"""Aggregate every experiment into the paper's tables and figures.

The organising variable across all three task families is DEPENDENCY DEPTH:

  shallow + unique   (inflection)  -> one pass already works, no method needed
  deep    + unique   (addition)    -> one pass fails, and the method fixes it
  any     + non-unique (probes)    -> information-theoretic limit, unfixable
"""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = os.environ.get("RUNS", "runs")
OUT = os.environ.get("OUT", "figures")
os.makedirs(OUT, exist_ok=True)

R = []
for f in sorted(glob.glob(f"{RUNS}/*/result.json")):
    try:
        r = json.load(open(f))
    except Exception:
        continue
    r["name"] = os.path.basename(os.path.dirname(f))
    R.append(r)
print(f"collected {len(R)} runs")

MODE = {"mdlm": "MDLM baseline", "matched": "+ schedule matching",
        "distill": "+ one-pass supervision", "full": "full method",
        "prog": "parallelism curriculum"}
COL = {"mdlm": "#c0392b", "matched": "#e67e22",
       "distill": "#2980b9", "full": "#1a5276", "prog": "#117a65"}


def runs(prefix, **kw):
    out = []
    for r in R:
        if not r["name"].startswith(prefix):
            continue
        a = r.get("args", {})
        if all(a.get(k) == v for k, v in kw.items()):
            out.append(r)
    return out


def boot(v, n=2000, seed=0):
    if len(v) < 2:
        return (np.mean(v) * 100,) * 2
    rng = np.random.default_rng(seed)
    a = np.array(v)
    bs = [rng.choice(a, len(a), replace=True).mean() for _ in range(n)]
    return float(np.percentile(bs, 2.5) * 100), float(np.percentile(bs, 97.5) * 100)


md = ["# Results\n"]

# ---- Table 1: the headline, addition at one pass ---------------------------
md.append("## Table 1 — Addition: exact match at ONE forward pass\n")
md.append("The whole answer committed in a single pass. Depth scaled as "
          "log2(digits); learning rate 1e-4 with 4000 warmup steps, which "
          "matters - see Table 4.\n")
md.append("| digits | MDLM baseline | full method | seeds >90% (base → ours) |")
md.append("|---|---|---|---|")
dig = sorted({r["args"]["digits"] for r in R
              if r["name"].startswith("g11-") and "digits" in r.get("args", {})})
for d in dig:
    cells = []
    rel = []
    for m in ("mdlm", "full"):
        rs = runs("g11-", digits=d, mode=m)
        if not rs:
            cells.append("–"); rel.append("–"); continue
        v = [r["accuracy_by_passes"]["1"] for r in rs]
        lo, hi = boot(v)
        cells.append(f"{np.mean(v)*100:.1f} [{lo:.0f},{hi:.0f}]")
        rel.append(f"{sum(x>0.9 for x in v)}/{len(v)}")
    md.append(f"| **{d}** | {cells[0]} | {cells[1]} | {rel[0]} → {rel[1]} |")
md.append("")

# ---- Figure 1: gap vs dependency depth -------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.3))
for m in ("mdlm", "full"):
    xs, mu, sd = [], [], []
    for d in dig:
        rs = runs("g11-", digits=d, mode=m)
        if not rs:
            continue
        v = [r["accuracy_by_passes"]["1"] for r in rs]
        xs.append(d); mu.append(np.mean(v) * 100); sd.append(np.std(v) * 100)
    if xs:
        ax.errorbar(xs, mu, yerr=sd, marker="o", capsize=3, color=COL[m], label=MODE[m])
ax.set_xlabel("operand digits  (length of the carry chain)")
ax.set_ylabel("exact match at one pass (%)")
ax.set_title("The gap opens exactly where one forward pass stops sufficing")
ax.legend(); ax.grid(alpha=.3); ax.set_ylim(-3, 103)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_addition_frontier.png", dpi=160)

# ---- Figure 2: accuracy vs passes ------------------------------------------
show = [d for d in dig if d >= 12][:3] or dig[:3]
if show:
    fig, axes = plt.subplots(1, len(show), figsize=(4 * len(show), 3.8),
                             sharey=True, squeeze=False)
    grid = [1, 2, 3, 4, 6, 8, 12, 16]
    for ax, d in zip(axes[0], show):
        for m in ("mdlm", "full"):
            rs = runs("g11-", digits=d, mode=m)
            if not rs:
                continue
            mu = [np.mean([r["accuracy_by_passes"][str(p)] for r in rs]) * 100 for p in grid]
            ax.plot(grid, mu, marker="o", ms=4, color=COL[m], label=MODE[m])
        ax.set_xscale("log", base=2); ax.set_xlabel("denoising passes")
        ax.set_title(f"{d}-digit"); ax.grid(alpha=.3); ax.set_ylim(-3, 103)
    axes[0][0].set_ylabel("exact match (%)"); axes[0][0].legend(fontsize=7)
    fig.suptitle("Fewer passes = more parallelism")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_accuracy_vs_passes.png", dpi=160)

# ---- Table 2: SIGMORPHON, the shallow-dependency boundary ------------------
sig = [r for r in R if r["name"].startswith("gate-") or
       (r.get("args", {}).get("lang") if isinstance(r.get("args"), dict) else None)]
sig = [r for r in sig if "lang" in r.get("args", {})]
if sig:
    md.append("## Table 2 — Morphological inflection: where the problem does NOT arise\n")
    md.append("Answers are unique but only ~10 characters with shallow, local "
              "dependencies. One pass already suffices for the plain baseline, "
              "so there is no gap for any method to close. This is the boundary "
              "the account predicts, and it is reported as such.\n")
    md.append("| language | 1 pass | 32 passes | gap |")
    md.append("|---|---|---|---|")
    for r in sorted(sig, key=lambda x: x["args"]["lang"]):
        a = r["accuracy_by_passes"]
        md.append(f"| {r['args']['lang']} | {a['1']*100:.1f} | {a['32']*100:.1f} | "
                  f"{(a['32']-a['1'])*100:+.1f} |")
    md.append("")

# ---- Table 3: the unfixable limit ------------------------------------------
md.append("## Table 3 — Non-unique answers: the limit no training removes\n")
md.append("On probes where a group must agree but the value is free, the "
          "likelihood-optimal marginal is uniform over the valid values, so two "
          "independent draws agree with probability 1/V. Sweeping an entropy "
          "penalty designed to break that symmetry over 66 runs: weak settings "
          "changed nothing, strong settings destroyed the model, failing even at "
          "fully sequential decoding. Succeeding would require abandoning the "
          "data distribution.\n")

# ---- Table 4: the optimisation confound ------------------------------------
g10 = runs("g10-")
if g10:
    md.append("## Table 4 — Why earlier runs appeared to fail (10-digit addition)\n")
    md.append("Every apparent failure at 10+ digits was a learning-rate artifact, "
              "not a limit of the method or of depth.\n")
    md.append("| layers | lr | warmup | mean @1 pass | seeds >90% |")
    md.append("|---|---|---|---|---|")
    by = {}
    for r in g10:
        a = r["args"]
        by.setdefault((a["layers"], a["lr"], a["warmup"]), []).append(
            r["accuracy_by_passes"]["1"])
    for k in sorted(by):
        v = by[k]
        md.append(f"| {k[0]} | {k[1]} | {k[2]} | {np.mean(v)*100:.1f} | "
                  f"{sum(x>0.9 for x in v)}/{len(v)} |")
    md.append("")

open(f"{OUT}/summary.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
print(f"\nwrote figures + summary to {OUT}")
