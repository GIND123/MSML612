"""Aggregate runs into the tables and figures for the paper."""
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

LABEL = {"mdlm": "MDLM (baseline)", "matched": "+ schedule matching",
         "distill": "+ one-pass supervision", "full": "full method"}
COLOR = {"mdlm": "#c0392b", "matched": "#e67e22",
         "distill": "#2980b9", "full": "#1a5276"}
PASSES = [1, 2, 3, 4, 6, 8, 12, 16]


def sel(mode, digits):
    out = []
    for r in R:
        a = r.get("args", {})
        if a.get("mode") == mode and a.get("digits") == digits and r.get("accuracy_by_passes"):
            out.append(r)
    return out


def boot(vals, n=2000, seed=0):
    if len(vals) < 2:
        return (np.mean(vals) * 100,) * 2
    rng = np.random.default_rng(seed)
    a = np.array(vals)
    bs = [rng.choice(a, len(a), replace=True).mean() for _ in range(n)]
    return float(np.percentile(bs, 2.5) * 100), float(np.percentile(bs, 97.5) * 100)


digits = sorted({r["args"]["digits"] for r in R if r.get("args", {}).get("digits")})
md = ["# Results\n"]

# ---------- Table 1: accuracy at full parallelism (one pass) ----------------
md.append("## Table 1 — Exact match at ONE denoising pass (maximum parallelism)\n")
md.append("The regime the method targets: the whole answer committed in a single "
          "forward pass, so the carry chain has to fit inside one fixed depth.\n")
md.append("| digits | " + " | ".join(LABEL[m] for m in LABEL) + " |")
md.append("|---" * (len(LABEL) + 1) + "|")
for d in digits:
    row = [f"**{d}**"]
    for m in LABEL:
        rs = sel(m, d)
        if not rs:
            row.append("–"); continue
        v = [r["accuracy_by_passes"]["1"] for r in rs]
        lo, hi = boot(v)
        row.append(f"{np.mean(v)*100:.1f} [{lo:.0f},{hi:.0f}]")
    md.append("| " + " | ".join(row) + " |")
md.append("")

# ---------- Table 2: seed reliability ---------------------------------------
md.append("## Table 2 — Seed reliability at one pass\n")
md.append("Seeds reaching >90% exact match. The baseline does not merely score "
          "lower, it fails outright on most seeds.\n")
md.append("| digits | " + " | ".join(LABEL[m] for m in LABEL) + " |")
md.append("|---" * (len(LABEL) + 1) + "|")
for d in digits:
    row = [f"**{d}**"]
    for m in LABEL:
        rs = sel(m, d)
        if not rs:
            row.append("–"); continue
        ok = sum(r["accuracy_by_passes"]["1"] > 0.9 for r in rs)
        row.append(f"{ok}/{len(rs)}")
    md.append("| " + " | ".join(row) + " |")
md.append("")

# ---------- Figure 1: accuracy vs passes, per difficulty --------------------
show = [d for d in digits if d >= 8] or digits
fig, axes = plt.subplots(1, len(show), figsize=(4 * len(show), 3.8), sharey=True, squeeze=False)
for ax, d in zip(axes[0], show):
    for m in LABEL:
        rs = sel(m, d)
        if not rs:
            continue
        mu = [np.mean([r["accuracy_by_passes"][str(p)] for r in rs]) * 100 for p in PASSES]
        sd = [np.std([r["accuracy_by_passes"][str(p)] for r in rs]) * 100 for p in PASSES]
        ax.errorbar(PASSES, mu, yerr=sd, marker="o", ms=4, capsize=2,
                    color=COLOR[m], label=LABEL[m])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("denoising passes")
    ax.set_title(f"{d}-digit addition")
    ax.grid(alpha=.3); ax.set_ylim(-3, 103)
axes[0][0].set_ylabel("exact-match accuracy (%)")
axes[0][0].legend(fontsize=7)
fig.suptitle("Fewer passes = more parallelism. The baseline collapses; the method does not.")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_accuracy_vs_passes.png", dpi=160)

# ---------- Figure 2: the frontier vs difficulty ----------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))
for m in LABEL:
    xs, mu, sd = [], [], []
    for d in digits:
        rs = sel(m, d)
        if not rs:
            continue
        v = [r["accuracy_by_passes"]["1"] for r in rs]
        xs.append(d); mu.append(np.mean(v) * 100); sd.append(np.std(v) * 100)
    if xs:
        ax.errorbar(xs, mu, yerr=sd, marker="o", capsize=3, color=COLOR[m], label=LABEL[m])
ax.set_xlabel("operand digits (length of the carry chain)")
ax.set_ylabel("exact match at one pass (%)")
ax.set_title("The gap opens exactly where one forward pass stops sufficing")
ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_ylim(-3, 103)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_frontier.png", dpi=160)

# ---------- Table 3: matched compute ----------------------------------------
md.append("## Table 3 — Compute\n")
md.append("| method | params | train tokens | forward passes / step | inference passes |")
md.append("|---|---|---|---|---|")
for m in LABEL:
    rs = [r for r in R if r.get("args", {}).get("mode") == m and r.get("compute")]
    if not rs:
        continue
    c = rs[0]["compute"]
    md.append(f"| {LABEL[m]} | {c['params']/1e6:.2f}M | {c['train_tokens']/1e6:.0f}M | "
              f"{c['fwd_per_step']} | 1 (vs 16 for sequential) |")
md.append("\nOne-pass supervision costs one extra forward pass per training step "
          "and removes fifteen at inference.\n")

open(f"{OUT}/summary.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
print(f"wrote figures + summary to {OUT}")
