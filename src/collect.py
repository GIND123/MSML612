"""Aggregate every experiment into the paper's tables and figures.

The organising variable across all three task families is DEPENDENCY DEPTH:

  shallow + unique   (inflection)  -> one pass already works, no method needed
  deep    + unique   (addition)    -> one pass fails, and the method fixes it
  any     + non-unique (probes)    -> information-theoretic limit, unfixable
"""
import json, glob, math, os
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


# ---- Table 5: the definitive frontier, one configuration throughout ---------
# Grid 11 mixed configurations across digit counts, which makes the curve a
# statement about the configurations as much as about the method. Grid 16 reruns
# every length at a single setting so the frontier means one thing.
g16 = runs("g16-")
if g16:
    md.append("## Table 5 — Addition frontier at ONE fixed configuration\n")
    md.append("12 layers, 150k steps, lr 1e-4, 3 seeds, identical at every "
              "length, so the curve reflects the method and not the config.\n")
    md.append("| digits | MDLM baseline | matched (ours) | seeds >90% |")
    md.append("|---|---|---|---|")
    for d in sorted({r["args"]["digits"] for r in g16}):
        cells, rel = [], []
        for m in ("mdlm", "full"):
            rs = runs("g16-", digits=d, mode=m)
            if not rs:
                cells.append("–"); rel.append("–"); continue
            v = [r["accuracy_by_passes"]["1"] for r in rs]
            lo, hi = boot(v)
            cells.append(f"{np.mean(v)*100:.1f} [{lo:.0f},{hi:.0f}]")
            rel.append(f"{sum(x>0.9 for x in v)}/{len(v)}")
        md.append(f"| **{d}** | {cells[0]} | {cells[1]} | {rel[0]} → {rel[1]} |")
    md.append("")

# ---- Table 6: the depth sweep, the theory's falsifiable prediction ----------
g14 = runs("g14-")
if g14:
    md.append("## Table 6 — Minimum depth for one-pass addition\n")
    md.append("The account in THEORY.md predicts minimum depth growing as "
              "log2(n), since the carry is a prefix scan. A ripple-carry account "
              "predicts linear growth; an impossibility account predicts no "
              "depth suffices.\n")
    lays = sorted({r["args"]["layers"] for r in g14})
    md.append("| digits | " + " | ".join(f"{l}L" for l in lays) +
              " | log2(n) | min depth >90% |")
    md.append("|---" * (len(lays) + 3) + "|")
    for d in sorted({r["args"]["digits"] for r in g14}):
        cells, mind = [], None
        for l in lays:
            rs = runs("g14-", digits=d, layers=l)
            if not rs:
                cells.append("–"); continue
            mu = np.mean([r["accuracy_by_passes"]["1"] for r in rs]) * 100
            cells.append(f"{mu:.0f}")
            if mu > 90 and mind is None:
                mind = l
        md.append(f"| **{d}** | " + " | ".join(cells) +
                  f" | {math.log2(d):.1f} | {mind or '–'} |")
    md.append("")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ds = sorted({r["args"]["digits"] for r in g14})
    obs = []
    for d in ds:
        mind = None
        for l in lays:
            rs = runs("g14-", digits=d, layers=l)
            if rs and np.mean([r["accuracy_by_passes"]["1"] for r in rs]) > 0.9:
                mind = l; break
        obs.append(mind)
    ok = [(d, m) for d, m in zip(ds, obs) if m]
    if ok:
        ax.plot([d for d, _ in ok], [m for _, m in ok], "o-", color="#1a5276",
                label="measured minimum depth")
    ax.plot(ds, [math.log2(d) for d in ds], "--", color="#117a65",
            label="log2(n)  — prefix-scan prediction")
    ax.plot(ds, [d / 4 for d in ds], ":", color="#c0392b",
            label="n/4  — ripple-carry prediction")
    ax.set_xlabel("operand digits"); ax.set_ylabel("layers needed for >90% at one pass")
    ax.set_title("Does required depth grow like log n or like n?")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_depth_scaling.png", dpi=160)

# ---- Table 7: component ablation on inflection -----------------------------
g15 = runs("g15-")
if g15:
    md.append("## Table 7 — Which component carries the gain (inflection)\n")
    modes = ["mdlm", "matched", "distill", "full"]
    langs = sorted({r["args"]["lang"] for r in g15})
    md.append("| language | " + " | ".join(modes) + " | (gap 1→32) |")
    md.append("|---" * (len(modes) + 2) + "|")
    for L in langs:
        cells, gaps = [], []
        for m in modes:
            rs = runs("g15-", lang=L, mode=m)
            if not rs:
                cells.append("–"); gaps.append("–"); continue
            one = np.mean([r["accuracy_by_passes"]["1"] for r in rs]) * 100
            many = np.mean([r["accuracy_by_passes"]["32"] for r in rs]) * 100
            cells.append(f"{one:.1f}"); gaps.append(f"{many-one:+.1f}")
        md.append(f"| {L} | " + " | ".join(cells) + " | " + " / ".join(gaps) + " |")
    md.append("")

# ---- Table 8: text8, where the two fixes have to compose -------------------
t8 = [r for r in R if r["name"].startswith("t8-")]
if t8:
    md.append("## Table 8 — text8: quality against compute\n")
    md.append("Generative perplexity under a from-scratch autoregressive "
              "character evaluator, against the number of function evaluations "
              "actually used. Entropy and distinct-4-gram rate are reported "
              "beside it because generative perplexity alone is gamed by "
              "degenerate repetition; the real-text reference row is the "
              "target, not zero.\n")
    ref = t8[0].get("reference", {})
    md.append(f"Real text8 reference — ppl {ref.get('gen_ppl', float('nan')):.2f}, "
              f"entropy {ref.get('entropy', float('nan')):.3f}, "
              f"distinct-4 {ref.get('distinct4', float('nan')):.3f}\n")
    md.append("| training | decoding | NFE | gen ppl | entropy | distinct-4 | valid BPC |")
    md.append("|---|---|---|---|---|---|---|")
    by = {}
    for r in t8:
        by.setdefault(r["args"]["mode"], []).append(r)
    for mode in ("mdlm", "matched", "full"):
        for r in by.get(mode, [])[:1]:
            for key, row in sorted(r.get("decode", {}).items()):
                if row.get("gen_ppl") is None:
                    continue
                md.append(f"| {mode} | {key} | {row['nfe']:.1f} | "
                          f"{row['gen_ppl']:.2f} | {row['entropy']:.3f} | "
                          f"{row['distinct4']:.3f} | {r.get('valid_bpc', float('nan')):.4f} |")
    md.append("")

    # quality/compute frontier: the plot the claim lives or dies on
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    style = {"mdlm": ("#c0392b", "uniform-t training"),
             "matched": ("#e67e22", "matched training"),
             "full": ("#1a5276", "matched + high-ratio")}
    for mode, rs in by.items():
        c, lbl = style.get(mode, ("#555", mode))
        for fam, mk, ls in (("fixedK_", "o", "-"), ("entbudget_", "s", "--")):
            pts = []
            for r in rs:
                for k, row in r.get("decode", {}).items():
                    if k.startswith(fam) and row.get("gen_ppl"):
                        pts.append((row["nfe"], row["gen_ppl"]))
            if pts:
                pts.sort()
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk,
                        ms=4, ls=ls, color=c, alpha=.85,
                        label=f"{lbl}, {'fixed-K' if fam[0]=='f' else 'entropy budget'}")
    if ref.get("gen_ppl"):
        ax.axhline(ref["gen_ppl"], color="k", lw=1, ls=":", label="real text8")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("forward passes (NFE)"); ax.set_ylabel("generative perplexity")
    ax.set_title("Quality against compute: do the two fixes compose?")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_text8_quality_vs_compute.png", dpi=160)

# ---- Table 9: failed to parallelise, or failed to learn? -------------------
sc = os.path.join(OUT, "seq_check_g13.json")
if os.path.exists(sc):
    rows = json.load(open(sc))
    md.append("## Table 9 — Did the baseline fail to parallelise, or fail to learn?\n")
    md.append("A 20-digit answer occupies 21 slots, so the standard 16-pass grid "
              "is not a sequential decode. Re-decoded at one token per pass, the "
              "two explanations separate.\n")
    md.append("| digits | mode | @1 pass | fully sequential | reading |")
    md.append("|---|---|---|---|---|")
    agg = {}
    for r in rows:
        agg.setdefault((r["digits"], r["mode"]), []).append(r)
    for k in sorted(agg):
        v = agg[k]
        one = np.mean([x["acc_1pass"] for x in v]) * 100
        seq = np.mean([x["acc_sequential"] for x in v]) * 100
        read = ("never learned the task" if seq < 5 else
                "learned it, cannot parallelise" if one < seq - 5 else
                "learned it and parallelises")
        md.append(f"| {k[0]} | {k[1]} | {one:.1f} | {seq:.1f} | {read} |")
    md.append("")

open(f"{OUT}/summary.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
print(f"\nwrote figures + summary to {OUT}")
