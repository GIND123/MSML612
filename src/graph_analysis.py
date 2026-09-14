"""Statistics for the graph experiments, done properly.

Two things the training script reports naively and this one fixes.

1. "% of the ceiling" invites over-reading. One-pass validity is a binomial
   proportion over n_gen samples, and at the ceilings involved (V* below 1% for
   the strongly constrained families) the sampling error is comparable to V*
   itself. A run scoring 0.93% against V* = 0.82% has NOT beaten an
   information-theoretic bound - it is one standard error away from it. What
   matters is whether the confidence interval CONTAINS V*, which is the
   statement "the model is at the limit".

2. Uniqueness as unique/valid is meaningless for small families. The perfect
   matchings on 6 nodes number exactly 15, so a model that recovers the whole
   family reports 15/4072 = 0.4% "uniqueness" and looks degenerate while being
   flawless. Coverage - distinct valid graphs as a fraction of |F| - is the
   correct metric here.

Also assembles the quality-vs-compute frontier, which is the comparison the
entropy-budget rule exists to win: at equal or lower NFE, does it hold more
validity than fixed-K?
"""
import glob, json, math, os, sys
from collections import defaultdict

import numpy as np

RUNS = os.environ.get("RUNS", "runs")


def wilson(k, n, z=1.96):
    """Wilson score interval - correct for proportions near 0, unlike normal
    approximation, which is exactly the regime V* lives in."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


runs = defaultdict(list)
for f in sorted(glob.glob(f"{RUNS}/gr1-*/result.json")):
    r = json.load(open(f))
    runs[(r["args"]["family"], r["args"]["mode"])].append(r)

if not runs:
    sys.exit("no gr1 results yet")

print("=" * 96)
print("ONE-PASS VALIDITY AGAINST THE EXACT CEILING V*")
print("=" * 96)
print("V* is the best any model can do at one pass: it already assumes perfect")
print("marginals. The question is whether the interval contains it.\n")
print(f"{'family':<13}{'TC':>6}{'V*':>9}  {'mode':<13}{'observed':>9}"
      f"{'95% CI':>20}{'verdict':>24}")

summary = {}
for (fam, mode) in sorted(runs):
    rs = runs[(fam, mode)]
    ex = rs[0]["exact"]
    V = ex["one_pass_ceiling"]
    n = sum(r["args"]["n_gen"] for r in rs)
    k = sum(r["decode"]["fixedK_1"]["validity"] * r["args"]["n_gen"] for r in rs)
    p = k / n
    lo, hi = wilson(k, n)
    if lo <= V <= hi:
        verdict = "at the limit"
    elif p > V:
        verdict = "ABOVE limit (check!)"
    else:
        verdict = "below limit (mode i)"
    summary[(fam, mode)] = dict(V=V, p=p, lo=lo, hi=hi, verdict=verdict)
    print(f"{fam:<13}{ex['total_correlation_bits']:>6.2f}{V*100:>8.3f}%  {mode:<13}"
          f"{p*100:>9.3f}%  [{lo*100:>5.3f},{hi*100:>5.3f}]%{verdict:>24}")

print()
print("=" * 96)
print("COVERAGE: distinct valid graphs as a fraction of the family")
print("=" * 96)
print("unique/valid is misleading here - the matching family has only 15 members,")
print("so recovering all of them shows up as 0.4% 'uniqueness'.\n")
print(f"{'family':<13}{'|F|':>7}  {'mode':<13}{'rule':<18}{'valid':>9}"
      f"{'distinct':>10}{'coverage':>10}")
for (fam, mode) in sorted(runs):
    rs = runs[(fam, mode)]
    M = rs[0]["exact"]["size"]
    for rule in ("fixedK_1", "fixedK_8", "entbudget_0.25"):
        if rule not in rs[0]["decode"]:
            continue
        val = np.mean([r["decode"][rule]["validity"] for r in rs])
        uq = np.mean([r["decode"][rule]["unique"] for r in rs])
        ngen = rs[0]["args"]["n_gen"]
        distinct = val * ngen * uq
        print(f"{fam:<13}{M:>7}  {mode:<13}{rule:<18}{val*100:>8.2f}%"
              f"{distinct:>10.0f}{min(1.0, distinct/M)*100:>9.1f}%")

print()
print("=" * 96)
print("QUALITY vs COMPUTE: does the entropy budget dominate fixed-K?")
print("=" * 96)
print("The rule exists to spend passes where the dependence actually is. It wins")
print("only if it holds more validity at the SAME or LOWER number of passes.\n")
for (fam, mode) in sorted(runs):
    rs = runs[(fam, mode)]
    pts = defaultdict(list)
    for r in rs:
        for k_, v in r["decode"].items():
            fam_kind = "entropy-budget" if k_.startswith("entbudget") else "fixed-K"
            pts[fam_kind].append((v["nfe"], v["validity"]))
    def curve(kind):
        # average validity across seeds at each distinct NFE, ordered by NFE
        out = []
        for n in sorted({p[0] for p in pts[kind]}):
            vs = [v for m, v in pts[kind] if m == n]
            out.append((float(n), float(np.mean(vs))))
        return out

    fk, eb = curve("fixed-K"), curve("entropy-budget")
    print(f"-- {fam} / {mode}")
    print(f"   {'fixed-K':<16}" + "  ".join(f"{n:.0f}->{v*100:.1f}%" for n, v in fk))
    print(f"   {'entropy-budget':<16}" + "  ".join(f"{n:.0f}->{v*100:.1f}%" for n, v in eb))
    # a dominating point: at most the NFE and strictly more validity
    wins = [(n, v) for n, v in eb
            if any(n <= fn and v > fv + 1e-9 for fn, fv in fk)]
    print(f"   entropy-budget points dominating some fixed-K point: "
          f"{[(int(n), round(v*100,1)) for n, v in wins] or 'none'}")

json.dump({f"{k[0]}/{k[1]}": v for k, v in summary.items()},
          open(os.path.join(os.environ.get("OUT", "figures"),
                            "graph_ceiling.json"), "w"), indent=2, default=float)
print("\nwrote graph_ceiling.json")
