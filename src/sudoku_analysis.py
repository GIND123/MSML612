"""Sudoku results: the decoding comparison, and the benchmark comparison.

Two separate questions, kept separate because they have different standards of
evidence.

1. THE INTERNAL CLAIM. Does committing by certainty beat committing a fixed
   share per pass? This is a controlled comparison - same trained model, same
   test split, only the decoder changed - so it is clean regardless of how the
   absolute numbers land, and it is the claim the framework actually makes.

2. THE BENCHMARK COMPARISON. SATNet (Wang et al., ICML 2019) reports 98.3%
   board-level accuracy on this split. Our number is comparable only for the
   un-augmented arm, since SATNet did not use digit-relabelling augmentation.
   The augmented arm is reported separately and must not be quoted against them.

Board accuracy means all 81 cells correct. Cell accuracy over blanks and the
fraction of outputs that are valid Sudoku are reported alongside, because a
solver can score well per-cell while almost never completing a board.
"""
import glob, json, os, sys
from collections import defaultdict

import numpy as np

RUNS = os.environ.get("RUNS", "runs")
SATNET = 0.983

runs = defaultdict(list)
for f in sorted(glob.glob(f"{RUNS}/sud1-*/result.json")):
    r = json.load(open(f))
    a = r["args"]
    runs[(a["mode"], "aug" if a.get("augment") else "plain")].append(r)

if not runs:
    sys.exit("no sudoku results yet")


def rows(rs, prefix):
    out = {}
    for r in rs:
        for k, v in r["decode"].items():
            if k.startswith(prefix):
                out.setdefault(k, []).append(v)
    return out


print("=" * 88)
print("SUDOKU (SATNet benchmark, 1000-puzzle test split) — BOARD accuracy = all 81 cells")
print("=" * 88)

best_overall = {}
for key in sorted(runs):
    rs = runs[key]
    print(f"\n--- {key[0]} / {key[1]}   ({len(rs)} seeds)")
    print(f"    {'rule':<24}{'NFE':>7}{'BOARD':>10}{'cell':>9}{'valid':>9}")
    for prefix, label in (("fixedK_", "fixed-K"),
                          ("entbudget_", "entropy-budget"),
                          ("conf_", "confidence")):
        got = rows(rs, prefix)
        for k in sorted(got, key=lambda s: float(s.split("_")[1])):
            v = got[k]
            b = np.mean([x["board"] for x in v])
            print(f"    {k:<24}{np.mean([x['nfe'] for x in v]):>7.1f}"
                  f"{b*100:>9.2f}%{np.mean([x['cell'] for x in v])*100:>8.2f}%"
                  f"{np.mean([x['valid_sudoku'] for x in v])*100:>8.2f}%")
            fam = best_overall.setdefault((key, label), {"board": -1})
            if b > fam["board"]:
                best_overall[(key, label)] = {
                    "board": b, "rule": k,
                    "nfe": float(np.mean([x["nfe"] for x in v]))}

print()
print("=" * 88)
print("1. THE CONTROLLED CLAIM — same model, same data, only the decoder changes")
print("=" * 88)
print(f"{'arm':<22}{'fixed-K best':>16}{'entropy-budget':>17}{'gap':>10}{'conf best':>12}")
for key in sorted(runs):
    fk = best_overall.get((key, "fixed-K"))
    eb = best_overall.get((key, "entropy-budget"))
    cf = best_overall.get((key, "confidence"))
    if not (fk and eb):
        continue
    print(f"{key[0] + '/' + key[1]:<22}{fk['board']*100:>15.2f}%{eb['board']*100:>16.2f}%"
          f"{(eb['board']-fk['board'])*100:>+9.2f}"
          f"{(cf['board']*100 if cf else float('nan')):>11.2f}%")

print()
print("=" * 88)
print("2. THE BENCHMARK COMPARISON — like-for-like is the UN-AUGMENTED arm only")
print("=" * 88)
print(f"  SATNet (Wang et al., ICML 2019): {SATNET*100:.1f}% board accuracy\n")
for key in sorted(runs):
    eb = best_overall.get((key, "entropy-budget"))
    if not eb:
        continue
    tag = "comparable" if key[1] == "plain" else "NOT comparable (augmented)"
    delta = (eb["board"] - SATNET) * 100
    verdict = f"{delta:+.2f} pts vs SATNet" if key[1] == "plain" else "—"
    print(f"  {key[0] + '/' + key[1]:<20}{eb['board']*100:>7.2f}%  at {eb['nfe']:>5.1f} passes"
          f"   [{tag}]  {verdict}")

print()
print("Note: a decoder is only as good as the model it decodes. The controlled")
print("claim in section 1 holds regardless of whether section 2 clears 98.3%.")
