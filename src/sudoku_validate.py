"""Is the Sudoku result meaningful, or is the benchmark simply saturated?

A neural solver scoring 100% on a test split proves nothing on its own. The
questions that decide whether the number means anything:

1. HOW HARD ARE THESE PUZZLES? If plain constraint propagation - no search, no
   backtracking, the first thing anyone writes - already solves every test
   puzzle, then the benchmark is trivial and 100% is the expected result for any
   competent method rather than evidence of anything. This is the test that
   matters most, and it is run first.

2. IS THE MARGIN STATISTICALLY REAL? 100% against a published 98.3% on 1000
   puzzles: exact binomial, not eyeballing.

3. DOES IT HOLD PER SEED? A mean of 100% could hide a split.

4. DOES IT GENERALISE TO HARDER PUZZLES? SATNet's puzzles carry 31-42 clues.
   Stripping clues while VERIFYING the solution stays unique gives strictly
   harder instances from the same distribution, and shows whether the model
   learned Sudoku or learned this clue-density.

Every puzzle generated for (4) is checked to have exactly one solution by an
exhaustive counter, so "accuracy" remains well defined.
"""
import argparse, glob, json, os, sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sudoku import GRID, CELLS, load_satnet, split

PEERS = None


def _build_peers():
    """For each cell, the 20 cells sharing its row, column or box."""
    peers = []
    for i in range(CELLS):
        r, c = divmod(i, GRID)
        br, bc = 3 * (r // 3), 3 * (c // 3)
        s = set()
        for k in range(GRID):
            s.add(r * GRID + k)
            s.add(k * GRID + c)
        for dr in range(3):
            for dc in range(3):
                s.add((br + dr) * GRID + bc + dc)
        s.discard(i)
        peers.append(sorted(s))
    return peers


def candidates(grid):
    """Per-cell candidate sets as 9-bit masks; None if a contradiction appears.

    Values are forced to Python ints: numpy integer scalars lack .bit_length(),
    and mixing the two silently produces numpy masks that then fail.
    """
    g = [int(v) for v in grid]
    cand = [0x1FF if v == 0 else (1 << (v - 1)) for v in g]
    for i, v in enumerate(g):
        if v:
            bit = 1 << (v - 1)
            for p in PEERS[i]:
                if cand[p] & bit:
                    cand[p] &= ~bit
                    if cand[p] == 0:
                        return None
    return cand


def propagate(grid):
    """Constraint propagation only: naked singles + hidden singles to fixpoint.

    This is the baseline every Sudoku solver starts from and it uses no search.
    If it solves the whole test split, the split is trivial.
    """
    g = [int(v) for v in grid]
    while True:
        cand = candidates(g)
        if cand is None:
            return None
        changed = False
        for i in range(CELLS):
            if g[i] == 0 and bin(cand[i]).count("1") == 1:
                g[i] = cand[i].bit_length()
                changed = True
        if not changed:
            # hidden singles: a digit with only one home in some unit
            for unit in UNITS:
                for d in range(9):
                    bit = 1 << d
                    spots = [i for i in unit if g[i] == 0 and cand[i] & bit]
                    if len(spots) == 1:
                        g[spots[0]] = d + 1
                        changed = True
        if not changed:
            return g


def _units():
    us = []
    for r in range(GRID):
        us.append([r * GRID + c for c in range(GRID)])
    for c in range(GRID):
        us.append([r * GRID + c for r in range(GRID)])
    for br in range(0, GRID, 3):
        for bc in range(0, GRID, 3):
            us.append([(br + dr) * GRID + bc + dc for dr in range(3) for dc in range(3)])
    return us


def count_solutions(grid, limit=2):
    """Exhaustive count of solutions, stopping at `limit`. Used for uniqueness."""
    g = [int(v) for v in grid]

    def rec():
        cand = candidates(g)
        if cand is None:
            return 0
        best, bi = 10, -1
        for i in range(CELLS):
            if g[i] == 0:
                n = bin(cand[i]).count("1")
                if n < best:
                    best, bi = n, i
                    if n == 1:
                        break
        if bi < 0:
            return 1
        total = 0
        for d in range(9):
            if cand[bi] & (1 << d):
                g[bi] = d + 1
                total += rec()
                g[bi] = 0
                if total >= limit:
                    return total
        return total

    return rec()


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def binom_p(k, n, p0):
    """One-sided exact binomial: P(X >= k) under H0: rate = p0."""
    from math import comb
    return float(sum(comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(k, n + 1)))


PEERS = _build_peers()
UNITS = _units()

p = argparse.ArgumentParser()
p.add_argument("--root", default="data/sudoku")
p.add_argument("--runs", default="runs")
p.add_argument("--n_hard", type=int, default=200)
p.add_argument("--out", default="figures/sudoku_validate.json")
a = p.parse_args()

X, Y, tok = load_satnet(a.root)
(Xtr, Ytr), (Xte, Yte) = split(X, Y)
out = {}

print("=" * 78)
print("1. HOW HARD IS THIS BENCHMARK? (the test that decides if 100% means anything)")
print("=" * 78)
cp_solved = 0
for i in range(len(Xte)):
    g = propagate(Xte[i])
    if g is not None and all(v > 0 for v in g) and np.array_equal(g, Yte[i]):
        cp_solved += 1
cp_rate = cp_solved / len(Xte)
out["classical_propagation"] = cp_rate
print(f"  constraint propagation alone, NO search: {cp_solved}/{len(Xte)} = {cp_rate*100:.2f}%")
print(f"  clues per test puzzle: min={(Xte>0).sum(1).min()} "
      f"max={(Xte>0).sum(1).max()} mean={(Xte>0).sum(1).mean():.1f}")
if cp_rate > 0.99:
    print("  >>> The split is SATURATED: a search-free classical solver already")
    print("  >>> solves it. 100% is the expected result for any competent method,")
    print("  >>> and the neural number should NOT be presented as a capability claim.")
else:
    print(f"  >>> Not saturated: propagation alone leaves {(1-cp_rate)*100:.1f}% unsolved,")
    print("  >>> so the benchmark does discriminate between methods.")

print()
print("=" * 78)
print("2. IS THE MARGIN STATISTICALLY REAL?")
print("=" * 78)
runs = defaultdict(list)
for f in sorted(glob.glob(os.path.join(a.runs, "sud1-*/result.json"))):
    r = json.load(open(f))
    arg = r["args"]
    runs[(arg["mode"], "aug" if arg.get("augment") else "plain")].append(r)

for key in sorted(runs):
    rs = runs[key]
    best = max((max(r["decode"].values(), key=lambda v: v["board"]) for r in rs),
               key=lambda v: v["board"])
    per_seed = [max(r["decode"].values(), key=lambda v: v["board"])["board"] for r in rs]
    k = int(round(np.mean(per_seed) * len(Xte)))
    lo, hi = wilson(k, len(Xte))
    pv = binom_p(k, len(Xte), 0.983)
    print(f"  {key[0]}/{key[1]:<6} best {np.mean(per_seed)*100:6.2f}%  "
          f"95% CI [{lo*100:.2f}, {hi*100:.2f}]  per-seed {[round(x*100,2) for x in per_seed]}"
          f"  p(vs 98.3%)={pv:.2g}")
    out[f"{key[0]}/{key[1]}"] = {"mean": float(np.mean(per_seed)),
                                 "per_seed": per_seed, "ci": [lo, hi], "p_vs_satnet": pv}

print()
print("=" * 78)
print(f"3. DOES IT GENERALISE TO HARDER PUZZLES? (stripping clues, uniqueness verified)")
print("=" * 78)
rng = np.random.default_rng(0)
hard = {}
for target in (30, 26, 22):
    made, tries = [], 0
    for idx in range(len(Xte)):
        if len(made) >= a.n_hard or tries > a.n_hard * 12:
            break
        tries += 1
        sol = Yte[idx]
        cells = list(rng.permutation(CELLS))
        g = list(sol)
        removed = 0
        for c in cells:
            if CELLS - removed <= target:
                break
            keep = g[c]
            g[c] = 0
            if count_solutions(g, 2) == 1:
                removed += 1
            else:
                g[c] = keep
        if sum(1 for v in g if v) <= target + 2:
            made.append((np.array(g), sol))
    if made:
        P = np.stack([m[0] for m in made])
        S = np.stack([m[1] for m in made])
        cp = sum(1 for i in range(len(P))
                 if (lambda r: r is not None and all(v > 0 for v in r)
                     and np.array_equal(r, S[i]))(propagate(P[i])))
        hard[target] = {"n": len(P), "clues": float((P > 0).sum(1).mean()),
                        "classical_propagation": cp / len(P)}
        print(f"  target ~{target} clues: built {len(P)} unique puzzles "
              f"(mean {(P>0).sum(1).mean():.1f} clues)   "
              f"constraint propagation solves {cp/len(P)*100:.1f}%")
        np.save(os.path.join(os.path.dirname(a.out) or ".", f"hard_{target}_puz.npy"), P)
        np.save(os.path.join(os.path.dirname(a.out) or ".", f"hard_{target}_sol.npy"), S)
out["harder"] = hard

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(out, open(a.out, "w"), indent=2, default=float)
print(f"\nwrote {a.out}")
print("harder puzzle sets saved next to it for the model to be evaluated on")
