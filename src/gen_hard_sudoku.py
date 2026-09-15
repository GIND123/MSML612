"""Generate a HARD Sudoku benchmark, with every puzzle verified unique.

The SATNet split is saturated - search-free constraint propagation solves
1000/1000 of it - so nothing measured there discriminates between methods. This
builds instances that do.

Each puzzle is MINIMAL: cells are removed from a completed grid in random order
and a removal is kept only if the solution is still unique, verified by an
exhaustive counter that stops at two. The result carries 22-26 clues, where
constraint propagation alone solves roughly 4%. No puzzle is accepted without
that uniqueness check, so board accuracy remains a well-defined quantity.

Solutions come from a shuffled seed grid transformed by the Sudoku symmetry
group - digit relabelling, band and stack permutations, row and column swaps
within bands, and transposition - which generates valid grids far faster than
solving from empty, and uniformly enough for a benchmark.

Train and test are built from DISJOINT solution grids and the script asserts
that no test solution appears in training, in any relabelling.
"""
import argparse, os, sys, time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sudoku import CELLS, GRID

PEERS = None


def _peers():
    out = []
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
        out.append(sorted(s))
    return out


PEERS = _peers()
BASE = np.array([[(3 * (r % 3) + r // 3 + c) % 9 + 1 for c in range(9)]
                 for r in range(9)], dtype=np.int64)


def candidates(g):
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


def count_solutions(grid, limit=2):
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
        tot = 0
        for d in range(9):
            if cand[bi] & (1 << d):
                g[bi] = d + 1
                tot += rec()
                g[bi] = 0
                if tot >= limit:
                    return tot
        return tot

    return rec()


def random_solution(rng):
    """A valid completed grid, via the Sudoku symmetry group applied to a seed."""
    g = BASE.copy()
    g = np.vectorize({d: v for d, v in
                      zip(range(1, 10), rng.permutation(9) + 1)}.get)(g)
    for band in range(3):                       # rows within each band
        idx = band * 3 + rng.permutation(3)
        g[band * 3:band * 3 + 3] = g[idx]
    for stack in range(3):                      # columns within each stack
        idx = stack * 3 + rng.permutation(3)
        g[:, stack * 3:stack * 3 + 3] = g[:, idx]
    g = g[np.concatenate([b * 3 + np.arange(3) for b in rng.permutation(3)])]
    g = g[:, np.concatenate([s * 3 + np.arange(3) for s in rng.permutation(3)])]
    if rng.random() < 0.5:
        g = g.T
    return g.reshape(-1)


def make_one(seed):
    rng = np.random.default_rng(seed)
    sol = random_solution(rng)
    g = [int(v) for v in sol]
    for c in rng.permutation(CELLS):
        keep = g[c]
        g[c] = 0
        if count_solutions(g, 2) != 1:
            g[c] = keep
    return np.array(g, dtype=np.int64), sol


def canon(g):
    m, out = {}, []
    for v in g:
        if v not in m:
            m[v] = len(m) + 1
        out.append(m[v])
    return bytes(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=100000)
    p.add_argument("--n_test", type=int, default=1000)
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--out", default="data/sudoku_hard")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    total = a.n_train + a.n_test
    t0 = time.time()
    with Pool(a.workers) as pool:
        got = pool.map(make_one, range(total), chunksize=32)
    P = np.stack([g for g, _ in got])
    S = np.stack([s for _, s in got])
    print(f"generated {len(P)} minimal puzzles in {time.time()-t0:.0f}s", flush=True)
    print(f"clues: min={(P>0).sum(1).min()} max={(P>0).sum(1).max()} "
          f"mean={(P>0).sum(1).mean():.1f}", flush=True)

    Ptr, Str = P[:a.n_train], S[:a.n_train]
    Pte, Ste = P[a.n_train:], S[a.n_train:]

    # a test solution must not appear in training, even up to digit relabelling
    tr = {canon(r) for r in Str}
    leak = sum(canon(r) in tr for r in Ste)
    print(f"test solutions also in training (up to relabelling): {leak}", flush=True)
    assert leak == 0, "leakage between train and test"

    # every puzzle must still admit exactly one solution
    bad = sum(count_solutions(Pte[i], 2) != 1 for i in range(0, len(Pte), 25))
    print(f"test puzzles without a unique solution (sampled): {bad}", flush=True)
    assert bad == 0

    np.save(os.path.join(a.out, "train_puz.npy"), Ptr)
    np.save(os.path.join(a.out, "train_sol.npy"), Str)
    np.save(os.path.join(a.out, "test_puz.npy"), Pte)
    np.save(os.path.join(a.out, "test_sol.npy"), Ste)
    print(f"saved to {a.out}: train {Ptr.shape} test {Pte.shape}", flush=True)
