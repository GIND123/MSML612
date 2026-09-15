"""Generate a HARD Sudoku benchmark, with every puzzle verified unique.

The SATNet split is saturated - search-free constraint propagation solves
1000/1000 of it - so nothing measured there discriminates between methods. This
builds instances that do.

Each puzzle is MINIMAL: cells are removed from a completed grid in random order
and a removal is kept only if the solution is still unique, verified by an
exhaustive counter that stops at two. The result carries roughly 19-30 clues
(mean 24), where constraint propagation alone solves about 4%. No puzzle is accepted without
that uniqueness check, so board accuracy remains a well-defined quantity.

Solutions come from a randomised backtracking fill of an EMPTY grid, which
reaches the full space of roughly 6.7e21 valid grids.

An earlier version applied the Sudoku symmetry group to a single seed grid
instead. That was much faster and completely wrong: every grid it produced lay
in one equivalence class, whose size modulo digit relabelling is only about 3.4
million, so 101,000 draws collided constantly and 767 of 1000 test solutions
already appeared in training. The assertion below caught it. It is kept, and it
is the reason the slower fill is used.

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
    """A uniformly-ish random completed grid, by randomised backtracking fill.

    The previous version applied the Sudoku symmetry group to a single seed grid.
    Every grid it produced therefore lay in ONE equivalence class, whose size
    modulo digit relabelling is only about 3.4 million - so drawing 101,000
    samples collided constantly, and 767 of 1000 test solutions turned out to
    already appear in training. Filling an empty grid instead reaches the full
    space of roughly 6.7e21 valid grids, where collisions are impossible in
    practice.
    """
    g = [0] * CELLS
    while True:
        cand = candidates(g)
        if cand is None:
            return None
        best, bi = 10, -1
        for i in range(CELLS):
            if g[i] == 0:
                n = bin(cand[i]).count("1")
                if n < best:
                    best, bi = n, i
        if bi < 0:
            return np.array(g, dtype=np.int64)
        digits = [d + 1 for d in range(9) if cand[bi] & (1 << d)]
        rng.shuffle(digits)
        placed = False
        for d in digits:
            g[bi] = d
            if candidates(g) is not None:
                placed = True
                break
            g[bi] = 0
        if not placed:
            # dead end: restart rather than backtrack; restarts are rare and a
            # fresh fill is cheaper than unwinding a deep search
            g = [0] * CELLS


def make_one(seed):
    rng = np.random.default_rng(seed)
    sol = None
    while sol is None:
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
