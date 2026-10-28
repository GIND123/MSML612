"""Build test sets at a fixed clue count, for a like-for-like comparison.

Our own benchmark is made of *minimal* puzzles, so its clue count is whatever
minimality happens to leave (21-28, mean 24.3). The NCM paper instead defines
difficulty by a fixed clue count: 40 easy, 35 medium, 30 hard. Comparing a
model trained at 30 clues against a test set at 24 confounds architecture with
distribution shift, so this script generates sets at their clue counts using
the same solution sampler and the same uniqueness guarantee as our generator.

The difference from `src/gen_hard_sudoku.py` is only the stopping rule: cells
are removed until the target clue count is reached rather than until nothing
more can be removed. Uniqueness is still re-verified after every removal, so
board accuracy stays well defined.

    python ncm/make_clue_sets.py --clues 30 --n 1000 --out data/sudoku_c30

Pass --train_sol to assert the new set shares no solution with our training
split, including under digit relabelling.
"""
import argparse
import os
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from gen_hard_sudoku import CELLS, canon, count_solutions, random_solution  # noqa: E402


def make_one(seed, clues):
    """One puzzle with exactly `clues` givens, unique at every step.

    Returns (puzzle, solution) or None if this grid could not be thinned that
    far while staying unique, which the caller retries with another seed.
    """
    rng = np.random.default_rng(seed)
    sol = random_solution(rng)
    g = sol.copy()
    order = rng.permutation(CELLS)
    remaining = CELLS

    for i in order:
        if remaining <= clues:
            break
        keep = g[i]
        g[i] = 0
        if count_solutions(g, 2) == 1:
            remaining -= 1
        else:
            g[i] = keep

    return (g, sol) if remaining == clues else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clues", type=int, required=True, help="givens per puzzle, e.g. 30")
    ap.add_argument("--n", type=int, default=1000, help="puzzles to produce")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", help="file prefix, e.g. test -> test_puz.npy")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed0", type=int, default=0, help="first worker seed; change for a disjoint set")
    ap.add_argument("--train_sol", default="", help="training solutions to check against")
    a = ap.parse_args()

    if not 17 <= a.clues <= 60:
        raise SystemExit("--clues outside the sensible range 17-60")

    t0 = time.time()
    got, seed, batch = [], a.seed0, max(a.n, 64)
    with Pool(a.workers) as pool:
        while len(got) < a.n:
            seeds = range(seed, seed + batch)
            seed += batch
            for r in pool.map(partial(make_one, clues=a.clues), seeds, chunksize=4):
                if r is not None:
                    got.append(r)
            print(f"  {len(got)}/{a.n} after {time.time() - t0:.0f}s", flush=True)

    got = got[:a.n]
    P = np.stack([g for g, _ in got]).astype(np.uint8)
    S = np.stack([s for _, s in got]).astype(np.uint8)

    counts = (P > 0).sum(1)
    assert counts.min() == counts.max() == a.clues, f"clue counts drifted: {sorted(set(counts))}"

    # every puzzle must still admit exactly one solution
    bad = sum(count_solutions(P[i], 2) != 1 for i in range(0, len(P), 25))
    assert bad == 0, f"{bad} sampled puzzles are not unique"

    # the solutions themselves must be distinct, up to digit relabelling
    canon_sols = {canon(s) for s in S}
    print(f"distinct solutions up to relabelling: {len(canon_sols)}/{len(S)}")

    if a.train_sol:
        tr = {canon(s) for s in np.load(a.train_sol)}
        leak = sum(canon(s) in tr for s in S)
        print(f"solutions also in training (up to relabelling): {leak}")
        assert leak == 0, "leakage against the training split"

    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, f"{a.split}_puz.npy"), P)
    np.save(os.path.join(a.out, f"{a.split}_sol.npy"), S)
    print(f"saved {P.shape} puzzles at {a.clues} clues to {a.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
