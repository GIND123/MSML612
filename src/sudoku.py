"""Sudoku as a constraint-satisfaction benchmark for masked diffusion.

Uses the SATNet dataset (Wang et al., ICML 2019), which is the standard 9x9
Sudoku benchmark for neural solvers: 10,000 puzzles with the conventional
9,000 / 1,000 train/test split, scored by BOARD-level accuracy - every one of the
81 cells correct, nothing partial. That metric is unambiguous and it is what the
published numbers report, so a comparison means something.

Why this task. Sudoku is the cleanest possible case of the regime this project
is about: the answer is UNIQUE, so there is no information-theoretic obstacle at
all, and the dependence between cells is enormous, since each cell is
constrained by twenty others through its row, column and box. Our framework says
the entire difficulty is therefore mode (i) - and, more sharply, that a decoder
which commits cells in order of certainty is performing classical constraint
propagation. Filling the forced cells first, recomputing, and repeating IS the
textbook algorithm, and it is exactly what an entropy-budgeted commit rule does
without being told to.

The prediction, registered before any run: fixed-K parallel decoding should fail
badly on this task while entropy-budgeted decoding should succeed, and the gap
should be far larger than anything seen on text, where dependence is diffuse.
"""
import os

import numpy as np

GRID = 9
CELLS = GRID * GRID          # 81
N_TRAIN, N_TEST = 9000, 1000  # the conventional SATNet split


class SudokuTokenizer:
    """Ten symbols: the digits 1-9 and the absorbing state.

    There is no separate 'blank' symbol. A blank cell is simply one that starts
    masked, which is what makes this a conditional masked-diffusion problem
    rather than a classification one: clues are observed context, blanks are the
    positions to fill.
    """

    def __init__(self):
        self.itos = [str(d) for d in range(1, 10)] + ["[MASK]"]
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.mask = self.stoi["[MASK]"]
        self.pad = self.mask

    def __len__(self):
        return len(self.itos)


def _onehot_to_int(t):
    """(N, 9, 9, 9) one-hot -> (N, 81) with 0 for an empty cell, 1-9 otherwise."""
    filled = t.sum(-1) > 0
    digits = t.argmax(-1) + 1
    return np.where(filled, digits, 0).reshape(len(t), CELLS).astype(np.int64)


def is_valid_solution(grid):
    """Does a completed 81-cell grid satisfy every Sudoku constraint?

    Checked rather than assumed: if the dataset's labels were not valid Sudoku
    the entire benchmark would be meaningless, and the failure would be silent.
    """
    g = np.asarray(grid).reshape(GRID, GRID)
    target = set(range(1, 10))
    for i in range(GRID):
        if set(g[i]) != target or set(g[:, i]) != target:
            return False
    for bi in range(0, GRID, 3):
        for bj in range(0, GRID, 3):
            if set(g[bi:bi + 3, bj:bj + 3].ravel()) != target:
                return False
    return True


def consistent_with_clues(pred, puzzle):
    """Does a prediction keep every given clue? (It should: clues are observed.)"""
    p, q = np.asarray(pred), np.asarray(puzzle)
    given = q > 0
    return bool((p[given] == q[given]).all())


def load_satnet(root="data/sudoku"):
    """Return (puzzles, solutions, tokenizer) as (N, 81) integer arrays.

    puzzles:   0 where the cell is blank, 1-9 where a clue is given
    solutions: 1-9 everywhere
    """
    import torch
    base = os.path.join(root, "sudoku")
    if not os.path.isdir(base):
        base = root
    fx = os.path.join(base, "features.pt")
    fy = os.path.join(base, "labels.pt")
    if not (os.path.exists(fx) and os.path.exists(fy)):
        raise FileNotFoundError(
            f"SATNet Sudoku not found under {root}. Fetch once with:\n"
            f"  mkdir -p {root} && cd {root} && curl -sLO https://powei.tw/sudoku.zip "
            f"&& unzip -q sudoku.zip")
    X = _onehot_to_int(torch.load(fx, map_location="cpu").numpy())
    Y = _onehot_to_int(torch.load(fy, map_location="cpu").numpy())
    return X, Y, SudokuTokenizer()


def split(X, Y):
    """The conventional 9,000 / 1,000 split, in dataset order."""
    return (X[:N_TRAIN], Y[:N_TRAIN]), (X[N_TRAIN:N_TRAIN + N_TEST],
                                        Y[N_TRAIN:N_TRAIN + N_TEST])


def board_accuracy(pred, gold):
    """Fraction of boards with ALL 81 cells correct - the published metric."""
    return float((np.asarray(pred) == np.asarray(gold)).all(axis=1).mean())


def cell_accuracy(pred, gold, blanks):
    """Per-cell accuracy over blanks only, reported alongside as context."""
    p, g, b = np.asarray(pred), np.asarray(gold), np.asarray(blanks)
    return float((p[b] == g[b]).mean())


if __name__ == "__main__":
    root = os.environ.get("SUDOKUROOT", "data/sudoku")
    X, Y, tok = load_satnet(root)
    (Xtr, Ytr), (Xte, Yte) = split(X, Y)
    print(f"puzzles {X.shape}  solutions {Y.shape}  vocab {len(tok)}")
    print(f"train {Xtr.shape[0]}  test {Xte.shape[0]}  (conventional split)")

    clues = (X > 0).sum(1)
    print(f"clues per puzzle: min={clues.min()} max={clues.max()} "
          f"mean={clues.mean():.1f}")

    # every label must be a valid Sudoku solution, or the benchmark is meaningless
    bad = sum(not is_valid_solution(Y[i]) for i in range(0, len(Y), 7))
    print(f"labels that are NOT valid Sudoku (sampled): {bad}  (must be 0)")

    # every clue must agree with its solution
    dis = sum(not consistent_with_clues(Y[i], X[i]) for i in range(0, len(X), 7))
    print(f"puzzles whose clues disagree with the label (sampled): {dis}  (must be 0)")

    # train and test must not overlap
    tr = {bytes(r) for r in Xtr[::3]}
    ov = sum(bytes(r) in tr for r in Xte)
    print(f"test puzzles also seen in the training sample: {ov}  (must be 0)")
