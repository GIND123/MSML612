"""Board encoding for the public NCM Sudoku exports, plus scoring.

The exported graphs take a single `prefix` of 91 int64 tokens and return the
codes and tokens of all 12 levels. The encoding is not documented anywhere
except in the demo code that ships with the exports, so it is reproduced here
and checked against the shipped fixtures by `verify_export.py` before any
number from these models is trusted.

Layout: EDGE, then nine rows of nine cells separated by SEP, then EDGE.
A blank cell is 0 and a clue is its digit 1-9, which is exactly how our own
`test_puz.npy` stores a puzzle, so no conversion is needed on our side.
"""
import numpy as np

SEP, EDGE = 10, 11
GRID, CELLS = 9, 81
PREFIX_LEN = 1 + CELLS + (GRID - 1) + 1      # 91
LEVEL_LEN = CELLS + (GRID - 1)               # 89, one decoded level


def encode_board(cells):
    """81 cells (0 for blank) -> the 91-token prefix the export expects."""
    if len(cells) != CELLS:
        raise ValueError(f"expected {CELLS} cells, got {len(cells)}")
    out = [EDGE]
    for r in range(GRID):
        if r:
            out.append(SEP)
        out.extend(int(v) for v in cells[r * GRID:(r + 1) * GRID])
    out.append(EDGE)
    if len(out) != PREFIX_LEN:
        raise AssertionError(f"prefix is {len(out)} tokens, expected {PREFIX_LEN}")
    return out


def decode_cells(tokens):
    """Strip the separators from one level and return its 81 cells."""
    cells = [int(t) for t in tokens if t not in (SEP, EDGE)]
    if len(cells) != CELLS:
        raise ValueError(f"decoded {len(cells)} cells, expected {CELLS}")
    return cells


def levels_from_output(tokens):
    """Split a flat `tokens` output into one row per level."""
    flat = np.asarray(tokens).reshape(-1)
    if flat.size % LEVEL_LEN:
        raise ValueError(f"output of {flat.size} tokens is not a multiple of {LEVEL_LEN}")
    return flat.reshape(-1, LEVEL_LEN)


def is_legal(cells):
    """Every row, column and box holds each digit 1-9 exactly once.

    This is the validity-only metric the NCM paper reports as Valid Puzzle
    Rate. It is weaker than board accuracy: a legal board that ignores the
    clues still counts as valid under it.
    """
    g = np.asarray(cells, dtype=int).reshape(GRID, GRID)
    if g.min() < 1 or g.max() > 9:
        return False
    want = set(range(1, 10))
    if any(set(row) != want for row in g):
        return False
    if any(set(col) != want for col in g.T):
        return False
    for r in (0, 3, 6):
        for c in (0, 3, 6):
            if set(g[r:r + 3, c:c + 3].reshape(-1)) != want:
                return False
    return True


def keeps_clues(cells, puzzle):
    """The output agrees with every given clue."""
    return all(p == 0 or int(p) == int(c) for p, c in zip(puzzle, cells))


def score(cells, puzzle, solution):
    """The three numbers we report per puzzle.

    `board` is our metric: all 81 cells match the unique solution. `legal`
    and `clues` are reported beside it so the comparison with the NCM paper's
    validity-only metric is explicit rather than implied.
    """
    cells = [int(v) for v in cells]
    return {
        "board": all(int(s) == c for s, c in zip(solution, cells)),
        "legal": is_legal(cells),
        "clues": keeps_clues(cells, puzzle),
    }
