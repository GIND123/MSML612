"""Evaluate a published NCM Sudoku checkpoint on one of our test sets.

This scores somebody else's trained model on our benchmark. It is a baseline
measurement, not part of our own system: no weights from here are used by, or
transferred into, any model we train.

The checkpoints were trained on the S-FLM clue counts (40 easy, 35 medium, 30
hard). Our generated benchmark sits at 21-28 clues, so running them on it is
out of distribution for them and the result must be reported that way. Use
`make_clue_sets.py` to build matched-difficulty sets for a fair curve.

    python ncm/run_ncm.py sudoku-hard --data data/sudoku_hard --out results/ncm_hard.json

Results are written as JSON so `compare.py` can tabulate several runs without
re-running inference.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ncm_encoding import encode_board, decode_cells, levels_from_output, score
from verify_export import REPO, load_session


def load_split(data_dir, split):
    d = Path(data_dir)
    puz = np.load(d / f"{split}_puz.npy")
    sol = np.load(d / f"{split}_sol.npy")
    if puz.shape != sol.shape:
        raise ValueError(f"{puz.shape} puzzles but {sol.shape} solutions")
    return puz, sol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["sudoku-easy", "sudoku-medium", "sudoku-hard"])
    ap.add_argument("--data", required=True, help="directory holding <split>_puz.npy / <split>_sol.npy")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N puzzles")
    ap.add_argument("--out", default="", help="where to write the JSON result")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--every", type=int, default=100, help="progress interval")
    a = ap.parse_args()

    puz, sol = load_split(a.data, a.split)
    if a.limit:
        puz, sol = puz[:a.limit], sol[:a.limit]
    sess, meta = load_session(a.task, a.repo)
    name = sess.get_inputs()[0].name

    n = len(puz)
    tally = {"board": 0, "legal": 0, "clues": 0}
    by_clue = defaultdict(lambda: {"n": 0, "board": 0})
    t0 = time.time()

    for i in range(n):
        board = puz[i].reshape(-1)
        prefix = np.asarray(encode_board(board.tolist()), dtype=np.int64).reshape(1, -1)
        outs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {name: prefix})))
        final = decode_cells(levels_from_output(outs["tokens"])[-1])

        s = score(final, board.tolist(), sol[i].reshape(-1).tolist())
        for k in tally:
            tally[k] += int(s[k])
        clues = int((board > 0).sum())
        by_clue[clues]["n"] += 1
        by_clue[clues]["board"] += int(s["board"])

        if a.every and (i + 1) % a.every == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1}/{n}  board {tally['board'] / (i + 1):.1%}  {rate:.1f} puzzles/s", flush=True)

    secs = time.time() - t0
    result = {
        "task": a.task,
        "repo": a.repo,
        "data": str(a.data),
        "split": a.split,
        "n": n,
        "nfe": len(meta["levels"]),
        "levels": meta["levels"],
        "board_accuracy": tally["board"] / n,
        "legal_rate": tally["legal"] / n,
        "clue_consistent_rate": tally["clues"] / n,
        "clues_min": int((puz > 0).sum(1).min()),
        "clues_max": int((puz > 0).sum(1).max()),
        "clues_mean": float((puz > 0).sum(1).mean()),
        "by_clue": {str(k): v for k, v in sorted(by_clue.items())},
        "seconds": secs,
    }

    print(f"\n{a.task} on {a.data} ({a.split}, n={n}, {len(meta['levels'])} NFE)")
    print(f"  board accuracy      {result['board_accuracy']:.2%}")
    print(f"  legal boards        {result['legal_rate']:.2%}")
    print(f"  clues preserved     {result['clue_consistent_rate']:.2%}")
    print(f"  clues per puzzle    {result['clues_min']}-{result['clues_max']} "
          f"(mean {result['clues_mean']:.1f})")
    print(f"  {secs:.0f}s total")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(a.out, "w"), indent=2)
        print(f"  written to {a.out}")


if __name__ == "__main__":
    main()
