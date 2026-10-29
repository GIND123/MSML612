"""Tabulate finished runs so the comparison can be rebuilt without re-running.

Reads the JSON written by `run_ncm.py` (and anything else following the same
schema, so our own model's results can be dropped in beside them) and prints a
markdown table ordered by clue count, which is the axis that actually separates
these systems.

    python ncm/compare.py ncm/results/*.json
"""
import argparse
import json
from pathlib import Path


def row(r):
    clues = (f"{r['clues_min']}-{r['clues_max']}"
             if r["clues_min"] != r["clues_max"] else str(r["clues_min"]))
    return {
        "model": r.get("model", r.get("task", "?")),
        "params": r.get("params", "25M"),
        "nfe": r.get("nfe", "-"),
        "clues": clues,
        "mean": r.get("clues_mean", 0.0),
        "n": r["n"],
        "board": r["board_accuracy"],
        "legal": r.get("legal_rate"),
        "data": Path(r.get("data", "")).name,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="JSON files written by run_ncm.py")
    a = ap.parse_args()

    rows = [row(json.load(open(f))) for f in a.results]
    rows.sort(key=lambda x: (-x["mean"], x["model"]))

    print("| Model | Params | NFE | Clues | n | Board accuracy | Legal boards |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        legal = f"{r['legal']:.2%}" if r["legal"] is not None else "-"
        print(f"| {r['model']} | {r['params']} | {r['nfe']} | {r['clues']} "
              f"| {r['n']} | {r['board']:.2%} | {legal} |")

    print("\nBoard accuracy is all 81 cells matching the unique solution. Legal boards is "
          "the weaker validity-only metric the NCM paper reports, shown for comparability.")


if __name__ == "__main__":
    main()
