"""Evaluate timed-out runs from their checkpoints.

The R-MDM comparison runs hit their wall-clock limit at step 80,000 of 110,000,
so no result.json was written. All three stopped at the SAME step, which keeps
the comparison internally valid - it is simply at 80k steps rather than 110k, and
that is stated wherever the numbers are used.

Configuration is reconstructed from the run name rather than a config file,
because the config is only written at the end of a successful run.
"""
import argparse, glob, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import RecurrentDenoiser
import diffusion as dfn
from sudoku import CELLS, SudokuTokenizer

p = argparse.ArgumentParser()
p.add_argument("--hard", default="data/sudoku_hard")
p.add_argument("--runs", default="runs")
p.add_argument("--pattern", default="rmdm-*")
p.add_argument("--eval_bs", type=int, default=200)
p.add_argument("--out", default="figures/rmdm_compare.json")
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = SudokuTokenizer()
P = np.load(os.path.join(a.hard, "test_puz.npy"))
S = np.load(os.path.join(a.hard, "test_sol.npy"))
Pt, St = torch.from_numpy(P), torch.from_numpy(S - 1)
blank = Pt == 0
print(f"test set: {len(P)} puzzles, clues {(P>0).sum(1).min()}-{(P>0).sum(1).max()} "
      f"(mean {(P>0).sum(1).mean():.1f})\n", flush=True)


def cfg_from_name(n):
    """All three runs share d=128, 1 layer, 4 heads, R=32 train / 64 infer.
    They differ only in whether input injection is on."""
    return {"d": 128, "layers": 1, "heads": 4, "pe": "ape", "R": 32, "R_infer": 64,
            "step_embed": True, "inject": "faithful" not in n}


res = {}
for d in sorted(glob.glob(os.path.join(a.runs, a.pattern))):
    ck = os.path.join(d, "ckpt.pt")
    if not os.path.exists(ck):
        continue
    name = os.path.basename(d)
    c = cfg_from_name(name)
    st = torch.load(ck, map_location=dev)
    m = RecurrentDenoiser(len(tok), c["d"], c["layers"], c["heads"], c["pe"],
                          max_len=CELLS + 8, recurrences=c["R_infer"],
                          inject=c["inject"], step_embed=c["step_embed"]).to(dev)
    m.load_state_dict(st["model"])
    m.eval()
    print(f"--- {name}   step {st['step']:,}   injection={c['inject']}   "
          f"step_embed={c['step_embed']}   params {m.n_params()/1e3:.0f}k", flush=True)
    row = {"step": int(st["step"]), "inject": c["inject"],
           "step_embed": c["step_embed"], "decode": {}}
    print(f"    {'passes':>8}{'board':>10}{'cell':>9}", flush=True)
    for K in (1, 4, 8, 16, 32, 61):
        preds = []
        with torch.no_grad():
            for i in range(0, len(Pt), a.eval_bs):
                gold = St[i:i + a.eval_bs].to(dev)
                bl = blank[i:i + a.eval_bs].to(dev)
                x = torch.where(bl, torch.full_like(gold, tok.mask), gold)
                y, _ = dfn.fixed_k_decode(m, x, tok, dev, K, fillable=bl)
                preds.append(y.cpu().numpy() + 1)
        pred = np.concatenate(preds)
        board = float((pred == S).all(axis=1).mean())
        cell = float((pred[P == 0] == S[P == 0]).mean())
        row["decode"][str(K)] = {"board": board, "cell": cell}
        print(f"    {K:>8}{board*100:>9.2f}%{cell*100:>8.2f}%", flush=True)
    row["best"] = max(v["board"] for v in row["decode"].values())
    res[name] = row
    del m
    torch.cuda.empty_cache()
    print(flush=True)

if res:
    print("=" * 72)
    print("HEAD-TO-HEAD: does input injection help once R-MDM's step embedding is present?")
    print("=" * 72)
    off = [v["best"] for v in res.values() if not v["inject"]]
    on = [v["best"] for v in res.values() if v["inject"]]
    print(f"  R-MDM faithful (injection OFF) : {np.mean(off)*100:6.2f}%   "
          f"n={len(off)}  {[round(x*100,2) for x in off]}")
    if on:
        print(f"  + input injection              : {np.mean(on)*100:6.2f}%   "
              f"n={len(on)}  {[round(x*100,2) for x in on]}")
        d = (np.mean(on) - np.mean(off)) * 100
        print(f"\n  input injection is worth {d:+.2f} points here")
        if d <= 0.5:
            print("  >>> Injection does NOT help once the step embedding is present.")
            print("  >>> The earlier +5.60 was confounded with batch size and with the")
            print("  >>> absence of R-MDM's per-loop conditioning. This must be reported.")
        else:
            print("  >>> Injection still helps with their conditioning present.")
    print("\n  All runs stopped at step 80,000 of 110,000 (wall-clock limit), so they")
    print("  are matched to each other but undertrained relative to the headline runs.")

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(res, open(a.out, "w"), indent=2)
print(f"\nwrote {a.out}")
