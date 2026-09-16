"""Masked diffusion on Sudoku: does committing by certainty solve constraints?

The benchmark is SATNet's 9x9 Sudoku (Wang et al., ICML 2019) with its
conventional 9,000 / 1,000 split, scored by BOARD-level accuracy - all 81 cells
correct - which is the metric the published numbers use.

The claim under test is specific. Sudoku has a unique answer, so there is no
information-theoretic obstacle whatsoever: every cell's true conditional is a
point mass and TC is zero. The entire difficulty is therefore mode (i), plus the
ORDER in which cells are committed. A decoder that commits the lowest-entropy
cells first, recomputes, and repeats is doing classical constraint propagation -
fill the forced cells, propagate, repeat - without being told to. A fixed-K
decoder that commits a fixed fraction per pass regardless of certainty is not.

Prediction, registered before the runs: entropy-budgeted decoding should beat
fixed-K decoding by a very large margin here, larger than anywhere else in this
project, because Sudoku is the case where commit ORDER carries almost all of the
information. Confidence-thresholding is included as the published competitor.
"""
import argparse, json, math, os, time
import numpy as np
import torch

from model import RecurrentDenoiser, Transformer
import diffusion as dfn
from sudoku import (CELLS, SudokuTokenizer, board_accuracy, cell_accuracy,
                    consistent_with_clues, is_valid_solution, load_satnet, split)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["mdlm", "matched", "full"], default="mdlm")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--pe", default="ape", choices=["ape", "rope", "sin"])
    p.add_argument("--recurrent", action="store_true",
                   help="weight-shared block applied R times with deep supervision")
    p.add_argument("--R", type=int, default=32, help="training recurrences")
    p.add_argument("--R_infer", type=int, default=0,
                   help="inference recurrences (0 = same as training)")
    p.add_argument("--root", default="data/sudoku")
    p.add_argument("--hard", default="",
                   help="directory of the generated HARD benchmark (npy files); "
                        "overrides --root. Minimal puzzles at ~24 clues, where "
                        "constraint propagation alone solves about 4%")
    p.add_argument("--augment", action="store_true",
                   help="relabel digits by a random permutation (a Sudoku symmetry)")
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--layers", type=int, default=10)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--steps", type=int, default=80000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_bs", type=int, default=200)
    p.add_argument("--ckpt_every", type=int, default=4000,
                   help="steps between checkpoints; 0 disables")
    p.add_argument("--out", default="runs/sud")
    return p.parse_args()


a = get_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(a.out, exist_ok=True)

if a.hard:
    # the generated benchmark: minimal puzzles, uniqueness verified per removal,
    # train/test solutions disjoint even up to digit relabelling
    Xtr = np.load(os.path.join(a.hard, "train_puz.npy"))
    Ytr = np.load(os.path.join(a.hard, "train_sol.npy"))
    Xte = np.load(os.path.join(a.hard, "test_puz.npy"))
    Yte = np.load(os.path.join(a.hard, "test_sol.npy"))
    tok = SudokuTokenizer()
    print(f"HARD sudoku: train {Xtr.shape} test {Xte.shape}  "
          f"clues mean {(Xtr>0).sum(1).mean():.1f} "
          f"(min {(Xtr>0).sum(1).min()} max {(Xtr>0).sum(1).max()})", flush=True)
else:
    X, Y, tok = load_satnet(a.root)
    (Xtr, Ytr), (Xte, Yte) = split(X, Y)
    print(f"sudoku: train {Xtr.shape} test {Xte.shape} vocab {len(tok)}  "
          f"clues/puzzle mean {(X>0).sum(1).mean():.1f}", flush=True)

# tokens are the SOLUTION digits shifted to 0-8; blanks are what gets masked
sol_tr = torch.from_numpy(Ytr - 1)
blank_tr = torch.from_numpy(Xtr == 0)
sol_te = torch.from_numpy(Yte - 1)
blank_te = torch.from_numpy(Xte == 0)

if a.recurrent:
    model = RecurrentDenoiser(len(tok), a.d, a.layers, a.heads, a.pe,
                              max_len=CELLS + 8, recurrences=a.R).to(dev)
    print(f"params {model.n_params()/1e3:.0f}k  RECURRENT R={a.R} "
          f"layers={a.layers} d={a.d} mode={a.mode} pe={a.pe}", flush=True)
else:
    model = Transformer(len(tok), a.d, a.layers, a.heads, a.pe, causal=False,
                        max_len=CELLS + 8).to(dev)
    print(f"params {model.n_params()/1e6:.2f}M  mode={a.mode} pe={a.pe}", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / a.warmup) *
                   0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))

# ---- checkpoint / resume ---------------------------------------------------
# The class allocation bills an H100 at 144/min against 9/min for a 5GB MIG
# slice, so long runs belong on the cheap partition - which has shorter time
# limits. Resuming from a checkpoint is what makes a long run survive being
# split across several short jobs, and it costs no extra GPU-time: the
# allocation charges GPU-seconds consumed, not jobs submitted.
CKPT = os.path.join(a.out, "ckpt.pt")
start_step = 0
if os.path.exists(CKPT):
    st = torch.load(CKPT, map_location=dev)
    model.load_state_dict(st["model"])
    opt.load_state_dict(st["opt"])
    sched.load_state_dict(st["sched"])
    start_step = st["step"]
    torch.set_rng_state(st["torch_rng"].cpu())
    np.random.set_state(st["np_rng"])
    print(f"resumed from {CKPT} at step {start_step}", flush=True)


def save_ckpt(step):
    tmp = CKPT + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "step": step,
                "torch_rng": torch.get_rng_state(),
                "np_rng": np.random.get_state()}, tmp)
    os.replace(tmp, CKPT)      # atomic: a job killed mid-write leaves the old one


rng = np.random.default_rng(a.seed)
t0 = time.time()
for step in range(start_step, a.steps):
    idx = torch.randint(0, len(sol_tr), (a.bs,))
    xb, ab = sol_tr[idx].clone(), blank_tr[idx]
    if a.augment:
        # relabelling the nine digits maps a valid grid to a valid grid, and
        # leaves the constraint structure - the thing being learned - untouched
        perm = torch.from_numpy(rng.permutation(9))
        xb = perm[xb]
    xb, ab = xb.to(dev), ab.to(dev)
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
        if a.recurrent:
            loss = dfn.recurrent_cond_loss(model, xb, ab, tok, a.mode, a.K, a.lam)
        else:
            loss = dfn.pmd_loss(model, xb, ab, tok, a.mode, a.K, a.lam)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 5000 == 0:
        print(f"step {step:6d} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    if a.ckpt_every and step and step % a.ckpt_every == 0:
        save_ckpt(step)
save_ckpt(a.steps)

model.eval()
if a.recurrent and a.R_infer:
    model.recurrences = a.R_infer
    print(f"inference recurrences set to {a.R_infer} (trained with {a.R})", flush=True)
res = {"args": vars(a), "decode": {}}


@torch.no_grad()
def evaluate(fn):
    """Board accuracy over the full test split, plus the diagnostics."""
    preds, nfes = [], []
    for i in range(0, len(sol_te), a.eval_bs):
        gold = sol_te[i:i + a.eval_bs].to(dev)
        bl = blank_te[i:i + a.eval_bs].to(dev)
        x = torch.where(bl, torch.full_like(gold, tok.mask), gold)
        y, nfe = fn(x, bl)
        preds.append(y.cpu()); nfes.append(nfe)
    P = torch.cat(preds).numpy() + 1
    G = Yte
    board = board_accuracy(P, G)
    cell = cell_accuracy(P, G, Xte == 0)
    valid = float(np.mean([is_valid_solution(P[i]) for i in range(len(P))]))
    kept = float(np.mean([consistent_with_clues(P[i], Xte[i]) for i in range(len(P))]))
    return {"board": board, "cell": cell, "valid_sudoku": valid,
            "clues_kept": kept, "nfe": float(np.mean(nfes))}


def report(tag, key, r):
    res["decode"][key] = r
    print(f"  {tag:<26}{r['nfe']:>7.1f}{r['board']*100:>10.2f}%{r['cell']*100:>9.2f}%"
          f"{r['valid_sudoku']*100:>9.2f}%", flush=True)


print("\n=== board accuracy against decoding budget ===", flush=True)
print(f"  {'rule':<26}{'NFE':>7}{'BOARD':>10}{'cell':>9}{'valid':>9}", flush=True)

print("fixed-K (commits a fixed share per pass, ignoring certainty):", flush=True)
for K in (1, 2, 4, 8, 16, 32, 61):
    report(f"fixed-K K={K}", f"fixedK_{K}",
           evaluate(lambda x, bl, K=K: dfn.fixed_k_decode(
               model, x, tok, dev, K, fillable=bl)))

print("entropy-budget (= constraint propagation):", flush=True)
for B in (0.01, 0.05, 0.2, 0.5, 1.0, 2.0):
    report(f"entropy-budget B={B}", f"entbudget_{B}",
           evaluate(lambda x, bl, B=B: dfn.entropy_budget_decode(
               model, x, tok, dev, B, max_steps=CELLS + 2, fillable=bl)))

print("confidence-threshold (published inference-time method):", flush=True)
for tau in (0.9, 0.99, 0.999):
    report(f"confidence tau={tau}", f"conf_{tau}",
           evaluate(lambda x, bl, tau=tau: dfn.confidence_threshold_decode(
               model, x, tok, dev, tau, max_steps=CELLS + 2, fillable=bl)))

best = max(res["decode"].values(), key=lambda r: r["board"])
res["best_board"] = best["board"]
print(f"\nbest board accuracy: {best['board']*100:.2f}%  at {best['nfe']:.1f} passes",
      flush=True)
print(f"SATNet (Wang et al. ICML 2019) reports 98.3% board accuracy on this split.",
      flush=True)

torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=2)
print(f"saved to {a.out}", flush=True)
