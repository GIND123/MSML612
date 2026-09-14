"""Did the baseline fail to PARALLELISE, or fail to LEARN?

At 20 digits the baseline scores 0.0% at one pass and 0.0% at sixteen. It is
tempting to read that as a very large parallel-decoding gap, but the standard
pass grid stops at sixteen while a 20-digit answer occupies 21 slots - so that
number was never a fully sequential decode, and the two explanations are not
distinguished by it:

  (a) the baseline learned the task and merely cannot commit it in parallel
      -> it should succeed once decoding is fully sequential;
  (b) the baseline never learned the task at all
      -> it fails at every budget, and the claim is about sample efficiency
         rather than about parallelism.

These are different papers. This script decodes every run at a budget equal to
the number of answer positions, which is genuinely one token per pass, and
reports accuracy at that budget next to the one-pass number.
"""
import argparse, glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data import build_addition
from model import Transformer
import diffusion as dfn
from train_add import exact_match

p = argparse.ArgumentParser()
p.add_argument("--runs", default="runs")
p.add_argument("--pattern", default="g13-*")
p.add_argument("--n_eval", type=int, default=200)
p.add_argument("--eval_bs", type=int, default=50)
p.add_argument("--out", default="figures/seq_check.json")
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
rows = []

for run_dir in sorted(glob.glob(os.path.join(a.runs, a.pattern))):
    rf = os.path.join(run_dir, "result.json")
    mf = os.path.join(run_dir, "model.pt")
    if not (os.path.exists(rf) and os.path.exists(mf)):
        continue
    r = json.load(open(rf))
    cfg = r["args"]
    d, SL = cfg["digits"], cfg["seq_len"]

    X, A, tok = build_addition(a.n_eval, d, SL, seed=91_000, exact=True)
    X, A = torch.from_numpy(X), torch.from_numpy(A)
    n_slots = int(A[0].sum())

    m = Transformer(len(tok), cfg["d"], cfg["layers"], cfg["heads"], cfg["pe"],
                    causal=False, max_len=SL + 8).to(dev)
    m.load_state_dict(torch.load(mf, map_location=dev))
    m.eval()

    accs = {}
    for steps in (1, n_slots):
        vals = []
        for i in range(0, len(X), a.eval_bs):
            xb, ab = X[i:i + a.eval_bs], A[i:i + a.eval_bs]
            out = dfn.cond_decode(m, xb, ab, tok, steps, dev)
            vals.append(exact_match(out.cpu(), xb, ab, tok))
        accs[steps] = float(np.mean(vals))

    rows.append({"run": os.path.basename(run_dir), "digits": d,
                 "mode": cfg["mode"], "seed": cfg["seed"], "slots": n_slots,
                 "acc_1pass": accs[1], "acc_sequential": accs[n_slots]})
    print(f"{rows[-1]['run']:<22} {d:>3}d {cfg['mode']:<6} "
          f"1-pass {accs[1]*100:6.2f}   fully-sequential ({n_slots} passes) "
          f"{accs[n_slots]*100:6.2f}", flush=True)

print("\n=== does the baseline learn the task at ANY budget? ===")
agg = {}
for r in rows:
    agg.setdefault((r["digits"], r["mode"]), []).append(r)
print(f"{'digits':>7} {'mode':<7}{'1 pass':>9}{'sequential':>12}  verdict")
for k in sorted(agg):
    v = agg[k]
    one = np.mean([x["acc_1pass"] for x in v]) * 100
    seq = np.mean([x["acc_sequential"] for x in v]) * 100
    if seq < 5:
        verdict = "never learned the task (sample efficiency, not parallelism)"
    elif one < seq - 5:
        verdict = "learned it, cannot parallelise (a true parallelism gap)"
    else:
        verdict = "learned it and parallelises"
    print(f"{k[0]:>7} {k[1]:<7}{one:8.1f} {seq:11.1f}  {verdict}")

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(rows, open(a.out, "w"), indent=2)
print(f"\nwrote {a.out}")
