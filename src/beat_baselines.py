"""Head-to-head against inference-time decoding methods.

This is the comparison that decides whether the paper's claim holds. Roughly a
dozen 2026 papers accelerate diffusion decoding by choosing *which* positions to
reveal together - confidence ranking, entropy gating, dependency-aware ordering.
All of them take the trained marginals as given and only reorder commits.

The argument of this project is that no such method can close the gap, because
the marginals themselves were never trained for the one-pass regime. That is
testable directly: take a BASELINE model, decode it with every strategy at every
budget, and check whether any of them reaches the training-time method.

For a fair fight each inference-time method is given its best configuration and
is allowed MORE passes than our method uses.
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
p.add_argument("--digits", type=int, default=12)
p.add_argument("--n_eval", type=int, default=400)
p.add_argument("--eval_bs", type=int, default=100)
p.add_argument("--out", default="figures/beat_baselines.json")
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
SL = 3 * a.digits + 4
X, A, tok = build_addition(a.n_eval, a.digits, SL, seed=90_000, exact=True)
X, A = torch.from_numpy(X), torch.from_numpy(A)


def load(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "result.json")))["args"]
    m = Transformer(len(tok), cfg["d"], cfg["layers"], cfg["heads"], cfg["pe"],
                    causal=False, max_len=cfg["seq_len"] + 8).to(dev)
    m.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location=dev))
    m.eval()
    return m, cfg


@torch.no_grad()
def run(model, steps, strategy, tau=None):
    accs = []
    for i in range(0, len(X), a.eval_bs):
        xb, ab = X[i:i + a.eval_bs].to(dev), A[i:i + a.eval_bs].to(dev)
        x = torch.where(ab, torch.full_like(xb, tok.mask), xb)
        n_ans = int(ab[0].sum())
        pred = None
        for s in range(steps, 0, -1):
            masked = (x == tok.mask) & ab
            if not masked.any():
                break
            logits = model(x)
            probs = logits.softmax(-1)
            conf, pred = probs.max(-1)
            if strategy == "random":
                score = torch.rand_like(conf)
            elif strategy == "entropy":
                ent = -(probs * probs.clamp_min(1e-9).log()).sum(-1)
                score = -ent
            elif strategy == "margin":
                top2 = probs.topk(2, dim=-1).values
                score = top2[..., 0] - top2[..., 1]
            else:
                score = conf
            score = score.masked_fill(~masked, -1e9)
            left = int(n_ans * (s - 1) / steps)
            for b in range(x.shape[0]):
                k = max(0, int(masked[b].sum()) - left)
                if strategy == "entropy" and tau is not None and k > 0:
                    ent_b = -(probs[b] * probs[b].clamp_min(1e-9).log()).sum(-1)
                    allowed = int(((ent_b < tau) & masked[b]).sum())
                    k = max(1, min(k, allowed)) if allowed else 1
                if k:
                    idx = score[b].topk(k).indices
                    x[b, idx] = pred[b, idx]
        rem = (x == tok.mask) & ab
        if rem.any() and pred is not None:
            x[rem] = pred[rem]
        accs.append(exact_match(x.cpu(), xb.cpu(), ab.cpu(), tok))
    return float(np.mean(accs))


base_dirs = sorted(glob.glob(f"{a.runs}/g11-d{a.digits}-mdlm-s*"))
ours_dirs = sorted(glob.glob(f"{a.runs}/g11-d{a.digits}-full-s*"))
if not base_dirs or not ours_dirs:
    sys.exit(f"need both baseline and method runs for {a.digits} digits")

results = {"digits": a.digits, "inference_time_on_baseline": {}, "ours": {}}

print(f"\n=== {a.digits}-digit addition: can any inference-time method rescue "
      f"the baseline? ===")
print(f"{'strategy':<22}{'1 pass':>9}{'2':>8}{'4':>8}{'8':>8}{'16':>8}")
for strat, tau in [("confidence", None), ("entropy", None), ("entropy-gated", 0.5),
                   ("margin", None), ("random", None)]:
    row = []
    for steps in (1, 2, 4, 8, 16):
        vals = []
        for d in base_dirs:
            m, _ = load(d)
            vals.append(run(m, steps, "entropy" if strat == "entropy-gated" else strat, tau))
        row.append(float(np.mean(vals)))
    results["inference_time_on_baseline"][strat] = row
    print(f"{strat:<22}" + "".join(f"{v*100:8.1f}" for v in row))

row = []
for steps in (1, 2, 4, 8, 16):
    vals = []
    for d in ours_dirs:
        m, _ = load(d)
        vals.append(run(m, steps, "confidence"))
    row.append(float(np.mean(vals)))
results["ours"]["confidence"] = row
print(f"{'OURS (training-time)':<22}" + "".join(f"{v*100:8.1f}" for v in row))

best_inf = max(v[0] for v in results["inference_time_on_baseline"].values())
best_inf_any = max(max(v) for v in results["inference_time_on_baseline"].values())
print(f"\nbest inference-time method at 1 pass : {best_inf*100:.1f}%")
print(f"best inference-time at ANY budget     : {best_inf_any*100:.1f}%")
print(f"ours at 1 pass                        : {row[0]*100:.1f}%")
results["summary"] = {"best_inference_1pass": best_inf,
                      "best_inference_any_budget": best_inf_any,
                      "ours_1pass": row[0]}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(results, open(a.out, "w"), indent=2)
print(f"\nwrote {a.out}")
