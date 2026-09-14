"""QM9 molecule generation: validity against denoising budget.

Phase 2. Where the synthetic graph families give an exactly computable ceiling
but an artificial task, QM9 gives a real benchmark whose metric is unambiguous:
RDKit sanitisation either succeeds or it does not, and it enforces valency at
every atom simultaneously. That makes it a joint constraint of exactly the kind
parallel decoding breaks - mode (ii) - on data nobody designed for this project.

The axis of comparison is the one the graph-diffusion literature competes on:
how much validity survives as the number of denoising steps falls. Discrete
graph diffusion conventionally uses hundreds of steps; the claim under test is
that budget-conditioned training plus an entropy-budgeted commit rule holds
validity at far fewer.

All arms share the model, the data and the training budget, so the comparison
isolates the schedule and the decoding rule rather than scale.
"""
import argparse, json, math, os, time
import numpy as np
import torch

from model import Transformer
import diffusion as dfn
from qm9 import MolTokenizer, SEQ_LEN, load_qm9, validity


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["mdlm", "anybudget", "anybudget_nc"],
                   default="mdlm")
    p.add_argument("--budget_bins", type=int, default=8)
    p.add_argument("--root", default="data/qm9")
    p.add_argument("--limit", type=int, default=0, help="0 = all molecules")
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_gen", type=int, default=4096)
    p.add_argument("--gen_bs", type=int, default=512)
    p.add_argument("--out", default="runs/moldev")
    return p.parse_args()


a = get_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(a.out, exist_ok=True)

X, tok, train_smiles = load_qm9(a.root, a.limit or None)
print(f"qm9 sequences {X.shape}  vocab {len(tok)}  train SMILES {len(train_smiles)}",
      flush=True)
Xt = torch.from_numpy(X)
ALLOWED = torch.from_numpy(tok.allowed_mask()).to(dev)

COND = a.mode == "anybudget"
model = Transformer(len(tok), a.d, a.layers, a.heads, "rope", causal=False,
                    max_len=SEQ_LEN + 8,
                    budget_bins=a.budget_bins if COND else 0).to(dev)
print(f"params {model.n_params()/1e6:.2f}M  mode={a.mode}", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / a.warmup) *
                   0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
BUDGETS = tuple(2 ** i for i in range(int(math.log2(SEQ_LEN)) + 2))

t0 = time.time()
for step in range(a.steps):
    xb = Xt[torch.randint(0, len(Xt), (a.bs,))].to(dev)
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
        if a.mode in ("anybudget", "anybudget_nc"):
            loss = dfn.uncond_any_budget_loss(model, xb, tok, choices=BUDGETS,
                                              condition=COND)
        else:
            loss = dfn.uncond_mdlm_loss(model, xb, tok, "uniform")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 5000 == 0:
        print(f"step {step:6d} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)

model.eval()
res = {"args": vars(a), "decode": {}}


@torch.no_grad()
def gen(fn):
    outs, nfes = [], []
    for _ in range(max(1, a.n_gen // a.gen_bs)):
        blank = torch.full((a.gen_bs, SEQ_LEN), tok.mask, dtype=torch.long)
        x, nfe = fn(blank)
        outs.append(x.cpu()); nfes.append(nfe)
    x = torch.cat(outs).numpy()
    v, smis = validity(x, tok)
    uniq = len(set(smis)) / max(1, len(smis))
    novel = sum(s not in train_smiles for s in set(smis)) / max(1, len(set(smis)))
    return {"validity": v, "nfe": float(np.mean(nfes)), "unique": uniq,
            "novel": novel}


def bud(K):
    return torch.full((a.gen_bs,), float(K), device=dev) if COND else None


print("\n=== validity / uniqueness / novelty against denoising budget ===", flush=True)
print(f"{'rule':<24}{'NFE':>7}{'valid':>9}{'unique':>9}{'novel':>9}", flush=True)


def report(tag, key, r):
    res["decode"][key] = r
    print(f"{tag:<24}{r['nfe']:>7.1f}{r['validity']*100:>8.2f}%"
          f"{r['unique']*100:>8.1f}%{r['novel']*100:>8.1f}%", flush=True)


for K in (1, 2, 4, 8, 16, 32, SEQ_LEN):
    report(f"fixed-K K={K}", f"fixedK_{K}",
           gen(lambda b, K=K: dfn.fixed_k_decode(model, b, tok, dev, K,
                                                 budget_cond=bud(K), allowed=ALLOWED)))

for B in (0.05, 0.25, 1.0, 4.0):
    report(f"entropy-budget B={B}", f"entbudget_{B}",
           gen(lambda b, B=B: dfn.entropy_budget_decode(
               model, b, tok, dev, B, max_steps=SEQ_LEN + 2,
               budget_cond=bud(8), allowed=ALLOWED)))

for tau in (0.7, 0.9, 0.99):
    report(f"confidence tau={tau}", f"conf_{tau}",
           gen(lambda b, tau=tau: dfn.confidence_threshold_decode(
               model, b, tok, dev, tau, max_steps=SEQ_LEN + 2,
               budget_cond=bud(8), allowed=ALLOWED)))

torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=2)
print(f"\nsaved to {a.out}", flush=True)
