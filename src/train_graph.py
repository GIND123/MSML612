"""Masked diffusion on structured graphs, where the ceiling is known exactly.

This is the experiment the project could not run before. On every earlier task
the best achievable parallel-decoding accuracy was unknown, so a gap could
always be blamed on undertraining. Here it is computable in closed form:

    V* = sum over valid graphs of prod_i p_i(g_i)   and   V* >= 2^(-TC),

with equality when the family has a fixed edge count. V* is the probability
that independent draws from the TRUE marginals land on a valid graph, so it is
the one-pass ceiling for any model whatsoever - it already assumes the marginals
are perfect.

That makes the logic airtight in a way it was not on addition or text:

  * if a trained model reaches V* at one pass, the remaining gap to 100% is
    PROVABLY information-theoretic - mode (ii) - and no training objective can
    recover it;
  * if it falls short of V*, the shortfall is exactly mode (i), the trainable
    part, and budget conditioning should close it.

The two failure modes are therefore separated numerically rather than argued
about, on the same task, at the same time.

A graph on n nodes is a binary sequence of n(n-1)/2 edge indicators, so the
masked-diffusion machinery is used unchanged - only the data and the validity
check are new.
"""
import argparse, json, math, os, time
import numpy as np
import torch

from model import Transformer
import diffusion as dfn
from graphs import (EdgeTokenizer, exact_stats, is_valid, n_edges, sample)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--family", default="matching",
                   choices=["matching", "2regular", "tree", "trianglefree",
                            "bipartite"])
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--mode", choices=["mdlm", "anybudget", "anybudget_nc"],
                   default="mdlm")
    p.add_argument("--budget_bins", type=int, default=8)
    p.add_argument("--pe", default="rope", choices=["rope", "ape", "sin", "alibi", "nope"],
                   help="position encoding; decisive at the fully-masked state")
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_gen", type=int, default=4096)
    p.add_argument("--gen_bs", type=int, default=512)
    p.add_argument("--out", default="runs/gdev")
    return p.parse_args()


a = get_args()
torch.manual_seed(a.seed)
np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(a.out, exist_ok=True)

E = n_edges(a.n)
tok = EdgeTokenizer()
stats = exact_stats(a.family, a.n)
print(f"family={a.family} n={a.n} edges={E} |family|={stats['size']}", flush=True)
print(f"  sum_i H(e_i) = {stats['sum_marginal_entropy_bits']:.3f} bits   (the bound)", flush=True)
print(f"  H(S)         = {stats['joint_entropy_bits']:.3f} bits", flush=True)
print(f"  TC           = {stats['total_correlation_bits']:.3f} bits", flush=True)
print(f"  V* ceiling   = {stats['one_pass_ceiling']*100:.4f}%   "
      f"(2^-TC = {2**-stats['total_correlation_bits']*100:.4f}%)", flush=True)

X = torch.from_numpy(sample(a.family, a.n, 200_000, seed=a.seed))

COND = a.mode == "anybudget"
model = Transformer(len(tok), a.d, a.layers, a.heads, a.pe, causal=False,
                    max_len=E + 8, budget_bins=a.budget_bins if COND else 0).to(dev)
print(f"params {model.n_params()/1e6:.2f}M  mode={a.mode}  pe={a.pe}", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / a.warmup) *
                   0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))

t0 = time.time()
for step in range(a.steps):
    idx = torch.randint(0, len(X), (a.bs,))
    xb = X[idx].to(dev)
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
        if a.mode in ("anybudget", "anybudget_nc"):
            loss = dfn.uncond_any_budget_loss(
                model, xb, tok, choices=tuple(2 ** i for i in range(int(math.log2(E)) + 1)),
                condition=COND)
        else:
            loss = dfn.uncond_mdlm_loss(model, xb, tok, "uniform")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 5000 == 0:
        print(f"step {step:6d} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)

model.eval()
res = {"args": vars(a), "exact": {k: v for k, v in stats.items() if k != "marginals"}}


@torch.no_grad()
def gen(fn):
    outs, nfes = [], []
    for _ in range(max(1, a.n_gen // a.gen_bs)):
        blank = torch.full((a.gen_bs, E), tok.mask, dtype=torch.long)
        x, nfe = fn(blank)
        outs.append(x.cpu()); nfes.append(nfe)
    x = torch.cat(outs).numpy()
    ok = is_valid(x, a.family, a.n)
    uniq = len({tuple(r) for r in x[ok]}) / max(1, int(ok.sum()))
    return float(ok.mean()), float(np.mean(nfes)), float(uniq)


def bud(K):
    return torch.full((a.gen_bs,), float(K), device=dev) if COND else None


print("\n=== validity against decoding budget ===", flush=True)
print(f"{'rule':<22}{'NFE':>7}{'validity':>11}{'vs V*':>10}{'uniq':>8}", flush=True)
res["decode"] = {}
V = stats["one_pass_ceiling"]

for K in sorted({1, 2, 4, 8, E}):
    v, nfe, u = gen(lambda b, K=K: dfn.fixed_k_decode(model, b, tok, dev, K,
                                                      budget_cond=bud(K)))
    res["decode"][f"fixedK_{K}"] = {"validity": v, "nfe": nfe, "unique": u}
    rel = f"{v/V:8.2f}x" if K == 1 else f"{'—':>9}"
    print(f"{'fixed-K K=' + str(K):<22}{nfe:>7.1f}{v*100:>10.2f}%{rel}{u*100:>7.1f}%", flush=True)

for B in (0.05, 0.25, 1.0, 4.0):
    v, nfe, u = gen(lambda b, B=B: dfn.entropy_budget_decode(model, b, tok, dev, B,
                                                             max_steps=E + 2,
                                                             budget_cond=bud(8)))
    res["decode"][f"entbudget_{B}"] = {"validity": v, "nfe": nfe, "unique": u}
    print(f"{'entropy-budget B=' + str(B):<22}{nfe:>7.1f}{v*100:>10.2f}%{'—':>9}"
          f"{u*100:>7.1f}%", flush=True)

one = res["decode"]["fixedK_1"]["validity"]
res["reached_ceiling"] = one >= 0.9 * V
print(f"\nV* (exact one-pass ceiling) : {V*100:.4f}%", flush=True)
print(f"achieved at one pass        : {one*100:.4f}%  "
      f"({one/V*100:.1f}% of the ceiling)", flush=True)
print("reading: " + ("the model is at the information-theoretic limit, so the "
                     "residual gap is mode (ii) and NOT trainable"
                     if res["reached_ceiling"] else
                     "the model falls short of the ceiling, so part of the gap "
                     "is mode (i) and IS trainable"), flush=True)

torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=2)
print(f"\nsaved to {a.out}", flush=True)
