"""Parallelism-Matched Diffusion on addition: a unique-answer, chained task.

Addition is the right testbed because the answer is fully determined by the
prompt. Every position's true marginal is a point mass, so there is no
information-theoretic obstacle to parallel decoding - unlike free-choice tasks,
where two independent draws simply cannot agree and no training can help. What
is left is a computation problem, and computation problems are trainable.

Headline axis: exact-match accuracy against the number of denoising passes,
from fully parallel (1) to fully sequential.
"""
import argparse, json, math, os, time
import numpy as np
import torch

from data import build_addition
from model import Transformer
import diffusion as dfn


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",
                   choices=["mdlm", "matched", "distill", "full", "prog",
                            "anybudget", "anybudget_nc"],
                   default="mdlm",
                   help="anybudget = sample K and condition on it (the method); "
                        "anybudget_nc = same K sampling WITHOUT conditioning, "
                        "the ablation that isolates what conditioning buys")
    p.add_argument("--budget_bins", type=int, default=10,
                   help="log2 bins for budget conditioning; 0 disables")
    p.add_argument("--K_max", type=int, default=16)
    p.add_argument("--K", type=int, default=8, help="parallelism the schedule is matched to")
    p.add_argument("--lam", type=float, default=1.0, help="one-pass supervision weight")
    p.add_argument("--digits", type=int, default=6)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--n_train", type=int, default=200000)
    p.add_argument("--n_eval", type=int, default=500)
    p.add_argument("--eval_bs", type=int, default=100)
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--pe", default="rope")
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/dev")
    return p.parse_args()


PASS_GRID = [1, 2, 3, 4, 6, 8, 12, 16]


def exact_match(pred, gold, amask, tok):
    ok = 0
    for p, g, a in zip(pred.tolist(), gold.tolist(), amask.tolist()):
        pa = [p[i] for i in range(len(a)) if a[i]]
        ga = [g[i] for i in range(len(a)) if a[i]]
        ok += pa == ga
    return ok / len(gold)


@torch.no_grad()
def evaluate(model, a, tok, device):
    """Accuracy across the whole pass grid.

    The grid now runs out to a FULLY SEQUENTIAL decode (one token per pass), not
    just 16: a 20-digit answer has 21 slots, so stopping at 16 never measures
    sequential decoding at all and cannot tell "failed to parallelise" apart
    from "failed to learn".
    """
    model.eval()
    out = {}
    X, A, _ = build_addition(a.n_eval, a.digits, a.seq_len, seed=90_000, exact=True)
    X, A = torch.from_numpy(X), torch.from_numpy(A)
    cond = getattr(model, "budget_bins", 0) > 0
    grid = sorted(set(PASS_GRID) | {int(A[0].sum())})
    for steps in grid:
        accs = []
        for i in range(0, len(X), a.eval_bs):
            xb, ab = X[i:i + a.eval_bs].to(device), A[i:i + a.eval_bs].to(device)
            pred = (dfn.cond_decode_budget(model, xb, ab, tok, steps, device)
                    if cond else dfn.cond_decode(model, xb, ab, tok, steps, device))
            accs.append(exact_match(pred.cpu(), xb.cpu(), ab.cpu(), tok))
        out[str(steps)] = float(np.mean(accs))
    model.train()
    return out


def main():
    a = get_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)

    X, A, tok = build_addition(a.n_train, a.digits, a.seq_len, seed=a.seed)
    X, A = torch.from_numpy(X), torch.from_numpy(A)

    cond = a.mode == "anybudget"
    model = Transformer(len(tok), a.d, a.layers, a.heads, a.pe,
                        causal=False, max_len=a.seq_len + 8,
                        budget_bins=a.budget_bins if cond else 0).to(dev)
    print(f"[{a.mode}] params={model.n_params()/1e6:.2f}M digits<={a.digits} "
          f"K={a.K} device={dev}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) *
                       0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    amp = dict(device_type="cuda", dtype=torch.bfloat16) if dev == "cuda" \
        else dict(device_type="cpu", enabled=False)

    t0 = time.time()
    for step in range(a.steps):
        idx = torch.randint(0, len(X), (a.bs,))
        xb, ab = X[idx].to(dev), A[idx].to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(**amp):
            if a.mode == "prog":
                loss, K_now, lam_now = dfn.pmd_progressive(
                    model, xb, ab, tok, step / max(1, a.steps), a.K_max, a.lam)
            elif a.mode in ("anybudget", "anybudget_nc"):
                loss = dfn.cond_any_budget_loss(model, xb, ab, tok,
                                                condition=cond)
            else:
                loss = dfn.pmd_loss(model, xb, ab, tok, a.mode, a.K, a.lam)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 2000 == 0:
            extra = f" K={K_now} lam={lam_now:.2f}" if a.mode == "prog" else ""
            print(f"step {step:6d} loss {loss.item():.4f}{extra} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    acc = evaluate(model, a, tok, dev)
    n_par = model.n_params()
    json.dump({"args": vars(a), "accuracy_by_passes": acc,
               "minutes": (time.time() - t0) / 60,
               "compute": {"params": n_par,
                           "train_tokens": a.steps * a.bs * a.seq_len,
                           "fwd_per_step": 2 if a.mode in ("distill", "full") else 1}},
              open(os.path.join(a.out, "result.json"), "w"), indent=2)
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    print("FINAL", json.dumps(acc), flush=True)


if __name__ == "__main__":
    main()
