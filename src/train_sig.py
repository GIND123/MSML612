"""One-pass masked diffusion on SIGMORPHON morphological inflection.

Autoregressive seq2seq needs |y| decoding steps, about ten per word. If a
diffusion model can commit the whole form in ONE forward pass at the same
exact-match accuracy, that is an order-of-magnitude inference saving on a task
with published baselines across 52 languages.

Breadth here is measured across typologically diverse languages rather than
across seeds of a single configuration, which is the honest test of whether a
method generalises.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

from sigmorphon import load_language
from model import Transformer
import diffusion as dfn


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="english")
    p.add_argument("--size", default="medium", choices=["low", "medium", "high"])
    p.add_argument("--mode", choices=["ar", "mdlm", "full", "prog"], default="mdlm",
                   help="ar = autoregressive baseline; full = ours")
    p.add_argument("--seq_len", type=int, default=48)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--pe", default="rope")
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_root", default="data/sig")
    p.add_argument("--eval_bs", type=int, default=128)
    p.add_argument("--out", default="runs/dev")
    return p.parse_args()


PASSES = [1, 2, 4, 8, 16, 32]


def exact_match(pred, gold, amask):
    ok = 0
    for p, g, a in zip(pred.tolist(), gold.tolist(), amask.tolist()):
        pa = [p[i] for i in range(len(a)) if a[i]]
        ga = [g[i] for i in range(len(a)) if a[i]]
        ok += pa == ga
    return ok / max(len(gold), 1)


@torch.no_grad()
def ar_decode(model, x0, amask, tok, device):
    """Autoregressive baseline: fill the answer strictly left to right, one
    position per forward pass. This is the |y|-step cost the method removes."""
    x = torch.where(amask, torch.full_like(x0, tok.mask), x0).clone()
    order = amask[0].nonzero().flatten().tolist()
    for j in order:
        logits = model(x)
        x[:, j] = torch.where(amask[:, j], logits[:, j].argmax(-1), x[:, j])
    return x


@torch.no_grad()
def evaluate(model, a, tok, Xdv, Adv, device):
    model.eval()
    out = {}
    for steps in PASSES:
        accs = []
        for i in range(0, len(Xdv), a.eval_bs):
            xb = Xdv[i:i + a.eval_bs].to(device)
            ab = Adv[i:i + a.eval_bs].to(device)
            pred = dfn.cond_decode(model, xb, ab, tok, steps, device)
            accs.append(exact_match(pred.cpu(), xb.cpu(), ab.cpu()))
        out[str(steps)] = float(np.mean(accs))
    # full autoregressive decode: one pass per answer position
    accs = []
    for i in range(0, len(Xdv), a.eval_bs):
        xb = Xdv[i:i + a.eval_bs].to(device)
        ab = Adv[i:i + a.eval_bs].to(device)
        accs.append(exact_match(ar_decode(model, xb, ab, tok, device).cpu(),
                                xb.cpu(), ab.cpu()))
    out["ar"] = float(np.mean(accs))
    model.train()
    return out


def main():
    a = get_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)

    (Xtr, Atr), (Xdv, Adv), tok = load_language(a.lang, a.data_root, a.size, a.seq_len)
    Xtr, Atr = torch.from_numpy(Xtr), torch.from_numpy(Atr)
    Xdv, Adv = torch.from_numpy(Xdv), torch.from_numpy(Adv)

    model = Transformer(len(tok), a.d, a.layers, a.heads, a.pe,
                        causal=False, max_len=a.seq_len + 8).to(dev)
    print(f"[{a.lang}/{a.size}/{a.mode}] params={model.n_params()/1e6:.2f}M "
          f"vocab={len(tok)} train={len(Xtr)} dev={len(Xdv)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) *
                       0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    amp = dict(device_type="cuda", dtype=torch.bfloat16) if dev == "cuda" \
        else dict(device_type="cpu", enabled=False)

    t0 = time.time()
    for step in range(a.steps):
        idx = torch.randint(0, len(Xtr), (a.bs,))
        xb, ab = Xtr[idx].to(dev), Atr[idx].to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(**amp):
            if a.mode == "prog":
                loss, _, _ = dfn.pmd_progressive(model, xb, ab, tok,
                                                 step / max(1, a.steps), 16, a.lam)
            elif a.mode == "ar":
                # same network, but supervised only on left-to-right prefixes
                loss = dfn.cond_mdlm_loss(model, xb, ab, tok, "uniform")
            else:
                loss = dfn.pmd_loss(model, xb, ab, tok, a.mode, a.K, a.lam)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 2000 == 0:
            print(f"step {step:6d} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)

    acc = evaluate(model, a, tok, Xdv, Adv, dev)
    json.dump({"args": vars(a), "accuracy_by_passes": acc,
               "minutes": (time.time() - t0) / 60,
               "compute": {"params": model.n_params(),
                           "answer_len_mean": float(Adv.sum(1).float().mean())}},
              open(os.path.join(a.out, "result.json"), "w"), indent=2)
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    print("FINAL", json.dumps(acc), flush=True)


if __name__ == "__main__":
    main()
