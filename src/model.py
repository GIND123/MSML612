"""Transformer denoiser for masked diffusion.

Bidirectional by default (a masked diffusion denoiser must see both sides);
`causal=True` gives the matched autoregressive baseline. Rotary is the default
positional encoding - it was the strongest option in our previous study.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal(max_len, d):
    pe = torch.zeros(max_len, d)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def alibi_slopes(n_heads):
    start = 2 ** (-(2 ** -(math.log2(n_heads) - 3))) if math.log2(n_heads).is_integer() else 0.5
    return torch.tensor([start * (start ** i) for i in range(n_heads)])


def apply_rope(x, cos, sin):
    # x: (B, H, L, Dh)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rx = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rx.flatten(-2)


class Attention(nn.Module):
    def __init__(self, d, n_heads, pe, causal, max_len):
        super().__init__()
        self.h, self.dh, self.pe, self.causal = n_heads, d // n_heads, pe, causal
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        if pe == "alibi":
            self.register_buffer("slopes", alibi_slopes(n_heads), persistent=False)
        if pe == "rope":
            inv = 1.0 / (10000 ** (torch.arange(0, self.dh, 2).float() / self.dh))
            ang = torch.arange(max_len).float()[:, None] * inv[None, :]
            self.register_buffer("cos", ang.cos(), persistent=False)
            self.register_buffer("sin", ang.sin(), persistent=False)

    def forward(self, x, pad_mask, pos_ids=None):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.h, self.dh).transpose(1, 2)
        k = k.view(B, L, self.h, self.dh).transpose(1, 2)
        v = v.view(B, L, self.h, self.dh).transpose(1, 2)

        if self.pe == "rope":
            if pos_ids is None:
                cos, sin = self.cos[:L][None, None], self.sin[:L][None, None]
            else:
                cos, sin = self.cos[pos_ids][:, None], self.sin[pos_ids][:, None]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)

        if self.pe == "alibi":
            if pos_ids is None:
                pos = torch.arange(L, device=x.device)
                rel = (pos[None, :] - pos[:, None]).abs().float()[None]
            else:
                rel = (pos_ids[:, None, :] - pos_ids[:, :, None]).abs().float()
            att = att - self.slopes.view(1, -1, 1, 1) * rel[:, None]

        if self.causal:
            cm = torch.ones(L, L, device=x.device, dtype=torch.bool).tril()
            att = att.masked_fill(~cm[None, None], float("-inf"))
        if pad_mask is not None:
            att = att.masked_fill(~pad_mask[:, None, None, :], float("-inf"))

        att = att.softmax(-1)
        out = (att @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d, n_heads, pe, causal, max_len):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, n_heads, pe, causal, max_len)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, pad_mask, pos_ids=None):
        x = x + self.attn(self.n1(x), pad_mask, pos_ids)
        return x + self.mlp(self.n2(x))


class Transformer(nn.Module):
    def __init__(self, vocab, d=384, n_layers=6, n_heads=6, pe="ape", causal=False,
                 max_len=256, budget_bins=0):
        super().__init__()
        self.pe_kind, self.causal = pe, causal
        self.budget_bins = budget_bins
        self.emb = nn.Embedding(vocab, d)
        # Budget conditioning. A K-pass decode only visits mask ratios
        # {1, (K-1)/K, ..., 1/K}, so a model trained for one K is mis-specified
        # at every other K - measured directly: our 20-digit model scores 99.8%
        # at one pass and 85.7% at twenty. Telling the model which budget it is
        # being decoded at lets ONE network serve every budget instead of a
        # family of networks each good at a single point.
        # budget_bins=0 keeps the parameter absent entirely, so checkpoints
        # trained before this existed still load.
        if budget_bins:
            self.budget_emb = nn.Embedding(budget_bins, d)
        # Segment embedding: which part of the equation a token belongs to.
        # Place-value ids deliberately collide across operands and answer, so a
        # bidirectional model needs this to tell those tokens apart; a causal
        # model gets the same information free from the attention mask.

        if pe == "ape":
            self.pos = nn.Embedding(max_len, d)
        elif pe == "sin":
            self.register_buffer("pos_sin", sinusoidal(max_len, d), persistent=False)
        self.blocks = nn.ModuleList([Block(d, n_heads, pe, causal, max_len) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def budget_bin(K, n_bins, device):
        """Map a decoding budget K to a bin index by log2, clamped.

        Budgets are used on a log scale (1, 2, 4, ... L), so log2 is the natural
        parameterisation and keeps the table small.
        """
        if not torch.is_tensor(K):
            K = torch.as_tensor(K, device=device)
        K = K.to(device).float().clamp_min(1.0)
        return torch.log2(K).round().long().clamp_(0, n_bins - 1)

    def forward(self, idx, pad_mask=None, pos_ids=None, budget=None):
        B, L = idx.shape
        x = self.emb(idx)
        if self.budget_bins and budget is not None:
            b = self.budget_bin(budget, self.budget_bins, idx.device)
            if b.ndim == 0:
                b = b.expand(B)
            x = x + self.budget_emb(b)[:, None, :]
        if self.pe_kind == "ape":
            p = torch.arange(L, device=idx.device)[None] if pos_ids is None else pos_ids
            x = x + self.pos(p)
        elif self.pe_kind == "sin":
            p = torch.arange(L, device=idx.device)[None] if pos_ids is None else pos_ids
            x = x + self.pos_sin[p]
        for b in self.blocks:
            x = b(x, pad_mask, pos_ids)
        return self.head(self.norm(x))

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RecurrentDenoiser(nn.Module):
    """A weight-shared block applied R times, supervised at every step.

    Motivated by a measurement, not a hunch. On hard Sudoku a 37.86M-parameter
    12-layer denoiser reaches 89.4%, while the Recurrent Transformer of Yang et
    al. (2023) reaches 99.5% on a comparable split with 211k parameters - 180x
    smaller. The difference is not capacity, it is that constraint propagation is
    an ITERATIVE algorithm: solving a hard Sudoku takes tens of rounds of
    "eliminate, propagate, repeat". A fixed 12-layer feedforward network cannot
    express thirty rounds of propagation at any width; one block applied thirty
    times can.

    This is the same argument THEORY.md makes for addition - that a K-pass decode
    has effective depth L*K and "borrows depth from the decoding loop" - applied
    to the architecture instead of to the decoder.

    Two details carry the method:

      INPUT INJECTION. The token embedding is re-added at every recurrence.
      Without it the input signal decays through thirty applications of the same
      block and the model forgets the clues it is solving for.

      DEEP SUPERVISION. Training loss is taken at EVERY recurrence, not only the
      last. That forces each application to make progress on its own rather than
      letting the stack learn one entangled thirty-step function, and it is what
      lets inference run MORE recurrences than training - the per-step objective
      is the same at every step, so the map is iterable beyond where it was fit.
    """

    def __init__(self, vocab, d=128, n_layers=1, n_heads=4, pe="ape",
                 max_len=128, recurrences=32, hidden_mult=4, inject=True):
        super().__init__()
        self.pe_kind, self.causal = pe, False
        self.recurrences = recurrences
        # `inject` exists to separate two things the headline comparison
        # otherwise conflates: weight sharing, and re-supplying the input at
        # every application. Turning it off keeps the architecture and the depth
        # identical and changes only the injection, which is the clean isolation.
        self.inject = inject
        self.emb = nn.Embedding(vocab, d)
        if pe == "ape":
            self.pos = nn.Embedding(max_len, d)
        elif pe == "sin":
            self.register_buffer("pos_sin", sinusoidal(max_len, d), persistent=False)
        self.blocks = nn.ModuleList([Block(d, n_heads, pe, False, max_len)
                                     for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(Transformer._init)

    def _inject(self, idx):
        x = self.emb(idx)
        if self.pe_kind == "ape":
            x = x + self.pos(torch.arange(idx.shape[1], device=idx.device))[None]
        elif self.pe_kind == "sin":
            x = x + self.pos_sin[torch.arange(idx.shape[1], device=idx.device)][None]
        return x

    def forward(self, idx, pad_mask=None, pos_ids=None, budget=None,
                recurrences=None, return_all=False):
        """Returns final logits, or the list of logits from every recurrence."""
        R = recurrences or self.recurrences
        inp = self._inject(idx)
        h = inp
        outs = []
        for _ in range(R):
            if self.inject:
                h = h + inp                  # input injection: keep the clues alive
            for b in self.blocks:
                h = b(h, pad_mask, pos_ids)
            outs.append(self.head(self.norm(h)))
        return outs if return_all else outs[-1]

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
