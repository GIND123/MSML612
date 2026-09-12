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
    def __init__(self, vocab, d=384, n_layers=6, n_heads=6, pe="ape", causal=False, max_len=256):
        super().__init__()
        self.pe_kind, self.causal = pe, causal
        self.emb = nn.Embedding(vocab, d)
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

    def forward(self, idx, pad_mask=None, pos_ids=None):
        B, L = idx.shape
        x = self.emb(idx)
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
