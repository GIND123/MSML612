"""Datasets: text8 for language modelling, plus dependency-controlled probes.

text8 is the standard small-scale benchmark for discrete diffusion language
models (D3PM, SEDD, MDLM all report bits-per-character on it), which is what
makes a from-scratch comparison meaningful.

The probes exist because text8 alone cannot isolate the phenomenon we study.
Parallel decoding fails when simultaneously-revealed tokens depend on each
other, so we need tasks where that dependence is a knob rather than a property
of English.
"""
import os
import numpy as np

TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
# text8 is lowercase letters plus space: a 27-symbol alphabet, no tokenizer.
ALPHABET = " abcdefghijklmnopqrstuvwxyz"
SPECIALS = ["[MASK]", "[PAD]"]


class CharTokenizer:
    def __init__(self):
        self.itos = SPECIALS + list(ALPHABET)
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.mask = self.stoi["[MASK]"]
        self.pad = self.stoi["[PAD]"]

    def __len__(self):
        return len(self.itos)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


def load_text8(root, seq_len=256):
    """Standard 90M/5M/5M character split used throughout the literature."""
    raw = os.path.join(root, "text8")
    if not os.path.exists(raw):
        os.makedirs(root, exist_ok=True)
        import zipfile, io, requests   # only needed on first download
        r = requests.get(TEXT8_URL, timeout=300)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extract("text8", root)
    text = open(raw).read()
    assert len(text) == 100_000_000, f"unexpected text8 length {len(text)}"

    tok = CharTokenizer()
    cache = os.path.join(root, "text8.npy")
    if os.path.exists(cache):
        ids = np.load(cache)
    else:
        ids = np.array(tok.encode(text), dtype=np.uint8)
        np.save(cache, ids)

    tr, va = ids[:90_000_000], ids[90_000_000:95_000_000]
    te = ids[95_000_000:]

    def chunk(a):
        n = len(a) // seq_len
        return a[: n * seq_len].reshape(n, seq_len).astype(np.int64)

    return chunk(tr), chunk(va), chunk(te), tok


# ---------------------------------------------------------------------------
# Dependency-controlled probes.
#
# Each item is a sequence whose tokens must agree with each other. `coupling`
# sets how many positions are bound together, which is the exact quantity that
# breaks parallel decoding: revealing two mutually-dependent positions in one
# step forces the model to guess them independently.
# ---------------------------------------------------------------------------

def build_probe(n, seq_len, coupling, seed=0, task="copy"):
    """
    copy    : second half must equal the first half. Each answer position is
              fixed by one context position, so answer positions are mutually
              INDEPENDENT given the context - parallel decoding should be safe.
    agree   : within each group of `coupling` positions all tokens must be equal
              AND adjacent groups must differ. The value of a group is free, so
              its members are mutually DEPENDENT: revealing two of them in one
              step is a 1/V coin flip. This is the failure mode, isolated.
    parity  : groups are a run of bits whose last element is the XOR of the
              rest, with adjacent groups forced to differ, so the group is
              jointly constrained without any member being determined alone.

    The adjacent-groups-differ rule matters: without it a model that emits one
    constant token everywhere satisfies "all members agree" perfectly, and an
    untrained model scores 100% at maximum parallelism. That degenerate solution
    made the first version of this metric meaningless.
    """
    tok = CharTokenizer()
    rng = np.random.default_rng(seed)
    V = 2 + 26                       # specials + letters
    lo, hi = 2, V                    # sample letters only

    X = np.zeros((n, seq_len), dtype=np.int64)
    for i in range(n):
        if task == "copy":
            half = seq_len // 2
            src = rng.integers(lo, hi, size=half)
            X[i, :half] = src
            X[i, half:half * 2] = src
        elif task == "agree":
            g, prev = coupling, -1
            for s in range(0, seq_len, g):
                v = int(rng.integers(lo, hi))
                while v == prev:                      # adjacent groups must differ
                    v = int(rng.integers(lo, hi))
                X[i, s:s + g] = v
                prev = v
        elif task == "parity":
            g, prev = coupling, None
            for s in range(0, seq_len, g):
                end = min(s + g, seq_len)
                if end - s < 2:
                    X[i, s:end] = lo
                    continue
                while True:
                    bits = list(rng.integers(0, 2, size=end - s - 1))
                    vals = bits + [int(sum(bits) % 2)]
                    if vals != prev:                  # adjacent groups must differ
                        break
                X[i, s:end] = [lo + v for v in vals]
                prev = vals
        else:
            raise ValueError(task)
    return X, tok
