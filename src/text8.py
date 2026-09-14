"""text8: the standard character-level language-modelling benchmark.

100M characters of cleaned English Wikipedia, vocabulary of 27 symbols (a-z and
space). The conventional split is the first 90M characters for training, the
next 5M for validation and the last 5M for test; it is used unchanged here so
bits-per-character is comparable with published numbers.

This is the task that separates the two failure modes of parallel decoding.
Addition has unique answers, so every conditional is a point mass and all of the
damage comes from marginals that were never trained at high mask ratios. Natural
text has both kinds of position at once: closing a word, a suffix or a function
word is nearly deterministic given the context, while the choice of the next
content word carries real entropy. A method that claims to work "across all
things" has to handle a sequence in which both appear together.
"""
import os
import numpy as np

URL = "http://mattmahoney.net/dc/text8.zip"
VOCAB = " abcdefghijklmnopqrstuvwxyz"      # 27 symbols, exactly as distributed

# the conventional text8 split, in characters
N_TRAIN, N_VALID, N_TEST = 90_000_000, 5_000_000, 5_000_000


class Text8Tokenizer:
    """Character vocabulary plus the absorbing [MASK] state.

    [MASK] is appended after the 27 real characters, so ids 0..26 are the data
    symbols and id 27 is the absorbing state. No padding symbol is needed: every
    training sequence is a fixed-length contiguous window of the corpus.
    """

    def __init__(self):
        self.itos = list(VOCAB) + ["[MASK]"]
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.mask = self.stoi["[MASK]"]
        self.pad = self.mask          # unused; kept for interface compatibility

    def __len__(self):
        return len(self.itos)

    def encode(self, s):
        return np.array([self.stoi[c] for c in s if c in self.stoi], dtype=np.int64)

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def _raw(root):
    path = os.path.join(root, "text8")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Fetch it once with:\n"
            f"  mkdir -p {root} && cd {root} && curl -sLO {URL} && unzip -q text8.zip")
    with open(path, "r") as f:
        return f.read()


def load_text8(root="data/text8", seq_len=256):
    """Return (train, valid, test) integer arrays of shape (n_windows, seq_len).

    Windows are non-overlapping and contiguous, which keeps the evaluation
    splits strictly disjoint from training: no window can straddle a split
    boundary, so no test character is ever seen during training.
    """
    tok = Text8Tokenizer()
    text = _raw(root)
    if len(text) < N_TRAIN + N_VALID + N_TEST:
        raise ValueError(f"text8 is {len(text)} chars, expected 100,000,000")

    ids = tok.encode(text)
    tr = ids[:N_TRAIN]
    va = ids[N_TRAIN:N_TRAIN + N_VALID]
    te = ids[N_TRAIN + N_VALID:N_TRAIN + N_VALID + N_TEST]

    def windows(a):
        n = len(a) // seq_len
        return a[:n * seq_len].reshape(n, seq_len)

    return windows(tr), windows(va), windows(te), tok


def batches(arr, bs, seed=0, shuffle=True):
    """Infinite batch iterator over pre-windowed data."""
    rng = np.random.default_rng(seed)
    n = len(arr)
    while True:
        idx = rng.permutation(n) if shuffle else np.arange(n)
        for i in range(0, n - bs + 1, bs):
            yield arr[idx[i:i + bs]]
