"""CoNLL-SIGMORPHON 2017 morphological inflection.

Format per line:  lemma <TAB> inflected form <TAB> semicolon-separated tags
e.g.              swim        swam            V;PST

Why this is the right benchmark for one-pass decoding:

  * the answer is UNIQUE given lemma and tags, so every position's true marginal
    is a point mass and there is no information-theoretic barrier to committing
    the whole form in a single pass;
  * forms are short (about ten characters) with shallow, local dependencies, so
    unlike 16-digit addition the dependency chain fits comfortably inside one
    forward pass;
  * exact-match accuracy is the official metric - deterministic, no judges;
  * it is a shared task with published baselines across 52 languages, so breadth
    is measured across typologically diverse data rather than across seeds of
    one configuration.
"""
import os
import numpy as np

BASE = "https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1"
SPECIALS = ["[PAD]", "[MASK]", "[SEP]", "[EOS]"]


def download(lang, split, size, root):
    """split in {train, dev, test}; size in {low, medium, high} (train only)."""
    fn = f"{lang}-{split}" + (f"-{size}" if split == "train" else "")
    path = os.path.join(root, fn)
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(root, exist_ok=True)
        urllib.request.urlretrieve(f"{BASE}/{fn}", path)
    return path


def read(path):
    out = []
    for line in open(path, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 3:
            out.append((parts[0], parts[1], parts[2]))
    return out


class InflectionTokenizer:
    """Character vocabulary plus one symbol per morphological tag, built from
    the training split only - no pretrained tokenizer anywhere."""

    def __init__(self, rows):
        chars, tags = set(), set()
        for lemma, form, tag in rows:
            chars |= set(lemma) | set(form)
            tags |= set(tag.split(";"))
        self.itos = SPECIALS + sorted(chars) + [f"<{t}>" for t in sorted(tags)]
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.pad, self.mask = self.stoi["[PAD]"], self.stoi["[MASK]"]
        self.sep, self.eos = self.stoi["[SEP]"], self.stoi["[EOS]"]

    def __len__(self):
        return len(self.itos)

    def encode_src(self, lemma, tag):
        ids = [self.stoi[c] for c in lemma if c in self.stoi]
        ids += [self.stoi[f"<{t}>"] for t in tag.split(";") if f"<{t}>" in self.stoi]
        return ids

    def encode_tgt(self, form):
        return [self.stoi[c] for c in form if c in self.stoi] + [self.eos]

    def decode(self, ids):
        out = []
        for i in ids:
            s = self.itos[i]
            if s == "[EOS]":
                break
            if s not in SPECIALS and not s.startswith("<"):
                out.append(s)
        return "".join(out)


def build(rows, tok, seq_len):
    """One flat sequence: source, [SEP], then the answer region.

    The prompt stays visible throughout decoding and only the answer region is
    masked, exactly as in the addition setup.
    """
    X = np.full((len(rows), seq_len), tok.pad, dtype=np.int64)
    A = np.zeros((len(rows), seq_len), dtype=bool)
    kept = 0
    for lemma, form, tag in rows:
        src = tok.encode_src(lemma, tag)
        tgt = tok.encode_tgt(form)
        if len(src) + 1 + len(tgt) > seq_len:
            continue                      # drop the rare over-long item
        ids = src + [tok.sep] + tgt
        X[kept, : len(ids)] = ids
        A[kept, len(src) + 1: len(ids)] = True
        kept += 1
    return X[:kept], A[:kept]


def load_language(lang, root, size="medium", seq_len=48):
    tr = read(download(lang, "train", size, root))
    dv = read(download(lang, "dev", None, root))
    tok = InflectionTokenizer(tr)
    Xtr, Atr = build(tr, tok, seq_len)
    Xdv, Adv = build(dv, tok, seq_len)
    return (Xtr, Atr), (Xdv, Adv), tok
