"""QM9 molecular graphs as flat token sequences.

QM9 is the standard small-molecule generation benchmark and the one DiGress and
the discrete graph-diffusion line report on. Molecules have at most 9 heavy
atoms drawn from {C, N, O, F}, with bonds in {single, double, triple} once
kekulised, so a molecule is fully described by

    9 node slots   (atom type, or absent)
  + 36 edge slots  (bond type, or absent)      = 45 tokens.

That is small enough to train from scratch cheaply, which is the point: the
claim under test is about the training schedule and the decoding rule, not about
model scale.

Why molecules are the right Phase 2. Validity here is a genuine JOINT
constraint - valency must balance at every atom simultaneously - so mode (ii) is
both dominant and, unlike natural text, *checkable*: RDKit says yes or no. The
metric is therefore unambiguous in a way generative perplexity is not, and the
field's live axis (validity at few denoising steps) is exactly the
quality-vs-compute frontier this project is about.

Node and edge slots take disjoint symbol sets, which is a property of the
representation rather than something a model should spend capacity
rediscovering; `allowed_mask` expresses it and the decoders enforce it.
"""
import csv
import os

import numpy as np

MAX_ATOMS = 9
ATOMS = ["C", "N", "O", "F"]
# Kekulised: aromatic rings are stored as alternating single/double bonds, so
# no aromaticity perception is needed to rebuild a molecule. Storing AROMATIC
# directly loses the per-ATOM aromatic flags, and reconstruction then fails with
# "Can't kekulize mol" on every aromatic species - 7% of QM9, and 100% of the
# round-trip failures measured before this fix.
BONDS = ["SINGLE", "DOUBLE", "TRIPLE"]
N_EDGE = MAX_ATOMS * (MAX_ATOMS - 1) // 2      # 36
SEQ_LEN = MAX_ATOMS + N_EDGE                   # 45

EDGES = [(i, j) for i in range(MAX_ATOMS) for j in range(i + 1, MAX_ATOMS)]


class MolTokenizer:
    """One vocabulary, but node and edge positions draw from disjoint parts.

      0            no atom / no bond  (shared "absent" symbol)
      1..4         C N O F                       (node slots only)
      5..7         single, double, triple        (edge slots only)
      8            [MASK]
    """

    def __init__(self):
        self.itos = (["<none>"] + ATOMS + BONDS + ["[MASK]"])
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.mask = self.stoi["[MASK]"]
        self.pad = self.mask
        self.none = 0
        self.atom_ids = [self.stoi[a] for a in ATOMS]
        self.bond_ids = [self.stoi[b] for b in BONDS]

    def __len__(self):
        return len(self.itos)

    def allowed_mask(self):
        """(SEQ_LEN, V) boolean: which symbols each position may take."""
        m = np.zeros((SEQ_LEN, len(self)), dtype=bool)
        m[:MAX_ATOMS, self.none] = True
        for i in self.atom_ids:
            m[:MAX_ATOMS, i] = True
        m[MAX_ATOMS:, self.none] = True
        for i in self.bond_ids:
            m[MAX_ATOMS:, i] = True
        return m


def mol_to_seq(mol, tok):
    """RDKit molecule -> (SEQ_LEN,) token array, or None if it cannot be encoded.

    Rejects molecules this representation cannot express exactly, rather than
    encoding them lossily: anything over MAX_ATOMS, anything carrying a formal
    charge (the atom vocabulary is uncharged), and anything that will not
    kekulise. Encoding must be lossless or generated-sample validity is measured
    against a target the encoding itself cannot hit.
    """
    from rdkit import Chem
    n = mol.GetNumAtoms()
    if n > MAX_ATOMS:
        return None
    if any(a.GetFormalCharge() != 0 for a in mol.GetAtoms()):
        return None
    try:
        mol = Chem.Mol(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        return None
    seq = np.zeros(SEQ_LEN, dtype=np.int64)
    for a in mol.GetAtoms():
        s = a.GetSymbol()
        if s not in tok.stoi:
            return None
        seq[a.GetIdx()] = tok.stoi[s]
    pos = {e: k for k, e in enumerate(EDGES)}
    for b in mol.GetBonds():
        i, j = sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        bt = str(b.GetBondType())
        if bt not in tok.stoi:
            return None
        seq[MAX_ATOMS + pos[(i, j)]] = tok.stoi[bt]
    return seq


def seq_to_mol(seq, tok):
    """Token array -> RDKit molecule (unsanitised), or None if incoherent."""
    from rdkit import Chem
    bond_map = {tok.stoi["SINGLE"]: Chem.BondType.SINGLE,
                tok.stoi["DOUBLE"]: Chem.BondType.DOUBLE,
                tok.stoi["TRIPLE"]: Chem.BondType.TRIPLE}
    m = Chem.RWMol()
    idx = {}
    for i in range(MAX_ATOMS):
        t = int(seq[i])
        if t in tok.atom_ids:
            idx[i] = m.AddAtom(Chem.Atom(tok.itos[t]))
    if not idx:
        return None
    for k, (i, j) in enumerate(EDGES):
        t = int(seq[MAX_ATOMS + k])
        if t in bond_map:
            # a bond to an absent atom is incoherent, not merely invalid
            if i not in idx or j not in idx:
                return None
            m.AddBond(idx[i], idx[j], bond_map[t])
    return m


def validity(seqs, tok):
    """Fraction that sanitise, plus canonical SMILES of those that do.

    Sanitisation is the standard validity criterion for this benchmark: it
    enforces valency at every atom at once, which is precisely the joint
    constraint parallel decoding risks breaking.
    """
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    smiles = []
    for s in seqs:
        m = seq_to_mol(s, tok)
        if m is None:
            continue
        try:
            mm = m.GetMol()
            Chem.SanitizeMol(mm)
            smi = Chem.MolToSmiles(mm)
            if smi:
                smiles.append(smi)
        except Exception:
            continue
    return len(smiles) / max(1, len(seqs)), smiles


def load_qm9(root="data/qm9", limit=None):
    """Return (sequences, tokenizer, train_smiles_set).

    The SMILES set is kept so novelty can be measured against the training data
    rather than only uniqueness within a sample.
    """
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    path = os.path.join(root, "qm9.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Fetch once with:\n  mkdir -p {root} && curl -sL -o {path} "
            "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/qm9.csv")
    tok = MolTokenizer()
    seqs, smis = [], set()
    n_kept, n_seen = [0], 0
    with open(path) as f:
        for r in csv.DictReader(f):
            n_seen += 1
            smi = r.get("smiles")
            if not smi:
                continue
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            s = mol_to_seq(m, tok)
            if s is None:
                continue
            seqs.append(s)
            smis.add(Chem.MolToSmiles(m))
            n_kept[0] += 1
            if limit and len(seqs) >= limit:
                break
    print(f"qm9: kept {n_kept[0]}/{n_seen} molecules "
          f"({100*n_kept[0]/max(1,n_seen):.1f}%); the remainder are charged, "
          f"over {MAX_ATOMS} heavy atoms, or will not kekulise")
    return np.stack(seqs), tok, smis


if __name__ == "__main__":
    X, tok, smis = load_qm9(os.environ.get("QM9ROOT", "data/qm9"), limit=5000)
    print(f"sequences {X.shape}  vocab {len(tok)}  unique SMILES {len(smis)}")
    am = tok.allowed_mask()
    print(f"allowed mask {am.shape}: node slots allow {am[0].sum()} symbols, "
          f"edge slots allow {am[MAX_ATOMS].sum()}")
    # round-trip: every training molecule must re-sanitise from its own tokens
    v, sm = validity(X[:2000], tok)
    print(f"round-trip validity on real data: {v*100:.2f}%  "
          f"(must be ~100% or the encoding is lossy)")
