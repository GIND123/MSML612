"""Check a public NCM export against the fixtures shipped beside it.

The ONNX graphs are a third-party conversion of the official checkpoints, so
before any number from them goes into a report we confirm that (a) our prefix
encoding reproduces the fixture inputs exactly and (b) the graph reproduces the
fixture outputs exactly. If this script does not print VERIFIED, the numbers
from `run_ncm.py` mean nothing and the official PyTorch checkpoints should be
used instead.

    python ncm/verify_export.py sudoku-hard
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from ncm_encoding import PREFIX_LEN, encode_board, decode_cells, levels_from_output, is_legal

REPO = "mrfakename/ncm-webgpu-weights"
MODELS = Path(__file__).resolve().parent / ".models"


def load_session(task, repo=REPO):
    """Download the graph and its external weights, then open a CPU session.

    onnxruntime cannot follow the symlinks in the Hugging Face cache to reach
    external weight files, so both parts are pulled into a real directory.
    """
    meta = json.load(open(hf_hub_download(repo, f"{task}/meta.json")))
    model = hf_hub_download(repo, f"{task}/model.onnx", local_dir=MODELS)
    hf_hub_download(repo, f"{task}/model.onnx.data", local_dir=MODELS)
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    return sess, meta


def run_prefix(sess, prefix):
    """One board through the graph. Returns the decoded levels, coarse to fine."""
    name = sess.get_inputs()[0].name
    feed = {name: np.asarray(prefix, dtype=np.int64).reshape(1, PREFIX_LEN)}
    outs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feed)))
    return levels_from_output(outs["tokens"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["sudoku-easy", "sudoku-medium", "sudoku-hard"])
    ap.add_argument("--repo", default=REPO)
    a = ap.parse_args()

    fixtures = json.load(open(hf_hub_download(a.repo, f"{a.task}/fixtures.json")))
    fixtures = fixtures if isinstance(fixtures, list) else fixtures.get("fixtures", [fixtures])
    sess, meta = load_session(a.task, a.repo)

    print(f"{a.task}: {len(meta['levels'])} levels {meta['levels']}, prefix {meta['prefix_len']}")
    print(f"inputs  {[(i.name, i.shape) for i in sess.get_inputs()]}")
    print(f"outputs {[(o.name, o.shape) for o in sess.get_outputs()]}")

    ok = True
    for n, fx in enumerate(fixtures):
        prefix = list(fx["prefix"])
        puzzle = decode_cells(prefix)

        # (a) our encoding must rebuild the fixture prefix token for token
        if encode_board(puzzle) != prefix:
            print(f"fixture {n}: ENCODING MISMATCH")
            ok = False
            continue

        # (b) the graph must reproduce the reference tokens
        levels = run_prefix(sess, prefix)
        if "tokens" in fx:
            ref = np.asarray(fx["tokens"]).reshape(-1)
            got = levels.reshape(-1)
            same = got.shape == ref.shape and bool((got == ref).all())
            print(f"fixture {n}: tokens match {same}")
            ok &= same

        final = decode_cells(levels[-1])
        print(f"fixture {n}: {len(levels)} levels, final board legal {is_legal(final)}")

    print("VERIFIED" if ok else "MISMATCH - do not use these weights for reported numbers")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
