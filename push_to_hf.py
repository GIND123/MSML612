#!/usr/bin/env python3
"""Stage results from Zaratan and push them to the Hugging Face Hub.

Token is read from ../.env (chmod 600, never committed, never copied to the
shared cluster filesystem).
"""
import json, os, subprocess, sys, glob, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = HERE.parent / ".env"
REMOTE = "/scratch/zt1/project/msml612/user/govind02/pcmd"
REPO = os.environ.get("HF_REPO", "GOVINDFROM/parallel-consistent-masked-diffusion")
PRIVATE = os.environ.get("HF_PRIVATE", "1") == "1"
STAGE = HERE / "hf_stage"
SSH = ["-o", "ControlMaster=auto",
       "-o", f"ControlPath={Path.home()}/.ssh/cm-zaratan-%r@%h-%p",
       "-o", "ControlPersist=8h"]


def token():
    for line in ENV.read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip()
    sys.exit("HF_TOKEN not found in .env")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def rrun(cmd):
    return subprocess.run(["ssh", *SSH, "zaratan", "bash", "-l", "-c", cmd],
                          capture_output=True, text=True).stdout


def fetch():
    STAGE.mkdir(exist_ok=True)
    print("regenerating figures on the cluster ...")
    rrun(f"source /etc/profile; module load pytorch/2.0.1 >/dev/null 2>&1; cd {REMOTE}/src && "
         f"RUNS={REMOTE}/runs OUT={REMOTE}/figures python collect.py")

    for sub in ("figures", "results", "weights", "src"):
        (STAGE / sub).mkdir(exist_ok=True)

    print("copying figures + summary ...")
    subprocess.run(["scp", *SSH, "-q", f"zaratan:{REMOTE}/figures/*", str(STAGE / "figures")])

    print("copying result json ...")
    rrun("mkdir -p ~/pcmd-backup/results && "
         "for d in %s/runs/*/; do n=$(basename $d); "
         "[ -f \"$d/result.json\" ] && cp -f \"$d/result.json\" "
         "\"$HOME/pcmd-backup/results/$n.json\"; done" % REMOTE)
    subprocess.run(["scp", *SSH, "-q", "-r",
                    "zaratan:pcmd-backup/results/.", str(STAGE / "results")])
    got = list((STAGE / "results").glob("*.json"))
    print(f"  {len(got)} result files")
    names = [g.stem for g in got]

    # weights: seed 0 of each configuration keeps the upload to a sane size
    wanted = [n for n in names if n.endswith("-s0") or "-s0" in n]
    wanted = [n for n in wanted if n.startswith("coup-")] or wanted[:8]
    print(f"copying {len(wanted)} checkpoints ...")
    for n in wanted:
        subprocess.run(["scp", *SSH, "-q", f"zaratan:{REMOTE}/runs/{n}/model.pt",
                        str(STAGE / "weights" / f"{n}.pt")], capture_output=True)

    for f in glob.glob(str(HERE / "src" / "*.py")):
        shutil.copy(f, STAGE / "src")


def card():
    rows = []
    for f in sorted((STAGE / "results").glob("*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if r.get("final"):
            rows.append((f.stem, r["args"], r["final"]))
    summary = (STAGE / "figures" / "summary.md")
    table = summary.read_text() if summary.exists() else "_(pending)_"
    return f"""---
license: apache-2.0
tags: [masked-diffusion, discrete-diffusion, length-generalization, positional-encoding, from-scratch]
---

# Length Generalization in Masked Diffusion Language Models

From-scratch masked diffusion language models (MDLM-style absorbing-state
diffusion) and matched autoregressive baselines, trained on multi-digit addition
and evaluated on operand lengths never seen in training.

**No pretrained weights or tokenizers are used anywhere.** Every model is trained
from random initialization with a symbol-level vocabulary built from the data.

## What is here

- `figures/` — plots and the summary table
- `results/` — one JSON per run (config + accuracy on every test length)
- `weights/` — checkpoints (~10.7M parameters each)
- `src/` — full training, evaluation and plotting code

## Method

**Significance-aligned position ids.** Instead of numbering tokens by sequence
index, every digit is numbered by its place value, so digits that must be
combined share an id:

```
 4   7   +   8   5   =   1   3   2
 2   1   0   2   1   0   3   2   1
```

The rule "combine equal ids, carry into id+1" does not depend on operand length,
which is what allows extrapolation. A random per-example offset is added to all
ids so the model keys on relative place value and encounters large ids during
training.

## Findings

1. **Positional encodings do not transfer between the two architectures.**
   Removing positional information entirely, and sinusoidal encoding, both give
   100% in-distribution accuracy for autoregressive models and **0% for masked
   diffusion** — the diffusion model cannot learn the task at all (3 seeds each).
   A diffusion model attends bidirectionally and so has no implicit order to
   fall back on.

2. **Length-generalization accuracy has very high seed variance** on this task
   (individual seeds span 3%–68% for one configuration), so single-seed
   comparisons in this setting are unreliable.

## Results

{table}

## Setup

6 layers, 384 hidden dimension, 6 heads, ~10.7M parameters. AdamW, learning rate
1e-4, cosine schedule, bfloat16, effective batch 256. Trained on operands of 1-5
digits; evaluated at 5, 6, 7, 8, 10 and 12 digits. Exact-match accuracy — the
entire answer must be correct.

Trained on the University of Maryland Zaratan cluster (A100 MIG slices).
"""


def main():
    from huggingface_hub import HfApi
    tok = token()
    fetch()
    (STAGE / "README.md").write_text(card())

    api = HfApi(token=tok)
    api.create_repo(REPO, repo_type="model", private=PRIVATE, exist_ok=True)
    print(f"uploading to {REPO} (private={PRIVATE}) ...")
    api.upload_folder(folder_path=str(STAGE), repo_id=REPO, repo_type="model",
                      commit_message="Add results, figures and checkpoints")
    print(f"done: https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
