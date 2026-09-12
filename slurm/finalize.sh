#!/bin/bash
# Runs on the cluster after a grid finishes (SLURM dependency), so results are
# collected, backed up and published even if the laptop is closed, the network
# drops, or the session that launched the work is long gone.
#
#   1. regenerate figures + tables from every run
#   2. back up to HOME (the only nightly-backed-up tier)
#   3. back up weights to SHELL (1 TB, no 90-day purge unlike scratch)
#   4. push to the Hugging Face Hub, with retries
#
#SBATCH -J finalize
#SBATCH -t 03:00:00
#SBATCH -c 4
#SBATCH --mem=16g
#SBATCH -p standard
#SBATCH -o logs/finalize-%j.out
#SBATCH --open-mode=append

source /etc/profile
module load pytorch/2.0.1

USER_SCR=/scratch/zt1/project/msml612/user/$USER
STAR=$USER_SCR/pcmd
export PYTHONPATH=$USER_SCR/pylibs:$PYTHONPATH
# Compute nodes cannot reach HF's Xet CAS endpoint; force the plain LFS path.
export HF_HUB_DISABLE_XET=1

HOME_BK=$HOME/pcmd-backup
SHELL_BK=/afs/shell.umd.edu/project/msml612/user-$USER/msml612-weights

echo "=== 1. collect + plot ==="
cd $STAR/src
RUNS=$STAR/runs OUT=$STAR/figures python collect.py || echo "collect failed (continuing)"

echo "=== 2. back up to HOME (backed up nightly) ==="
mkdir -p $HOME_BK/{results,figures,src,slurm}
cp -f $STAR/figures/* $HOME_BK/figures/ 2>/dev/null
cp -f $STAR/src/*.py $HOME_BK/src/ 2>/dev/null
cp -f $STAR/*.sh $HOME_BK/slurm/ 2>/dev/null
for d in $STAR/runs/*/; do
  n=$(basename $d)
  [ -f "$d/result.json" ] && cp -f "$d/result.json" "$HOME_BK/results/$n.json"
done
echo "  results backed up: $(ls $HOME_BK/results | wc -l)"

echo "=== 3. back up weights to SHELL (not purged) ==="
if mkdir -p $SHELL_BK 2>/dev/null; then
  for d in $STAR/runs/*/; do
    n=$(basename $d)
    [ -f "$d/model.pt" ] && cp -f "$d/model.pt" "$SHELL_BK/$n.pt" 2>/dev/null
  done
  echo "  weights backed up: $(ls $SHELL_BK 2>/dev/null | wc -l)"
else
  echo "  SHELL not writable from this node - weights remain on scratch"
fi

echo "=== 4. push to Hugging Face (with retries) ==="
# A transient network failure must not mean the results never reach the Hub, so
# retry with backoff rather than giving up on the first error.
for attempt in 1 2 3 4 5; do
  python - <<'PY' && break
import os, pathlib, sys
from huggingface_hub import HfApi

tok = pathlib.Path(os.path.expanduser("~/.hf_token")).read_text().strip()
star = os.path.expanduser(f"/scratch/zt1/project/msml612/user/{os.environ['USER']}/pcmd")
stage = pathlib.Path(os.path.expanduser("~/pcmd-backup"))
repo = "GOVINDFROM/parallel-consistent-masked-diffusion"

summary = pathlib.Path(star, "figures", "summary.md")
table = summary.read_text() if summary.exists() else "_(pending)_"

(stage / "README.md").write_text(f"""---
license: apache-2.0
tags: [masked-diffusion, discrete-diffusion, length-generalization, positional-encoding, from-scratch]
---

# Length Generalization in Masked Diffusion Language Models

From-scratch masked diffusion language models (MDLM-style absorbing state) and
matched autoregressive baselines, trained on arithmetic and algorithmic tasks and
evaluated far beyond the lengths seen in training.

**No pretrained weights or tokenizers are used anywhere.**

Code: https://github.com/GIND123/MSML612

## Results

{table}
""")

api = HfApi(token=tok)
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
api.upload_folder(folder_path=str(stage), repo_id=repo, repo_type="model",
                  commit_message="Automated push from Zaratan")
print("pushed:", f"https://huggingface.co/{repo}")
PY
  echo "  push attempt $attempt failed; sleeping $((attempt * 60))s"
  sleep $((attempt * 60))
done

echo "=== finalize complete ==="
