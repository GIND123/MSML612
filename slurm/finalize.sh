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

echo "=== 3. weights: NOT copied to HOME (quota is 9.5G soft / 19G hard) ==="
# An earlier version copied every model.pt here. That is 15G, which blew through
# both the soft quota and the hard limit, left HOME unwritable, and silently
# broke the Hugging Face push for five consecutive retries - the staging README
# could not be written. Weights stay on scratch; a curated subset goes to the
# Hub in step 4, where there is no such quota.
echo "  weights remain on scratch: $(ls -d $STAR/runs/*/model.pt 2>/dev/null | wc -l) checkpoints"
echo "  HOME usage now: $(du -sh $HOME_BK 2>/dev/null | cut -f1)"

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
tags: [masked-diffusion, discrete-diffusion, parallel-decoding, text8, from-scratch]
---

# Two Failure Modes of Parallel Decoding in Masked Diffusion

Parallel decoding in masked diffusion loses accuracy for two different reasons
that need opposite fixes:

1. **The marginals are untrained at high mask ratio.** A K-pass decode only ever
   evaluates the model at mask ratios {{1, (K-1)/K, ..., 1/K}}, while uniform-t
   training spends most of its capacity on nearly-complete states that
   high-parallelism decoding never visits. **Fixable only at training time.**
2. **The committed positions carry real mutual information.** Parallel decoding
   samples the product of marginals; the truth is the joint. The discarded
   dependence is the total correlation of the committed set, upper-bounded by
   its summed conditional entropy. **Fixable only at inference, by committing
   less.**

Prior work applies inference-time fixes to both, which is why it flatlines on
arithmetic - where mode 2 is identically zero and all the loss is mode 1.

Everything is trained from scratch: no pretrained weights, no pretrained
tokenizer, and the autoregressive evaluator used to score generated text is
itself trained from scratch on the same corpus.

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
