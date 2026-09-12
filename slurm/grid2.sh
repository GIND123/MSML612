#!/bin/bash
# Grid 2 — the method vs its ablations vs the baseline, on all three probes.
#
# Objectives
#   mdlm        ELBO only (standard masked diffusion)
#   selfcorrupt ELBO + self-conditioning        (component 1)
#   coord       ELBO + coordination             (component 2, the core)
#   cmd         ELBO + both                     (full method)
#
# The probes make the claim falsifiable: `copy` has independent answer
# positions, so the method must NOT hurt there; `agree` and `parity` have
# jointly-dependent positions, which is where it must win.
#SBATCH -J g2
#SBATCH -p gpu-h100
#SBATCH --gpus=h100:1
#SBATCH -c 8
#SBATCH --mem=64g
#SBATCH -t 08:00:00
#SBATCH -o logs/g2-%A_%a.out
#SBATCH --array=0-23%12

source /etc/profile
SCR=/scratch/zt1/project/msml612/user/$USER
module load python/3.10.10
export PYTHONPATH=
PY=$SCR/hf_env/bin/python
cd $SCR/pcmd/src

I=$SLURM_ARRAY_TASK_ID
TASKS=(copy agree parity); OBJ=(mdlm selfcorrupt coord cmd); SEEDS=(0 1)
T=${TASKS[$((I / 8))]}
O=${OBJ[$(((I % 8) / 2))]}
S=${SEEDS[$((I % 2))]}

NAME="g2-$T-$O-s$S"
echo "=== $NAME ==="
$PY train.py --task $T --objective $O --coupling 4 --seed $S \
  --seq_len 128 --n_train 200000 --n_eval 500 --eval_bs 100 \
  --d 384 --layers 6 --heads 6 --bs 64 --steps 20000 --lr 3e-4 \
  --alpha 0.5 --beta 1.0 --group 4 --coord_every 4 --ramp_frac 0.3 \
  --out $SCR/pcmd/runs/$NAME
