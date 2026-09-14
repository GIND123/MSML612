#!/bin/bash
# Grid 9 - THE FALSIFIABLE TEST of the theory in THEORY.md.
#
# Addition's carry is a prefix scan, computable in O(log n) depth by carry
# lookahead, and constant-depth transformers sit inside uniform TC0 which
# contains addition. So one-pass decoding should become possible once depth
# reaches ~log2(n), and the minimum depth should grow LOGARITHMICALLY with the
# number of digits.
#
#   log-depth account  : min depth ~ log2(n)   -> 2, 3, 4, 5
#   ripple-carry account: min depth ~ n        -> 4, 8, 16, 32
#   "transformers cannot do this": no depth suffices
#
# These are cleanly distinguishable. Sweep depth x digits and read off where
# one-pass accuracy crosses 90%.
#SBATCH -J g9
#SBATCH -p gpu-h100
#SBATCH --gpus=h100:1
#SBATCH -c 8
#SBATCH --mem=64g
#SBATCH -t 12:00:00
#SBATCH -o logs/g9-%A_%a.out
#SBATCH --array=0-47%12
source /etc/profile
SCR=/scratch/zt1/project/msml612/user/$USER
module load python/3.10.10
export PYTHONPATH=
PY=$SCR/hf_env/bin/python
cd $SCR/pcmd/src

I=$SLURM_ARRAY_TASK_ID
DIG=(4 8 16 32); LAYERS=(2 3 4 6 8 12); SEEDS=(0 1)
D=${DIG[$((I / 12))]}
L=${LAYERS[$(((I % 12) / 2))]}
S=${SEEDS[$((I % 2))]}
SL=$(( 3 * D + 4 ))

NAME="g9-d$D-L$L-s$S"
echo "=== $NAME ==="
$PY train_add.py --mode prog --digits $D --seq_len $SL --seed $S \
  --n_train 500000 --n_eval 400 --eval_bs 100 \
  --d 384 --layers $L --heads 6 --bs 128 --steps 60000 --lr 3e-4 \
  --K_max 16 --lam 1.0 --out $SCR/pcmd/runs/$NAME
