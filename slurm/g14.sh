#!/bin/bash
# Grid 14 - THE THEORY TEST, now at a learning rate that is not broken.
#
# THEORY.md predicts the minimum depth for one-pass decoding grows as log2(n),
# because addition's carry is a prefix scan computable by carry lookahead in
# O(log n) depth, and constant-depth transformers sit in TC0 which contains
# addition.
#
#   log-depth (ours)   : min depth ~ 2, 3, 4, 4  for 4, 8, 12, 16 digits
#   ripple-carry       : min depth ~ 4, 8, 12, 16
#   impossibility      : no depth suffices
#
# The first attempt at this sweep was cancelled on discovering it would have run
# at lr 3e-4 with 500 warmup, where deep models fail for optimisation reasons and
# would have faked a clean "depth does not help" curve.
#SBATCH -J g14
#SBATCH -p gpu-h100
#SBATCH --gpus=h100:1
#SBATCH -c 8
#SBATCH --mem=48g
#SBATCH -t 08:00:00
#SBATCH -o logs/g14-%A_%a.out
#SBATCH --array=0-47%12
source /etc/profile
SCR=/scratch/zt1/project/msml612/user/$USER
module load python/3.10.10
export PYTHONPATH=
cd $SCR/pcmd/src
DIG=(4 8 12 16); LAYERS=(2 3 4 6 8 12); SEEDS=(0 1)
I=$SLURM_ARRAY_TASK_ID
D=${DIG[$((I / 12))]}; L=${LAYERS[$(((I % 12) / 2))]}; S=${SEEDS[$((I % 2))]}
SL=$(( 3 * D + 4 ))
NAME="g14-d$D-L$L-s$S"
echo "=== $NAME ==="
$SCR/hf_env/bin/python train_add.py --mode full --digits $D --seq_len $SL --seed $S \
  --n_train 800000 --n_eval 400 --eval_bs 100 \
  --d 384 --layers $L --heads 6 --bs 128 --steps 60000 --lr 1e-4 --warmup 4000 \
  --K 4 --lam 1.0 --out $SCR/pcmd/runs/$NAME
