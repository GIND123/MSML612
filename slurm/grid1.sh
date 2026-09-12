#!/bin/bash
# Grid 1 — learnability gate and the first quality-vs-parallelism curves.
#
# Doubles as a gate: if a model cannot solve a probe even at FULLY SEQUENTIAL
# decoding (128 passes for a 128-token canvas), the task is not learnable at
# this scale and no parallel-decoding claim built on it would mean anything.
# Launching a large grid before establishing this is the mistake that cost a
# day on the previous project.
#SBATCH -J g1
#SBATCH -p gpu-h100
#SBATCH --gpus=h100:1
#SBATCH -c 8
#SBATCH --mem=64g
#SBATCH -t 04:00:00
#SBATCH -o logs/g1-%A_%a.out
#SBATCH --array=0-11%12

source /etc/profile
SCR=/scratch/zt1/project/msml612/user/$USER
module load python/3.10.10
export PYTHONPATH=
PY=$SCR/hf_env/bin/python
cd $SCR/pcmd/src

I=$SLURM_ARRAY_TASK_ID
TASKS=(copy agree parity); OBJ=(mdlm selfcorrupt); SEEDS=(0 1)
T=${TASKS[$((I / 4))]}
O=${OBJ[$(((I % 4) / 2))]}
S=${SEEDS[$((I % 2))]}

NAME="g1-$T-$O-s$S"
echo "=== $NAME ==="
$PY train.py --task $T --objective $O --coupling 4 --seed $S \
  --seq_len 128 --n_train 200000 --n_eval 500 --eval_bs 100 \
  --d 384 --layers 6 --heads 6 --bs 64 --steps 20000 --lr 3e-4 \
  --out $SCR/pcmd/runs/$NAME
