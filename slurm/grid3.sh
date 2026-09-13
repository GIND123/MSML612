#!/bin/bash
# Grid 3 — trace the likelihood <-> parallel-decodability trade-off.
#
# Grid 2 showed coordination alone is cancelled by the ELBO: the sequential
# teacher picks an arbitrary value for a free group, so averaged over batches
# its target is still uniform, while the ELBO actively pulls back to uniform.
# The weight IS the experiment. gamma penalises marginal entropy (explicit
# symmetry breaking); elbo_w down-weights the term fighting it.
#SBATCH -J g3
#SBATCH -p gpu-h100
#SBATCH --gpus=h100:1
#SBATCH -c 8
#SBATCH --mem=64g
#SBATCH -t 06:00:00
#SBATCH -o logs/g3-%A_%a.out
#SBATCH --array=0-29%12

source /etc/profile
SCR=/scratch/zt1/project/msml612/user/$USER
module load python/3.10.10
export PYTHONPATH=
PY=$SCR/hf_env/bin/python
cd $SCR/pcmd/src

I=$SLURM_ARRAY_TASK_ID
GAMMA=(0.0 0.1 0.5 1.0 2.0); ELBOW=(1.0 0.2 0.05); SEEDS=(0 1)
G=${GAMMA[$((I / 6))]}
E=${ELBOW[$(((I % 6) / 2))]}
S=${SEEDS[$((I % 2))]}

NAME="g3-gam$G-elbo$E-s$S"
echo "=== $NAME ==="
$PY train.py --task agree --objective coord --coupling 4 --seed $S \
  --seq_len 128 --n_train 200000 --n_eval 500 --eval_bs 100 \
  --d 384 --layers 6 --heads 6 --bs 64 --steps 20000 --lr 3e-4 \
  --beta 1.0 --gamma $G --elbo_w $E --group 4 --coord_every 4 --ramp_frac 0.2 \
  --out $SCR/pcmd/runs/$NAME
