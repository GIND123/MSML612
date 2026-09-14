#!/bin/bash
# Grid 18 - does budget conditioning remove the crossover?
#
# At 20 digits, same config as g13, we already have both endpoints measured:
#   mdlm  0.0 @1 pass, 0.0 @sequential   (never learned the task)
#   full 99.8 @1 pass, 85.7 @sequential  (wins at 1, LOSES at 20)
# so this grid only needs the two new arms to complete a 4-way comparison at a
# single configuration.
#
#   anybudget     sample K per example AND condition the model on it
#   anybudget_nc  sample K identically but do NOT condition - the ablation that
#                 separates "mixing budgets" from "knowing the budget"
#
# Prediction: anybudget is strong at BOTH ends; anybudget_nc improves the
# sequential end but gives back the one-pass end, because without conditioning
# one set of weights has to serve contradictory targets.
#SBATCH -J g18
#SBATCH -p gpu-h100
#SBATCH --gpus=h100:1
#SBATCH -c 8
#SBATCH --mem=64g
#SBATCH -t 8:00:00
#SBATCH -o logs/g18-%A_%a.out
#SBATCH --array=0-5%6
source /etc/profile
SCR=/scratch/zt1/project/msml612/user/$USER
module load python/3.10.10
export PYTHONPATH=
cd $SCR/pcmd/src
MODES=(anybudget anybudget_nc); SEEDS=(0 1 2)
I=$SLURM_ARRAY_TASK_ID
M=${MODES[$((I / 3))]}; S=${SEEDS[$((I % 3))]}
NAME="g18-d20-$M-s$S"
echo "=== $NAME ==="
$SCR/hf_env/bin/python -u train_add.py --mode $M --digits 20 --seq_len 64 --seed $S \
  --n_train 1500000 --n_eval 400 --eval_bs 100 \
  --d 512 --layers 12 --heads 8 --bs 128 --steps 150000 --lr 1e-4 --warmup 6000 \
  --out $SCR/pcmd/runs/$NAME
