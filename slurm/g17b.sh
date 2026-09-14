#!/bin/bash
# Grid 17 - COMPUTE-MATCHED CONTROL.
#
# `full` evaluates the model twice per training step (matched-schedule term plus
# high-ratio term), so at equal step counts it spends roughly twice the training
# FLOPs of the baseline. Any reviewer will ask whether the gap is the method or
# simply the extra compute. This gives the BASELINE the larger budget: 300k
# steps of plain MDLM against 150k steps of the method, which is if anything
# generous to the baseline since the method's second pass is on a shared batch.
#SBATCH -J g17
#SBATCH -p gpu-h100
#SBATCH --gpus=h100:1
#SBATCH -c 8
#SBATCH --mem=64g
#SBATCH -t 24:00:00
#SBATCH -o logs/g17-%A_%a.out
#SBATCH --array=0-3%4
source /etc/profile
SCR=/scratch/zt1/project/msml612/user/$USER
module load python/3.10.10
export PYTHONPATH=
cd $SCR/pcmd/src
DIG=(16 20); SEEDS=(0 1)
I=$SLURM_ARRAY_TASK_ID
D=${DIG[$((I / 2))]}; S=${SEEDS[$((I % 2))]}
SL=$(( 3 * D + 4 ))
NAME="g17-d$D-mdlm2x-s$S"
echo "=== $NAME : baseline with DOUBLE the training steps ==="
$SCR/hf_env/bin/python -u train_add.py --mode mdlm --digits $D --seq_len $SL --seed $S \
  --n_train 1500000 --n_eval 400 --eval_bs 100 \
  --d 512 --layers 12 --heads 8 --bs 128 --steps 300000 --lr 1e-4 --warmup 6000 \
  --out $SCR/pcmd/runs/$NAME
