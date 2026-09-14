#!/bin/bash
# Grid 15 - component ablation on inflection. Which part carries the gain, and
# does the answer hold across typologically diverse languages rather than just
# on addition?
#SBATCH -J g15
#SBATCH -p gpu-h100
#SBATCH --gpus=h100:1
#SBATCH -c 8
#SBATCH --mem=32g
#SBATCH -t 06:00:00
#SBATCH -o logs/g15-%A_%a.out
#SBATCH --array=0-47%12
source /etc/profile
SCR=/scratch/zt1/project/msml612/user/$USER
module load python/3.10.10
export PYTHONPATH=
cd $SCR/pcmd/src
LANGS=(german turkish finnish russian navajo georgian)
MODES=(mdlm matched distill full); SEEDS=(0 1)
I=$SLURM_ARRAY_TASK_ID
L=${LANGS[$((I / 8))]}; M=${MODES[$(((I % 8) / 2))]}; S=${SEEDS[$((I % 2))]}
NAME="g15-$L-$M-s$S"
echo "=== $NAME ==="
$SCR/hf_env/bin/python train_sig.py --lang $L --mode $M --size medium --seed $S \
  --steps 15000 --d 256 --layers 6 --lr 1e-4 --warmup 1000 \
  --data_root $SCR/pcmd/data/sig --out $SCR/pcmd/runs/$NAME
