#!/bin/bash
# Grid 16 - THE DEFINITIVE FRONTIER, one consistent configuration throughout.
#
# Grid 11 (10 layers, 60k steps) understated the method: it gave 75% at sixteen
# digits, while grid 13 (12 layers, 150k steps) reached 99.9% at TWENTY digits
# with the baseline at exactly zero. Mixing configurations across digit counts
# makes the curve meaningless, so this reruns every length at one setting.
#SBATCH -J g16
#SBATCH -p gpu-h100
#SBATCH --gpus=h100:1
#SBATCH -c 8
#SBATCH --mem=64g
#SBATCH -t 16:00:00
#SBATCH -o logs/g16-%A_%a.out
#SBATCH --array=0-11%6
source /etc/profile
SCR=/scratch/zt1/project/msml612/user/$USER
module load python/3.10.10
export PYTHONPATH=
cd $SCR/pcmd/src
DIG=(12 16); MODES=(mdlm full); SEEDS=(0 1 2)
I=$SLURM_ARRAY_TASK_ID
D=${DIG[$((I / 6))]}; M=${MODES[$(((I % 6) / 3))]}; S=${SEEDS[$((I % 3))]}
SL=$(( 3 * D + 4 ))
NAME="g16-d$D-$M-s$S"
echo "=== $NAME ==="
$SCR/hf_env/bin/python train_add.py --mode $M --digits $D --seq_len $SL --seed $S \
  --n_train 1500000 --n_eval 400 --eval_bs 100 \
  --d 512 --layers 12 --heads 8 --bs 128 --steps 150000 --lr 1e-4 --warmup 6000 \
  --K 4 --lam 1.0 --out $SCR/pcmd/runs/$NAME
