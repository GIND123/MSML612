#!/bin/bash
# The capstone experiment: does the training-side fix compose with the
# inference-side fix on data that contains BOTH failure modes at once?
#SBATCH -J t8
#SBATCH -p gpu-h100
#SBATCH --gpus=h100:1
#SBATCH -c 8
#SBATCH --mem=64g
#SBATCH -t 16:00:00
#SBATCH -o logs/t8-%A_%a.out
#SBATCH --array=0-5
source /etc/profile
SCR=/scratch/zt1/project/msml612/user/$USER
module load python/3.10.10
export PYTHONPATH=
cd $SCR/pcmd/src
MODES=(mdlm matched anybudget); SEEDS=(0 1)
I=$SLURM_ARRAY_TASK_ID
M=${MODES[$((I / 2))]}; S=${SEEDS[$((I % 2))]}
NAME="t8-$M-s$S"
echo "=== $NAME ==="
$SCR/hf_env/bin/python -u train_text8.py --mode $M --seed $S \
  --root $SCR/pcmd/data/text8 --seq_len 256 \
  --d 512 --layers 12 --heads 8 --bs 128 --steps 60000 --lr 1e-4 --warmup 4000 \
  --K 8 --lam 1.0 --n_gen 256 --gen_bs 64 \
  --evallm $SCR/pcmd/runs/evallm --out $SCR/pcmd/runs/$NAME
