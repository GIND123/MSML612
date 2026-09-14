#!/bin/bash
# The shared judge: one autoregressive character LM, trained from scratch on
# text8, used to score every diffusion run's samples. Trained once so that all
# systems are measured against exactly the same yardstick.
#SBATCH -J t8eval
#SBATCH -p gpu-h100
#SBATCH --gpus=h100:1
#SBATCH -c 8
#SBATCH --mem=64g
#SBATCH -t 8:00:00
#SBATCH -o logs/t8eval-%j.out
source /etc/profile
SCR=/scratch/zt1/project/msml612/user/$USER
module load python/3.10.10
export PYTHONPATH=
cd $SCR/pcmd/src
$SCR/hf_env/bin/python -u train_eval_lm.py --root $SCR/pcmd/data/text8 \
  --seq_len 256 --d 512 --layers 8 --heads 8 --bs 128 \
  --steps 50000 --lr 3e-4 --warmup 2000 --out $SCR/pcmd/runs/evallm
