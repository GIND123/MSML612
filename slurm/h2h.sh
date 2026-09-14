#!/bin/bash
#SBATCH -J h2h
#SBATCH -p gpu-h100
#SBATCH --gpus=h100:1
#SBATCH -c 4
#SBATCH --mem=32g
#SBATCH -t 3:00:00
#SBATCH -o logs/h2h-%j.out
source /etc/profile
SCR=/scratch/zt1/project/msml612/user/$USER
module load python/3.10.10
export PYTHONPATH=
cd $SCR/pcmd/src
for D in 12 16; do
  $SCR/hf_env/bin/python -u beat_baselines.py --runs $SCR/pcmd/runs --digits $D \
    --n_eval 400 --eval_bs 100 --out $SCR/pcmd/figures/beat_d$D.json
done
