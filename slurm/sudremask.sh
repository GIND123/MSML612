#!/bin/bash
#SBATCH -J sudrm
#SBATCH -p gpu-h100
#SBATCH --gpus=h100:1
#SBATCH -c 8
#SBATCH --mem=48g
#SBATCH -t 2:00:00
#SBATCH -o logs/sudrm-%j.out
source /etc/profile
SCR=/scratch/zt1/project/msml612/user/$USER
module load python/3.10.10
export PYTHONPATH=
cd $SCR/pcmd/src
$SCR/hf_env/bin/python -u sudoku_remask.py --hard $SCR/pcmd/data/sudoku_hard \
  --runs $SCR/pcmd/runs --pattern 'sudh-*' --out $SCR/pcmd/figures/sudoku_remask.json
