#!/bin/bash
#SBATCH -J molfair
#SBATCH -p gpu-h100
#SBATCH --gpus=h100:1
#SBATCH -c 8
#SBATCH --mem=48g
#SBATCH -t 2:00:00
#SBATCH -o logs/molfair-%j.out
source /etc/profile
SCR=/scratch/zt1/project/msml612/user/$USER
module load python/3.10.10
export PYTHONPATH=
cd $SCR/pcmd/src
$SCR/hf_env/bin/python -u mol_fair.py --runs $SCR/pcmd/runs --root $SCR/pcmd/data/qm9 \
  --n_gen 2048 --gen_bs 512 --out $SCR/pcmd/figures/mol_fair.json
