#!/bin/bash
#SBATCH -J seqchk
#SBATCH -p gpu-h100
#SBATCH --gpus=h100:1
#SBATCH -c 4
#SBATCH --mem=32g
#SBATCH -t 2:00:00
#SBATCH -o logs/seqchk-%j.out
source /etc/profile
SCR=/scratch/zt1/project/msml612/user/$USER
module load python/3.10.10
export PYTHONPATH=
cd $SCR/pcmd/src
$SCR/hf_env/bin/python -u seq_check.py --runs $SCR/pcmd/runs --pattern 'g13-*' \
  --out $SCR/pcmd/figures/seq_check_g13.json
