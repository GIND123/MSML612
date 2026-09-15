#!/bin/bash
# Hard Sudoku: minimal puzzles at ~24 clues, where search-free constraint
# propagation solves about 4%. Unlike the SATNet split - which a classical
# solver saturates at 100% - this discriminates.
#
# Every puzzle was verified to have exactly one solution at generation time, and
# train/test solutions are disjoint even up to digit relabelling (an earlier
# generator leaked 767/1000 and the assertion caught it).
#
# Baselines to beat, measured on this exact test set:
#   constraint propagation, no search : ~4%
#   backtracking search               : 100%, but it is search
# The question is what a masked diffusion model reaches in a fixed, small number
# of forward passes.
#SBATCH -J sudhard
#SBATCH -p gpu-h100
#SBATCH --gpus=h100:1
#SBATCH -c 8
#SBATCH --mem=48g
#SBATCH -t 4:00:00
#SBATCH -o logs/sudhard-%A_%a.out
#SBATCH --array=0-3%4
source /etc/profile
SCR=/scratch/zt1/project/msml612/user/$USER
module load python/3.10.10
export PYTHONPATH=
cd $SCR/pcmd/src
MODES=(mdlm full); SEEDS=(0 1)
I=$SLURM_ARRAY_TASK_ID
M=${MODES[$((I / 2))]}; S=${SEEDS[$((I % 2))]}
NAME="sudh-$M-s$S"
echo "=== $NAME ==="
$SCR/hf_env/bin/python -u train_sudoku.py --mode $M --seed $S --pe ape \
  --hard $SCR/pcmd/data/sudoku_hard --augment \
  --d 512 --layers 12 --heads 8 --bs 256 --steps 150000 --lr 1e-4 --warmup 4000 \
  --K 8 --lam 1.0 --eval_bs 200 --ckpt_every 10000 \
  --out $SCR/pcmd/runs/$NAME
