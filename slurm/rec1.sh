#!/bin/bash
# Recurrent masked diffusion with deep supervision, on hard Sudoku.
#
# The gap this targets is measured, not guessed. Our 37.86M-parameter 12-layer
# denoiser reaches 89.4%; Yang et al. (2023) reach 99.5% on a comparable split
# with 211k parameters - 180x smaller - by applying ONE block 32 times with loss
# at every recurrence. Constraint propagation is iterative, and a fixed-depth
# network cannot express thirty rounds of it at any width.
#
# Config follows theirs (d=128, 1 layer, 4 heads, R=32) so the comparison is
# about the diffusion framing rather than about scale. Inference runs R=64,
# more than training, which deep supervision is what makes possible.
#SBATCH -J rec1
#SBATCH -p gpu-h100
#SBATCH --gpus=h100:1
#SBATCH -c 8
#SBATCH --mem=48g
#SBATCH -t 5:00:00
#SBATCH -o logs/rec1-%A_%a.out
#SBATCH --array=0-3%4
source /etc/profile
SCR=/scratch/zt1/project/msml612/user/$USER
module load python/3.10.10
export PYTHONPATH=
cd $SCR/pcmd/src
MODES=(mdlm full); SEEDS=(0 1)
I=$SLURM_ARRAY_TASK_ID
M=${MODES[$((I / 2))]}; S=${SEEDS[$((I % 2))]}
NAME="rec1-$M-s$S"
echo "=== $NAME ==="
$SCR/hf_env/bin/python -u train_sudoku.py --recurrent --R 32 --R_infer 64 \
  --mode $M --seed $S --pe ape --hard $SCR/pcmd/data/sudoku_hard --augment \
  --d 128 --layers 1 --heads 4 --bs 256 --steps 120000 --lr 3e-4 --warmup 3000 \
  --K 8 --lam 1.0 --eval_bs 250 --ckpt_every 10000 \
  --out $SCR/pcmd/runs/$NAME
