#!/bin/bash
# Sudoku on the SATNet benchmark (Wang et al., ICML 2019), 9000/1000 split,
# scored by BOARD-level accuracy - all 81 cells correct - which is the published
# metric. SATNet reports 98.3%.
#
# The registered prediction: Sudoku has a unique answer, so TC = 0 and there is
# no information-theoretic obstacle at all; the whole difficulty is mode (i) plus
# the ORDER of commitment. An entropy-budgeted decoder commits the forced cells
# first, recomputes and repeats, which IS constraint propagation. A fixed-K
# decoder commits a fixed share per pass regardless of certainty. The gap between
# them should be the largest in this project.
#
# Augmentation is by digit relabelling, a genuine Sudoku symmetry that leaves the
# constraint structure untouched. Both arms are run because SATNet did not
# augment, so the un-augmented number is the like-for-like one.
#SBATCH -J sud1
#SBATCH -p gpu-h100
#SBATCH --gpus=h100:1
#SBATCH -c 8
#SBATCH --mem=48g
#SBATCH -t 4:00:00
#SBATCH -o logs/sud1-%A_%a.out
#SBATCH --array=0-7%4
source /etc/profile
SCR=/scratch/zt1/project/msml612/user/$USER
module load python/3.10.10
export PYTHONPATH=
cd $SCR/pcmd/src
MODES=(mdlm full); AUG=(plain aug); SEEDS=(0 1)
I=$SLURM_ARRAY_TASK_ID
M=${MODES[$((I / 4))]}; A=${AUG[$(((I % 4) / 2))]}; S=${SEEDS[$((I % 2))]}
FLAG=""; [ "$A" = "aug" ] && FLAG="--augment"
NAME="sud1-$M-$A-s$S"
echo "=== $NAME ==="
$SCR/hf_env/bin/python -u train_sudoku.py --mode $M $FLAG --seed $S --pe ape \
  --root $SCR/pcmd/data/sudoku --d 512 --layers 10 --heads 8 --bs 128 \
  --steps 60000 --lr 1e-4 --warmup 2000 --K 8 --lam 1.0 --eval_bs 200 \
  --out $SCR/pcmd/runs/$NAME
