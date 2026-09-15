#!/bin/bash
# QM9 molecular generation - validity retained as denoising steps fall.
#
# This is the axis the discrete graph-diffusion literature competes on. DiGress
# and its successors use hundreds of denoising steps; the question is how much
# validity survives at few steps, and that depends on the training schedule and
# the decoding rule rather than on model scale - which is why it is winnable at
# this budget when text8 bits-per-character is not.
#
# Both of the project's findings are under test here at once, and QM9 is the
# strongest possible case for the second. Its marginals are extremely
# position-dependent: node slot 8 is NEVER occupied, and edge slots range from
# never-bonded to always-bonded, giving a mean total-variation distance of 0.435
# between a slot and the average slot. Finding 2 therefore predicts that a
# relative-only encoding badly caps few-step generation here, and that absolute
# position information produces a large gain. Registered before the runs.
#SBATCH -J mol1
#SBATCH -p gpu-h100
#SBATCH --gpus=h100:1
#SBATCH -c 8
#SBATCH --mem=48g
#SBATCH -t 3:00:00
#SBATCH -o logs/mol1-%A_%a.out
#SBATCH --array=0-7%8
source /etc/profile
SCR=/scratch/zt1/project/msml612/user/$USER
module load python/3.10.10
export PYTHONPATH=
cd $SCR/pcmd/src
MODES=(mdlm matched); PES=(rope ape); SEEDS=(0 1)
I=$SLURM_ARRAY_TASK_ID
M=${MODES[$((I / 4))]}; P=${PES[$(((I % 4) / 2))]}; S=${SEEDS[$((I % 2))]}
NAME="mol1-$M-$P-s$S"
echo "=== $NAME ==="
$SCR/hf_env/bin/python -u train_mol.py --mode $M --pe $P --seed $S --K 8 \
  --root $SCR/pcmd/data/qm9 --d 384 --layers 8 --heads 8 --bs 256 \
  --steps 60000 --lr 1e-4 --warmup 2000 --n_gen 4096 --gen_bs 512 \
  --out $SCR/pcmd/runs/$NAME
