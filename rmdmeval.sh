#!/bin/bash
# Recover the R-MDM comparison from checkpoints, and the one missing figure.
#
# All three R-MDM arms hit the 5h wall clock at step 80,000 of 110,000, so the
# trainer never reached its evaluation block and no result.json was written.
# The weights are intact and all three stopped at the SAME step, so the
# comparison between them is still matched - it is simply at 80k steps.
#
# Figure 7 (accuracy vs. clue count) is folded into this job rather than a
# second allocation: it needs the same GPU and the same test set, and the
# remaining budget is 3,030 billing-minutes.
#SBATCH -J rmdmeval
#SBATCH -p gpu-a100_1g.5gb
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH -c 4
#SBATCH --mem=16g
#SBATCH -t 0:45:00
#SBATCH -o logs/rmdmeval-%j.out
source /etc/profile
SCR=/scratch/zt1/project/msml612/user/$USER
module load python/3.10.10
export PYTHONPATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SCR/pcmd/src

echo "############ R-MDM head-to-head, from step-80k checkpoints ############"
$SCR/hf_env/bin/python -u eval_ckpt.py \
  --hard $SCR/pcmd/data/sudoku_hard \
  --runs $SCR/pcmd/runs \
  --out $SCR/pcmd/figures/rmdm_compare.json

echo
echo "############ figure 7: accuracy vs. puzzle difficulty ############"
RUNS=$SCR/pcmd/runs OUT=$SCR/pcmd/figures HARD=$SCR/pcmd/data/sudoku_hard \
  $SCR/hf_env/bin/python -u plot_sudoku_grids.py

echo
echo "############ done ############"
