#!/bin/bash
# Grid 2 (graphs) - THE EXPRESSIVITY TEST.
#
# Claim: at t = 1 every input token is [MASK], so the input sequence is
# CONSTANT. With a purely relative position encoding the attention score between
# i and j depends only on (i-j), and when all value vectors are identical a
# layer's output is v * sum(attn) = v - the same at every position. Boundary
# effects aside, a RoPE-only model therefore cannot express position-DEPENDENT
# marginals at the fully-masked state. That is an EXPRESSIVITY limit, so no
# amount of training fixes it.
#
# The graph results already fit this exactly:
#   matching / 2-regular / tree  -> vertex-transitive, marginals CONSTANT across
#                                   edge slots -> RoPE suffices -> at ceiling
#   bipartite                    -> within-half p=0, cross p=0.53, marginals
#                                   POSITION-DEPENDENT -> RoPE cannot -> 5.9%
#                                   against a 93.6% ceiling
#
# Prediction: absolute position embeddings close the bipartite gap and leave
# matching unchanged. Matching is the control - if APE "helps" there too, the
# mechanism is not what is claimed.
#SBATCH -J gr2
#SBATCH -p gpu-a100_1g.5gb
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH -c 4
#SBATCH --mem=16g
#SBATCH -t 2:00:00
#SBATCH -o logs/gr2-%A_%a.out
#SBATCH --array=0-11%6
source /etc/profile
SCR=/scratch/zt1/project/msml612/user/$USER
module load python/3.10.10
export PYTHONPATH=
cd $SCR/pcmd/src
FAMS=(bipartite matching); PES=(rope ape sin); SEEDS=(0 1)
I=$SLURM_ARRAY_TASK_ID
F=${FAMS[$((I / 6))]}; P=${PES[$(((I % 6) / 2))]}; S=${SEEDS[$((I % 2))]}
NAME="gr2-$F-$P-s$S"
echo "=== $NAME ==="
$SCR/hf_env/bin/python -u train_graph.py --family $F --n 6 --mode mdlm --pe $P --seed $S \
  --d 256 --layers 6 --heads 8 --bs 256 --steps 30000 --lr 1e-4 --warmup 1000 \
  --n_gen 4096 --gen_bs 512 --out $SCR/pcmd/runs/$NAME
