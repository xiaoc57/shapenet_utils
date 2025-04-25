#!/bin/bash
#SBATCH --job-name=feat_448_gs_32_debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=gpujl
#SBATCH --gres=gpu:4
#SBATCH --mem=256G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fvdb

cd /home/jiangyun/documents/Sp2Sl

# 设置环境变量
export OMP_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export MKL_NUM_THREADS=8

wandb login

python -m sp2sl.train +experiments=feat_448_gs_32_debug  2>&1 | tee logs/feat_448_gs_32_debug.log