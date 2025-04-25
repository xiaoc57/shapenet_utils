#!/bin/bash
#SBATCH --job-name=rs5000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpujl
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fvdb

cd /home/jiangyun/documents/Sp2Sl/ShapeSplat-Gaussian_MAE/render_scripts

# 设置环境变量
export OMP_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export MKL_NUM_THREADS=8



python -m sp2sl.train +experiments=feat_448_gs_32_debug  2>&1 | tee logs/feat_448_gs_32_debug.log