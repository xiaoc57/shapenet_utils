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


python3 render_shapenet.py \
--start_idx=0 --end_idx=5000 --model_root_dir=./ShapeNetCore.v1 \
--render_root_dir=./ShapeNetv1_render --blender_location=../blender_install/blender-3.6.13-linux-x64/blender \
--shapenetversion=v1  2>&1 | tee render_logs/5000.log