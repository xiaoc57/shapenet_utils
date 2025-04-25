
# 将shapenetv1体素化

import os
import json
import torch
import utils3d
import torch.nn.functional as F

import fvdb
import fvdb.nn as fvnn

from tqdm import tqdm
from einops import rearrange
from torchvision import transforms

import os
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
import numpy as np
import open3d as o3d
import utils3d

def _voxelize(ip, out):
    mesh = o3d.io.read_triangle_mesh(ip)
    # clamp vertices to the range [-0.5, 0.5]
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    num_points=1000000
    sampled_points = mesh.sample_points_uniformly(number_of_points=num_points)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        sampled_points, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    assert np.all(vertices >= 0) and np.all(vertices < 64), "Some vertices are out of bounds"
    vertices = (vertices + 0.5) / 64 - 0.5
    utils3d.io.write_ply(out, vertices)
    # return {'sha256': sha256, 'voxelized': True, 'num_voxels': len(vertices)}


def main():

    # voxel_path 
    voxel_path = "/home/jiangyun/documents/Sp2Sl/ShapeSplat-Gaussian_MAE/render_scripts/voxels"
    image_path = "/home/jiangyun/documents/Sp2Sl/ShapeSplat-Gaussian_MAE/render_scripts/ShapeNetv1_render"
    error_log = "process_errors.txt"  # 错误日志文件

    os.makedirs(voxel_path, exist_ok=True)

    file_list = [f for f in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, f))]

    for f in tqdm(file_list, total=len(file_list)):
        try:
            vx = os.path.join(voxel_path, f + ".ply")
            ip = os.path.join(image_path, f, "mesh.ply")
            _voxelize(ip, vx)
        except Exception as e:
            # 记录错误信息
            with open(error_log, 'a') as log_file:
                error_message = f"Error processing {f}: {str(e)}\n"
                log_file.write(error_message)
            
            # 清理可能的GPU变量
            try:
                torch.cuda.empty_cache()
            except:
                pass
            
            print(f"Error processing {f}: {str(e)}")  # 打印错误信息
            continue  # 继续处理下一个文件
        
        # 每次成功处理后清理GPU
        torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()
