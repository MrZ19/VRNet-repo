#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import open3d as o3d
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

def genereate_correspondence_line_set(src, target):
    points = np.concatenate((src, target), axis=0)
    lines = [[i,i+src.shape[0]] for i in range(src.shape[0])]
    colors = [[0.47, 0.53, 0.7] for i in range(len(lines))] # 0.69, 0.76, 0.87 / 0.9, 0, 0
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def visualize_pc(src, target, target_backup, src_transformed, correspondings):
    
    src = src.permute(0,2,1).cpu().numpy()
    src_transformed = src_transformed.permute(0,2,1).cpu().detach().numpy()
    target = target.permute(0,2,1).cpu().numpy()
    target_backup = target_backup.permute(0,2,1).cpu().numpy()
    correspondings = correspondings.permute(0,2,1).cpu().numpy()
    
    for _src, _src_transformed, _target, _target_backup, _correspondings in \
        zip(src, src_transformed, target, target_backup, correspondings):
        
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        _src[:,0] -= 1.5
        #_correspondings[:,0] -= 0.25
        _target[:,0] += 1.5
        _src_transformed[:,0] += 1.5

        line = genereate_correspondence_line_set(_src[::3], _correspondings[::3])

        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(_src)
        src_pcd.colors = o3d.utility.Vector3dVector(np.repeat(np.asarray([[0, 0, 1]]), _src.shape[0], axis=0)) #red

        target_backup_pcd = o3d.geometry.PointCloud()
        _target_backup[:,0] -= 1.5
        target_backup_pcd.points = o3d.utility.Vector3dVector(_target_backup)
        target_backup_pcd.colors = o3d.utility.Vector3dVector(np.repeat(np.asarray([[0, 0, 1]]), _target_backup.shape[0], axis=0)) #blue

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(_target)
        target_pcd.colors = o3d.utility.Vector3dVector(np.repeat(np.asarray([[1, 0, 0]]), _target.shape[0], axis=0)) #blue

        src_transformed_pcd = o3d.geometry.PointCloud()
        src_transformed_pcd.points = o3d.utility.Vector3dVector(_src_transformed)
        src_transformed_pcd.colors = o3d.utility.Vector3dVector(np.repeat(np.asarray([[0, 1.0, 0]]), _src_transformed.shape[0], axis=0)) #green

        corresponding_pcd = o3d.geometry.PointCloud()
        corresponding_pcd.points = o3d.utility.Vector3dVector(_correspondings)
        corresponding_pcd.colors = o3d.utility.Vector3dVector(np.repeat(np.asarray([[1.0, 0.0, 1.0]]), _correspondings.shape[0], axis=0)) #green
        o3d.visualization.draw_geometries([src_pcd, target_pcd, src_transformed_pcd, corresponding_pcd]) #, target_pcd, src_transformed_pcd
        #o3d.visualization.draw_geometries([src_pcd, target_backup_pcd, target_pcd, src_transformed_pcd, corresponding_pcd, line])
        pass