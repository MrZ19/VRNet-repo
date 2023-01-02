#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.spatial.transform import Rotation


def matchingPerformance(src, tgt):
    "function to test performance of matching: including RMSE_distance, MAE_distance, matching accuracy"
    "according to predicted matching matrix and registration result respectively" "tgt = R_pre * src + t_pre"
    '''args:
            src: source point cloud <N,3,768>
            tgt: target point cloud <N,3,768>
            R_pre: predicted rotation matrix <N,3,3>  
            t_pre: predicted translation vector <N,3> 
    '''
    N,d,n = src.shape # N,3,768
    print("src shape :{}".format(src.shape))
    #---------------------------------------------------------------------------------
    performance_predicted_Rt(src, tgt)

    return 0


def performance_predicted_Rt(source, target):
    src = np.transpose(source,[0,2,1]) #N,768,3
    tgt = np.transpose(target,[0,2,1]) #N,768,3
    src_corr_pre = []
    for _src_transfored, _tgt in \
        zip(src, tgt):
        src_corr_pre_i = estimate_correspondence(_src_transfored, _tgt) #768,3;768,3 -> 768,3
        src_corr_pre.append(src_corr_pre_i.detach().cpu().numpy()) 
    src_corr_pre = np.stack(src_corr_pre, axis=0) #N,768,3

    rmse_dis = np.sqrt(np.sum((src-src_corr_pre)**2, axis=2)) #N,768
    interval = 10
    num_ = int(src.shape[0]/interval)
    total_points = num_ * interval * src.shape[1]
    for i in range(2,20,2):
        inliers_num = 0
        for j in range(num_-1):
            index_state = j * interval
            index_end = (j+1)*interval
            src_j = src[index_state:index_end]
            rmse_dis_j = rmse_dis[index_state:index_end]

            threshold = estimate_threshold(src_j, i)
            flag_j = rmse_dis_j <= threshold
            inliers_num_j = np.sum(flag_j)
            inliers_num = inliers_num_j + inliers_num
        inliers_rate = inliers_num / total_points
        print("for threshold K: {}, inliers_ratio is : {}".format(i,inliers_rate))
    
def estimate_correspondence(A,B):
    # A: 1024,3
    # B: 1024,3
    A = torch.from_numpy(A).cuda()
    B = torch.from_numpy(B).cuda()
    distance, indices = nearest_neighbor(A.permute(1,0),B.permute(1,0),1) #3,1024;3,1024 ->
    corre = B[indices].squeeze() #1024,3
    
    return corre

def nearest_neighbor(src, dst, k):
    inner = -2 * torch.matmul(src.transpose(1, 0).contiguous(), dst)  # src, dst (num_dims, num_points)
    distances = -torch.sum(src ** 2, dim=0, keepdim=True).transpose(1, 0).contiguous() - inner - torch.sum(dst ** 2,
                                                                                                           dim=0,
                                                                                                           keepdim=True)
    distances, indices = distances.topk(k=k, dim=-1)
    return distances, indices

def estimate_threshold(B,k):
    # B,N,3
    B = torch.from_numpy(B).cuda()
    distance, indices = knn(B.permute(0,2,1),k) # B,3,N
    distance_matrix = torch.sqrt(torch.abs(distance)) #B,N,K
    threshold = torch.sum(distance_matrix,dim=2)/(k)
    return threshold.detach().cpu().numpy()


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    dis, idx = pairwise_distance.topk(k=k, dim=-1)  # (B, num_points, k)
    return dis, idx