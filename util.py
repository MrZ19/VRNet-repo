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


# Part of the code is referred from: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py

def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def transform_point_cloud(point_cloud, rotation, translation):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_dcm(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')


def unsupervisedloss(src, tgt, alpha=1.0, beta=1.0, gamma=1.0):
    "the unsupervised loss function including three part: 1) keep distance consisitency loss; 2) keep rigid motion consisitency loss; 3) keep global consisitency loss"
    '''args:
        src,tgt: point clouds matched in order
        alpha: weight for keep distance
        beta: weight for rigid motion
        gamma: weight for keep global
    '''
    w_dis = alpha / (alpha + beta + gamma)
    w_rgd = beta / (alpha + beta + gamma)
    w_glb = gamma / (alpha + beta + gamma)
    loss1 = KDCLoss(src,tgt)
    loss2 =KRMCLoss(src,tgt,Num=10)
    loss3 =KGCLoss(src,tgt)
    Loss = w_dis * loss1 + w_rgd * loss2 + w_glb * loss3

    return Loss


def KDCLoss(src,tgt):
    dis_matrix_src = knn(src)
    dis_matrix_tgt = knn(tgt)
    mse = torch.mean((dis_matrix_src - dis_matrix_tgt)**2)
    return mse


def KRMCLoss(src, tgt, Num):
    #input: 
    #   scr: point cloud; B,3,num_points
    #   tgt: corresponding point cloud; B,3,num_points
    #   Num: the number of siamese structure; int
    #output: 
    #   loss: difference loss; float

    src = src.cuda()
    tgt = tgt.cuda()    
        
    batch_size = src.shape[0]
    num_points = src.shape[2]
    n = np.floor(num_points / Num).astype(int)
    #print("n: {}".format(n))
    loss = 0
    identity = torch.eye(3).cuda()
    for i in range(batch_size):
        loss_i = 0
        src_one_batch = src[i]
        tgt_one_batch = tgt[i] #3,num_points
        R_global, t_global = motion_esti(src_one_batch, tgt_one_batch)
        #R_gb, t_gb = motion_esti(src_one_batch, tgt_one_batch) #3,3;3
        s = torch.randperm(src_one_batch.shape[1])
        src_one_batch_perm = src_one_batch.transpose(1,0)[s,:] # 3,n -> n,3 -> ... 
        tgt_one_batch_perm = tgt_one_batch.transpose(1,0)[s,:] # 3,n -> n,3 -> ... 
        for j in range(Num):
            src_local = src_one_batch_perm[j*n:(j+1)*n,:] # n,3
            tgt_local = tgt_one_batch_perm[j*n:(j+1)*n,:] # n,3
            R_local, t_local = motion_esti(src_local.transpose(1,0), tgt_local.transpose(1,0))
            loss_j = F.mse_loss(torch.matmul(R_local.transpose(1,0), R_global), identity) + F.mse_loss(t_local, t_global)
            loss_i = loss_i + loss_j

        loss = loss + loss_i   
        
    return loss


def KGCLoss(src,tgt):
    batch_size = src.shape[0]
    R_batch = []
    t_batch = []
    for i in range(batch_size):
        R,t = motion_esti(src[i], tgt[i])
        R_batch.append(R)
        t_batch.append(t)
    R_batch = torch.stack(R_batch,dim=0)
    t_batch = torch.stack(t_batch,dim=0)

    #src_motion = (torch.matmul(R, src) + t.repeat(1,1,src.shape[2])) #b,3,n
    
    src_motion = (torch.matmul(R_batch, src) + t_batch.repeat(1,1,src.shape[2])) #b,3,n 
    mse = torch.mean((src_motion - tgt) ** 2) 
    return mse


def motion_esti(src, tgt):
    #input: 
    #   scr: point cloud; 3,num_points
    #   tgt: corresponding point cloud; 3,num_points
    #output: 
    #   R: Rotation matrix; 3,3 
    #   t: translation matrix; 3
    #src = src.transpose(1,0)
    #tgt = tgt.transpose(1,0)
    src_centered = src - src.mean(dim=1, keepdim=True) #3,n
    tgt_centered = tgt - tgt.mean(dim=1, keepdim=True) #3,n

    H = torch.matmul(src_centered, tgt_centered.transpose(1,0).contiguous()).cpu()

    u, s, v = torch.svd(H)
    r = torch.matmul(v, u.transpose(1, 0)).contiguous()
    r_det = torch.det(r).item()
    diag = torch.from_numpy(np.array([[1.0, 0, 0],
                                    [0, 1.0, 0],
                                    [0, 0, r_det]]).astype('float32')).to(v.device)
    r = torch.matmul(torch.matmul(v, diag), u.transpose(1, 0)).contiguous().cuda()
    t = torch.matmul(-r, src.mean(dim=1, keepdim=True)) + tgt.mean(dim=1, keepdim=True)  
    
    return r, t


def knn(x):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous() #b,n,n

    return pairwise_distance

def supervisedloss(I_gt, I_pre):
    #input:
    #   pre: prediction matching matrix ; b,M,N 
    #   gt:  ground truth matching matrix; b,M,N
    #log_pre = torch.log(pre)
    I_gt = I_gt.cuda()
    I_pre = I_pre.cuda()
    loss_up = torch.sum(torch.mul(I_gt,I_pre))
    loss_down = torch.sum(I_gt)
    loss = -1.0*loss_up/loss_down
    #print("matching loss: {}".format(loss))

    return loss

def Chamfer_distance(preds, gts):
    preds = torch.tensor(preds).cuda()
    gts = torch.tensor(gts).cuda()
    sum_ = 0
    interval = 10
    num = int(preds.shape[0] / interval)
    for i in range(num-1):
        index_start = interval * i 
        index_end = interval * (i+1)
        gts_i = gts[index_start:index_end]
        preds_i = preds[index_start:index_end]
        P = batch_pairwise_dist(gts_i, preds_i)
        mins, _ = torch.min(P, 1)
        dis_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        dis_2 = torch.sum(mins)
        dis_i = (dis_1 + dis_2)/interval
        sum_ = sum_ + dis_i
    CD = sum_ / num

    return CD / preds.shape[1]

def batch_pairwise_dist(x, y):
    bs, num_points_x, points_dim = x.size()
    b2, num_points_y, p2 = y.size()

    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))

    dtype = torch.cuda.LongTensor
    diag_ind_x = torch.arange(0, num_points_x).type(dtype)
    diag_ind_y = torch.arange(0, num_points_y).type(dtype)

    rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P 

def Chamfer_dis(transformed_src, targets):
    transformed_src = transformed_src.cuda()
    targets = targets.cuda()
    dist_src = torch.min(square_distance(src_transformed, ref_clean), dim=-1)[0]
    dist_ref = torch.min(square_distance(points_ref, src_clean), dim=-1)[0]
    chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

    return chamfer_dist

def square_distance(src, dst):
        return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)

def ab_angle(rotation_pred, rotation_gt):
    angAB_sum = 0
    num_e = 0
    for i in range(rotation_pred.shape[0]):
        # train_angAB_i = np.arccos((np.trace(np.matmul(np.linalg.inv(train_rotations_ab_pred[i]),train_rotations_ab[i]))-1)/2.0)
        frac1 = np.matmul(rotation_pred[i].transpose(), rotation_gt[i])
        frac2 = (np.trace(frac1) - 1) / 2.0
        angAB_i = np.arccos(frac2)
        
        if np.isnan(angAB_i):
            continue
        #print(angAB_i)
        angAB_sum = angAB_sum + angAB_i
        num_e = num_e + 1
    angAB = angAB_sum / num_e

    return angAB * 180 / 3.14