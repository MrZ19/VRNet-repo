import os
import gc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from data_finetune import ModelNet40
from data_stanford import Stanford
from util import transform_point_cloud, npmat2euler, unsupervisedloss, supervisedloss, Chamfer_dis, Chamfer_distance, ab_angle
                                                     
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from matching import matchingPerformance
from vis import visualize_pc
import time
import h5py

def test_one_epoch(args, net, test_loader):
    net.eval()
    
    total_loss = 0
    num_examples = 0
    rotations = []
    translations = []
    rotations_pred = []
    translations_pred = []

    corres = []
    targets = []
    transformed_src = []

    Time_total = 0
    Num = 0
    i = 0
    I_gts = []
    transformed_src_gt = []
    for src, target, rotation, translation, I_gt, _, _ in tqdm(test_loader):
        src = src.cuda()
        target = target.cuda()
        rotation = rotation.cuda()
        translation = translation.cuda()


        batch_size = src.size(0)
        num_examples += batch_size

        torch.cuda.synchronize()
        time_start=time.time()
        rotation_pred, translation_pred, corre_src, scores = net(src, target)
        torch.cuda.synchronize()
        time_end=time.time()
        diff_time = time_end - time_start
        if i>10 and i<150:
            Num = Num + batch_size
            Time_total = Time_total+diff_time
        i = i+1

        ## save rotation and translation
        rotations.append(rotation.detach().cpu().numpy())
        translations.append(translation.detach().cpu().numpy())
        rotations_pred.append(rotation_pred.detach().cpu().numpy())
        translations_pred.append(translation_pred.detach().cpu().numpy())
        
        namta = 100.0
        unloss = unsupervisedloss(src, corre_src)
        suloss = supervisedloss(I_gt, scores)
        ###########################
        loss = unloss + namta * suloss
        #loss = namta * suloss

        total_loss += loss.item() * batch_size
        ## for visualization
        if args.eval:
            transformed_src_batch = torch.matmul(rotation_pred, src) + translation_pred.unsqueeze(2)
            transformed_src_batch_gt = torch.matmul(rotation, src) + translation.unsqueeze(2)
            visualize_pc(src, target, target, transformed_src_batch, corre_src)

            transformed_src.append(transformed_src_batch.detach().cpu().numpy())
            transformed_src_gt.append(transformed_src_batch_gt.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
            corres.append(corre_src.detach().cpu().numpy())
            I_gts.append(I_gt.detach().cpu().numpy())

    #print("total time:{}; Num:{}; average: {}".format(Time_total, Num, Time_total/Num))
    ## for inliers ratio changes
    if args.eval:
        transformed_src = np.concatenate(transformed_src, axis=0)  
        transformed_src_gt = np.concatenate(transformed_src_gt, axis=0) 

        targets = np.concatenate(targets, axis=0)
        corres = np.concatenate(corres, axis=0)
        I_gts = np.concatenate(I_gts, axis=0)  
        num_ = int(I_gts.shape[0] / 10)
        I_gts_ = I_gts[:(num_*10)]
        inliers_ratio = np.sum(I_gts_)/(num_*10.0 *I_gts.shape[1])
        print("ground truth inliers ratio: {}".format(inliers_ratio))
        CD = Chamfer_distance(transformed_src.transpose((0,2,1)), corres.transpose((0,2,1)))
        print("chamfer distance is: {}".format(CD))
        CD2 = Chamfer_distance(transformed_src_gt.transpose((0,2,1)), targets.transpose((0,2,1)))
        print("chamfer gt distance is: {}".format(CD2))

        
        #matchingPerformance(transformed_src_gt, corres)

        
        print("the following is real inliers ratio:")
        #matchingPerformance(transformed_src_gt, targets)

        print("the following is predict  inliers ratio:")
        #matchingPerformance(transformed_src, corres)
        #matchingPerformance(transformed_src, targets)
        

    rotations = np.concatenate(rotations, axis=0)
    translations = np.concatenate(translations, axis=0)
    rotations_pred = np.concatenate(rotations_pred, axis=0)
    translations_pred = np.concatenate(translations_pred, axis=0)

    return total_loss * 1.0 / num_examples, rotations, translations, rotations_pred, translations_pred

def test_one_epoch_for_sun3d(args, net, test_loader):
    net.eval()
    
    total_loss = 0
    num_examples = 0
    rotations = []
    translations = []
    rotations_pred = []
    translations_pred = []

    corres = []
    targets = []
    transformed_src = []

    Time_total = 0
    Num = 0
    i = 0
    I_gts = []
    for src, target, rotation, translation, I_gt, idx1, idx2 in tqdm(test_loader):
        src = src.cuda()
        target = target.cuda()
        rotation = rotation.cuda()
        translation = translation.cuda()


        batch_size = src.size(0)
        num_examples += batch_size

        torch.cuda.synchronize()
        time_start=time.time()
        rotation_pred, translation_pred, corre_src, scores = net(src, target)
        torch.cuda.synchronize()
        time_end=time.time()
        diff_time = time_end - time_start
        if i>10 and i<150:
            Num = Num + batch_size
            Time_total = Time_total+diff_time
        i = i+1

        if args.eval:
            for i in range(batch_size):
                name = str(idx1[i].detach().cpu().numpy())+'_'+str(idx2[i].detach().cpu().numpy())
                save_path = os.path.join('./sun3d_pose_har/',name+'.h5')
                with h5py.File(save_path, 'w') as h5file:
                    h5file.create_dataset('R', data=rotation_pred[i].detach().cpu().numpy(), compression="gzip", compression_opts=9)
                    h5file.create_dataset('t', data=translation_pred[i].detach().cpu().numpy(), compression="gzip", compression_opts=9)
            print("save successfully")

        ## save rotation and translation
        rotations.append(rotation.detach().cpu().numpy())
        translations.append(translation.detach().cpu().numpy())
        rotations_pred.append(rotation_pred.detach().cpu().numpy())
        translations_pred.append(translation_pred.detach().cpu().numpy())
        
        namta = 100.0
        unloss = unsupervisedloss(src, corre_src)
        suloss = supervisedloss(I_gt, scores)
        ###########################
        loss = unloss + namta * suloss
        #loss = namta * suloss

        total_loss += loss.item() * batch_size
        if args.eval:
            transformed_src_batch = torch.matmul(rotation_pred, src) + translation_pred.unsqueeze(2)

            #visualize_pc(src, target, target, transformed_src_batch, corre_src)

            transformed_src.append(transformed_src_batch.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
            corres.append(corre_src.detach().cpu().numpy())
            I_gts.append(I_gt.detach().cpu().numpy())

    #print("total time:{}; Num:{}; average: {}".format(Time_total, Num, Time_total/Num))
    if args.eval:
        transformed_src = np.concatenate(transformed_src, axis=0)   
        targets = np.concatenate(targets, axis=0)
        corres = np.concatenate(corres, axis=0)
        I_gts = np.concatenate(I_gts, axis=0)  
        num_ = int(I_gts.shape[0] / 10)
        I_gts_ = I_gts[:(num_*10)]
        inliers_ratio = np.sum(I_gts_)/(num_*10.0 *I_gts.shape[1])
        print("ground truth inliers ratio: {}".format(inliers_ratio))
        CD = Chamfer_distance(transformed_src.transpose((0,2,1)), targets.transpose((0,2,1)))
        print("chamfer distance is: {}".format(CD))
        matchingPerformance(transformed_src, corres)
        

    rotations = np.concatenate(rotations, axis=0)
    translations = np.concatenate(translations, axis=0)
    rotations_pred = np.concatenate(rotations_pred, axis=0)
    translations_pred = np.concatenate(translations_pred, axis=0)

    return total_loss * 1.0 / num_examples, rotations, translations, rotations_pred, translations_pred

def train_one_epoch(args, net, train_loader, opt):
    net.train()

    total_loss = 0

    num_examples = 0
    rotations = []
    translations = []
    rotations_pred = []
    translations_pred = []

    for src, target, rotation, translation, I_gt in tqdm(train_loader):
        src = src.cuda()
        target = target.cuda()
        rotation = rotation.cuda()
        translation = translation.cuda()

        batch_size = src.size(0)
        opt.zero_grad()
        num_examples += batch_size
        rotation_pred, translation_pred, corre_src, scores = net(src, target)

        ## save rotation and translation
        rotations.append(rotation.detach().cpu().numpy())
        translations.append(translation.detach().cpu().numpy())
        rotations_pred.append(rotation_pred.detach().cpu().numpy())
        translations_pred.append(translation_pred.detach().cpu().numpy())
        
        # transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)

        # transformed_target = transform_point_cloud(target, rotation_ba_pred, translation_ba_pred)

        
        namta = 100.0
        unloss = unsupervisedloss(src, corre_src)
        suloss = supervisedloss(I_gt, scores)
        ###########################
        loss = unloss + namta * suloss
        #loss = namta * suloss

        loss.backward()
        opt.step()
        total_loss += loss.item() * batch_size



    rotations = np.concatenate(rotations, axis=0)
    translations = np.concatenate(translations, axis=0)
    rotations_pred = np.concatenate(rotations_pred, axis=0)
    translations_pred = np.concatenate(translations_pred, axis=0)

    return total_loss * 1.0 / num_examples, rotations, translations, rotations_pred, translations_pred


def test(args, net, test_loader, boardio, textio):
    with torch.no_grad():
        test_loss, test_rotations, test_translations, test_rotations_pred, test_translations_pred = test_one_epoch(args, net, test_loader)
    
    test_rotations_pred_euler = npmat2euler(test_rotations_pred)
    test_rotations_euler = npmat2euler(test_rotations)

    test_r_mse = np.mean((test_rotations_pred_euler - test_rotations_euler) ** 2)
    test_r_rmse = np.sqrt(test_r_mse)
    test_r_mae = np.mean(np.abs(test_rotations_pred_euler - test_rotations_euler))

    test_t_mse = np.mean((test_translations - test_translations_pred) ** 2)
    test_t_rmse = np.sqrt(test_t_mse)
    test_t_mae = np.mean(np.abs(test_translations - test_translations_pred))

    angle_error = ab_angle(test_rotations_pred, test_rotations)
    textio.cprint('==FINAL TEST==')
    textio.cprint('A--------->B')
    textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, '
                  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f, Angle_error: %f'
                  % (-1, test_loss, test_r_mse, test_r_rmse, test_r_mae, test_t_mse, test_t_rmse, test_t_mae, angle_error))


def train(args, net, train_loader, test_loader, boardio, textio):
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1)


    best_test_loss = np.inf
    
    best_test_r_mse = np.inf
    best_test_r_rmse = np.inf
    best_test_r_mae = np.inf
    best_test_t_mse = np.inf
    best_test_t_rmse = np.inf
    best_test_t_mae = np.inf

    
    for epoch in range(args.epochs):
        scheduler.step()
        train_loss, train_rotations, train_translations, train_rotations_pred, train_translations_pred = train_one_epoch(args, net, train_loader, opt)

        with torch.no_grad():
            test_loss, test_rotations, test_translations, test_rotations_pred, test_translations_pred = test_one_epoch(args, net, test_loader)
        
        train_rotations_pred_euler = npmat2euler(train_rotations_pred)
        train_rotations_euler = npmat2euler(train_rotations)

        train_r_mse = np.mean((train_rotations_pred_euler - train_rotations_euler) ** 2)
        train_r_rmse = np.sqrt(train_r_mse)
        train_r_mae = np.mean(np.abs(train_rotations_pred_euler - train_rotations_euler))
        train_t_mse = np.mean((train_translations - train_translations_pred) ** 2)
        train_t_rmse = np.sqrt(train_t_mse)
        train_t_mae = np.mean(np.abs(train_translations - train_translations_pred))

        
        test_rotations_pred_euler = npmat2euler(test_rotations_pred)
        test_rotations_euler = npmat2euler(test_rotations)

        test_r_mse = np.mean((test_rotations_pred_euler - test_rotations_euler) ** 2)
        test_r_rmse = np.sqrt(test_r_mse)
        test_r_mae = np.mean(np.abs(test_rotations_pred_euler - test_rotations_euler))
        test_t_mse = np.mean((test_translations - test_translations_pred) ** 2)
        test_t_rmse = np.sqrt(test_t_mse)
        test_t_mae = np.mean(np.abs(test_translations - test_translations_pred))

        
        if best_test_loss >= test_loss:
            best_test_loss = test_loss
            
            best_test_r_mse = test_r_mse
            best_test_r_rmse = test_r_rmse
            best_test_r_mae = test_r_mae

            best_test_t_mse = test_t_mse
            best_test_t_rmse = test_t_rmse
            best_test_t_mae = test_t_mae

            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
            else:
                torch.save(net.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)

        textio.cprint('==TRAIN==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, train_loss, train_r_mse, train_r_rmse, train_r_mae, train_t_mse, train_t_rmse, train_t_mae))
        

        textio.cprint('==TEST==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, test_loss, test_r_mse, test_r_rmse, test_r_mae, test_t_mse, test_t_rmse, test_t_mae))
        

        textio.cprint('==BEST TEST==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, best_test_loss, best_test_r_mse, best_test_r_rmse,
                         best_test_r_mae, best_test_t_mse, best_test_t_rmse, best_test_t_mae))
        
        boardio.add_scalar('A->B/train/loss', train_loss, epoch)
        
        boardio.add_scalar('A->B/train/rotation/MSE', train_r_mse, epoch)
        boardio.add_scalar('A->B/train/rotation/RMSE', train_r_rmse, epoch)
        boardio.add_scalar('A->B/train/rotation/MAE', train_r_mae, epoch)
        boardio.add_scalar('A->B/train/translation/MSE', train_t_mse, epoch)
        boardio.add_scalar('A->B/train/translation/RMSE', train_t_rmse, epoch)
        boardio.add_scalar('A->B/train/translation/MAE', train_t_mae, epoch)

        ############TEST
        boardio.add_scalar('A->B/test/loss', test_loss, epoch)
        
        boardio.add_scalar('A->B/test/rotation/MSE', test_r_mse, epoch)
        boardio.add_scalar('A->B/test/rotation/RMSE', test_r_rmse, epoch)
        boardio.add_scalar('A->B/test/rotation/MAE', test_r_mae, epoch)
        boardio.add_scalar('A->B/test/translation/MSE', test_t_mse, epoch)
        boardio.add_scalar('A->B/test/translation/RMSE', test_t_rmse, epoch)
        boardio.add_scalar('A->B/test/translation/MAE', test_t_mae, epoch)

        ############BEST TEST
        boardio.add_scalar('A->B/best_test/loss', best_test_loss, epoch)
        
        boardio.add_scalar('A->B/best_test/rotation/MSE', best_test_r_mse, epoch)
        boardio.add_scalar('A->B/best_test/rotation/RMSE', best_test_r_rmse, epoch)
        boardio.add_scalar('A->B/best_test/rotation/MAE', best_test_r_mae, epoch)
        boardio.add_scalar('A->B/best_test/translation/MSE', best_test_t_mse, epoch)
        boardio.add_scalar('A->B/best_test/translation/RMSE', best_test_t_rmse, epoch)
        boardio.add_scalar('A->B/best_test/translation/MAE', best_test_t_mae, epoch)

        
        if torch.cuda.device_count() > 1:
            torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        else:
            torch.save(net.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        gc.collect()

