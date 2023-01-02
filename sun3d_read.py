import os
import numpy as np
import pickle
import glob
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from util import npmat2euler

def load_data(mode):
    print('Loading %s data' % mode)
    prefix = '*.pickle'

    data_dir = os.path.join(os.getcwd(), 'sun3d', mode)
    file_name_list = sorted(glob.glob(os.path.join(data_dir, '*.pickle')))[1]

    data = {}

    for filename in [file_name_list]:
        file = open(filename, 'rb')
        data_temp = pickle.load(file)
        file.close()

        if not data:
            data = data_temp
        else:
            for key in data:
                data[key] += data_temp[key]
    length = 3000
    flag_seq = data['flag']
    #for i in range(len(flag_seq)): # [0,3218]
    for i in range(30): # [0,3218]
        pc_size = flag_seq[i].shape[0]
        if pc_size < length:
            length = pc_size
    data_real = {}
    x1, x2, R, t, Flag, index1, index2 = [], [], [], [], [], [], []
    #for i in range(len(flag_seq)):
    for i in range(30): # [0,3218]
        x1.append(data['x1'][i][:length])
        x2.append(data['x2'][i][:length])
        R.append(data['R'][i][:length])
        t.append(data['t'][i][:length])
        Flag.append(np.diag(data['flag'][i][:length]))
        index1.append(data['idx1'][i])
        index2.append(data['idx2'][i])
    x1 = np.array(x1)
    x2 = np.array(x2)
    R = np.array(R)
    t = np.array(t)
    Flag = np.array(Flag)
    index1 = np.array(index1)
    index2 = np.array(index2)
    data_real['x1'] = x1 
    data_real['x2'] = x2 
    data_real['R'] = R
    data_real['t'] = t
    data_real['Flag'] = Flag
    data_real['idx1'] = index1
    data_real['idx2'] = index2
    return data_real

class Sun3d(Dataset):
    def __init__(self, partition='test'):
        self.data_real = load_data(partition)
        self.partition = partition

    def __getitem__(self, item):
        pc1 = self.data_real['x1'][item]
        pc2 = self.data_real['x2'][item]
        R_ab = self.data_real['R'][item]
        translation_ab = self.data_real['t'][item]
        I_gt = self.data_real['Flag'][item]
        idx1 = self.data_real['idx1'][item]
        idx2 = self.data_real['idx2'][item]
        num_points,dim = pc1.shape
        if self.partition == 'train':
            select1 = np.random.permutation(num_points)
            select2 = np.random.permutation(num_points)
            pc1_select = pc1[select1]
            pc2_select = pc2[select2]

            I_gt_temp = I_gt[select1].T
            I_gt = I_gt_temp[select2].T
        else:
            pc1_select = pc1 
            pc2_select = pc2


        return pc1_select.T.astype('float32'), pc2_select.T.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), I_gt.astype('float32'), idx1, idx2
    def __len__(self):
        return self.data_real['t'].shape[0]

def main():
    data = load_data('train')

if __name__ == "__main__":
    main()
