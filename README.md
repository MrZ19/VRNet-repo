# VRNet-repo

## Related information
Here is the code of "VRNet: Learning the Rectified Virtual Corresponding Points for 3D Point Cloud Registration" (``https://ieeexplore.ieee.org/abstract/document/9681904``), which proposes to rectify the virtual corresponding points to avoid the degeneration problem.

<!--Note: the code is being prepared. -->

## Implementation
The code is tested with Pytorch 1.6.0 with CUDA 10.2.89. Prerequisites include scipy, h5py, tqdm, etc. Your can install them by yourself.

The ModelNet40 dataset can be download from:
```
https://github.com/WangYueFt/dcp
```

Start training with the command:
```
python main.py 
```

Start testing with the command:
```
python main.py --eval True --mdoel_path YOUR_CHECKPOINT_DIRECTORY
```

## Acknowledgement
The code is insipred by DCP, PRNet, RPMNet, etc.

## Please cite:
```
@article{vrnet_tcsv_zhang,
  author    = {Zhiyuan Zhang and Jiadai Sun and Yuchao Dai and Bin Fan and Mingyi He},
  title     = {VRNet: Learning the Rectified Virtual Corresponding Points for 3D Point Cloud Registration},
  journal   = {IEEE Transactions on Circuits and Systems for Video Technology},
  volume    = {32},
  number    = {8},
  pages     = {4997--5010},
  year      = {2022}
}
```
