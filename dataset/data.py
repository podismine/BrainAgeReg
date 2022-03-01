#coding:utf8
import os
import glob
import random
import numpy as np
import nibabel as nib


from random import gauss
from torch.utils import data
from sklearn.utils import shuffle
from transformations import rotation_matrix
from sklearn.model_selection import StratifiedKFold
from scipy.ndimage.interpolation import map_coordinates



def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]

def coordinateTransformWrapper(X_T1,maxDeg=0,maxShift=7.5,mirror_prob = 0.):
    randomAngle = np.radians(maxDeg*2*(random.random()-0.5))
    unitVec = tuple(make_rand_vector(3))
    shiftVec = [maxShift*2*(random.random()-0.5),
                maxShift*2*(random.random()-0.5),
                maxShift*2*(random.random()-0.5)]
    X_T1 = coordinateTransform(X_T1,randomAngle,unitVec,shiftVec)
    return X_T1

def coordinateTransform(vol,randomAngle,unitVec,shiftVec,order=1,mode='constant'):
    '''
    Implemented based on  https://github.com/benniatli/BrainAgePredictionResNet

    @article{jonsson2019brain,
        title={Brain age prediction using deep learning uncovers associated sequence variants},
        author={J{\'o}nsson, Benedikt Atli and Bjornsdottir, Gyda and Thorgeirsson, TE and Ellingsen, Lotta Mar{\'\i}a and Walters, G Bragi and Gudbjartsson, DF and Stefansson, Hreinn and Stefansson, Kari and Ulfarsson, MO},
        journal={Nature communications},
        volume={10},
        number={1},
        pages={1--10},
        year={2019},
        publisher={Nature Publishing Group}
    }
    '''
    ax = (list(vol.shape))
    ax = [ ax[i] for i in [1,0,2]]
    coords=np.meshgrid(np.arange(ax[0]),np.arange(ax[1]),np.arange(ax[2]))

    # stack the meshgrid to position vectors, center them around 0 by substracting dim/2
    xyz=np.vstack([coords[0].reshape(-1)-float(ax[0])/2,     # x coordinate, centered
               coords[1].reshape(-1)-float(ax[1])/2,     # y coordinate, centered
               coords[2].reshape(-1)-float(ax[2])/2,     # z coordinate, centered
               np.ones((ax[0],ax[1],ax[2])).reshape(-1)])    # 1 for homogeneous coordinates
    
    mat=rotation_matrix(randomAngle,unitVec)

    transformed_xyz=np.dot(mat, xyz)

    x=transformed_xyz[0,:]+float(ax[0])/2+shiftVec[0]
    y=transformed_xyz[1,:]+float(ax[1])/2+shiftVec[1]
    z=transformed_xyz[2,:]+float(ax[2])/2+shiftVec[2]
    x=x.reshape((ax[1],ax[0],ax[2]))
    y=y.reshape((ax[1],ax[0],ax[2]))
    z=z.reshape((ax[1],ax[0],ax[2]))
    new_xyz=[y,x,z]
    new_vol=map_coordinates(vol,new_xyz, order=order,mode=mode)
    return new_vol

def generate_label(label,sigma = 2, bin_step = 1):
    labelset = np.array([i * bin_step + 12 for i in range(int(88 / bin_step))])

    dis = np.exp(-1/2. * np.power((labelset - label)/sigma/sigma, 2))
    dis = dis / dis.sum()
    return dis, labelset


def make_train_test(length, fold_idx, seed = 0, ns_splits = 5):
    assert 0 <= fold_idx and fold_idx < 5, "fold_idx must be from 0 to 4."
    skf = StratifiedKFold(n_splits=ns_splits, shuffle=True, random_state=seed)
    labels = np.zeros((length))

    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]
    return train_idx, test_idx

class AllData(data.Dataset):
    '''
    The processed data are named by Index[str]_Age[float]_sex[int: 0_Female/1_Male].npy/nii eg.I355689_69.0_0.npy
    '''
    def __init__(self, root, train=True, fold = 0):
        self.train = train
        self.root = root
        all_files = shuffle(sorted(glob.glob(os.path.join(self.root , "*"))), random_state = 1111) 
        all_files = [f for f in all_files if f.endswith(".nii.gz") or f.endswith(".npy")]

        #all_files = [f for f in all_files if float(f.split("/")[-1].split("_")[1]) > 18 and float(f.split("/")[-1].split("_")[1]) < 94] 
        assert len(all_files) > 0, "No images found"
        train_idx, test_idx = make_train_test(len(all_files),fold)

        if train:
            self.imgs = np.array(all_files)[train_idx]
            self.lbls = [float(f.split("/")[-1].split("_")[1]) for f in self.imgs]

            print("Total files: ", len(self.imgs), min(self.lbls), max(self.lbls))
        else:
            self.imgs = np.array(all_files)[test_idx]
            self.lbls = [float(f.split("/")[-1].split("_")[1]) for f in self.imgs]

    def __getitem__(self,index):
        if self.imgs[index].endswith(".nii.gz"):
            img = nib.load(self.imgs[index]).get_fdata()
        elif self.imgs[index].endswith(".npy"):
            img = np.load(self.imgs[index])
        else:
            print("failed loading... Please check files.")
            exit()

        lbl = self.lbls[index]
        lbl_y3, lbl_bc3 = generate_label(lbl, sigma = 2, bin_step= 4)
        
        if self.train:
            img = coordinateTransformWrapper(img,maxDeg=10,maxShift=5, mirror_prob = 0)
        else:
            img = img

        img = img[np.newaxis,...]

        return img, lbl, lbl_y3, lbl_bc3, index
    
    def __len__(self):
        return len(self.imgs)

import torch
class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, self.next_y, self.next_bc, self.next_indices = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_y = None
            self.next_bc = None
            self.next_indices = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True).float()
            self.next_target = self.next_target.cuda(non_blocking=True).float()
            self.next_y = self.next_y.cuda(non_blocking=True).float()
            self.next_bc = self.next_bc.cuda(non_blocking=True).float()
            self.next_indices = self.next_indices.cuda(non_blocking=True).long()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        y = self.next_y
        bc = self.next_bc
        indices = self.next_indices

        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        if y is not None:
            y.record_stream(torch.cuda.current_stream())
        if bc is not None:
            bc.record_stream(torch.cuda.current_stream())
        if indices is not None:
            indices.record_stream(torch.cuda.current_stream())

        self.preload()

        return input, target, y, bc, indices

