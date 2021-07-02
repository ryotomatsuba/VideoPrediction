import torch
import numpy as np
import seaborn as sns
sns.set()
import cv2
from math import *
from scipy import signal
import unittest
from utils.draw import save_gif
from omegaconf import DictConfig
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
class MnistDataset(Dataset):
    
    def __init__(self,cfg):
        mnist = np.load("/home/data/ryoto/Datasets/row/mnist.npz")
        num_data=cfg.dataset.num_data
        self.data = mv_mnist(num_data, mnist['X'].reshape(60000, 28, 28))
        self.input_num=cfg.dataset.input_num

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        X=self.data[index]
        X=torch.from_numpy(X).type(torch.FloatTensor)
        return X


def mv_mnist(num_sample, mnist):
    _, mnist_h, mnist_w = mnist.shape
    len_seq, output_h, output_w = 10, 128, 128 
    seeds = np.arange(0, num_sample, 1)
    data = np.zeros((num_sample, len_seq, output_h, output_w))
    for i in range(num_sample):
        np.random.seed(seeds[i])
        transition_mnist, rotation_mnist, growth_decay_mnist = np.zeros((3, len_seq, output_h*2, output_w*2)) # data for 
        x_rot, y_rot = np.random.randint(64, 128+64-28, size=2) # rotate position
        x_grow, y_grow = np.random.randint(64, 128+64-28, size=2) # growth/decay position
        x_trans, y_trans = np.random.randint(80, 160, size=2) # trainsition position
        v0_x, v0_y = np.random.randint(-3, 3, size=2)
        a_x, a_y = 0, 0 # no acceraration
        ang_0 = np.random.randint(-6, 6)
        a_ang = int(np.random.normal()*0.1)     
        mask = np.random.rand(1).reshape(1, 1) 
        mask /= np.sum(mask) if np.sum(mask)!=0 else mask
        mask = (1 + np.random.uniform(-0.5, 0.5)) * mask
        growth_decay_mnist[0, y_grow:y_grow+28, x_grow:x_grow+28] = mnist[seeds[i]+20000]
        for t in range(len_seq):
            # rotation processing
            angles = ang_0*t + a_ang*t**2
            rotation_matrix = cv2.getRotationMatrix2D((20, 20), angles, 1)
            mn = np.zeros((40, 40))
            mn[6:34, 6:34] = mnist[seeds[i]] # put mnist image on (40,40) canvas
            rotation_mnist[t, x_rot:x_rot+40, y_rot:y_rot+40]  = cv2.warpAffine(mn, rotation_matrix, (40, 40)) # rotate mnist image
            # trainsition processing
            x = x_trans + v0_x*t + a_x*t**2
            y = y_trans + v0_y*t + a_y*t**2
            try:
                transition_mnist[t, x:x+28, y:y+28] = mnist[seeds[i]+30000] # put on mnist image
            except:
                transition_mnist = np.zeros_like(transition_mnist)
                print(i, t, x_trans, y_trans)
                break
            # growth and decay processing
            if t>0:
                da_t = signal.convolve2d(growth_decay_mnist[t-1], mask, mode = 'same')
            else:
                da_t = signal.convolve2d(growth_decay_mnist[t], mask, mode = 'same')                
            f = np.fft.fft2(da_t)
            fshift = np.fft.fftshift(f)
            fshift[256-256//4:, 256-256//4:] = 0
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            growth_decay_mnist[t] = img_back
        growth_decay_mnist = growth_decay_mnist/(growth_decay_mnist.max())*255
        frames = transition_mnist + rotation_mnist + growth_decay_mnist
        data[i] = frames[:, 64:192, 64:192] # crop center
    return data      


class Test(unittest.TestCase):
    def test(self):
        cfg={"dataset":{"num_data":10,"input_num":4}}
        cfg=DictConfig(cfg)
        self.dataset=MnistDataset(cfg)
        self.assertEqual(self.dataset[0].shape,(10,128,128))
        self.assertEqual(len(self.dataset),cfg.dataset.num_data)
        save_gif(self.dataset[0],self.dataset[1])


if __name__=="__main__":
    unittest.main()
    