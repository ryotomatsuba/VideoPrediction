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
        da, da2 = np.zeros((2, len_seq, output_h*2, output_w*2)) # data for transition 
        da3 = np.zeros((len_seq, output_h, output_w)) # data for rotation
        x2, y2 = np.random.randint(64, 128+64-28, size=2)
        x3, y3 = np.random.randint(0, 100, size=2)
        x_start, y_start = np.random.randint(80, 160, size=2)
        x_0v, y_0v = np.random.randint(-3, 3, size=2)
        x_a, y_a = 0, 0 # no acceraration
        x, y = x_start, y_start
        ang_0 = np.random.randint(-6, 6)
        ang_a = int(np.random.normal()*0.1)     
        angles = 0
        mask = np.random.rand(1).reshape(1, 1) 
        mask /= np.sum(mask) if np.sum(mask)!=0 else mask
        mask = (1 + np.random.uniform(-0.5, 0.5)) * mask
        da3[0, y3:y3+28, x3:x3+28] = mnist[seeds[i]+20000]
        for t in range(len_seq):
            ang = ang_0 + ang_a*t
            angles += ang
            trans = cv2.getRotationMatrix2D((20, 20), angles, 1)
            mn = np.zeros((40, 40))
            mn[6:34, 6:34] = mnist[seeds[i]] # put mnist image on (40,40) canvas
            da2[t, x2:x2+40, y2:y2+40]  = cv2.warpAffine(mn, trans, (40, 40)) # rotate mnist image
            x_v = x_0v + x_a*t
            y_v = x_0v + y_a*t 
            x += x_v
            y += y_v
            try:
                da[t, x:x+28, y:y+28] = mnist[seeds[i]+30000] # put on mnist image
            except:
                da = np.zeros_like(da)
                print(i, t, x_start, y_start)
                break
            if t>0:
                da_t = signal.convolve2d(da3[t-1], mask, mode = 'same')
            else:
                da_t = signal.convolve2d(da3[t], mask, mode = 'same')                
            f = np.fft.fft2(da_t)
            fshift = np.fft.fftshift(f)
            fshift[128-128//4:128+128//4, 128-128//4:128+128//4] = 0
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            da3[t] = img_back
        da3 = da3/(da3.max())*255
        data[i] = da[:, 64:192, 64:192] + da2[:, 64:192, 64:192] + da3
    return data      


class Test(unittest.TestCase):
    def test(self):
        cfg={"dataset":{"num_data":10,"input_num":4}}
        cfg=DictConfig(cfg)
        self.dataset=MnistDataset(cfg)
        self.assertEqual(self.dataset[0].shape,(10,128,128))
        self.assertEqual(len(self.dataset),cfg.dataset.num_data)
    def tearDown(self) -> None:
        save_gif(self.dataset[0],self.dataset[1])
if __name__=="__main__":
    unittest.main()
    