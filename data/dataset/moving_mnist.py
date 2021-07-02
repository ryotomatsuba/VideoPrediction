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
import random
class MnistDataset(Dataset):
    
    def __init__(self,cfg):
        
        num_data=cfg.dataset.num_data
        self.data = mv_mnist(num_data,cfg.dataset.motions)
        self.input_num=cfg.dataset.input_num

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        X=self.data[index]
        X=torch.from_numpy(X).type(torch.FloatTensor)
        return X


def mv_mnist(num_sample, choice=["transition", "rotation", "growth_decay"]):
    """
    make moving mnist videos
    Params:
    num_sample: number of data
    choice: which motion to use (transition, rotation, growth_decay)
    """
    mnist = np.load("/home/data/ryoto/Datasets/row/mnist.npz")['X'].reshape(60000, 28, 28)
    mnist_num, mnist_h, mnist_w = mnist.shape
    len_seq, output_h, output_w = 10, 128, 128 
    seeds = np.arange(0, num_sample, 1)
    data = np.zeros((num_sample, len_seq, output_h, output_w))
    for i in range(num_sample):
        np.random.seed(seeds[i])
        for j in range(3):
            motion=random.choice(choice)
            index=np.random.choice(mnist_num)
            if motion=="transition":
                data[i]+=make_transition_movie(mnist[index]) # overrap videos
            if motion=="rotation":
                data[i]+=make_rotation_movie(mnist[index]) # overrap videos
            if motion=="growth_decay":
                data[i]+=make_growth_decay_movie(mnist[index]) # overrap videos
    return data      


def make_rotation_movie(image, len_seq=10, angle_range=30):
    """
    Params: image = mnist image shape (28,28)
    Return: rotation movie shape(len_seq,128,128)
    """
    angles=np.random.randint(-angle_range, angle_range)
    rotation_movie = np.zeros((len_seq, 128, 128))
    x, y = np.random.randint(0, 128-28, size=2) # rotate center
    for t in range(0, len_seq):
        rotation_matrix = cv2.getRotationMatrix2D((14,14), angles*t, 1)
        rotation_movie[t, x:x+28, y:y+28] = cv2.warpAffine(image, rotation_matrix, (28, 28))
    return rotation_movie

def make_transition_movie(image, len_seq=10, v_range=3, a_range=0):
    """
    Params: image = mnist image shape (28,28)
    Return: transition movie shape(len_seq,128,128)
    """
    transition_movie = np.zeros((len_seq, 256, 256))
    x_trans, y_trans = np.random.randint(80, 160, size=2) # trainsition position
    v_x, v_y = np.random.randint(-v_range, v_range, size=2)
    if a_range:
        a_x, a_y = np.random.randint(-a_range, a_range, size=2)
    else:
        a_x, a_y = 0, 0
    for t in range(len_seq):
        x = x_trans + v_x*t + a_x*t**2
        y = y_trans + v_y*t + a_y*t**2
        try:
            transition_movie[t, x:x+28, y:y+28] = image # put on mnist image
        except:
            transition_movie = np.zeros_like(transition_movie)
            break
    transition_movie=transition_movie[:, 64:192, 64:192] # center crop       
    return transition_movie

def make_growth_decay_movie(image, len_seq=10):
    """
    Params: image = mnist image shape (28,28)
    Return: growth/decay movie shape(len_seq,128,128)
    """
    growth_decay_mnist = np.zeros((len_seq, 128, 128)) # data for growth_decay
    x_grow, y_grow = np.random.randint(0, 128-28, size=2) # growth/decay position
    mask = np.random.rand(1).reshape(1, 1) 
    mask /= np.sum(mask) if np.sum(mask)!=0 else mask
    mask = (1 + np.random.uniform(-0.5, 0.5)) * mask
    growth_decay_mnist[0, y_grow:y_grow+28, x_grow:x_grow+28] = image
    for t in range(len_seq):        
        # growth and decay processing
        if t>0:
            da_t = signal.convolve2d(growth_decay_mnist[t-1], mask, mode = 'same')
        else:
            da_t = signal.convolve2d(growth_decay_mnist[t], mask, mode = 'same')                
        f = np.fft.fft2(da_t)
        fshift = np.fft.fftshift(f)
        fshift[128*3//4:, 128*3//4:] = 0
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        growth_decay_mnist[t] = img_back
    return  growth_decay_mnist/(growth_decay_mnist.max())*255

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
    