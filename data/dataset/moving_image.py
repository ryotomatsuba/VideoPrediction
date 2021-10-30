import random
import unittest
import urllib.request
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from omegaconf import DictConfig
from scipy import signal
from torch.utils.data import Dataset
from utils.draw import save_gif


class MovingImageDataset(Dataset):
    """
    Moving Image Dataset
    """
    
    def __init__(self,cfg):
        
        num_data=cfg.dataset.num_data
        
        # select image_type
        if cfg.dataset.image_type=="mnist":
            images=get_mnist_images()
        elif cfg.dataset.image_type=="cifar10":
            images=get_cifar10_images()
        else:
            raise ValueError(f'image_type {cfg.dataset.image_type} is not supported')
        self.data = get_moing_image_video(images,num_data,cfg.dataset.num_frames,cfg.dataset.motions)
        self.data *= cfg.dataset.max_intensity/255

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        X=self.data[index]
        X=torch.from_numpy(X).type(torch.FloatTensor)
        return X

def get_mnist_images():
    """get mnist images
    Return: 
        mnist_images: shape(60000, 28, 28)
    """
    req = urllib.request.Request('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz')
    with urllib.request.urlopen(req) as response:
        data = response.read()
        mnist_images = np.load(BytesIO(data),allow_pickle=True)['x_train'].reshape(60000, 28, 28)
    return mnist_images

def get_cifar10_images(num_images=1000):
    """get cifar10 images
    Return: 
        cifar10_images: shape(num_images, 28, 28)
    """
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((28,28)),
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
    ])
    cifar10_dataset=torchvision.datasets.CIFAR10(root='/data/Datasets/', download=True, transform=transform)
    cifar10_images=np.zeros((1000, 28, 28))
    for i in range(num_images):
        cifar10_images[i]=cifar10_dataset[i][0].numpy()
    return cifar10_images


def get_moing_image_video(images, num_sample, num_frames=10, choice=["transition", "rotation", "growth_decay"]):
    """
    make moving image videos
    Params:
        images = image shape (28,28)
        num_sample: number of data
        choice: which motion to use (transition, rotation, growth_decay)
    Return:
        data: shape(num_sample, num_frames, 128, 128)
    """

    
    image_num, image_h, image_w = images.shape
    output_h, output_w = 128, 128 
    seeds = np.arange(0, num_sample, 1)
    data = np.zeros((num_sample, num_frames, output_h, output_w))
    for i in range(num_sample):
        np.random.seed(seeds[i])
        for j in range(3):
            motion=random.choice(choice)
            index=np.random.choice(image_num)
            if motion=="transition":
                data[i]+=make_transition_movie(images[index], num_frames=num_frames) # overrap videos
            if motion=="rotation":
                data[i]+=make_rotation_movie(images[index], num_frames=num_frames) # overrap videos
            if motion=="growth_decay":
                data[i]+=make_growth_decay_movie(images[index], num_frames=num_frames) # overrap videos
    return data      


def make_rotation_movie(image, num_frames=10, angle_range=30):
    """
    Params: image = image shape (28,28)
    Return: rotation movie shape(num_frames,128,128)
    """
    angles=np.random.randint(-angle_range, angle_range)
    rotation_movie = np.zeros((num_frames, 128, 128))
    x, y = np.random.randint(0, 128-28, size=2) # rotate center
    for t in range(0, num_frames):
        rotation_matrix = cv2.getRotationMatrix2D((14,14), angles*t, 1)
        rotation_movie[t, x:x+28, y:y+28] = cv2.warpAffine(image, rotation_matrix, (28, 28))
    return rotation_movie

def make_transition_movie(image, num_frames=10, v_range=3, a_range=0):
    """
    Params: image = image shape (28,28)
    Return: transition movie shape(num_frames,128,128)
    """
    transition_movie = np.zeros((num_frames, 256, 256))
    x_trans, y_trans = np.random.randint(80, 160, size=2) # trainsition position
    v_x, v_y = np.random.randint(-v_range, v_range, size=2)
    if a_range:
        a_x, a_y = np.random.randint(-a_range, a_range, size=2)
    else:
        a_x, a_y = 0, 0
    for t in range(num_frames):
        x = x_trans + v_x*t + a_x*t**2
        y = y_trans + v_y*t + a_y*t**2
        try:
            transition_movie[t, x:x+28, y:y+28] = image # put image
        except:
            transition_movie = np.zeros_like(transition_movie)
            break
    transition_movie=transition_movie[:, 64:192, 64:192] # center crop       
    return transition_movie

def make_growth_decay_movie(image, num_frames=10):
    """
    Params: image = image shape (28,28)
    Return: growth/decay movie shape(num_frames,128,128)
    """
    growth_decay_movie = np.zeros((num_frames, 128, 128)) # data for growth_decay
    x_grow, y_grow = np.random.randint(0, 128-28, size=2) # growth/decay position
    mask = np.random.rand(1).reshape(1, 1) 
    mask /= np.sum(mask) if np.sum(mask)!=0 else mask
    mask = (1 + np.random.uniform(-0.5, 0.5)) * mask
    growth_decay_movie[0, y_grow:y_grow+28, x_grow:x_grow+28] = image
    for t in range(num_frames):        
        # growth and decay processing
        if t>0:
            da_t = signal.convolve2d(growth_decay_movie[t-1], mask, mode = 'same')
        else:
            da_t = signal.convolve2d(growth_decay_movie[t], mask, mode = 'same')                
        f = np.fft.fft2(da_t)
        fshift = np.fft.fftshift(f)
        fshift[128*3//4:, 128*3//4:] = 0
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        growth_decay_movie[t] = img_back
    return  growth_decay_movie/(growth_decay_movie.max())*255

class Test(unittest.TestCase):
    def test(self):
        cfg={
            "dataset":{
                "num_data":10,
                "num_frames": 5,
                "max_intensity": 1,
                "motions":["transition", "rotation", "growth_decay"]
            }
        }
        cfg=DictConfig(cfg)
        self.dataset=MovingImageDataset(cfg)
        self.assertEqual(self.dataset[0].shape,(cfg.dataset.num_frames,128,128))
        self.assertEqual(len(self.dataset),cfg.dataset.num_data)
        save_gif(self.dataset[0],self.dataset[1])


if __name__=="__main__":
    unittest.main()
    