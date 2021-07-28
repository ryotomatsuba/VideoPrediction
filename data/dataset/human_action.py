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
from torch.utils.data import Dataset, dataset
import matplotlib.pyplot as plt
from pathlib import Path

import cv2
from PIL import Image
import random
import os
import numpy as np
import socket
import torch
from scipy import misc

class Video(Dataset):
    """
    convert avi video to np image sequence
    """
    def __init__(self, video_path, grey_scale=True):
        self.grey_scale = grey_scale
        self.cap = cv2.VideoCapture(video_path)

    def __getitem__(self, index):
        # read the video frame
        ret, img = self.cap.read()
        # convert to grey scale
        if self.grey_scale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img

    def __len__(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

class ActionDataset(Dataset):
    """
    Human Action Dataset
    """
    
    def __init__(self,cfg):
        self.cfg=cfg

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        X=self.data[index]
        X=torch.from_numpy(X).type(torch.FloatTensor)
        return X


# define action dataset


if __name__=="__main__":
    video=Video("/home/data/ryoto/Datasets/KTH/avi_data/boxing/person01_boxing_d1_uncomp.avi")
    print(video[0].shape)
    plt.imsave("test.png",video[0])
    # save_gif(video[0:10],video[0:10], "./test.gif")
