import torch
import numpy as np

import cv2
from math import *
from scipy import signal
import unittest
from utils.draw import save_gif
from omegaconf import DictConfig
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random

class TyphoonDataset(Dataset):
    """
    Typhoon Mnist Dataset
    """
    
    def __init__(self,cfg):
        
        num_data=cfg.dataset.num_data
        self.data = mv_mnist(num_data,cfg.dataset.len_seq,cfg.dataset.motions)
        self.data *= cfg.dataset.max_intensity/255

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        X=self.data[index]
        X=torch.from_numpy(X).type(torch.FloatTensor)
        return X



class Test(unittest.TestCase):
    def test(self):
        cfg={
            "dataset":{
                "num_data":10,
                "len_seq": 5,
                "max_intensity": 1,
                "motions":["transition", "rotation", "growth_decay"]
            }
        }
        cfg=DictConfig(cfg)
        self.dataset=MnistDataset(cfg)
        self.assertEqual(self.dataset[0].shape,(cfg.dataset.len_seq,128,128))
        self.assertEqual(len(self.dataset),cfg.dataset.num_data)
        save_gif(self.dataset[0],self.dataset[1])


if __name__=="__main__":
    unittest.main()
    