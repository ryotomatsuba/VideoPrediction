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
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

    def __getitem__(self, index):
        # read the video frame
        ret, img = self.cap.read()
        if not ret:
            raise Exception(f'{index} is not a valid index for {self.video_path}')
        if index==self.__len__()-1:
            # release the video
            self.cap.release()
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
        self.get_action_dataset()

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        X=self.data[index]
        X=torch.from_numpy(X).type(torch.FloatTensor)
        return X
    
    def get_action_videos(self,action_name):
        """
        get video sequence of a specific action

        Args:
            action_name: string, name of the action
        Returns:
            video: numpy array, video sequence. ex: (id, num_frames, 224, 224)
        """
        videos=[]
        video_path=f"/home/data/ryoto/Datasets/KTH/avi_data/{action_name}"
        video_path=Path(video_path)
        video_paths=video_path.glob("*.avi")
        for video_path in video_paths:
            print(video_path)
            video=Video(str(video_path))
            video_sequence=[]
            for i in range(len(video)):
                print(i, end="\r")
                video_sequence.append(video[i])
            videos.append(video_sequence)
        return videos
    
    def augment_videos(self,videos,num_frames=10,shift=1):
        """
        augment a video sequence

        Args:
            video_sequence: numpy array, video sequence. ex: (id, long_frames, H, W)
            num_frames: int, number of frames
        Returns:
            videos: numpy array, video sequence. ex: (id, num_frames, H, W)
        """
        new_videos=[]
        for video in videos:
            for start in range(0,len(video)-num_frames+1,shift):
                new_video=video[start:start+num_frames]
                new_videos.append(new_video)
        return new_videos
    
    def get_action_dataset(self):
        """
        get action dataset
        """
        all_videos=[]
        for action_name in self.cfg.dataset.actions:
            videos=self.get_action_videos(action_name)
            videos=self.augment_videos(videos)
            all_videos.extend(videos)
        self.data=np.array(all_videos)



# define action dataset


if __name__=="__main__":
    # test ActionDataset class
    cfg=DictConfig({
        "dataset":{
            "len_seq":10,
            "img_width": 128,
            "img_channel": 1,
            "max_intensity": 1,
            "actions":["walking", "jogging", "running", "boxing", "handwaving", "handclapping"],
        }
    })
    dataset=ActionDataset(cfg)
    print(dataset[0].shape)
    save_gif(dataset[0],dataset[0],"test.gif")