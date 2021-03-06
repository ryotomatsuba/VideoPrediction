import torch
import numpy as np
import seaborn as sns
sns.set()
import cv2
from math import *
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np
import torch
from .video_data import Video

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
            video: numpy array, video sequence. ex: (id, num_frames, h, w)
        """
        videos=[]
        video_path=f"/data/Datasets/KTH/avi_data/{action_name}"
        video_path=Path(video_path)
        video_paths=video_path.glob("*.avi")
        for video_path in video_paths:
            video=Video(str(video_path))
            video_sequence=[]
            for i in range(len(video)):
                print(f"read {video_path}", end="\r")
                video_sequence.append(video[i])
            videos.append(video_sequence)
        return videos
    
    def augment_videos(self,videos,num_frames=10, num_crop=30):
        """
        augment a video sequence by shifting frames.
        if num_crop>0, then crop front and back frames.

        Args:
            video_sequence: numpy array, video sequence. ex: (id, long_frames, H, W)
            num_frames: int, number of frames
        Returns:
            videos: numpy array, video sequence. ex: (id, num_frames, h, w)
        """
        frames_shift = self.cfg.dataset.frames_shift
        new_videos=[]
        for video in videos:
            for start in range(num_crop,len(video)-num_frames-num_crop+1,frames_shift):
                new_video=video[start:start+num_frames]
                new_videos.append(new_video)
        return new_videos
    
    def resize_videos(self,videos,size=(128,128)):
        """
        resize a video sequence

        Args:
            videos: numpy array, video sequence. ex: (id, num_frames, h, w)
            size: tuple, size of the resized video
        Returns:
            videos: numpy array, video sequence. ex: (id, num_frames, h, w)
        """
        new_videos=[]
        for video in videos:
            new_video=[]
            for frame in video:
                frame=cv2.resize(frame,size)
                # intensity normalization
                frame=frame/255.0
                new_video.append(frame)
            new_videos.append(new_video)
        return new_videos

    def get_action_dataset(self):
        """
        get action dataset
        """
        size = (self.cfg.dataset.img_width, self.cfg.dataset.img_width)
        all_videos=[]
        for action_name in self.cfg.dataset.actions:
            videos=self.get_action_videos(action_name)
            videos=self.augment_videos(videos, num_frames=self.cfg.dataset.num_frames)
            videos=self.resize_videos(videos,size=size)
            all_videos.extend(videos)
        self.data=np.array(all_videos)
        print(f"dataset size is {self.data.shape}")



# define action dataset


if __name__=="__main__":
    pass