from torch.utils.data import Dataset
import cv2
import torch
from pathlib import Path
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Video(Dataset):
    """
    convert avi/mp4 video to np image sequence
    """
    def __init__(self, video_path, grey_scale=True):
        self.grey_scale = grey_scale
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

    def __getitem__(self, index):
        # read the video frame
        ret, img = self.cap.read()
        if not ret:
            raise IndexError(f'{index} is not a valid index for {self.video_path}')
        if index==self.__len__()-1:
            # release the video
            self.cap.release()
        # convert to grey scale
        if self.grey_scale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img

    def __len__(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))


class VideoDataset(Dataset):
    """
    Video Dataset
    """
    
    def __init__(self,cfg):
        self.cfg=cfg
        self.get_dataset()

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        X=self.data[index]
        X=torch.from_numpy(X).type(torch.FloatTensor)
        return X
    
    def get_videos_from_dir(self,dir_path):
        """
        get video sequence

        Args:
            dir_path: str, path to the video directory
        Returns:
            videos: numpy array, video sequence. ex: (id, num_frames, h, w)
        """
        videos=[]
        dir_path=Path(dir_path)
        video_paths=dir_path.glob("*")
        for video_path in video_paths:
            logger.info(f"read {video_path}")
            video=Video(str(video_path))
            video_sequence=[]
            for i in range(len(video)):
                try:
                    video_sequence.append(video[i])
                except:
                    # if the video is broken, skip it
                    break
            videos.append(video_sequence)
        return videos
    
    def augment_videos(self,videos,num_frames=10, num_crop=0):
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

    def get_dataset(self):
        """
        get dataset
        """
        size = (self.cfg.dataset.img_width, self.cfg.dataset.img_width)
    
        videos=self.get_videos_from_dir(self.cfg.dataset.dir_path)
        videos=self.augment_videos(videos, num_frames=self.cfg.dataset.num_frames)
        videos=self.resize_videos(videos,size=size)
        self.data=np.array(videos)
        print(f"dataset size is {self.data.shape}")
