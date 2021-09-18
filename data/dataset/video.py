from torch.utils.data import Dataset
import cv2


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
