from data.dataset import get_dataset
import unittest
from hydra.experimental import initialize, compose
from utils.draw import save_gif
import torch

class TestDataset(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
    
        self.overrides=[
            "experiment.name=unittest",
            "dataset.num_frames=10",
            "dataset.img_width=128"]
        super().__init__(methodName=methodName)
        return

    def tearDown(self) -> None:
        return

    
    def override_config(self, args: list) -> None:
        with initialize(config_path="../configs"):
            self.overrides.extend(args)
            self.cfg = compose(config_name="default", overrides=self.overrides)
    
    def check_intensity_range(self, dataset: torch.Tensor, min_val=-1e-6, max_val=1) -> None:
        """check if dataset is almost in range [min_val, max_val]"""
        q=torch.tensor([0.25,0.75])
        # if tensor is too large, then clip it
        if len(dataset)>10:
            dataset=dataset[:10]
        q1, q3 =torch.quantile(dataset[:],q)
        self.assertLess(min_val,q1)
        self.assertLess(q3, max_val)


    def test_moving_image(self):
        args=[
            "dataset=moving_image",
            "dataset.num_data=2",
            ]
        self.override_config(args)
        dataset=get_dataset(self.cfg)
        self.assertEqual(dataset[:].shape,(2,10,128,128))
        self.check_intensity_range(dataset[:])
        save_gif(dataset[0],dataset[1])

    def test_human_action(self):
        args=[
            "dataset=video_data",
            "dataset.dir_path=/data/Datasets/KTH/avi_data/walking",
            "dataset.frames_shift=3000",
            ]
        self.override_config(args)
        dataset=get_dataset(self.cfg)
        self.assertEqual(dataset[:].shape,(100,10,128,128)) # check shape
        self.check_intensity_range(dataset[:])
        save_gif(dataset[0],dataset[1],greyscale=True)

    def test_traffic_dataset(self):
        args=[
            "dataset=video_data",
            "dataset.dir_path=/data/Datasets/traffic/video",
            "dataset.frames_shift=10",
            ]
        self.override_config(args)
        dataset=get_dataset(self.cfg)
        self.assertEqual(dataset[:].shape,(1261,10,128,128))
        self.check_intensity_range(dataset[:])
        save_gif(dataset[0],dataset[1],greyscale=True)

    
    def test_video_data_mp4(self):
        args=[
            "dataset=video_data",
            "dataset.dir_path=/data/Datasets/MP4",
            "dataset.frames_shift=100",
            ]
        self.override_config(args)
        dataset=get_dataset(self.cfg)
        self.assertEqual(dataset[:].shape,(18,10,128,128))
        self.check_intensity_range(dataset[:])
        save_gif(dataset[0],dataset[1],greyscale=True)


