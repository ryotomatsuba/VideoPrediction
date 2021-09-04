from data.dataset import get_dataset
import unittest
from hydra.experimental import initialize, compose
from utils.draw import save_gif

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


    def test_moving_mnist(self):
        args=[
            "dataset=moving_mnist",
            "dataset.num_data=2",
            ]
        self.override_config(args)
        dataset=get_dataset(self.cfg)
        self.assertEqual(dataset[0].shape,(10,128,128))
        self.assertEqual(len(dataset),2)
        save_gif(dataset[0],dataset[1])

    def test_human_action(self):
        args=[
            "dataset=human_action",
            "dataset.frames_shift=3000",
            ]
        self.override_config(args)
        dataset=get_dataset(self.cfg)
        self.assertEqual(dataset[0].shape,(10,128,128))
        self.assertEqual(len(dataset),100)
        save_gif(dataset[0],dataset[1],greyscale=True)


