import unittest

import torch
from hydra.experimental import compose, initialize

from data.dataset.get_images import *
from utils.draw import save_gif


class TestImages(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
    
        super().__init__(methodName=methodName)
        return

    def tearDown(self) -> None:
        return

    def check_intensity_range(self,images):
        print(f"min:{images.min()},max:{images.max()}")
        self.assertLessEqual(0,images.min())
        self.assertLessEqual(images.max(),255)
        self.assertAlmostEqual(images[0].mean(), 255/2, delta=126)

    def test_mnist_images(self):
        images=get_mnist_images(num_images=10)
        self.assertEqual(images.shape,(10,28,28))
        self.check_intensity_range(images)

    
    def test_box_images(self):
        images=get_box_images(num_images=10)
        self.assertEqual(images.shape,(10,28,28))
        self.check_intensity_range(images)
    
    def test_cifar10_images(self):
        images=get_cifar10_images(num_images=10)
        self.assertEqual(images.shape,(10,28,28))
        self.check_intensity_range(images)


    def test_wave_images(self):
        images=get_wave_images(num_images=10)
        self.assertEqual(images.shape,(10,28,28))
        self.check_intensity_range(images)

    
    def test_mix_images(self):
        images1=get_box_images(num_images=10)
        images2=get_wave_images(num_images=20)

        images=get_mix_images(images1,images2)
        self.assertEqual(images.shape,(10,28,28))
        self.check_intensity_range(images)




