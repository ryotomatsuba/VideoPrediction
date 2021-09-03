import unittest
import torch
from hydra.experimental import initialize, compose
from trainers import UnetTrainer, PredRNNTrainer, STMoETrainer,get_trainer
import os

class TestTrainer(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../configs"):
            # config is relative to a module
            overrides=[
                "train=development",
                "model=unet",
                "dataset=moving_mnist",
                "dataset.num_data=4",
                "experiment.name=unittest"]
            self.cfg = compose(config_name="default", overrides=overrides)

        os.makedirs(".hydra", exist_ok=True)
        with open(".hydra/config.yaml","x") as f:
            f.write("")
        with open("train.log","x") as f:
            f.write("")
        return super().setUp()
    
    def test_train(self)-> None:
        trainer=get_trainer(self.cfg)
        trainer.train()
        self.assertTrue(os.path.exists("best_ckpt.pth"))
        self.assertTrue(os.path.exists("pred_train_0(0).gif"))

    


    def tearDown(self) -> None:
        os.remove(".hydra/config.yaml")
        os.remove("train.log")
        os.remove("best_ckpt.pth")
        os.remove("pred_train_0(0).gif")
