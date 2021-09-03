import unittest
import torch
from hydra.experimental import initialize, compose
from trainers import UnetTrainer, PredRNNTrainer, STMoETrainer,get_trainer
import os

class TestTrainer(unittest.TestCase):
    def __init__(self, methodName: str) -> None:

        # config is relative to a module
        self.overrides=[
            "train=development",
            "dataset=moving_mnist",
            "dataset.num_data=4",
            "experiment.name=unittest"]
        super().__init__(methodName=methodName)
    
    def setUp(self) -> None:
        
        os.makedirs(".hydra", exist_ok=True)
        with open(".hydra/config.yaml","x") as f:
            f.write("")
        with open("train.log","x") as f:
            f.write("")
        return super().setUp()
    
    def test_unet_trainer(self)-> None:
        self.override_model("unet")
        trainer=get_trainer(self.cfg)
        trainer.train()
        self.check_file_exists()


    def test_predrnn_trainer(self)-> None:
        self.override_model("predrnn")
        trainer=get_trainer(self.cfg)
        trainer.train()
        self.check_file_exists()

    def test_stmoe_trainer(self)-> None:
        self.override_model("stmoe")
        trainer=get_trainer(self.cfg)
        trainer.train()
        self.check_file_exists()

    def check_file_exists(self) -> None:
        self.assertTrue(os.path.exists("best_ckpt.pth"))
        self.assertTrue(os.path.exists("pred_train_0(0).gif"))
    
    def override_model(self, model_name: str) -> None:
        with initialize(config_path="../configs"):
            self.overrides.append(f"model={model_name}")
            self.cfg = compose(config_name="default", overrides=self.overrides)
        


    


    def tearDown(self) -> None:
        os.remove(".hydra/config.yaml")
        os.remove("train.log")
        os.remove("best_ckpt.pth")
        os.remove("pred_train_0(0).gif")
