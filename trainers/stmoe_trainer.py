# -*- coding: utf-8 -*-
"""STMoE Trainer"""
import trainers
from models.stmoe import STMoE
import logging
import torch.nn as nn
import torch
from torch import optim
from tqdm import tqdm
import numpy as np
from trainers.base_trainer import BaseTrainer
from data.dataset import MnistDataset,ActionDataset
from omegaconf import DictConfig
import unittest

log = logging.getLogger(__name__)


class STMoETrainer(BaseTrainer):
    """STMoE Trainer
    
    Attributes:
        self.cfg: Config of project.

    """

    def __init__(self, cfg: object) -> None:
        """Initialization
    
        Args:
            self.cfg: Config of project.

        """
        self.net = STMoE(input_num=cfg.model.input_num,n_expert=2,training=cfg.model.training) # define model
        super().__init__(cfg)

    def train(self) -> None:
        """Train

        Trains model.

        """

        super().train()
        epochs=self.cfg.train.epochs
        lr=self.cfg.train.lr


        global_step = 0

        logging.info('Starting training')

        optimizer = optim.RMSprop(self.net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

        self.criterion = nn.MSELoss()

        for epoch in range(epochs):

            logging.info(f'epoch: {epoch}')
            self.net.train()

            epoch_loss = 0
            # train
            for X in tqdm(self.train_loader, ncols=100):
                X = X.to(device=self.device, dtype=torch.float32)
                input_num=self.cfg.model.input_num
                total_num=X.shape[1]
                for t in range(input_num, total_num):
                    pred, weight = self.net(X[:,t-input_num:t])
                    loss = self.criterion(pred, X[:,[t]])
                    epoch_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(self.net.parameters(), 0.1)
                    optimizer.step()
                global_step += 1
            loss_ave = epoch_loss / len(self.train_loader) # average per batch
            self.loss["train"].append(loss_ave) 
            logging.info(f'Train MSE     : {loss_ave}')
            # val
            if epoch % 1 == 0: # checkpoint interval
                val_score = self.eval()
                scheduler.step(val_score)
                self.log_metrics(epoch)

            # save model if it performes better than it has done before
            if epoch==0 or (epoch > 1 and self.loss["val"][-1] < min(self.loss["val"][:-1])):
                torch.save(self.net.state_dict(),self.cfg.train.ckpt_path)
                self.save_gif(epoch)
                self.log_artifact(self.cfg.train.ckpt_path)
                logging.info(f'Checkpoint saved !')
        self.log_base_artifacts()



    def eval(self) -> float:
        super().eval()
        device = self.device

        self.net.eval()
        epoch_loss = 0 
        
        for X in tqdm(self.val_loader, ncols=100):
            X = X.to(device=self.device, dtype=torch.float32)
            input_num=self.cfg.model.input_num
            total_num=X.shape[1]
            for t in range(input_num,total_num):
                with torch.no_grad():
                    pred, weight= self.net(X[:,t-input_num:t])
                loss = self.criterion(pred, X[:,[t]])
                epoch_loss += loss.item()


        loss_ave=epoch_loss/len(self.val_loader)
        self.loss["val"].append(loss_ave)
        logging.info(f'Validation MSE: {loss_ave}')
        return loss_ave

    def save_gif(self,epoch):
        """
        save generated image sequences as gif file
        """
        device = self.device
        self.net.eval()
        draw_grey = self.cfg.dataset.grey_scale
        for phase in ["train", "val"]:
            data_loader = self.train_loader if phase == "train" else self.val_loader
            X = iter(data_loader).__next__()
            X = X.to(device=self.device, dtype=torch.float32)
            input_num = self.cfg.model.input_num
            num_expert = self.cfg.model.num_expert
            batch_size, total_num, height, width=X.shape
            preds=torch.empty(batch_size, 0, height, width).to(device=self.device)
            weights=torch.empty(batch_size, 0, num_expert, height, width).to(device=self.device)

            input_X = X[:,0:input_num]
            for t in range(input_num,total_num):
                with torch.no_grad():
                    pred, weight = self.net(input_X)
                input_X=torch.cat((input_X[:,1:],pred),dim=1) # use output image to pred next frame
                preds=torch.cat((preds,pred),dim=1) 
                weights=torch.cat((weights,weight[:,np.newaxis]),dim=1) 
            X, preds, weights= X.to(device="cpu"), preds.to(device="cpu"), weights.to(device="cpu")
            super().save_weight_gif(preds,weights, epoch, phase)
            super().save_gif(X, preds, epoch, phase)



class Test(unittest.TestCase):
    def test(self):
        cfg={
            "experiment":{"name":"test"},
            "dataset":{
                "num_data":10,
                "num_frames": 5,
                "max_intensity": 1,
                "motions":["transition", "rotation", "growth_decay"]
            },
            "train":{
                "epochs": 1,
                "batch_size": 2,
                "save_best_ckpt": True,
                "num_save_images": 1,
                "num_workers": 2,
                "ckpt_path": "best_ckpt.pth",
                "lr": 0.0001,
                "val_percent": 0.1,

            },
            "model":{
                "name": "unet",
                "input_num": 4,
                "load": False
            }
        }
        cfg=DictConfig(cfg)
        trainer = STMoETrainer(cfg)
        trainer.train()


if __name__=="__main__":
    unittest.main()
    