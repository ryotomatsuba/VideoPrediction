# -*- coding: utf-8 -*-
"""Unet Trainer"""
from models.unet import UNet
import logging
import torch.nn as nn
import torch
from torch import optim
import os
import numpy as np
from trainers.base_trainer import BaseTrainer
from torch.utils.data import DataLoader,random_split
from data.dataset import MnistDataset
from utils.draw import save_gif

log = logging.getLogger(__name__)


class UnetTrainer(BaseTrainer):
    """Unet Trainer
    
    Attributes:
        self.cfg: Config of project.

    """

    def __init__(self, cfg: object) -> None:
        """Initialization
    
        Args:
            self.cfg: Config of project.

        """
        # define dataset
        dataset = MnistDataset(cfg)
        n_val = int(len(dataset) * cfg.train.val_percent)
        n_train = len(dataset) - n_val
        train, val = random_split(dataset, [n_train, n_val])
        self.train_loader = DataLoader(train, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers, pin_memory=True)
        self.val_loader = DataLoader(val, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True)
        # define model
        device = torch.device(cfg.train.device)
        net = UNet(n_channels=4, n_classes=1, bilinear=True)
        net.to(device=device)
        logging.info(f'Network:\n'
                    f'\t{net.n_channels} input channels\n'
                    f'\t{net.n_classes} output channels (classes)\n'
                    f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

        if cfg.model.load:
            net.load_state_dict(
                torch.load(cfg.model.load, map_location=device)
            )
            logging.info(f'Model loaded from {cfg.load}')
        self.net=net
        super().__init__(cfg)


    def train(self) -> None:
        """Train

        Trains model.

        """

        super().train()
        epochs=self.cfg.train.epochs
        lr=self.cfg.train.lr
        img_scale=self.cfg.scale
        device = next(self.net.parameters()).device


        global_step = 0

        logging.info('Starting training')

        optimizer = optim.RMSprop(self.net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if self.net.n_classes > 1 else 'max', patience=2)

        self.criterion = nn.MSELoss()

        for epoch in range(epochs):

            logging.info(f'epoch: {epoch}')
            self.net.train()

            epoch_loss = 0
            # train
            for X in self.train_loader:
                X = X.to(device=device, dtype=torch.float32)
                input_num=self.cfg.dataset.input_num
                total_num=X.shape[1]
                for t in range(input_num, total_num):
                    pred = self.net(X[:,t-input_num:t])
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
        device = next(self.net.parameters()).device

        self.net.eval()
        epoch_loss = 0 

        for X in self.val_loader:
            X = X.to(device=device, dtype=torch.float32)
            input_num=self.cfg.dataset.input_num
            total_num=X.shape[1]
            for t in range(input_num,total_num):
                with torch.no_grad():
                    pred = self.net(X[:,t-input_num:t])
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
        device = next(self.net.parameters()).device
        self.net.eval()
        for phase in ["train", "val"]:
            data_loader = self.train_loader if phase == "train" else self.val_loader
            X = iter(data_loader).__next__()
            X = X.to(device=device, dtype=torch.float32)
            input_num = self.cfg.dataset.input_num
            batch_size, total_num, height, width=X.shape
            preds=torch.empty(batch_size, 0, height, width).to(device=device)
            input_X = X[:,0:input_num]
            for t in range(input_num,total_num):
                with torch.no_grad():
                    pred = self.net(input_X)
                input_X=torch.cat((input_X[:,1:],pred),dim=1) # use output image to pred next frame
                preds=torch.cat((preds,pred),dim=1) 
            X, preds = X.to(device="cpu"), preds.to(device="cpu")
            save_gif(X[0], preds[0], save_path = f"pred_{phase}_{epoch}.gif", suptitle=f"{phase}_{epoch}")
            self.log_artifact(f"pred_{phase}_{epoch}.gif")
