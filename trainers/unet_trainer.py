# -*- coding: utf-8 -*-
"""Unet Trainer"""

import logging
import torch.nn as nn
import torch
from torch import optim
import os
import numpy as np
from trainers.base_trainer import BaseTrainer
from torch.utils.data import DataLoader,random_split
from data.dataset import MnistDataset

log = logging.getLogger(__name__)


class UnetTrainer(BaseTrainer):
    """DefaultTrainer
    
    Attributes:
        self.cfg: Config of project.
        model: Model.
    
    """

    def __init__(self, cfg: object) -> None:
        """Initialization
    
        Args:
            self.cfg: Config of project.

        """
        dataset = MnistDataset(cfg)
        n_val = int(len(dataset) * cfg.train.val_percent)
        n_train = len(dataset) - n_val
        train, val = random_split(dataset, [n_train, n_val])
        self.train_loader = DataLoader(train, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers, pin_memory=True)
        self.val_loader = DataLoader(val, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True)
        
        super().__init__(cfg)


    def train(self,net) -> None:
        """Train

        Trains model.

        """

        super().train()
        epochs=self.cfg.train.epochs
        lr=self.cfg.train.lr
        img_scale=self.cfg.scale
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.to(device=device)

        global_step = 0

        logging.info('Starting training')

        optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)

        self.criterion = nn.MSELoss()

        for epoch in range(epochs):

            logging.info(f'epoch: {epoch}')
            net.train()

            epoch_loss = 0
            # train
            for X in self.train_loader:
                X = X.to(device=device, dtype=torch.float32)
                input_num=self.cfg.dataset.input_num
                total_num=X.shape[1]
                for t in range(input_num, total_num):
                    pred = net(X[:,t-input_num:t])
                    loss = self.criterion(pred, X[:,t])
                    epoch_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(net.parameters(), 0.1)
                    optimizer.step()
                global_step += 1
            loss_ave = epoch_loss / len(self.train_loader) # average per batch
            self.loss["train"].append(loss_ave) 
            logging.info(f'Train MSE     : {loss_ave}')
            # val
            if epoch % 1 == 0: # checkpoint interval
                val_score = self.eval(net)
                scheduler.step(val_score)
                self.log_metrics(epoch)

            # save model if it performes better than it has done before
            if epoch==0:
                torch.save(net.state_dict(),self.cfg.train.ckpt_path)
            elif self.loss["val"][-1]<min(self.loss["val"][:-1]):
                torch.save(net.state_dict(),self.cfg.train.ckpt_path)
                logging.info(f'Checkpoint saved !')
        self.log_artifacts()



    def eval(self,net) -> float:
        super().eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.to(device=device)
        net.eval()
        epoch_loss = 0 

        for X in self.val_loader:
            input_num=self.cfg.dataset.input_num
            total_num=X.shape[1]
            for t in range(input_num,total_num):
                with torch.no_grad():
                    pred = net(X[:,t-input_num:t])
                loss = self.criterion(pred, X[:,t])
                epoch_loss += loss.item()


        loss_ave=epoch_loss/len(self.val_loader)
        self.loss["val"].append(loss_ave)
        logging.info(f'Validation MSE: {loss_ave}')
        return loss_ave