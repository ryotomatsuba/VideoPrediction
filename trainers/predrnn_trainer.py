# -*- coding: utf-8 -*-
"""PredRNN Trainer"""

import logging
from models.predrnn import PredRNN
from models.predrnn.utils.sampling import schedule_sampling
import torch
from trainers.base_trainer import BaseTrainer
import torch.nn as nn
from torch import optim
log = logging.getLogger(__name__)
import numpy as np
import math
from tqdm import tqdm

class PredRNNTrainer(BaseTrainer):
    """DefaultTrainer
    
    Attributes:
        cfg: Config of project.
        model: Model.
    
    """

    def __init__(self, cfg: object) -> None:
        """Initialization
    
        Args:
            self.cfg: Config of project.

        """
        self.net = PredRNN(
            input_num=cfg.model.input_num,total_length=cfg.dataset.num_frames,
            img_channel=cfg.dataset.img_channel,img_width=cfg.dataset.img_width,
            patch_size=cfg.model.patch_size,num_hidden=cfg.model.num_hidden ,
            filter_size=cfg.model.filter_size,stride=cfg.model.stride,layer_norm=cfg.model.layer_norm) # define model
        logging.info('PredRNN Network Ready')
        super().__init__(cfg)

    def train(self) -> None:
        """Train

        Trains model.

        """

        super().train()
        epochs=self.cfg.train.epochs
        lr=self.cfg.train.lr
        iteration = 0

        logging.info('Starting training')

        optimizer = optim.Adam(self.net.parameters(), lr=lr,)
        self.criterion = nn.MSELoss()
        for epoch in range(epochs):

            logging.info(f'epoch: {epoch}')
            self.net.train()

            epoch_loss = 0
            eta = self.cfg.sampling.sampling_start_value
            # train
            for X in tqdm(self.train_loader, ncols=100):
                X = X.to(device=self.device, dtype=torch.float32)
                eta, mask_tensor = schedule_sampling(eta,iteration,batch=X.shape[0],input_num=4)
                self.net.set_mask(mask_tensor)
                optimizer.zero_grad()
                next_frames = self.net(X)
                loss = self.criterion(next_frames, X[:, self.cfg.model.input_num:])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() # loss per batch
                iteration += 1
            loss_ave = epoch_loss / len(self.train_loader) # average per batch
            self.loss["train"].append(loss_ave) 
            logging.info(f'Train MSE     : {loss_ave}')
            # val
            if epoch % 1 == 0: # checkpoint interval
                self.eval()
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

        self.net.eval()
        epoch_loss = 0 
        mask_tensor = torch.zeros((self.cfg.train.batch_size,self.cfg.dataset.num_frames - self.cfg.model.input_num - 1, 1, 1, 1))
        self.net.set_mask(mask_tensor)
        if self.cfg.sampling.reverse_scheduled_sampling == 1:
            mask_tensor[:, :self.cfg.model.input_num - 1, :, :] = 1.0
        for X in tqdm(self.val_loader, ncols=100):    
            X = torch.FloatTensor(X).to(self.device)
            with torch.no_grad():
                next_frames = self.net(X)
                loss = self.criterion(next_frames, X[:, self.cfg.model.input_num:])
            epoch_loss += loss.item() # loss per batch
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
        mask_true = torch.zeros((self.cfg.train.batch_size,self.cfg.dataset.num_frames - self.cfg.model.input_num - 1, 1, 1, 1))
        if self.cfg.sampling.reverse_scheduled_sampling == 1:
            mask_true[:, :self.cfg.model.input_num - 1, :, :] = 1.0
        self.net.set_mask(mask_true)

        for phase in ["train", "val"]:
            data_loader = self.train_loader if phase == "train" else self.val_loader
            X = iter(data_loader).__next__()
            X = torch.FloatTensor(X).to(self.device)
            with torch.no_grad():
                img_gen = self.net(X)
            img_gen = img_gen.detach().cpu().numpy() 
            output_length = self.cfg.dataset.num_frames - self.cfg.model.input_num
            pred = img_gen[:, -output_length:]
            super().save_gif(X, pred, epoch, phase)
            
