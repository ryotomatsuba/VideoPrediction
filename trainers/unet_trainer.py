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
from torch.utils.tensorboard import SummaryWriter

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
        data=np.load("/home/data/ryoto/Datasets/mnist/dev_data_10.npy")
        dataset = MnistDataset(data)
        n_val = int(len(dataset) * cfg.train.val_percent)
        n_train = len(dataset) - n_val
        train, val = random_split(dataset, [n_train, n_val])
        self.train_loader = DataLoader(train, batch_size=cfg.train.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        self.val_loader = DataLoader(val, batch_size=cfg.train.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
        
        super().__init__(cfg)


    def train(self,net) -> None:
        """Train

        Trains model.

        """

        super().train()
        dir_checkpoint = 'checkpoints/'
        save_cp=True
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
            for X,Y in self.train_loader:
                assert X.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {X.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                X = X.to(device=device, dtype=torch.float32)
                Y = Y.to(device=device, dtype=torch.float32)

                pred = net(X)
                loss = self.criterion(pred, Y)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                global_step += 1
            if epoch % 1 == 0: # checkpoint interval
                val_score = self.eval(net)
                scheduler.step(val_score)


                logging.info(f'Validation MSE: {val_score}')


                # writer.add_images('images', X, global_step)
                if net.n_classes == 1:
                    pass
            if save_cp:
                try:
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(net.state_dict(),
                        dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')




    def eval(self,net) -> float:
        super().eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.to(device=device)
        net.eval()
        mask_type = torch.float32 if net.n_classes == 1 else torch.long
        n_val = len(self.val_loader)  # the number of batch
        losses = [] 

        for X,Y in self.val_loader:
            imgs, truth = X,Y
            imgs = imgs.to(device=device, dtype=torch.float32)
            truth = truth.to(device=device, dtype=mask_type)

            with torch.no_grad():
                pred = net(imgs)
            loss = self.criterion(pred, truth)
            losses.append(loss.item())
        loss_ave=sum(losses)/len(losses)
        return loss_ave