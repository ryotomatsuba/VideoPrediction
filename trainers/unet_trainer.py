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
        logging.info(f'Using device {device}')
        net.to(device=device)

        global_step = 0

        logging.info('Starting training')

        optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
        if net.n_classes > 1:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()

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
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                Y = Y.to(device=device, dtype=mask_type)

                masks_pred = net(X)
                loss = criterion(masks_pred, Y)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                global_step += 1
            if epoch % 1 == 0: # checkpoint interval
                for tag, value in net.named_parameters():
                    tag = tag.replace('.', '/')
                val_score = 0
                scheduler.step(val_score)

                if net.n_classes > 1:
                    logging.info('Validation cross entropy: {}'.format(val_score))
                else:
                    logging.info('Validation Dice Coeff: {}'.format(val_score))

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
        with torch.no_grad():
            pass
        return 