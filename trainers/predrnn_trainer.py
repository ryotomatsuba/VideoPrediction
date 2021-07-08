# -*- coding: utf-8 -*-
"""Default Trainer"""

import logging
from models.predrnn import PredRNN
import torch
from data.dataset import MnistDataset
from trainers.base_trainer import BaseTrainer
from utils.draw import save_gif
from torch.utils.data import DataLoader,random_split
import torch.nn as nn
from torch import optim
from models.predrnn.utils import preprocess
log = logging.getLogger(__name__)
import numpy as np

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
        # define dataset
        dataset = MnistDataset(cfg)
        n_val = int(len(dataset) * cfg.train.val_percent)
        n_train = len(dataset) - n_val
        train, val = random_split(dataset, [n_train, n_val])
        self.train_loader = DataLoader(train, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers, pin_memory=True)
        self.val_loader = DataLoader(val, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True)
        # define model
        device = torch.device(cfg.train.device)
        net = PredRNN(cfg)
        net.to(device=device)
        logging.info('Network Ready')

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
        device = next(self.net.parameters()).device


        iteration = 0

        logging.info('Starting training')

        optimizer = optim.Adam(self.net.parameters(), lr=lr,)
        for epoch in range(epochs):

            logging.info(f'epoch: {epoch}')
            self.net.train()

            epoch_loss = 0
            eta = self.cfg.sampling.sampling_start_value
            # train
            for X in self.train_loader:
                X=preprocess.reshape_patch(X, self.cfg.model.patch_size)
                X = X.to(device=device, dtype=torch.float32)
                if self.cfg.sampling.reverse_scheduled_sampling == 1:
                    real_input_flag = self.reserve_schedule_sampling_exp(iteration)
                else:
                    eta, real_input_flag = self.schedule_sampling(eta, iteration)

                mask_tensor = torch.FloatTensor(real_input_flag).to(device)
                optimizer.zero_grad()
                next_frames, loss = self.net(X, mask_tensor)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                iteration += 1
            loss_ave = epoch_loss / len(self.train_loader) # average per batch
            self.loss["train"].append(loss_ave) 
            logging.info(f'Train MSE     : {loss_ave}')
            # val
            if epoch % 1 == 0: # checkpoint interval
                val_score = self.eval()
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
            input_num=self.cfg.model.input_num
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
            input_num = self.cfg.model.input_num
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

    def reserve_schedule_sampling_exp(self, itr):
        if itr < self.cfg.sampling.r_sampling_step_1:
            r_eta = 0.5
        elif itr < self.cfg.sampling.r_sampling_step_2:
            r_eta = 1.0 - 0.5 * math.exp(-float(itr - self.cfg.sampling.r_sampling_step_1) / args.r_exp_alpha)
        else:
            r_eta = 1.0

        if itr < self.cfg.sampling.r_sampling_step_1:
            eta = 0.5
        elif itr < self.cfg.sampling.r_sampling_step_2:
            eta = 0.5 - (0.5 / (self.cfg.sampling.r_sampling_step_2 - self.cfg.sampling.r_sampling_step_1)) * (itr - self.cfg.sampling.r_sampling_step_1)
        else:
            eta = 0.0

        r_random_flip = np.random.random_sample(
            (self.cfg.train.batch_size, self.cfg.dataset.len_seq - 1))
        r_true_token = (r_random_flip < r_eta)

        random_flip = np.random.random_sample(
            (self.cfg.train.batch_size, self.cfg.dataset.num_data - self.cfg.dataset.len_seq - 1))
        true_token = (random_flip < eta)

        ones = np.ones((self.cfg.dataset.img_width // self.cfg.model.patch_size,
                        self.cfg.dataset.img_width // self.cfg.model.patch_size,
                        self.cfg.model.patch_size ** 2 * self.cfg.dataset.img_channel))
        zeros = np.zeros((self.cfg.dataset.img_width // self.cfg.model.patch_size,
                        self.cfg.dataset.img_width // self.cfg.model.patch_size,
                        self.cfg.model.patch_size ** 2 * self.cfg.dataset.img_channel))

        real_input_flag = []
        for i in range(self.cfg.train.batch_size):
            for j in range(self.cfg.dataset.num_data - 2):
                if j < self.cfg.dataset.len_seq - 1:
                    if r_true_token[i, j]:
                        real_input_flag.append(ones)
                    else:
                        real_input_flag.append(zeros)
                else:
                    if true_token[i, j - (self.cfg.dataset.len_seq - 1)]:
                        real_input_flag.append(ones)
                    else:
                        real_input_flag.append(zeros)

        real_input_flag = np.array(real_input_flag)
        real_input_flag = np.reshape(real_input_flag,
                                    (self.cfg.train.batch_size,
                                    self.cfg.dataset.num_data - 2,
                                    self.cfg.dataset.img_width // self.cfg.model.patch_size,
                                    self.cfg.dataset.img_width // self.cfg.model.patch_size,
                                    self.cfg.model.patch_size ** 2 * self.cfg.dataset.img_channel))
        return real_input_flag


    def schedule_sampling(self, eta, itr):
        zeros = np.zeros((self.cfg.train.batch_size,
                        self.cfg.dataset.num_data - self.cfg.dataset.len_seq - 1,
                        self.cfg.dataset.img_width // self.cfg.model.patch_size,
                        self.cfg.dataset.img_width // self.cfg.model.patch_size,
                        self.cfg.model.patch_size ** 2 * self.cfg.dataset.img_channel))
        if not self.cfg.sampling.scheduled_sampling:
            return 0.0, zeros

        if itr < self.cfg.sampling.sampling_stop_iter:
            eta -= self.cfg.sampling.sampling_changing_rate
        else:
            eta = 0.0
        random_flip = np.random.random_sample(
            (self.cfg.train.batch_size, self.cfg.dataset.num_data - self.cfg.dataset.len_seq - 1))
        true_token = (random_flip < eta)
        ones = np.ones((self.cfg.dataset.img_width // self.cfg.model.patch_size,
                        self.cfg.dataset.img_width // self.cfg.model.patch_size,
                        self.cfg.model.patch_size ** 2 * self.cfg.dataset.img_channel))
        zeros = np.zeros((self.cfg.dataset.img_width // self.cfg.model.patch_size,
                        self.cfg.dataset.img_width // self.cfg.model.patch_size,
                        self.cfg.model.patch_size ** 2 * self.cfg.dataset.img_channel))
        real_input_flag = []
        for i in range(self.cfg.train.batch_size):
            for j in range(self.cfg.dataset.num_data - self.cfg.dataset.len_seq - 1):
                if true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
        real_input_flag = np.array(real_input_flag)
        real_input_flag = np.reshape(real_input_flag,
                                    (self.cfg.train.batch_size,
                                    self.cfg.dataset.num_data - self.cfg.dataset.len_seq - 1,
                                    self.cfg.dataset.img_width // self.cfg.model.patch_size,
                                    self.cfg.dataset.img_width // self.cfg.model.patch_size,
                                    self.cfg.model.patch_size ** 2 * self.cfg.dataset.img_channel))
        return eta, real_input_flag
