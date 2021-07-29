__author__ = 'yunbo'

import torch
import torch.nn as nn
from .SpatioTemporalLSTMCell import SpatioTemporalLSTMCell


class PredRNN(nn.Module):
    def __init__(self,cfg):
        super(PredRNN, self).__init__()

        self.cfg = cfg
        self.frame_channel = cfg.model.patch_size * cfg.model.patch_size * cfg.dataset.img_channel
        self.num_hidden = cfg.model.num_hidden
        self.num_layers = len(self.num_hidden)
        cell_list = []

        width = cfg.dataset.img_width // cfg.model.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(self.num_layers):
            in_channel = self.frame_channel if i == 0 else self.num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, self.num_hidden[i], width, cfg.model.filter_size,
                                       cfg.model.stride, cfg.model.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(self.num_hidden[self.num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.cfg.train.device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.cfg.train.device)

        for t in range(self.cfg.dataset.num_frames - 1):
            # reverse schedule sampling
            if self.cfg.sampling.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.cfg.model.input_num:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.cfg.model.input_num] * frames[:, t] + \
                          (1 - mask_true[:, t - self.cfg.model.input_num]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        return next_frames, loss

if __name__=="__main__":
    pass