__author__ = 'yunbo'

import torch
import torch.nn as nn
from models.predrnn.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell
from models.predrnn.utils import preprocess
from torchsummary import summary
import numpy as np


class PredRNN(nn.Module):
    def __init__(self,input_num=4,total_length=10,img_channel=1,img_width=128,patch_size=4,
            num_hidden=[64,64,64,64],filter_size=5,stride=1,layer_norm=1):
        super(PredRNN, self).__init__()
        self.input_num = input_num
        self.total_length = total_length
        self.patch_size = patch_size
        self.frame_channel = patch_size * patch_size * img_channel
        self.num_hidden = num_hidden
        self.reverse_scheduled_sampling=0
        self.num_layers = len(self.num_hidden)
        cell_list = []


        width = img_width // patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(self.num_layers):
            in_channel = self.frame_channel if i == 0 else self.num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, self.num_hidden[i], width, filter_size,
                                       stride, layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(self.num_hidden[self.num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true=None):
        device=frames_tensor.device
        frames_tensor=preprocess.reshape_patch(frames_tensor, self.patch_size)
        frames_tensor = torch.tensor(frames_tensor, device=device, dtype=torch.float32)

        if mask_true is None:
            real_input_flag = np.ones(frames_tensor.shape)
            mask_true = torch.tensor(real_input_flag,device=device,dtype=torch.float32)
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
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(device)

        for t in range(self.total_length - 1):
            # reverse schedule sampling
            if self.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.input_num:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.input_num] * frames[:, t] + \
                          (1 - mask_true[:, t - self.input_num]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        next_frames = preprocess.reshape_patch_back(next_frames, self.patch_size)

        return next_frames, 0

if __name__=="__main__":
    batch, total_length, h ,w= 1, 10, 128, 128
    net=PredRNN()
    summary(net,(total_length, h ,w, 1), device="cpu")
    input = torch.rand(batch, total_length, h ,w)
    output = net(input)