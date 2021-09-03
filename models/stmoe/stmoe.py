from torch import nn
from models.unet import UNet
import torch.nn.functional as F
import torch
import unittest
import numpy as np
from torchsummary import summary
class STMoE(nn.Module):
    """STMoE Network
    Params:
        input_num: number of frames to input
        n_channels: image channels. if color image, set this 3
        n_expert: number of expert
        train_model: which model to train expert1, expert2, gating or all

    """
    def __init__(self,input_num=4, n_channels=1, n_expert=2, train_model="all"):
        super(STMoE, self).__init__()
        self.n_expert = n_expert
        self.n_channels = n_channels
        assert train_model in ["expert1", "expert2", "gating","all"]
        self.train_model = train_model
        # define expert
        self.expert1=UNet(n_channels=input_num,n_classes=n_channels)
        self.expert2=UNet(n_channels=input_num,n_classes=n_channels)
        # define gating
        self.gating=UNet(n_channels=input_num,n_classes=n_expert)
        # fix expert parameters
        if train_model=="gating":
            self.fix_experts()

    def fix_experts(self):
        """fix experts weight"""
        for param in self.expert1.parameters():
            param.requires_grad = False
        for param in self.expert2.parameters():
            param.requires_grad = False



    def forward(self, x):
        """
        Params:
            x: Tensor(batch_size, input_num, height, width)
        Returns: 
            pred: next frame prediction. Tensor(batch_size, ch, height, width)
            weight: gating weight. Tensor(batch_size, n_expert, height, width)
        """
        # when training expert, gating weight is fixed to 0 or 1
        batch, input_num, h ,w = x.shape
        ones=torch.ones((batch,1, h ,w))
        zeros=torch.zeros(batch,1, h ,w)
        if self.train_model=="expert1":
            pred1 = self.expert1(x)
            gating_weight = torch.cat([ones,zeros],axis = 1).to(device=x.device)
            return pred1, gating_weight
        elif self.train_model=="expert2":
            pred2 = self.expert2(x)
            gating_weight = torch.cat([zeros,ones],axis = 1).to(device=x.device)
            return pred2, gating_weight
        elif self.train_model in ["gating","all"]:
            pred1 = self.expert1(x)
            pred2 = self.expert2(x)
            gating_weight = self.gating(x)
            gating_weight = F.softmax(gating_weight, dim=1)
             # set gating_weight on gpu 
            pred = gating_weight*torch.cat([pred1,pred2],axis = 1)
            pred = torch.sum(pred,dim=1)
            pred = pred[:,np.newaxis,:,:] # add channel axis
            return pred, gating_weight
        else:
            raise ValueError("train_model is not correct")
    



if __name__=="__main__":
    model=STMoE(input_num=4, n_channels=1, n_expert=2, train_model="all")
    summary(model,(4,128,128),device="cpu")