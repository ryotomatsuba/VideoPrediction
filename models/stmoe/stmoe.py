from torch import nn
from models.unet import UNet
import torch.nn.functional as F
import torch
import unittest
import numpy as np
class STMoE(nn.Module):
    """STMoE Network
    Params:
        input_num: number of frames to input
        n_channels: image channels. if color image, set this 3
        n_expert: number of expert
        training: which model to train expert1, expert2, gating or all

    """
    def __init__(self,input_num=4, n_channels=1, n_expert=2, training="all"):
        super(STMoE, self).__init__()
        self.n_expert = n_expert
        self.n_channels = n_channels
        assert training in ["expert1", "expert2", "gating","all"]
        self.training = training
        # define expert
        self.expert1=UNet(n_channels=input_num,n_classes=n_channels)
        self.expert2=UNet(n_channels=input_num,n_classes=n_channels)
        # define gating
        self.gating=UNet(n_channels=input_num,n_classes=n_expert)
        # fix expert parameters
        if training=="gating":
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
        pred1 = self.expert1(x)
        pred2 = self.expert2(x)
        # when training expert, gating weight is fixed to 0 or 1
        batch, input_num, h ,w = x.shape
        ones=torch.ones((batch,1, h ,w))
        zeros=torch.zeros(batch,1, h ,w)
        if self.training=="expert1":
            gating_weight = torch.cat([ones,zeros],axis = 1)
        if self.training=="expert2":
            gating_weight = torch.cat([zeros,ones],axis = 1)
        else:
            gating_weight = self.gating(x)
            gating_weight = F.softmax(gating_weight, dim=1)
        pred = gating_weight*torch.cat([pred1,pred2],axis = 1)
        pred = torch.sum(pred,dim=1)
        pred = pred[:,np.newaxis,:,:] # add channel axis
        return pred, gating_weight
    

class Test(unittest.TestCase):
    def test(self):
        for training in ["all","expert1","expert2","gating"]:
            self._test_training_param(training)
    
    def _test_training_param(self,training="all"):
        net=STMoE(training=training)
        n_expert=2
        batch, input_num, h ,w = 1, 4, 128, 128
        input = torch.rand(batch, input_num, h ,w)
        pred, weight = net(input)
        self.assertEqual(list(pred.shape),[batch, 1 ,h ,w])
        self.assertEqual(list(weight.shape),[batch,n_expert ,h ,w])




if __name__=="__main__":
    unittest.main()