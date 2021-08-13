from torch import nn
from models.unet import UNet
import torch.nn.functional as F
import torch
import unittest
import numpy as np
class STMoE(nn.Module):
    """STMoE Network

    """
    def __init__(self,input_num=4, n_channels=1, n_expert=2):
        super(STMoE, self).__init__()
        self.n_expert = n_expert
        self.n_channels = n_channels

        self.gating=UNet(n_channels=input_num,n_classes=n_expert)
        self.expert1=UNet(n_channels=input_num,n_classes=n_channels)
        self.expert2=UNet(n_channels=input_num,n_classes=n_channels)



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
        gating_weight = self.gating(x)
        gating_weight = F.softmax(gating_weight, dim=1)
        pred = gating_weight*torch.cat([pred1,pred2],axis = 1)
        pred = torch.sum(pred,dim=1)
        pred = pred[:,np.newaxis,:,:] # add channel axis
        return pred, gating_weight
    

class Test(unittest.TestCase):
    def test(self):
        net=STMoE()
        n_expert=2
        batch, input_num, h ,w = 1, 4, 128, 128
        input = torch.rand(batch, input_num, h ,w)
        pred, weight = net(input)
        self.assertEqual(list(pred.shape),[batch, 1 ,h ,w])
        self.assertEqual(list(weight.shape),[batch,n_expert ,h ,w])



if __name__=="__main__":
    unittest.main()