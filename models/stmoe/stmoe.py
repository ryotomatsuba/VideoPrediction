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
        pred1 = self.expert1(x)
        pred2 = self.expert2(x)
        # when training expert, gating weight is fixed to 0 or 1
        batch, input_num, h ,w = x.shape
        ones=torch.ones((batch,1, h ,w))
        zeros=torch.zeros(batch,1, h ,w)
        if self.train_model=="expert1":
            gating_weight = torch.cat([ones,zeros],axis = 1)
        elif self.train_model=="expert2":
            gating_weight = torch.cat([zeros,ones],axis = 1)
        else:
            gating_weight = self.gating(x)
            gating_weight = F.softmax(gating_weight, dim=1)
        # set gating_weight on gpu 
        gating_weight = gating_weight.to(x.device)
        pred = gating_weight*torch.cat([pred1,pred2],axis = 1)
        pred = torch.sum(pred,dim=1)
        pred = pred[:,np.newaxis,:,:] # add channel axis
        return pred, gating_weight
    
from utils.draw import save_weight_gif
class Test(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        self.batch, input_num, self.h ,self.w = 2, 4, 128, 128
        self.n_expert=2
        self.input = torch.rand(self.batch, input_num, self.h ,self.w)
        super().__init__(methodName=methodName)

    def test_all(self):
        net=STMoE(train_model="all")
        pred, weight = net(self.input)
        self.check_shape(pred, weight)
        self.save_gif(pred,weight)

    def test_gating(self):
        net=STMoE(train_model="gating")
        pred, weight = net(self.input)
        self.check_shape(pred, weight)
        self.save_gif(pred,weight)

    def test_expert_training(self):
        net=STMoE(train_model="expert2")
        pred, weight = net(self.input)
        summary(net,(4,128,128),device="cpu")
        self.check_shape(pred, weight)
        self.save_gif(pred,weight,save_name="expert.gif")
    
    def check_shape(self,pred,weight):
        self.assertEqual(list(pred.shape),[self.batch, 1 ,self.h ,self.w])
        self.assertEqual(list(weight.shape),[self.batch,self.n_expert ,self.h ,self.w])
    
    def save_gif(self,pred,weight,save_name="result.gif"):
        preds=torch.cat((pred,pred),dim=1) 
        weights=torch.cat((weight[:,np.newaxis],weight[:,np.newaxis]),dim=1) 
        save_weight_gif(preds[0],weights[0],save_name)


           





if __name__=="__main__":
    unittest.main()