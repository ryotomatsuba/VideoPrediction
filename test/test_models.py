import unittest
import torch
from torch.autograd import grad
from models.unet import UNet
from models.predrnn import PredRNN
from models.stmoe import STMoE
from utils.draw import save_weight_gif
import numpy as np
from models.predrnn.utils import preprocess


batch, total_length, input_frame, h ,w, c= 2, 10, 4, 128, 128, 1
input = torch.rand(batch, input_frame, h ,w)

class UnetTest(unittest.TestCase):
    def test_unet(self):
        net=UNet(n_channels=input_frame, n_classes=1)
        output = net(input)
        self.assertEqual(list(output.shape),[batch, 1 ,h ,w])
class PredRNNTest(unittest.TestCase):    
    def test_predrnn(self):
        net=PredRNN(input_num=input_frame,total_length=total_length)
        output = net(input)
        self.assertEqual(list(output.shape),[batch, 1, h ,w])
    
    def test_reshape_patch(self):
        input = torch.rand(batch, total_length, h ,w)
        reshaped_tensor = preprocess.reshape_patch(input, 4)
        reshaped_tensor = preprocess.reshape_patch_back(reshaped_tensor,4) 
        self.assertTrue(torch.equal(input, reshaped_tensor))

    def test_reshaped_patch_grad(self):
        patch_size=4
        input = torch.rand(batch, total_length, h ,w)
        true = torch.rand(batch, total_length, h ,w)
        criterion=torch.nn.MSELoss()
        weight = torch.rand(batch, total_length,h//patch_size,w//patch_size,patch_size*patch_size*c,requires_grad=True)
        optim=torch.optim.Adam([weight], lr=0.01)
        reshaped_input = preprocess.reshape_patch(input, patch_size)
        pred_patch=reshaped_input*weight
        # calc loss after reshape
        true_patch=preprocess.reshape_patch(true, patch_size)
        loss=criterion(pred_patch,true_patch)
        loss.backward(retain_graph=True)
        weight_grad_after_reshape=weight.grad.clone()
        optim.zero_grad()
        # calc loss before reshape
        pred=preprocess.reshape_patch_back(pred_patch, patch_size)
        loss=criterion(pred,true)
        loss.backward()
        weight_grad_before_reshape=weight.grad.clone()
        self.assertTrue(torch.equal(weight_grad_before_reshape, weight_grad_after_reshape))

class STMoETest(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        self.n_expert=2
        super().__init__(methodName=methodName)

    def test_all(self):
        net=STMoE(train_model="all")
        print(net.state_dict().keys())
        pred, weight = net(input)
        self.check_shape(pred, weight)
        self.save_gif(pred,weight)

    def test_gating(self):
        net=STMoE(train_model="gating")
        pred, weight = net(input)
        self.check_shape(pred, weight)
        self.save_gif(pred,weight)

    def test_expert(self):
        net=STMoE(train_model="expert2")
        print(net.state_dict().keys())
        pred, weight = net(input)
        self.check_shape(pred, weight)
        self.save_gif(pred,weight,save_name="expert.gif")
    
    def check_shape(self,pred,weight):
        self.assertEqual(list(pred.shape),[batch, 1 ,h ,w])
        self.assertEqual(list(weight.shape),[batch,self.n_expert ,h ,w])
    
    def save_gif(self,pred,weight,save_name="result.gif"):
        preds=torch.cat((pred,pred),dim=1) 
        weights=torch.cat((weight[:,np.newaxis],weight[:,np.newaxis]),dim=1) 
        save_weight_gif(preds[0],weights[0],save_name)


if __name__=="__main__":
    unittest.main()